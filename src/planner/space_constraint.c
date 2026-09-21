/*
 * This file and its contents are licensed under the Apache License 2.0.
 * Please see the included NOTICE for copyright information and
 * LICENSE-APACHE for a copy of the license.
 */

#include <postgres.h>
#include <access/xact.h>
#include <datatype/timestamp.h>
#include <nodes/makefuncs.h>
#include <nodes/nodeFuncs.h>
#include <nodes/pg_list.h>
#include <optimizer/optimizer.h>
#include <parser/parse_coerce.h>
#include <parser/parse_func.h>
#include <parser/parsetree.h>
#include <utils/fmgroids.h>
#include <utils/lsyscache.h>
#include <utils/typcache.h>

#include "cache.h"
#include "dimension.h"
#include "expression_utils.h"
#include "hypertable.h"
#include "hypertable_cache.h"
#include "partitioning.h"
#include "planner.h"

/*
 * Returns space dimension for a specific column. Returns NULL
 * if the column is not a space dimension.
 */
static const Dimension *
get_space_dimension(Oid relid, AttrNumber varattno)
{
	Hypertable *ht = ts_planner_get_hypertable(relid, CACHE_FLAG_CHECK);
	if (!ht)
	{
		return NULL;
	}

	return ts_hyperspace_get_dimension_by_attno(ht->space, DIMENSION_TYPE_CLOSED, varattno);
}

/*
 * Check if this operator is compatible with the constraints on the space
 * dimension. This is the equality of the btree operator family of the type.
 *
 * The type of the other operand does not matter, the family also covers
 * comparisons between different types, e.g. an int2 column against an integer
 * literal which defaults to int4.
 */
bool
ts_is_equality_operator(Oid opno, Oid lefttype)
{
	TypeCacheEntry *tce = lookup_type_cache(lefttype, TYPECACHE_BTREE_OPFAMILY);

	return get_op_opfamily_strategy(opno, tce->btree_opf) == BTEqualStrategyNumber;
}

/*
 * Check whether one side of an equality is a space partitioning column that can
 * be compared against the other side.
 *
 * relid restricts the check to the columns of that relation, 0 accepts any
 * relation of the range table. Returns the dimension of the column and sets var
 * to the column itself.
 */
static const Dimension *
space_partitioning_column(PlannerInfo *root, List *rtable, Index relid, Node *column, Node *other,
						  Var **var)
{
	/*
	 * A comparison between binary coercible types puts a relabel around the
	 * column, for example when a varchar column is compared against text.
	 */
	column = ts_strip_relabel_types(column);

	if (!IsA(column, Var))
	{
		return NULL;
	}

	Var *candidate = castNode(Var, column);

	if (candidate->varlevelsup != 0 || (relid != 0 && (Index) candidate->varno != relid))
	{
		return NULL;
	}

	/*
	 * The value has to come from outside the hypertable, e.g. a constant, a
	 * parameter of a prepared statement or a column of an outer relation of a
	 * join. When this runs before the outer references have been turned into
	 * parameters they are still plain Vars, so we only reject the columns of
	 * the hypertable itself.
	 */
	if (bms_is_member(candidate->varno, pull_varnos(root, other)))
	{
		return NULL;
	}

	Assert((int) candidate->varno <= list_length(rtable));
	RangeTblEntry *rte = rt_fetch(candidate->varno, rtable);
	const Dimension *dim = get_space_dimension(rte->relid, candidate->varattno);

	if (dim == NULL)
	{
		return NULL;
	}

	*var = candidate;

	return dim;
}

static FuncExpr *
make_partfunc_call(const Dimension *dim, Node *arg, Oid inputcollid)
{
	/* build FuncExpr to use in eval_const_expressions */
	return makeFuncExpr(dim->partitioning->partfunc.func_fmgr.fn_oid /* funcid */,
						dim->partitioning->partfunc.rettype /* rettype */,
						list_make1(arg) /* args */,
						InvalidOid /* funccollid */,
						inputcollid /* inputcollid */,
						COERCE_EXPLICIT_CALL /* fformat */);
}

/*
 * Check whether values of the two types are hashed the same way. All types of a
 * hash operator family hash equal values to the same value, which is what makes
 * them usable for hash joins between those types.
 */
static bool
hashes_match(Oid type1, Oid type2)
{
	TypeCacheEntry *tce1 = lookup_type_cache(type1, TYPECACHE_HASH_OPFAMILY);
	TypeCacheEntry *tce2 = lookup_type_cache(type2, TYPECACHE_HASH_OPFAMILY);

	return OidIsValid(tce1->hash_opf) && tce1->hash_opf == tce2->hash_opf;
}

/*
 * Apply the partitioning function to a value of the equality. The result is
 * folded when the value is already known, so it is not computed again for every
 * chunk whenever the plan runs.
 *
 * Returns NULL when the value cannot be coerced to the column type.
 */
static Node *
make_value_hash(PlannerInfo *root, const Dimension *dim, Var *var, Node *value)
{
	value = copyObject(value);

	/*
	 * The partitioning function hashes the value as the type it has, so a value
	 * of another type is only usable when both types are hashed the same way.
	 * Otherwise it has to be coerced to the column type. We only use implicit
	 * coercions because narrowing casts can fail at runtime.
	 */
	if (exprType(value) != var->vartype && !hashes_match(exprType(value), var->vartype))
	{
		value = coerce_to_target_type(NULL,
									  value,
									  exprType(value),
									  var->vartype,
									  -1,
									  COERCION_IMPLICIT,
									  COERCE_IMPLICIT_CAST,
									  -1);

		if (value == NULL)
		{
			return NULL;
		}
	}

	return eval_const_expressions(root, (Node *) make_partfunc_call(dim, value, var->varcollid));
}

/*
 * Find the space partitioning column of an equality clause.
 *
 * The column can be on either side of the equality. On success the dimension of
 * the column is returned and var and value are set to the two sides.
 */
static const Dimension *
space_equality_operands(PlannerInfo *root, List *rtable, Index relid, Expr *clause, Var **var,
						Node **value)
{
	if (!IsA(clause, OpExpr) || list_length(castNode(OpExpr, clause)->args) != 2)
	{
		return NULL;
	}

	OpExpr *op = castNode(OpExpr, clause);
	Node *left = linitial(op->args);
	Node *right = lsecond(op->args);

	/* the types are looked up in operand order as the operator is not commuted */
	if (!ts_is_equality_operator(op->opno, exprType(left)) || !op_strict(op->opno))
	{
		return NULL;
	}

	/*
	 * Under a nondeterministic collation values that hash to different
	 * partitions can be equal, so the hash of the value would exclude chunks
	 * with matching rows.
	 */
	if (OidIsValid(op->inputcollid) && !get_collation_isdeterministic(op->inputcollid))
	{
		return NULL;
	}

	const Dimension *dim = space_partitioning_column(root, rtable, relid, left, right, var);

	if (dim != NULL)
	{
		*value = right;
		return dim;
	}

	dim = space_partitioning_column(root, rtable, relid, right, left, var);

	if (dim != NULL)
	{
		*value = left;
		return dim;
	}

	return NULL;
}

/*
 * Build a clause on the partition hash for an equality between a space
 * partitioning column and a value from outside the hypertable:
 *
 * device_id = $1 => get_partition_hash(device_id) = get_partition_hash($1)
 *
 * The left side matches the constraints on the chunks, so chunk exclusion can
 * use the clause as soon as the value is known.
 *
 * relid restricts the clause to the columns of that relation, 0 accepts any
 * relation of the range table.
 *
 * Returns NULL when no such clause can be built.
 */
Expr *
ts_make_partition_hash_clause(PlannerInfo *root, List *rtable, Index relid, Expr *clause)
{
	Var *var;
	Node *value;
	const Dimension *dim = space_equality_operands(root, rtable, relid, clause, &var, &value);

	if (dim == NULL)
	{
		return NULL;
	}

	Node *value_hash = make_value_hash(root, dim, var, value);

	if (value_hash == NULL)
	{
		return NULL;
	}

	TypeCacheEntry *tce = lookup_type_cache(dim->partitioning->partfunc.rettype, TYPECACHE_EQ_OPR);

	/*
	 * The call on the column has to match the chunk constraint, which applies
	 * the partitioning function to the bare column.
	 */
	FuncExpr *column_hash = make_partfunc_call(dim, (Node *) copyObject(var), var->varcollid);

	return make_opclause(tce->eq_opr,
						 BOOLOID,
						 false,
						 (Expr *) column_hash,
						 (Expr *) value_hash,
						 InvalidOid,
						 InvalidOid);
}

/*
 * Transform a constraint like: device_id = 1
 * into
 * ((device_id = 1) AND (_timescaledb_functions.get_partition_hash(device_id) = 242423622))
 *
 * The hash of the value has to be known here because this clause is matched
 * against the constraints of the chunks while planning. The clause is tagged so
 * it can be removed from the finished plan again.
 *
 * Returns NULL when the constraint cannot be transformed.
 */
static OpExpr *
transform_space_constraint(PlannerInfo *root, List *rtable, OpExpr *op)
{
	Expr *clause = ts_make_partition_hash_clause(root, rtable, 0, (Expr *) op);

	if (clause == NULL || !IsA(lsecond(castNode(OpExpr, clause)->args), Const))
	{
		return NULL;
	}

	castNode(OpExpr, clause)->location = PLANNER_LOCATION_MAGIC;

	return castNode(OpExpr, clause);
}

/*
 * Transforms a constraint like: s1 = ANY ('{s1_2,s1_2}'::text[])
 * into
 * ((s1 = ANY ('{s1_2,s1_2}'::text[])) AND (_timescaledb_functions.get_partition_hash(s1) = ANY
 * ('{1583420735,1583420735}'::integer[])))
 *
 * Returns NULL when the constraint cannot be transformed.
 */
static ScalarArrayOpExpr *
transform_scalar_space_constraint(PlannerInfo *root, List *rtable, ScalarArrayOpExpr *op)
{
	if (list_length(op->args) != 2 || !IsA(lsecond(op->args), ArrayExpr))
	{
		return NULL;
	}

	Node *column = linitial(op->args);
	ArrayExpr *arr = castNode(ArrayExpr, lsecond(op->args));

	if (arr->multidims || !op->useOr)
	{
		return NULL;
	}

	if (!ts_is_equality_operator(op->opno, exprType(column)) || !op_strict(op->opno))
	{
		return NULL;
	}

	/*
	 * Under a nondeterministic collation values that hash to different
	 * partitions can be equal, so the hash of the values would exclude chunks
	 * with matching rows.
	 */
	if (OidIsValid(op->inputcollid) && !get_collation_isdeterministic(op->inputcollid))
	{
		return NULL;
	}

	Var *var;
	const Dimension *dim = space_partitioning_column(root, rtable, 0, column, (Node *) arr, &var);

	if (dim == NULL)
	{
		return NULL;
	}

	List *part_values = NIL;
	ListCell *lc;

	foreach (lc, arr->elements)
	{
		/*
		 * We can skip NULL here as elements are ORed and partitioning dimensions
		 * have NOT NULL constraint.
		 */
		if (IsA(lfirst(lc), Const) && lfirst_node(Const, lc)->constisnull)
		{
			continue;
		}

		Node *value_hash = make_value_hash(root, dim, var, lfirst(lc));

		/*
		 * Every element has to be known here, an unknown one would exclude
		 * chunks that can have matching rows.
		 */
		if (value_hash == NULL || !IsA(value_hash, Const))
		{
			return NULL;
		}

		part_values = lappend(part_values, value_hash);
	}

	Oid rettype = dim->partitioning->partfunc.rettype;
	TypeCacheEntry *tce = lookup_type_cache(rettype, TYPECACHE_EQ_OPR);

	ArrayExpr *hashes = makeNode(ArrayExpr);
	hashes->array_collid = InvalidOid;
	hashes->array_typeid = get_array_type(rettype);
	hashes->element_typeid = rettype;
	hashes->multidims = false;
	hashes->location = -1;
	hashes->elements = part_values;

	ScalarArrayOpExpr *ret = makeNode(ScalarArrayOpExpr);
	ret->opno = tce->eq_opr;
	ret->args =
		list_make2(make_partfunc_call(dim, (Node *) copyObject(var), var->varcollid), hashes);
	ret->inputcollid = InvalidOid;
	ret->useOr = true;
	ret->location = PLANNER_LOCATION_MAGIC;

	return ret;
}

/*
 * Transform constraints for hash-based partitioning columns to make
 * them usable by postgres constraint exclusion.
 *
 * If we have an equality condition on a space partitioning column, we add
 * a corresponding condition on get_partition_hash on this column. These
 * conditions match the constraints on chunks, so postgres' constraint
 * exclusion is able to use them and exclude the chunks.
 *
 */
Node *
ts_add_space_constraints(PlannerInfo *root, List *rtable, Node *node)
{
	Assert(node);

	switch (nodeTag(node))
	{
		case T_ScalarArrayOpExpr:
		{
			ScalarArrayOpExpr *hash_op =
				transform_scalar_space_constraint(root, rtable, castNode(ScalarArrayOpExpr, node));

			if (hash_op)
			{
				return (Node *) makeBoolExpr(AND_EXPR, list_make2(node, hash_op), -1);
			}
			break;
		}
		case T_OpExpr:
		{
			OpExpr *hash_op = transform_space_constraint(root, rtable, castNode(OpExpr, node));

			if (hash_op)
			{
				return (Node *) makeBoolExpr(AND_EXPR, list_make2(node, hash_op), -1);
			}
			break;
		}
		case T_BoolExpr:
		{
			ListCell *lc;
			BoolExpr *be = castNode(BoolExpr, node);

			if (be->boolop == AND_EXPR)
			{
				List *additions = NIL;
				/*
				 * If this is a top-level AND we can just append our transformed constraints
				 * to the list of ANDed expressions.
				 */
				foreach (lc, be->args)
				{
					Expr *hash_clause = NULL;

					switch (nodeTag(lfirst(lc)))
					{
						case T_OpExpr:
							hash_clause =
								(Expr *) transform_space_constraint(root,
																	rtable,
																	lfirst_node(OpExpr, lc));
							break;
						case T_ScalarArrayOpExpr:
							hash_clause = (Expr *)
								transform_scalar_space_constraint(root,
																  rtable,
																  lfirst_node(ScalarArrayOpExpr,
																			  lc));
							break;
						default:
							break;
					}

					if (hash_clause)
					{
						additions = lappend(additions, hash_clause);
					}
				}

				if (additions)
				{
					be->args = list_concat(be->args, additions);
				}
			}
			break;
		}
		default:
			break;
	}

	return node;
}
