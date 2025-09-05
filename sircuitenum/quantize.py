__doc__ = "quantize.py: contains functions used to produce symbolic hamiltonians"
__author__ = "Eli Weissler"
__version__ = "0.1.0"
__all__ = ["gen_cap_mat", "gen_ind_mat", "gen_junc_pot", "quantize_circuit"]

import itertools
import functools
import time

from typing import Union, Sequence

import sympy as sym
import numpy as np
import networkx as nx

from sympy import collect, expand_mul, Mul, Dummy
from sympy.core.add import Add

from func_timeout import func_timeout, FunctionTimedOut

from sircuitenum import utils



PERIODIC_CHARGE = "n"
PERIODIC_PHASE = "θ"
EXTENDED_CHARGE = "q"
EXTENDED_PHASE = "φ"
NODE_CHARGE = "q"
NODE_PHASE = "ϕ"
EXT_CHARGE = "n_g"
EXT_PHASE = "_{ext}"


# Cache of solved equations
CACHE_VARS = sym.symbols(",".join([f'x{i}' for i in range(100)]), real=True)
CACHE_CONST = sym.symbols(",".join([f'c{i}' for i in range(100)]), real=True)
SOLVE_CACHE = {}

def _independent_from(A, B):
    """
    Returns the maximum set of vectors in A such that 
    their span has no overlap with the span of B.
    Assumes A and B are sets of linearly independent column vectors.

    Args:
        A (list[sym.Matrix] or sym.Matrix): Set of linearly independent column vectors.
        B (list[sym.Matrix] or sym.Matrix): Set of linearly independent column vectors.

    Returns:
        list[sym.Matrix]: Basis vectors in span(A) not in span(B).
    """
    # Convert to list of column vectors if needed
    if isinstance(A, sym.Matrix):
        A = [A[:, j] for j in range(A.shape[1])]
    if isinstance(B, sym.Matrix):
        B = [B[:, j] for j in range(B.shape[1])]

    # If B is empty, all of A is independent
    if not B:
        return [v for v in A]

    # Stack B and A, get linearly independent columns
    mat = sym.Matrix.hstack(*B, *A)
    li_cols = _linearly_indep_cols(mat)
    # Only keep those columns that come from A
    offset = len(B)
    result = []
    for idx in li_cols:
        if idx >= offset:
            result.append(mat[:, idx])
    return result


def _vec_space_overlap(vecs1:Union[list[sym.Matrix], sym.Matrix],
                       vecs2:Union[list[sym.Matrix], sym.Matrix],
                       support:Union[list[sym.Matrix], sym.Matrix] = [],
                       return_decomp:bool = False
                       ) -> list[sym.Matrix]:
    """
    Calculates the overlap of two linearly independent sets of
    vectors vecs1, vecs2 using NULL(vecs1, -vecs2)

    Args:
        vecs1 (Union[list[sym.Matrix], sym.Matrix]): set of column vectors
        vecs2 (Union[list[sym.Matrix], sym.Matrix]): set of column vectors
        support (Union[list[sym.Matrix], sym.Matrix]): set of vectors that can be
                                                       included to make the overlap work,
                                                       if used in addition to the vectors
                                                       from the individual sets

    Returns:
        list[sym.Matrix]: list of vectors that span the overlap space.
    """
    # Convert matrices to list
    if isinstance(vecs1, sym.Matrix):
        vecs1 = [vecs1[:, j] for j in range(vecs1.shape[1])]
    if isinstance(vecs2, sym.Matrix):
        vecs2 = [vecs2[:, j] for j in range(vecs2.shape[1])]
    if isinstance(support, sym.Matrix):
        support = [support[:, j] for j in range(support.shape[1])]

    # If either one is empty, return no overlap
    if len(vecs1) == 0 or len(vecs2) == 0:
        if return_decomp:
            if len(support) > 0:
                return [], [], [], []
            else:
                return [], [], []
        else:
            return []

    # Assert vector sets are linearly independent and same length
    try:
        assert len(_linearly_indep_cols(sym.Matrix.hstack(*vecs1))) == len(vecs1)
        assert len(_linearly_indep_cols(sym.Matrix.hstack(*vecs2))) == len(vecs2)
        assert len(_linearly_indep_cols(sym.Matrix.hstack(*support))) == len(support)
        assert vecs1[0].shape[0] == vecs2[0].shape[0]
        if len(support) > 0:
            assert support[0].shape[0] == vecs1[0].shape[0]
    except:
        breakpoint()
    # Calculate the overlap of the two vector spaces
    divide = (len(vecs1), len(support) + len(vecs1))
    ns = sym.Matrix.hstack(*[v for v in vecs1],
                           *[-v for v in support],
                           *[-v for v in vecs2]).nullspace()
    
    # Gather the entries
    in_v1 = []
    in_v2 = []
    in_sup = []
    overlap_vecs = []
    mat = sym.Matrix.hstack(*vecs1)
    for vec in ns:
        v1_entry = vec[:divide[0], :]
        sup_entry = vec[divide[0]:divide[1], :]
        v2_entry = vec[divide[1]:, :]
        if not(v1_entry.is_zero_matrix or v2_entry.is_zero_matrix):
            in_v1.append(v1_entry)
            in_v2.append(v2_entry)
            in_sup.append(sup_entry)
            overlap_vecs.append(mat*v1_entry)

    if return_decomp:
        if len(support) == 0:
            return overlap_vecs, in_v1, in_v2
        else:
            return overlap_vecs, in_v1, in_v2, in_sup
    else:
        return overlap_vecs


def _linearly_indep_cols(X):
    rref, pivot_cols = X.rref()
    return pivot_cols


def _linearly_indep_rows(X):
    return _linearly_indep_cols(X.transpose())


def _linearly_indep_col_sets(X):
    # Check if all rows are linearly independent -> 1 answer
    if len(_linearly_indep_cols(X)) == X.shape[1]:
        return [_linearly_indep_cols(X)]
    # If not, try all permutations of column ordering to pick up
    # different combinations
    col_sets = []
    for perm in itertools.permutations(range(X.shape[1])):
        cols = _linearly_indep_cols(X[:, perm])
        cols_og = []
        for i in cols:
            cols_og.append(perm[i])
        cols_og = tuple(sorted(cols_og))
        if tuple(cols_og) not in col_sets:
            col_sets.append(cols_og)
    return col_sets


def _linearly_indep_row_sets(X):
    return _linearly_indep_col_sets(X.transpose())


def _det_fast(M):
    """
    Hardcoded determinant for small sympy matrices (up to 4x4)

    Falls back to M.det() for larger
    """
    n = M.shape[0]
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")

    if n == 1:
        return M[0,0]

    elif n == 2:
        return M[0,0]*M[1,1] - M[0,1]*M[1,0]

    elif n == 3:
        return (M[0,0]*(M[1,1]*M[2,2] - M[1,2]*M[2,1])
              - M[0,1]*(M[1,0]*M[2,2] - M[1,2]*M[2,0])
              + M[0,2]*(M[1,0]*M[2,1] - M[1,1]*M[2,0]))
    elif n == 4:
        return (
            M[0,0]*(M[1,1]*(M[2,2]*M[3,3] - M[2,3]*M[3,2])
                   -M[1,2]*(M[2,1]*M[3,3] - M[2,3]*M[3,1])
                   +M[1,3]*(M[2,1]*M[3,2] - M[2,2]*M[3,1]))
          - M[0,1]*(M[1,0]*(M[2,2]*M[3,3] - M[2,3]*M[3,2])
                   -M[1,2]*(M[2,0]*M[3,3] - M[2,3]*M[3,0])
                   +M[1,3]*(M[2,0]*M[3,2] - M[2,2]*M[3,0]))
          + M[0,2]*(M[1,0]*(M[2,1]*M[3,3] - M[2,3]*M[3,1])
                   -M[1,1]*(M[2,0]*M[3,3] - M[2,3]*M[3,0])
                   +M[1,3]*(M[2,0]*M[3,1] - M[2,1]*M[3,0]))
          - M[0,3]*(M[1,0]*(M[2,1]*M[3,2] - M[2,2]*M[3,1])
                   -M[1,1]*(M[2,0]*M[3,2] - M[2,2]*M[3,0])
                   +M[1,2]*(M[2,0]*M[3,1] - M[2,1]*M[3,0]))
        )
    else:
        return M.det()

def _equiv_cols(c1, c2, shifts=None):
    if shifts is None:
        # Just the global shift
        shifts = [sym.ones(c1.shape[0], 1)]
    for sign in [1, -1]:
        # Can you make the difference using the specified shifts?
        diff = sign*c1 - c2
        if diff.is_zero_matrix:
            return True
        if not shifts is None:
            if len(shifts) > 0:
                li_cols = _linearly_indep_cols(sym.Matrix.hstack(diff, *shifts))
                if len(li_cols) == len(shifts):
                    return True
    return False


def _equal_up_to_column_shift_and_sign(A: sym.Matrix, B: sym.Matrix, shifts: Union[sym.Matrix,list[sym.Matrix]] = None):
    """
    Assumes that the columns in A and B are linearly independent

    Args:
        A (sym.Matrix): _description_
        B (sym.Matrix): _description_
        shifts (list[sym.Matrix], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if A.shape != B.shape:
        return False
    if isinstance(shifts, sym.Matrix):
        shifts = [shifts[:, i] for i in range(shifts.shape[1])]
    if shifts is None:
        # Just the global shift
        shifts = [sym.ones(A.shape[0], 1)]
    elif len(shifts) > 0:
        # Get linearly independent set of shifts
        li_shifts = _linearly_indep_cols(sym.Matrix.hstack(*shifts))
        shifts = [shifts[i] for i in li_shifts]

    rows, cols = A.shape
    for j in range(cols):
        if not _equiv_cols(A[:, j], B[:, j], shifts):
            return False
    return True

def _find_equiv_cols(c1, cList, shifts=None):
    return [i for i in range(len(cList)) if _equal_up_to_column_shift_and_sign(c1, cList[i], shifts)]


def _find_equiv_mats(m1, mList, shifts=None):
    return [i for i in range(len(mList)) if _equal_up_to_column_swaps_and_shift_and_sign(m1, mList[i], shifts)]

def _equal_up_to_column_swaps_and_shift_and_sign(A: sym.Matrix, B: sym.Matrix, shifts: list[sym.Matrix] = None):
    """
 
    Args:
        A (sym.Matrix): _description_
        B (sym.Matrix): _description_

    Returns:
        _type_: _description_
    """

    if A.shape != B.shape:
        return False
    
    # Count unique columns and number of occurances
    colsA = []
    countsA = []
    colsB = []
    countsB = []
    for j in range(A.shape[1]):
        for mat, cols, counts in zip([A, B],
                                     [colsA, colsB],
                                     [countsA, countsB]):
            c = mat[:, j]
            equiv = _find_equiv_cols(c, cols, shifts)
            if len(equiv) == 0:
                cols.append(c)
                counts.append(1)
            else:
                counts[equiv[0]] += 1
    
    # Make sure there are equal sets of equal columns
    # Function that returns equivalent columns in an iterable
    for cA in colsA:
        # Identify equivalent columns
        eq_cols = _find_equiv_cols(cA, colsB, shifts)
        if len(eq_cols) > 1:
            raise ValueError("Columns are not unique")
        elif len(eq_cols) != 1:
            return False
    return True


def _nonzero_entries_str(X: Union[sym.Matrix, np.ndarray],
                         prepend_sum: bool = True,
                         full_mat: bool = False) -> str:
    """
    For a symmetric matrix X, returns the flattened indices
    of nonzero off-diagonal entries

    Args:
        X (Union[sym.Matrix, np.ndarray]): symmetric matrix
        prepend_sum (bool): prepends the sum of the binary string

    Returns:
        str: binary string 10101001 with 1 being the indices
             of nonzero entries and 0 being the indices of
             zero entries. optionally prepend the sum of the
             binary string 4-10101001
    """

    nz_idx = []
    n_nz = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if j > i or full_mat:
                if X[i,j] != 0:
                    nz_idx.append("1")
                    n_nz += 1
                else:
                    nz_idx.append("0")

    bin_str = "".join(nz_idx)
    if prepend_sum:
        bin_str = str(n_nz) + "-" + bin_str
    return bin_str 

def _sort_wT(wT: Union[sym.Matrix, np.ndarray]):
    if not isinstance(wT, np.ndarray):
        wT = np.array(wT).astype(int)
    keys = []
    for i in range(wT.shape[0]):
        keys.append(str(sum(wT[i, :] != 0)) +"_"+"-".join(wT[i, :].nonzero()[0].astype(str)))

    return wT[np.argsort(keys)]

def _maximize_wT(wT):
    """
    Need to sort before sending in for accurate row accounting

    Args:
        wT (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # Make sure all integer values
    # nz_min = np.min(np.abs(wT[np.nonzero(wT)]))
    # wT = (wT/nz_min).astype(int)
    ### TODO: Deal with non-integer values of wT


    # Sort to put rows with identical nz entries
    # next to each other
    wT = _sort_wT(wT)

    # Identify rows with identical nonzero entries
    nz_entries = [tuple(wT[i].nonzero()[0]) for i in range(wT.shape[0])]
    swap_rows = []
    segment = [0]
    for i in range(1, wT.shape[0]):
        if nz_entries[i] == nz_entries[i-1]:
            segment.append(i)
        else:
            swap_rows.append(segment)
            segment = [i]
    swap_rows.append(segment)

    # All different permutations of the swappable rows
    perms = list(itertools.product(*[itertools.permutations(rows, len(rows)) for rows in swap_rows]))

    best_w = None
    best_val = -np.inf
    best_key = ""

    # Choice of row swaps
    # Which columns to invert
    for cols in itertools.product([1, -1], repeat=wT.shape[1]):
        col_vec = np.array(cols).flatten()
        # Which rows to invert
        for rows in itertools.product([1, -1], repeat=wT.shape[0]):
            row_vec = np.array(rows).flatten()
            # Apply operations
            wT_mod = row_vec[:, np.newaxis]*wT*col_vec[np.newaxis, :]
            val = np.sum(wT_mod)
            # First check the sum of the elements
            if val >= best_val:
                for row_perm in perms:
                    row_order = list(itertools.chain.from_iterable(row_perm))
                    wT_perm = wT_mod[row_order, :]
                    # Flattened matrix in base 3 -- for canonical ordering
                    key = "".join((wT_perm + 1).flatten().astype(str))
                    if val > best_val or (val == best_val and key > best_key):
                        best_val = val
                        best_key = key
                        best_w = (wT_perm.copy(), row_vec.copy(),
                                  col_vec.copy(), row_order)

        
    return best_w[0], best_key, best_w[1:]


def _wT_key(wT: Union[sym.Matrix, np.ndarray], equalJ=False):

    if isinstance(wT, sym.Matrix):
        wT = np.array(wT).astype(float)

    # Number of coupled modes
    coup_mat = (wT[0][np.newaxis, :]*wT[0][:, np.newaxis]).astype(float)
    for i in range(1, wT.shape[0]):
        # outer product of row vectors
        term = wT[i][np.newaxis, :]*wT[i][:, np.newaxis]
        if not equalJ:
            term = np.random.random()*term
        coup_mat += term

    return _nonzero_entries_str(coup_mat)


def _var_col_perms(var_types,
                   dyn_modes=["compact", "extended", "harmonic"],
                   nd_modes=["free", "frozen", "sigma"],
                   dyn_only=False):
    
    mode_perms = []
    for dyn_mode in dyn_modes:
        if dyn_mode in var_types:
            mode_perms.append(tuple(itertools.permutations(var_types[dyn_mode], len(var_types[dyn_mode]))))
    
    if not dyn_only:
        for nd_mode in nd_modes:
            if nd_mode in var_types:
                mode_perms.append((tuple(var_types[nd_mode]),))

    perms = itertools.product(*mode_perms)
    return [tuple(itertools.chain.from_iterable(perm)) for perm in perms]


def _sub_equal_LC(X):
    # All C the same, different from Cj
    C = sym.symbols("C", positive=True, real=True)
    Cj = sym.symbols("C_J", positive=True, real=True)
    for x in X.free_symbols:
        if "c" in str(x).lower() and "j" not in str(x).lower():
            X = X.subs(x, C)
        if "c" in str(x).lower() and "j" in str(x).lower():
            X = X.subs(x, Cj)
    # All L the same
    L = sym.symbols("L", positive=True, real=True)
    for x in X.free_symbols:
        if "l" in str(x).lower():
            X = X.subs(x, L)
    return X

def _to_frozenset(eqs):
    return frozenset(frozenset((x.lhs, x.rhs)) for x in eqs)


def _to_eq_list(all_eq):
    # Convert dictionary to equation
    if isinstance(all_eq, dict):
        all_eq = [sym.Eq(x[0], x[1]) for x in all_eq.items()]
    # Convert list or frozenset or tuple of frozensets or expressions to equation
    elif isinstance(all_eq, list) or isinstance(all_eq, frozenset) or isinstance(all_eq, tuple):
        all_eq = list(all_eq)
        # Frozen set
        if isinstance(all_eq[0], frozenset):
            all_eq = [list(x) for x in all_eq]
            all_eq = [sym.Eq(x[0], x[1]) for x in all_eq if len(x) == 2]
        elif isinstance(all_eq[0], dict):
            all_eq = functools.reduce(lambda a,b: a+b, [[sym.Eq(x[0], x[1]) for x in eq.items()] for eq in all_eq], [])
        # expression
        elif not isinstance(all_eq[0], sym.Eq):
            new_eq = []
            for eq in all_eq:
                if eq.is_number:
                    if eq != 0:
                        return []
                else:
                    new_eq.append(sym.Eq(eq, 0))
            all_eq = new_eq

    return all_eq


COMPATIBLE_CACHE = {}
# TODO: ADD CACHE?
def _fully_compatible_set(assumptions, solve_vars, starting={}, nonzero=[], depth_first=False,
                          top_level=True):
    """
    Returns all sets of assumptions fully compatible with
    the starting one

    Args:
        starting (_type_): _description_
        assumptions (_type_): _description_

    Returns:
        _type_: _description_
    """
    # ASS_KEY = []
    # starting_eq = _to_eq_list(starting)
    # ASS_KEY += _to_frozenset(starting_eq)
    # for ass_set in assumptions:
    #     ASS_KEY += _to_frozenset(ass)

    # Record fixed substitutions for the starting assumptions
    old_eq = []
    const_subs = {}
    for var, val in starting.items():
        var, val = sym.simplify(var), sym.sympify(val)
        if var.is_number and val.is_number:
            if var != val:
                return []
        else:
            old_eq.append(sym.Eq(var, val))
        if val.is_number and isinstance(var, sym.Symbol):
            const_subs[var] = val
    if len(assumptions) == 0:
        return [starting]
    else:
        next_set = assumptions[0]
        all_res = []
        for ass in next_set:
            # quick check for incompatible constant values
            const_compat = True
            for var, val in ass.items():
                var, val = sym.simplify(var), sym.sympify(val)
                if var.is_number and val.is_number:
                    if var != val:
                        const_compat = False
                        break
                if val.is_number and isinstance(var, sym.Symbol):
                    if var in const_subs:
                        if val != const_subs[var]:
                            const_compat = False
                            break
                if not const_compat:
                    break
            if not const_compat:
                continue

            # If it's not obviously wrong, then solve
            new_eq = [sym.Eq(x[0], x[1]) for x in ass.items()]
            # sols = _cached_solve_dummy(new_eq + old_eq, solve_vars)
            sols = _cached_solve(new_eq + old_eq, solve_vars)
            # eq_set = frozenset(frozenset(x.))
            # sols = sym.solve(all_eq, solve_vars, dict=True, simplify=True)
            is_compat = len(sols) > 0
            if is_compat:
                for sol in sols:
                    # Make sure value isn't 0
                    this_nz = [sym.simplify(d.subs(sol)) for d in nonzero]
                    if any(d == 0 for d in this_nz):
                        continue
                    res = _fully_compatible_set(assumptions[1:], solve_vars, starting=sol, nonzero=this_nz, depth_first=depth_first)
                    if res:
                        if depth_first:
                            return res
                        else:
                            all_res += res    
        return all_res


def _cached_solve_dummy(all_eq, solve_vars, simplify=True):


    all_eq = _to_eq_list(all_eq)
    
    # Recursive base case
    if len(all_eq) == 0:
        return []
    
    var_by_eq = [[s for s in eq.free_symbols] for eq in all_eq]
    solve_var_present = [x for x in set(itertools.chain.from_iterable(var_by_eq)) if x in solve_vars]
    other_var_present = [x for x in set(itertools.chain.from_iterable(var_by_eq)) if x not in solve_vars]


    # Dummy variable mapping
    dummy_var = CACHE_VARS[:len(solve_var_present)]
    dummy_const = CACHE_CONST[:len(other_var_present)]
    mapping = dict(zip(solve_var_present, dummy_var)) | dict(zip(other_var_present, dummy_const))
    inv_mapping = {}
    for var, val in mapping.items():
        inv_mapping[val] = var
    all_eq_mapped = [eq.subs(mapping) for eq in all_eq]
    sols_w_dummy = _cached_solve(all_eq_mapped, dummy_var, simplify)
    sols = []
    for sol_dum in sols_w_dummy:
        sol = {}
        for var, val in sol_dum.items():
            sol[var.subs(inv_mapping)] = val.subs(inv_mapping)
        sols.append(sol)
    # breakpoint()
    return sols


def _cached_solve(all_eq, solve_vars, simplify=True):


    all_eq = _to_eq_list(all_eq)
    
    
    # Recursive base case
    if len(all_eq) == 0:
        return []
   
    eq_set = frozenset(frozenset((eq.lhs, eq.rhs)) for eq in all_eq)
    # eq_set = frozenset(frozenset((sym.simplify(eq.lhs), sym.simplify(eq.rhs))) for eq in all_eq)

    # Have we solved this set of equations before?
    # If we haven't, look for subsets of them that we have
    remaining_eq = [x for x in eq_set]
    solved_eq = []
    for solved in sorted(SOLVE_CACHE.keys(), key=len):
        # if len(solved) == 1:
        #     break
        solved_sols = SOLVE_CACHE[solved]
        # Strict subset of these equations were solved
        if eq_set == solved or (all(eq in eq_set for eq in solved) and solved_sols == []):
            SOLVE_CACHE[eq_set] = solved_sols
            return solved_sols
        elif all(eq in remaining_eq for eq in solved) and len(solved_sols) == 1:
            solved_eq += solved_sols
            for eq in solved:
                remaining_eq.remove(eq)
        if len(remaining_eq) == 0:
            simplified_eq = _to_eq_list(solved_eq)
            simplified_eq_set = frozenset(frozenset((eq.lhs, eq.rhs)) for eq in simplified_eq)
            sols = sym.solve(simplified_eq, solve_vars, dict=True, simplify=simplify)
            SOLVE_CACHE[eq_set] = sols 
            SOLVE_CACHE[simplified_eq_set] = sols  
            return sols

    if remaining_eq:
        remaining_eq = _to_eq_list(remaining_eq)
    if solved_eq:
        solved_eq = _to_eq_list(solved_eq)
    sols = sym.solve(remaining_eq+solved_eq, solve_vars, dict=True, simplify=simplify)
    # sols = sym.solve(all_eq, solve_vars, dict=True, simplify=simplify)
    
    SOLVE_CACHE[eq_set] = sols
    SOLVE_CACHE[_to_frozenset(remaining_eq+solved_eq)] = sols
    
    return sols
      

def _sol_indep_of_vars(expr, solve_vars, nonzero=[]):
    """
    Determines whether a solution to the given expr exists that only
    depends on the specified solve_variables

    Args:
        expr (sym.Expr): Expression to be set to 0
        solve_vars (list): Variables to solve for

    Returns:
        list[dict]: list of solutions that do not depend on bad_vars
    """

    # Simplify to rational form
    og_expr = expr
    expr = sym.together(expr, deep=True)
    numer, denom = expr.as_numer_denom()
    numer = sym.expand(numer)

    # No solutions if there are no solve variables
    if not any(v in numer.free_symbols for v in solve_vars):
        return []

    # Get unique symbolic products (excluding solve_vars)
    # No bad_vars in the expression
    var_combos = _unique_products(numer, exclude=solve_vars)

    # Anything without a bad var multiplied by it
    eqs = []
    eqs.append(var_combos.get(1,0))
    # Anything with a bad var multipled by it
    for v in var_combos:
        eqs.append(var_combos[v])
    eqs = [eq for eq in eqs if eq != 0]
    eqs = list(set(eqs))  # Remove duplicates

    # Get solutions for each equation and
    # check compatibility with each other
    sols = []
    for eq in eqs:
        si = _cached_solve([eq], solve_vars)
        # si = sym.solve(eq, solve_vars, dict=True, simplify=True)
        if si:
            sols.append(si)
        # At least one of them is unsolvable
        else:
            return []
    compat_sols = _fully_compatible_set(sols, solve_vars, nonzero=[])

    # Remove redundant solutions
    # i.e. ones where a
    # proper subset of the substitutions
    # also is a valid solution
    final_sols = []
    final_pairs = []
    for sol in sorted(compat_sols, key=lambda x: len(x)):
        is_redun = False
        subs_pairs = list(sol.items())
        for ref_pairs in final_pairs:
            # Are all pairs from a previously examined
            # substitution already there?
            if all(p in subs_pairs for p in ref_pairs):
                is_redun = True
                break
        if not is_redun:
            final_sol = {}
            for key, val in sol.items():
                if key != val:
                    final_sol[key] = val
            final_sols.append(final_sol)
            final_pairs.append(subs_pairs)
    # Make sure none of the nonzero conditions are violated
    final_sols = [s for s in final_sols if all(sym.simplify(d.subs(s)) != 0 for d in nonzero)]
    final_sols = [s for s in final_sols if sym.simplify(denom.subs(s)) != 0]
    return final_sols


def _unique_products(expr: sym.Expr, exclude: list[sym.Symbol] = []):
    """
    Return a list of unique products of free symbols
    from an expanded expression, excluding specified symbols.
    """
    expr = sym.expand(expr)
    products = {}
    for term in expr.as_ordered_terms():
        # Extract multiplicative factors
        factors = []
        if isinstance(term, Mul):
             #[f for f in term.args if isinstance(f, (sym.Symbol, sym.Pow)) and f not in exclude]
            for f in term.args:
                if isinstance(f, sym.Symbol) and f not in exclude:
                    factors.append(f)
                elif isinstance(f, sym.Pow):
                    if f.base not in exclude:
                        factors.append(f)
        elif isinstance(term, sym.Pow):
            if term.base not in exclude:
                factors.append(term)
        elif isinstance(term, sym.Symbol) and term not in exclude:
            factors.append(term)
        elif term.is_number:
            factors.append(term)
        factors = tuple(sorted(factors, key=lambda a: str(a))) + (1,)
        prod = functools.reduce(lambda a,b: a*b, factors, sym.sympify(1))
        coeff = term/prod
        if factors in products:
            products[factors].append(coeff)
        else:
            products[factors] = [coeff]
            
    to_return = {}
    for factors, coeffs in products.items():
        prod = functools.reduce(lambda a,b: a*b, factors, sym.sympify(1))
        coeff = functools.reduce(lambda a,b: a+b, coeffs, sym.sympify(0))
        to_return[prod] = sym.simplify(coeff)
    
    
    
    return to_return


def _expr_valid(valid_expr: list[sym.Expr], subs):
    """
    Determines whether the given list of substitutions
    are all "compatible," meaning they can be
    simultaneously fulfilled.

    Optionally include a sympy expression or matrix that must not
    have undefined values with the specified substitutions.

    Args:
        subs_list (list[dict]): list of substitutions mapping
                                sympy symbol to its value.

    Returns:
        (bool, list[dict]): first entry is whether the substitutions are compatible
                            second entry is a refined set of substitutions that
                            achieve them all simultaneously
    """

    is_finite = [x.subs(subs).simplify().is_finite for x in valid_expr]
    return all([x for x in is_finite if not x is None])

def _find_Z_instance(Z: Union[sym.Matrix, sym.Expr], var_list: list[sym.Expr],
                     vals=[0,1,-1,2,-2], all_real=True, sort=True, random=False,
                     nonzero=[]):

    # Variables to substitute concrete values in for
    to_sub = [x for x in sorted(Z.free_symbols, key=str) if x in var_list]
    # det = sym.simplify(Z.det())
    det = Z.det()
    # Random
    if random:
        # Random values backup
        subs = {}
        for v in to_sub:
            subs[v] = np.random.random()
        if det.subs(subs) != 0:
            Zsub = Z.subs(subs)
            if ((sym.im(Zsub).is_zero_matrix or not(all_real)) and
                all(sym.simplify(d.subs(subs)) != 0 for d in nonzero) and
                _expr_valid(Zsub, {})):
                return Zsub
        else:
            raise ValueError("No Valid Instance Present With Provided Values")


    best_val = ""
    best_Z = None
    assignments = itertools.combinations_with_replacement(vals, len(to_sub))
    if sort:
        assignments = sorted(assignments, key=lambda x: sum(abs(xi) for xi in x))
    for assign in assignments:
        subs = dict(zip(to_sub, assign))
        if det.subs(subs) != 0:
            Zsub = Z.subs(subs)
            if ((sym.im(Zsub).is_zero_matrix or not(all_real)) and
                all(sym.simplify(d.subs(subs)) != 0 for d in nonzero) and
                _expr_valid(Zsub, {})):
                return Zsub

    raise ValueError("No Valid Instance Present With Provided Values")


def compact_well_aligned(wT: sym.Matrix, n_c: int):

    # Nothing to worry about here
    if n_c == 0:
        return True
    
    wc = wT[:, :n_c]
    rows, cols = wc.shape

    # 1 periodic variable -- is there a 1 or -1 there
    # and all other entries are integers (i.e. same periodicity)
    if n_c == 1:
        nz_entries = [sym.Abs(wc[i,0]) for i in range(rows) if wc[i,0].is_nonzero]
        if len(nz_entries) == 0:
            return False
        elif 1 in nz_entries and all(x.is_integer for x in nz_entries):
            return True
        else:
            return False
        
    # 2 or more periodic variables --
    # Does each compact variable have at least one junction that
    # depends on it alone (among compact variables)
    # And do all the other rows have only integer entries
    else:
        # Select all rows with a single 1/-1 and rest 0's
        e_rows = []
        for i in range(wc.shape[0]):
            nz_entries = [sym.Abs(wc[i,j]) for j in range(cols) if wc[i,j].is_nonzero]
            if len(nz_entries) == 0:
                continue
            if len(nz_entries) == 1 and nz_entries[0] == 1:
                e_rows.append(i)
            elif any(not x.is_integer for x in nz_entries):
                return False
        # Check if all the single entry rows span the
        # number of compact variables: yes->well aligned
        return len(_linearly_indep_rows(wc[e_rows, :])) == n_c


def compact_alignment_transformation(wT:sym.Matrix, n_c:int):

    wc = wT[:, :n_c]
    all_Zc = []

    # Consider unchanged if it works
    if compact_well_aligned(wc, n_c):
        all_Zc.append(sym.eye(n_c))

    # TODO:
    # Think about row magnitudes, we want the rows with the
    # smallest magnitude
    # In general we want the lattice vectors to be the *shortest*
    # translations that give periodicity, but with linearly dependent
    # junction connectivity this will be a given
    # The last one will be a loop, so just adds to finish the loop
    # will always be a +/- 1 on the other loop junction variables
    # We can check if the transformation respected the overall periodicity
    # by checking if all other entries are integers

    # Try all different linearly independent row combinations
    row_sets = _linearly_indep_row_sets(wc)
    for rows in row_sets:
        # Get the appropriate transformation for each set
        Zc = wc[rows, :].pinv()
        # Verify it's well-aligned
        if compact_well_aligned(wc*Zc, n_c):
            if len(_find_equiv_mats(Zc, all_Zc, shifts=[])) == 0:
                all_Zc.append(Zc)

    # Return all possible transformations
    all_Z = []
    for Zc in all_Zc:
        Z = sym.eye(wT.shape[1])
        Z[:n_c, :n_c] = Zc
        all_Z.append(Z)

    return all_Z

# TODO: Add test
def decouple_column(v:sym.Matrix, nd_mat:sym.Matrix, mat:sym.Matrix):

    Z1 = sym.Matrix.hstack(v, nd_mat)
    # decouple from free modes -- this is guaranteed to not change w
    # Decouple each nondynamical mode individually
    i = nd_mat.shape[1]
    while i >= 1:
        Z1 = Z1*decoupling_transformation((Z1).transpose()*mat*(Z1), n_d=i)    
        i -= 1
    return sym.simplify(Z1[:, 0])
    

def decoupling_transformation(X:sym.Matrix, n_d:int):

    coupled = []
    for i in range(n_d, X.shape[0]):
        if any(X[i, n_d:]):
            coupled.append(i)

    # Make the transformation to uncouple them
    # (I 0)
    # (M I)
    # with M = -X_22^-1 X_21
    Z2 = sym.eye(X.shape[0])
    X22 = X[coupled, coupled]
    X21 = X[coupled, :n_d]
    M = -X22.inv()*X21
    for i, row in enumerate(coupled):
        Z2[row, :n_d] = M[i, :]
    return Z2


def H_hash(Z, var_types, cMat, lMat, wJ, equalJ=False,
           dyn_modes=["compact", "extended", "harmonic"],
           nd_modes=["free", "frozen", "sigma"], try_perms = True,
           eps=1e-10, ordering_matters:bool=True):
    """
    Produces a hash that represents in the specified basis

    1) a = [nCompact][nExtended][nHarmonic]
    2) c = wJ_trans in canonical form, in base 3 (-1 -> 0, 0 -> 1, 1 -> 2)
    Swapping columns to maximize:
    3) d = base 2 positions of nonzero capacitive intermode coupling (= 0 -> 0, != 0 -> 1)
    4) e = positions of nonzero inductive intermode coupling (= 0 -> 0, != 0 -> 1)

    hash = a_b_c_d_e (ex: 3_511_012)

    Keeps the variable transformation that minimizes the hash,
    after trying all possible permutations of the columns
    corresponding to different variable types.

    Args:
        circuit (_type_): _description_
        edges (_type_): _description_
        symmetric (bool, optional): _description_. Defaults to True.
        numerical (bool, optional):
        eps (float, optional): 
    """
    
    Z_og = Z

    # Record number of modes
    n_nodes = Z.shape[0]
    ext_var = var_types.get("extended", [])
    n_ext = len(ext_var)
    comp_var = var_types.get("compact", [])
    n_comp = len(comp_var)
    harm_var = var_types.get("harmonic", [])
    n_harm = len(harm_var)
    mode_str = f"{n_comp}{n_ext}{n_harm}"
    n_nd = n_nodes - n_ext - n_comp - n_harm
    n_nl = n_ext + n_comp

    # Truncate transformed matrices to dynamical modes
    if not cMat is None:
        cTrans = Z.transpose()*cMat*Z
        cTrans = cTrans[:-n_nd, :-n_nd]
    if wJ.shape[1] > 0:
        wJTrans = wJ.transpose()*Z
        wJTrans = sym.simplify(wJTrans[:, :-n_nd])
    lTrans = Z.transpose()*lMat*Z
    lTrans = lTrans[:-n_nd, :-n_nd]

    # Numerically treat the matrices
    if not cMat is None:
        C, cVals = num_subs(cTrans, symbol="C")
    if any("L" in str(x).upper() for x in lMat.free_symbols):
        L, lVals = num_subs(lTrans, symbol="L")
    else:
        L, lVals = num_subs(lTrans, symbol="J")
    
    # In case there was param dependance in the
    # transform
    if not cMat is None:
        C = C.subs(lVals)
        C = np.array(C).astype(float)
        L = L.subs(cVals)
    L = np.array(L).astype(float)
    if wJ.shape[1] > 0:
        wJTrans = np.array(wJTrans).astype(float)

    # All different ways to arrange columns
    # Interchanging like variable columns
    if try_perms and ordering_matters:
        perms = _var_col_perms(var_types, dyn_modes, nd_modes, dyn_only=True)
    else:
        perms = [tuple(range(n_nodes-n_nd))]

    lowest_hash = ""
    lowest_Z = None
    for perm in perms:

        L_tilde = L[:, perm]
    
        # Key for C, L = [n_coupled]-[nz entries of off diag]
        if np.abs(L_tilde).max() > 0:
            L_tilde[np.abs(L_tilde)/np.abs(L_tilde).max() < eps] = 0
        L_key = _nonzero_entries_str(L_tilde)

        # Invert capacitance matrix and trim small numerical values
        if not cMat is None:
            C_tilde = C[:, perm]
            C_tilde_inv = np.linalg.inv(C_tilde)
            C_tilde_inv[np.abs(C_tilde_inv)/np.abs(C_tilde_inv).max() < eps] = 0
            C_key =  _nonzero_entries_str(C_tilde_inv)
        else:
            C_key = _nonzero_entries_str(sym.zeros(*L_tilde.shape))
        
        if wJ.shape[1] > 0:
            wT_tilde = wJTrans[:, perm]
            # Put wT into cananocal ordering
            wT_tilde, _, _ = _maximize_wT(_sort_wT(wT_tilde))
            # Key for wT = [n_coupled]-[nz entries of off diag]
            w_key = _wT_key(wT_tilde, equalJ = equalJ)
        else:
            w_key = "0-"+"0"*(len(L_key)-2)

        # If ordering doesn't matter
        if not ordering_matters:
            w_key = w_key[:w_key.find("-")]
            C_key = C_key[:C_key.find("-")]
            L_key = L_key[:L_key.find("-")]

        # Make the hash string
        Z_hash = "_".join([mode_str, w_key, str(int(L_key[0])+int(C_key[0])), L_key, C_key])
        if lowest_hash == "" or Z_hash < lowest_hash:
            lowest_hash = Z_hash
            lowest_Z = Z_og[:, perm + tuple(range(n_nodes - n_nd, n_nodes))]

    return lowest_hash, lowest_Z


def incidence_to_square(w, vals):
    mat = sym.Matrix(np.zeros((w.shape[0], w.shape[0])))
    for j in range(w.shape[1]):
        mat += vals[j]*w[:, j]*w[:, j].transpose()
    return mat

def gen_cap_mat(circuit, edges, ground_node: list[int] = []):
    """
    Generates a capacitance matrix using Sympy for the given circuit.
    Energy C = Q_vec @ inv_cap @ Q_vec

    Args:
        circuit (list): a list of element labels for the desired circuit.
                        different numbers are treated as having different values.
                        e.g. [["J", "C1"],["L", "J"], ["C2"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]

    Returns:
        sym.Matrix of the capacitance matrix
    """

    edges, node_map = utils.renumber_nodes(edges, return_map=True)
    ground_node = [node_map[n] for n in ground_node]
    var_counts = utils.count_elems_mapped(circuit)

    # Generate incidence matrices for capacitors and junctions
    wC, c_vals = gen_w(circuit, edges, "C", return_params=True, ground_node=ground_node)
    wJ, cj_vals = gen_w(circuit, edges, "J", return_params=True, ground_node=ground_node,
                        param_func=lambda e: "C_{J" + e.replace("J", "") + "}")
    if var_counts["C"] > 0 and var_counts["J"] > 0:
        wC_total = sym.Matrix.hstack(wC, wJ)
        c_vals_total = c_vals + cj_vals
    elif var_counts["C"] > 0:
        wC_total = wC
        c_vals_total = c_vals
    elif var_counts["J"] > 0:
        wC_total = wJ
        c_vals_total = cj_vals
    else:
        raise ValueError("No capacitive elements in circuit")

    return incidence_to_square(wC_total, c_vals_total)


def gen_ind_mat(circuit, edges, ground_node: list[int] = []):
    """
    Generates an inductor matrix using Sympy for the given circuit. 
    Defined so Energy L = Phi_vec @ ind_mat @ Phi_vec

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]

    Returns:
        sym.Matrix of the capacitance matrix
    """
    edges, node_map = utils.renumber_nodes(edges, return_map=True)
    ground_node = [node_map[n] for n in ground_node]

    # Number of nodes excluding ground
    n_nodes = utils.get_num_nodes(edges) - len(ground_node)

    # Generate incidence matrices for inductors and get inverse of inductance values
    wL, l_vals = gen_w(circuit, edges, "L", return_params=True, ground_node=ground_node)
    l_vals = [1/l for l in l_vals]

    return incidence_to_square(wL, l_vals) if len(l_vals) > 0 else sym.zeros(n_nodes, n_nodes)

def gen_w(circuit: list, edges: list, w_elem: str = "J", return_params=False,
          ground_node: list[int]=[], param_func = lambda e: e):
    """
    Generate the junction (or inductor) incidence matrix.
    (n_nodes, n_jj) matrix where w_ij = 1,-1 for head/tail
    of junction and 0 otherwise.

    Head and tail are not done done consistently for elements in inductive
    loops, so this should not be used for assigning external fluxes.

    Args:
        circuit : list
        A list of element labels for the desired circuit.  
        Example: ``[["J"], ["L", "J"], ["C"]]``.
    edges : list
        A list of edge connections for the desired circuit.  
        Example: ``[(0, 1), (0, 2), (1, 2)]``.
    elem (str, optional): Element to generate incidence matrix for.
                           Defaults to "J".
    Returns:
        sym.Matrix of the incidence matrix
    """
    # Find loops -- if any are present assign directions
    # to each element
    # loops = find_loops(circuit, edges, loop_elem)

    # Fill in values
    edges, node_map = utils.renumber_nodes(edges, return_map=True)
    ground_node = [node_map[n] for n in ground_node]
    n_nodes = utils.get_num_nodes(edges) - len(ground_node)
    n_elem = utils.count_elems_mapped(circuit)[w_elem]
    if n_elem == 0:
        if return_params:
            return sym.zeros(n_nodes, 1), tuple()
        else:
            return sym.zeros(n_nodes, 1)
    w = sym.Matrix(np.zeros((n_nodes, n_elem), dtype=int))
    params = []
    w_count = 0
    for edge, elems in zip(edges, circuit):
        for e in elems:
            if w_elem in e:
                params.append(param_func(e))
                w[min(edge), w_count] = 1
                w[max(edge), w_count] = -1
                for n in ground_node:
                    w[n, w_count] = 0
                w_count += 1
    if return_params:
        param_symbols = sym.symbols(",".join(params), real=True)
        if len(params) == 1:
            param_symbols = (param_symbols,)
        return w, param_symbols
    else:
        return w

def var_trans_basis(circuit, edges, ground_node=[]):

    # Incidence matrices
    wC, c_vals = gen_w(circuit, edges, "C", return_params=True, ground_node=ground_node)
    _, cj_vals = gen_w(circuit, edges, "J", return_params=True, ground_node=ground_node,
                       param_func=lambda e: "C_{J" + e.replace("J", "") + "}")
    wL, l_vals = gen_w(circuit, edges, "L", return_params=True, ground_node=ground_node)
    wJ, j_vals = gen_w(circuit, edges, "J", return_params=True, ground_node=ground_node)

    # Combine capacitance incidence matrices if both are present
    var_counts = utils.count_elems_mapped(circuit)
    if var_counts["C"] > 0 and var_counts["J"] > 0:
        wC_total = sym.Matrix.hstack(wC, wJ)
        c_vals_total = c_vals + cj_vals
    elif var_counts["C"] > 0:
        wC_total = wC
        c_vals_total = c_vals
    elif var_counts["J"] > 0:
        wC_total = wJ
        c_vals_total = cj_vals

    return var_trans_basis_incidence_mat(wC_total, wL, wJ, c_vals_total, l_vals)


def var_trans_basis_incidence_mat(wC: sym.Matrix, wL: sym.Matrix, wJ: sym.Matrix,
                                  c_vals: list[sym.Symbol] = None, l_vals: list[sym.Symbol] = None):
    """
    Generates a basis for variable transformations that
    separate compact, extended, harmonic,
    free (no potential), frozen (no kinetic), and
    sigma (neither potential nor kinetic) variables.

    Produces variables a basis where dynamical and
    non-dynamical variables are decoupled in both
    charge and flux.

    The transformation must also maintain the periodicity of 
    the junction terms for compact variables.

    NOTE: Here we assume wC to include any junction capacitances
    and wJ to only include Josephson junctions.

    Orders variables in [compact, extended, harmonic,
                            free, frozen, sigma] order

    Args:
        circuit (_type_): _description_
        edges (_type_): _description_
    """
    # match indices with array indices
    if wC.shape[1] == 0:
        raise ValueError("No Capacitors in Circuit -- Cannot Define Variables")
    n_nodes = wC.shape[0]

    # Make C, L values if none are given
    if c_vals is None or len(c_vals) == 0:
        c_vals = sym.symbols(",".join([f"C{i+1}" for i in range(wC.shape[1])]), real=True)
        if isinstance(c_vals, sym.Symbol):
            c_vals = (c_vals,)
    if l_vals is None or len(l_vals) == 0:
        l_vals = sym.symbols(",".join([f"L{i+1}" for i in range(wL.shape[1])]), real=True)
        if isinstance(l_vals, sym.Symbol):
            l_vals = (l_vals,)
        l_vals = [1/l for l in l_vals]

    # Define Nullspaces of incidence matrices
    N_wC = wC.transpose().nullspace()
    N_wL = wL.transpose().nullspace()
    N_wJ = wJ.transpose().nullspace()

    # Capacitance and susceptance matrices for decoupling
    cMat = incidence_to_square(wC, c_vals)
    lMat = incidence_to_square(wL, l_vals)

    # labels of mode types
    mode_types = []
    # dynamical and nondynamical mode vectors
    dy_modes = []
    nd_modes = []
    
    # Nondynamical variable vectors
    sig_vec = _vec_space_overlap(N_wL, _vec_space_overlap(N_wJ, N_wC))
    n_sig = len(sig_vec)
    free_vec = _independent_from(_vec_space_overlap(N_wJ, N_wL), sig_vec)
    n_free = len(free_vec)
    froz_vec = _independent_from(_vec_space_overlap(N_wJ, N_wC), sig_vec)
    n_froz = len(froz_vec)

    nd_vec = sig_vec + free_vec + froz_vec
    nd_mat = sym.Matrix.hstack(*nd_vec) if len(nd_vec) > 0 else sym.Matrix([])
    n_nd = len(nd_vec)

    # Dynamical variable vectors
    harm_vec = _independent_from(N_wJ, nd_vec)
    n_harm = len(harm_vec)
    comp_vec = _independent_from(N_wL, nd_vec)
    n_comp = len(comp_vec)
    ext_vec = _independent_from(sym.eye(n_nodes), nd_vec + harm_vec + comp_vec)
    n_ext = len(ext_vec)


    # Categorize variable types
    var_types = {}
    var_types["compact"] = list(range(n_comp))
    var_types["extended"] = list(range(n_comp, n_comp+n_ext))
    var_types["harmonic"] = list(range(n_comp+n_ext, n_comp+n_ext+n_harm))
    var_types["free"] = list(range(n_comp+n_ext+n_harm, n_comp+n_ext+n_harm+n_free))
    var_types["frozen"] = list(range(n_comp+n_ext+n_harm+n_free, n_comp+n_ext+n_harm+n_free+n_froz))
    var_types["sigma"] = list(range(n_comp+n_ext+n_harm+n_free+n_froz, n_nodes))
    
    # Check number of modes adds up
    if n_nd + n_comp + n_harm + n_ext != n_nodes:
        raise ValueError("Did not identify correct number of modes")
    
    ## Decouple from nondynamical modes
    # Compact
    comp_vec = [decouple_column(v, nd_mat, cMat) for v in comp_vec]
    # Extended
    ext_vec = [decouple_column(v, nd_mat, cMat) for v in ext_vec]
    ext_vec = [decouple_column(v, nd_mat, lMat) for v in ext_vec]
    # Harmonic
    harm_vec = [decouple_column(v, nd_mat, cMat) for v in harm_vec]
    harm_vec = [decouple_column(v, nd_mat, lMat) for v in harm_vec]

    # Put into matrix form
    Z = sym.Matrix.hstack(*comp_vec, *ext_vec, *harm_vec, *free_vec, *froz_vec, *sig_vec)

    # If a sigma mode is present, scale the column to make
    # the defined variable an equal superposition of all node fluxes
    if len(sig_vec) > 0:
        Z_inv = sym.MutableDenseMatrix(Z.inv())
        for i in var_types["sigma"]:
            for j in range(Z_inv.shape[1]):
                Z_inv[i, j] = 1
        Z = sym.ImmutableDenseMatrix(Z_inv.inv())

    return Z, var_types


def secondary_decouple(Z0: sym.Matrix, var_types: dict[str, list[int]],
                       cMat: sym.Matrix, lMat: sym.Matrix,
                       return_instance: bool = False,
                       prefix="Z", Z_in: sym.Matrix = None,
                       param_symbols=["C", "L", "J"],
                       ordering_matters:bool = True,
                       wJ: sym.Matrix = sym.Matrix([])) -> sym.Matrix:
    """
    Performs a secondary decoupling transformation of
    the form
    [ I   Z_ce   0     0 ]
    [ 0   Z_e    0     0 ]
    [ 0   Z_eh  Z_hh   0 ]
    [ 0    0     0     I ]
    that maximizes the number of nonzero off diagonal elements in the
    transformed susceptance and inverse capacitance matrices.
    In case of a tie in number, prefers fewer inductive terms,
    in line with H_hash.

    Args:
        Z0 (sym.Matrix): Initial transformation that separates circuit into different
                         types of modes.
        var_types (dict[str, list[int]]): Dictionary mapping mode type to a list
                                          of indices that are that mode.
        cMat (sym.Matrix): Capacitance matrix
        lMat (sym.Matrix): Susceptance matrix
        return_instance: bool
            If there are free parameters in the final transformation,
            substitute in a valid set of 1's and 0's to complete it,
            which maximizes the number of 0's.

    Returns:
        sym.Matrix: Secondary transformation
    """

    # Record mode types
    eye = sym.eye(Z0.shape[0])
    comp = var_types.get("compact", [])
    n_comp = len(comp)
    ext = var_types.get("extended", [])
    n_ext = len(ext)
    harm = var_types.get("harmonic", [])
    n_harm = len(harm)
    sigma = var_types.get("sigma", [])
    n_sigma = len(sigma)
    n_d = n_comp + n_harm + n_ext
    
    # Possible to not include cMat to consider junction
    # decoupling
    if not cMat is None:
        cTrans = sym.simplify(Z0.transpose()*cMat*Z0)
    lTrans = sym.simplify(Z0.transpose()*lMat*Z0)

    # Possible Transformation
    if Z_in is None:
        Z = sym.eye(Z0.shape[0])
        var_list = []
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if ((i in comp and j in ext) or 
                    (i in ext and j in ext) or 
                    (i in harm and j in ext+harm)):
                    Z[i,j] = sym.Symbol(f"{prefix}{i}{j}", real=True)
                    var_list.append(Z[i,j])
    else:
        Z = Z_in
        var_list = [x for x in Z.free_symbols if prefix in str(x)]

    # Transformed Capacitance and Susceptance Matrices
    n_vars = len(var_list)
    if not cMat is None:
        cTrans2 = Z.transpose()*cTrans*Z
        cTrans2_trunc = cTrans2[:n_comp+n_ext+n_harm,:n_comp+n_ext+n_harm]
        # cTrans2_det = cTrans2_trunc.det()
        ## MAKE FAST SIMPLIFY HERE WITH TIMEOUT
        # cTrans2_det = sym.simplify(_det_fast(cTrans2_trunc))
        # Try to simplify but don't if it takes too long
        cTrans2_det = _det_fast(cTrans2_trunc)
        cTrans2_det_simp = utils.run_with_timeout(sym.simplify, (cTrans2_det,), timeout=5)
        if not cTrans2_det_simp is None:
            cTrans2_det = cTrans2_det_simp
        if cTrans2_trunc.shape[0] > 1:
            cTransInv2 = sym.inv_quick(cTrans2_trunc)
        else:
            cTransInv2 = cTrans2_trunc.inv()
    else:
        cTransInv2 = sym.eye(Z.shape[0])
        cTrans2 = sym.eye(Z.shape[0])
        cTrans2_det = sym.sympify(1)
        cTrans = None

    lTrans2 = (Z.transpose()*lTrans*Z)


    # Determine which entries are possible to decouple
    # without depending on system parameters
    params =  [x for x in cTransInv2.free_symbols if any(s in str(x).upper() for s in param_symbols)]
    params += [x for x in lTrans2.free_symbols if any(s in str(x).upper() for s in param_symbols)]
    coupling_c = []
    coupling_l = []
    coupling_terms = {}
    dets = [sym.simplify(Z.det()), cTrans2_det]
    for i1 in comp + ext + harm:
        for i2 in ext + harm:
            # Symmetric coupling
            if i2 <= i1:
                continue
            c_vars = [v for v in var_list if v in cTransInv2[i1, i2].free_symbols]
            sol_c = _sol_indep_of_vars(cTransInv2[i1, i2], c_vars, nonzero=dets)
            if sol_c:
                coupling_terms[("C", i1, i2)] = sol_c
            # L will be 0 if this is a real capacitance matrix
            l_vars = [v for v in var_list if v in lTrans2[i1, i2].free_symbols]
            sol_l = _sol_indep_of_vars(lTrans2[i1, i2], l_vars, nonzero=dets)
            if sol_l:
                coupling_terms[("L", i1, i2)] = sol_l

    # Remove empty solutions and those that make the transformation not inverible
    for k in list(coupling_terms.keys()):
        sols = coupling_terms[k]
        new_sols = []
        for sol in sols:
            # Invertible
            if all(sym.simplify(det.subs(sol)) != 0 for det in dets):
                new_sols.append(sol)
        # empty solution
        if len(new_sols) == 0:
            del coupling_terms[k]
        else:
            coupling_terms[k] = new_sols
        
    ## Maximize the number of coupling terms that go to zero
    n_couple = len(coupling_terms)
    already_tested = {}
    incompatible = set()
    compatible = {}
    compatible_keys = []
    # Start with comparison of not constraining the transformation
    best_Z = [Z]
    best_subs = [{}]
    # Hash from just the cTrans and lTrans
    Z_hash = _find_Z_instance(Z, var_list, random=True)
    nz_str = H_hash(Z_hash, var_types, cTrans, lTrans, wJ=sym.Matrix([]))[0]
    for i in range(1,len(nz_str)-1):
        if nz_str[i-1] == "_" and nz_str[i+1] == "_":
            nz_str = nz_str[i:]
            break
    best_nz_str = [nz_str]
    # best_nz = int(nz_str[0])
    best_nz = 0
    # initial_nz = best_nz
    # best_nz =
    for nz in range(1, n_couple+1):
        # Pidgeonhole principle, is it possible to to select nz keys that
        # might be compatible
        if len(incompatible) > 0:
            if nz > n_couple - max(len(incompat_keys) for incompat_keys in incompatible):
                break
        for keys in itertools.combinations(coupling_terms.keys(), nz):
            # Check if a subset of the keys are incompatible
            if any(all(k in keys for k in incompat_keys) for incompat_keys in incompatible):
                continue

            # Manually make hash for speed
            keys_L = [x for x in keys if "L" in x]
            keys_C = [x for x in keys if "C" in x]
            nz_str = []
            n_nz = 0
            for keyset in [keys_L, keys_C]:
                test = np.ones((n_d, n_d))
                for k in keyset:
                    test[k[1], k[2]] = 0
                nz_str.append(_nonzero_entries_str(test))
                n_nz += int(nz_str[-1][0])
            nz_str = str(n_nz) + "_" + "_".join(nz_str)

            # Fill in rule set with possible simplifications
            # from previous iterations
            keys_left = [k for k in keys]
            rules_sets = []
            for keyset in sorted(compatible.keys(), key=len)[::-1]:
                if all(k in keys_left for k in keyset):
                    rules_sets.append(compatible[keyset])
                    for k in keyset:
                        keys_left.remove(k)
                if keys_left == []:
                    break
            for k in keys_left:
                rules_sets.append(coupling_terms[k])
            
            # Identify a set of rules that are fully compatible
            res = _fully_compatible_set(rules_sets, var_list, depth_first=False)
            if len(res) == 0:
                incompatible.add(keys)
                continue
            good_subs = []
            for i, compat_sub in enumerate(res):
                if any(sym.simplify(d.subs(compat_sub)) == 0 for d in dets):
                    continue
                Ztest = sym.simplify(Z.subs(compat_sub))
                Z_free = list(Ztest.free_symbols)
                if nz > best_nz:
                    best_nz = nz
                    best_Z = [Ztest]
                    best_subs = [compat_sub]
                    best_nz_str = [nz_str]
                elif nz == best_nz:
                    best_Z.append(Ztest)
                    best_subs.append(compat_sub)
                    best_nz_str.append(nz_str)
                good_subs.append(compat_sub)
                # Save the solved compatibility for these keys
            if len(good_subs) == 0:
                incompatible.add(keys)
                continue
            if keys in compatible_keys:
                compatible[keys] += good_subs
            else:
                compatible[keys] = good_subs
                compatible_keys.append(keys)

    ## Remove any transformations that are "subsets" of
    ## another, meaning it has a proper subset of the terms present
    nz_Z = [X.shape[0]*X.shape[1] - int(_nonzero_entries_str(X, full_mat=True)[0]) for X in best_Z]
    sort_keys = np.array(list(zip(nz_Z, best_nz_str)), dtype=[("nz_Z", int), ("nz_str", f"<U{len(best_nz_str[0])+1}")])
    subs_keys = {}
    for i in np.argsort(sort_keys, order = ["nz_str", "nz_Z"]):
        key = frozenset(tuple(best_subs[i].items()))
        prev = [all(x in key for x in past_key) for past_key in subs_keys.keys()]
        if any(prev):
            continue
        else:
            subs_keys[key] = [best_Z[i], best_nz_str[i]]
    best_Z = []
    best_subs = []
    best_nz_str = []
    for sub_i, [Z_i, nz_str_i] in subs_keys.items():
        best_Z.append(Z_i)
        best_subs.append(sub_i)
        best_nz_str.append(nz_str_i)
    
    ## Remove any transformations that are column-wise
    ## permutations or *-1 of another
    unique_Z = []
    final_Z = []
    final_nz_str = []
    # Order columns by number of nonzero entries and
    # Then by row of nonzero entries
    perms = _var_col_perms(var_types, dyn_only=True)
    for Z_i, nz_str_i in zip(best_Z, best_nz_str):
        Z_set = []
        hashes = []
        to_add = True
        for perm in perms:
            Z_perm = Z_i[:, :n_d][:, perm]
            # Make dummy variables to allow for equality check
            subs = {}
            v_count = 1
            for j in range(Z_perm.shape[1]):
                for i in range(Z_perm.shape[0]):
                    for v in Z_perm[i,j].free_symbols:
                        if v not in subs:
                            subs[v] = sym.Symbol(f"v_{v_count}", real=True)
                        v_count += 1
            Z_perm = Z_perm.subs(subs)
            # Is there any past transformation equivalent to this one?
            if any(any(_equal_up_to_column_shift_and_sign(Z_past, Z_perm, shifts=[]) for Z_past in Z_past_set) for Z_past_set in unique_Z):
                to_add = False
                break
            Z_set.append(Z_perm)
        if to_add:
            unique_Z.append(Z_set)
            # Guaranteed to have selected lowest hash by previous sorting
            final_Z.append(Z_i)
            final_nz_str.append(nz_str_i)

    # Process transformation before returning
    if return_instance:
        final_Z = [_find_Z_instance(Z, var_list) for Z in final_Z]
    if ordering_matters:
        return final_Z[np.argmin(nz_str_i)]
    else:
        return final_Z


def choose_Z(circuit: list, edges: list, ground_node: list = [],
             return_instance: bool = True) -> tuple[sym.Matrix, dict[str, list[int]], str]:
    """
    Chooses a transformation phi_node = Z*phi_new that separates
    the circuit into compact, extended, harmonic, free, frozen, and cyclic modes.
    Chooses the transformation that minimizes the H_hash function, which minimizes
    the number of nonzero intermode coupling terms, while maintaining the
    periodicity of the compact and extended mode junction terms.

    Args:
        circuit : list
        A list of element labels for the desired circuit.  
        Example: ``[["J"], ["L", "J"], ["C"]]``.
        edges : list
        A list of edge connections for the desired circuit.  
        Example: ``[(0, 1), (0, 2), (1, 2)]``.
        return_instance: bool
        If there are free parameters in the final transformation,
        substitute in a valid set of 1's and 0's to complete it,
        which maximizes the number of 0's.

    Returns:
        tuple[sym.Matrix, dict[str, list[int]], str]: _description_
    """

    # Reset solve cache
    global SOLVE_CACHE
    SOLVE_CACHE = {}

    # Relabel nodes from 0
    edges, node_map = utils.renumber_nodes(edges, return_map=True)
    ground_node = [node_map[n] for n in ground_node]
    n_nodes = utils.get_num_nodes(edges) - len(ground_node)
     
    # Prep: Generate capacitance matrix, susceptance matrix, and incidence matrix
    cMat = gen_cap_mat(circuit, edges, ground_node=ground_node)
    lMat = gen_ind_mat(circuit, edges, ground_node=ground_node)
    wJ, EJ = gen_w(utils.add_elem_number(circuit), edges, w_elem="J", return_params=True)
    elem_counts = utils.count_elems_mapped(circuit)
    jMat = incidence_to_square(wJ, EJ) if elem_counts["J"] > 0 else sym.zeros(n_nodes, n_nodes)

    # Step 0: Separate into different variable types and decouple nondynamical
    Z0, var_types = var_trans_basis(circuit, edges, ground_node=ground_node)

    # Step 1: Enumerate different choices of compact variable
    wJT_trans = wJ.transpose()*Z0
    Z1 = [Z0*Z for Z in compact_alignment_transformation(wJT_trans, len(var_types["compact"]))]

    # Step 2: Minimize intermode coupling
    lowest_hash = ""
    lowest_Z = []
    for Z in Z1:
        if sym.simplify(Z.det()) == 0:
            raise ValueError("Invalid Transformation (Not Invertible)")
        # 2a: Decouple nonlinear degrees of freedom
        if elem_counts["J"] > 0:
            Z2_a = secondary_decouple(Z, var_types, None, jMat,
                                    return_instance=False, prefix="Z",
                                    ordering_matters=False)
        else:
            Z2_a = [None]
        t2 = time.time()
        # 2b: Decouple linear degrees of freedom
        for Z_in in Z2_a:
            Z2_b = secondary_decouple(Z, var_types, cMat, lMat,
                                    return_instance=False, prefix="Z", Z_in=Z_in,
                                    # ordering_matters=True, wJ=wJ)    
                                    ordering_matters=True)            
            Z_tot = Z*Z2_b
            var_list = [x for x in Z2_b.free_symbols if "Z" in str(x)]
            Z_hash = _find_Z_instance(Z_tot, var_list, random=True)
            val, Z_perm = H_hash(Z_hash, var_types, cMat, lMat, wJ)
            if val <= lowest_hash or lowest_hash == "":
                lowest_Z = [Z_tot]
                lowest_hash = val
    
    # If there are multiple Z with the same lowest hash
    # then see if they separate with equal parameter values
    # TODO: Could return the full list of lowest Z
    Z_final = lowest_Z[0]
    hash_final = lowest_hash
    for Z in lowest_Z[1:]:
        Z_equal = sym.simplify(_sub_equal_LC(Z))
        var_list = [x for x in Z_equal.free_symbols if "Z" in str(x)]
        val, Z_perm = H_hash(_find_Z_instance(Z_equal, var_list, random=True),
                            var_types, cMat, lMat, wJ, equalJ=True)
        if val < hash_final:
            hash_final = val
            Z_final = Z
    # Get a specific instance of the transformation
    if return_instance:
        # Try and identify a transformation that 
        # respects the periodicity of extended variable
        # junction terms
        var_list = [x for x in Z_final.free_symbols if "Z" in str(x)]
        det = Z_final.det()
        nz_list = [det]
        for i in range(Z_final.shape[0]):
            for j in range(Z_final.shape[1]):
                if len(Z_final[i,j].free_symbols) > 0:
                    nz_list.append(sym.simplify(Z_final[i,j].together().as_numer_denom()[1]))
        if elem_counts["J"] > 0 and len(var_types.get("extended", [])) > 0:
            ext = var_types.get("extended", [])
            wJT_trans = wJ.transpose()*Z_tot
            all_eqs = []
            for i in range(wJT_trans.shape[0]):
                for j in ext:
                    val = sym.simplify(wJT_trans[i,j])
                    if val != 0:
                        # = +1 and = -1 and = 0
                        sol_p1 = sym.solve(sym.Eq(val, 1), [v for v in var_list if v in val.free_symbols], dict=True)
                        sol_0 = sym.solve(sym.Eq(val, 0), [v for v in var_list if v in val.free_symbols], dict=True)
                        sol_n1 = sym.solve(sym.Eq(val, -1), [v for v in var_list if v in val.free_symbols], dict=True)
                        sols = []
                        eq_tuples = []
                        for sol in sol_p1+sol_n1+sol_0:
                            sols.append(sol)
                        all_eqs.append(sols)
            s2 = _fully_compatible_set(all_eqs, solve_vars=var_list, nonzero=nz_list, depth_first=True)
            if s2:
                Z_final = Z_final.subs(s2[0])
        if len(Z_final.free_symbols) > 0:
            Z_final = _find_Z_instance(Z_final, var_list, nonzero=nz_list)

    return Z_final, var_types, lowest_hash


def gen_junc_pot(circuit, edges, flux_vars, Z=None, eps=1e-10) -> sym.Matrix:
    """
    Generate the junction potential terms, optionally applying a change of basis.

    This function generates the junction potential terms for the given circuit and 
    optionally performs a change of basis to transform the flux variables based on 
    a provided transformation matrix.

    Parameters
    ----------
    circuit : list
        A list of element labels for the desired circuit.  
        Example: ``[["J"], ["L", "J"], ["C"]]``.
    edges : list
        A list of edge connections for the desired circuit.  
        Example: ``[(0, 1), (0, 2), (1, 2)]``.
    flux_vars : sym.Matrix
        A vector of flux variables for the circuit.
    Z : sym.Matrix
        A change of basis matrix used to transform the node flux variables.

    Returns
    -------
    sym.Matrix
        The capacitance matrix as a symbolic matrix, representing the junction potential terms.
    """
    
    if Z is None:
        Z = sym.eye(len(flux_vars))

    n_nodes = utils.get_num_nodes(edges)

    EJ = {}
    for elems in circuit:
        for elem in elems:
            if elem in EJ:
                continue
            elif "J" in elem:
                EJ[elem] = sym.Symbol("E_{"+str(elem)+"}", positive=True,
                                      real=True)
     
    # Add all the junction terms
    j_terms = 0
    for edge, elems in zip(edges, circuit):
        i, j = edge
        val = 0
        for elem in elems:
            if "J" in elem:
                val += -EJ[elem]
        if abs(val) > 0:
            wJi = sym.zeros(n_nodes,1)
            wJi[i,0] = -1
            wJi[j,0] = 1
            j_terms += val*sym.cos((wJi.transpose()*Z*flux_vars)[0])

    return j_terms


def num_subs(C, symbol="C", exclude = "", rand_range = (1,2), vals_in = {}, hermitify=True):

    vals = {}
    for x in C.free_symbols:
        x_str = str(x)
        if (symbol in x_str.upper()) or (symbol in x_str.lower()):
            if exclude == "" or exclude not in x_str:
                # Random value
                if x not in vals_in:
                    vals[x] = rand_range[0] + (rand_range[1]-rand_range[0])*np.random.random()
                # prescribed value
                else:
                    vals[x] = vals_in[x]
    
    C = C.subs(vals)
    if hermitify:
        C = C/2 + sym.conjugate(C.transpose())/2

    return C, vals


def collect_H_terms(H: Add, zero_ext: bool = True, 
                    periodic_charge=PERIODIC_CHARGE, periodic_phase=PERIODIC_PHASE,
                    extended_charge=EXTENDED_CHARGE, extended_phase=EXTENDED_PHASE,
                    ext_charge: str = EXT_CHARGE, ext_flux: str = EXT_PHASE,
                    no_coeff: bool = False, collect_phase: bool = True) -> Add:
    """
    Groups terms in the Hamiltonian
    
    (q and \varphi -> \theta).

    Args:
        H (Add): Hamiltonian
        zero_ext (bool, optional): Whether to zero all gate voltages/external
                                   fluxes. Defaults to True.
        periodic_charge (str, optional): symbol used for periodic charges.
                                         Defaults to "n".
        extended_charge (str, optional): symbol used for extended charges.
                                         Defaults to "Q".
        periodic_phase (str, optional): symbol used for periodic phases.
                                        Defaults to "θ".
        extended_phase (str, optional): symbol used for extended phases.
                                         Defaults to "θ".
        ext_charge (str, optional): symbol used in external charges.
                                    Defaults to "ng".
        ext_flux (str, optional): symbol used in external fluxes.
                                   Defaults to "_{ext}"
        no_coef (bool, optional): Remove all the coefficients,
                                  only leaving operators.
        collect_phase (bool, optional): for speed, don't collect the phase terms.
                                        slightly messier, but faster.

    Returns:
        Add: Hamiltonian with terms grouped
    """

    # List of variable types
    q_list = [q for q in H.free_symbols
              if extended_charge in str(q)]
    n_list = [q for q in H.free_symbols
              if periodic_charge in str(q) and
              ext_charge not in str(q)]
    theta_list = [th for th in H.free_symbols
                  if (periodic_phase in str(th) or
                      extended_phase in str(th)) and
                  ext_flux not in str(th)]
    ext_list = [q for q in H.free_symbols
                if ext_charge in str(q) or
                ext_flux in str(q)]
    n_modes = len(theta_list)

    # Set all external parameters to 0
    if zero_ext:
        for ext in ext_list:
            H = H.subs(ext, 0)

    # Terms to group
    # Q and n
    combosQ = {}
    for terms in itertools.product(q_list + n_list, repeat=2):
        combo = functools.reduce(lambda x, y: x*y, terms)
        indices = np.unique(["".join([c for c in str(x) if c.isdigit()]) for x in terms])
        combosQ[combo] = "E_{C"+''.join(indices)+"}"

    # Phase
    combos = []
    combos_trig = []
    if collect_phase:
        for num_terms in range(1, n_modes + 1):
            # Straight products
            combos += list(set([functools.reduce(lambda x, y: x*y, z)
                                for z in itertools.product(theta_list,
                                                           repeat=num_terms)]))
            # Trig products
            # Encoding signals cos or sin
            for encoding in itertools.product([0, 1], repeat=num_terms):
                # Modes is which num_terms modes are being considered
                for modes in itertools.combinations(range(n_modes), num_terms):
                    trig_prod = 1
                    for i, term in enumerate(encoding):
                        if term:
                            trig_prod *= sym.cos(theta_list[modes[i]])
                        else:
                            trig_prod *= sym.sin(theta_list[modes[i]])
                    combos_trig += [trig_prod]

        # Explicitly add theta squared terms if only one mode
        if n_modes == 1:
            combos += list(set([functools.reduce(lambda x, y: x*y, z)
                                for z in itertools.product(theta_list,
                                                           repeat=2)]))

    H = sym.expand(H)
    H = collect(H, list(combosQ.keys()) + combos, func=sym.ratsimp)
    if collect_phase:
        H = collect(H, combos_trig)

    if no_coeff:
        H = _remove_coeff(H, list(combosQ.keys()) + combos + combos_trig)

    return H, combos+combos_trig, combosQ


def _remove_coeff(H, all_combos):
    H_class = H.copy()
    for combo in all_combos:
        H_class = H_class.replace(lambda x: x.is_Mul
                                  # Dividing removes all the terms in combo
                                  and all([sym not in combo.free_symbols
                                           for sym in
                                           (x/combo).free_symbols])
                                  # And all theta/n terms in x are also in
                                  # combo
                                  and all([sym in combo.free_symbols
                                           for sym in x.free_symbols
                                           if sym in all_combos]),
                                  lambda x: -combo if str(x)[0] == "-"
                                  else combo)
    return H_class


def gen_variables(n_nodes, Z, periodic):

    Q_str = ""
    th_str = ""
    for n in range(1, n_nodes+1):
        if Z is None:
            th_str += NODE_PHASE + "_{"+str(n)+"}, "
            Q_str += NODE_CHARGE + "_{"+str(n)+"}, "
        elif n in periodic:
            th_str += PERIODIC_PHASE + "_{"+str(n)+"}, "
            Q_str += PERIODIC_CHARGE + "_{"+str(n)+"}, "
        else:
            th_str += EXTENDED_PHASE + "_{"+str(n)+"}, "
            Q_str += EXTENDED_CHARGE + "_{"+str(n)+"}, "

    Q_vec = sym.Matrix(sym.symbols(Q_str[:-1]))
    th_vec = sym.Matrix(sym.symbols(th_str[:-1]))

    return Q_vec, th_vec


def symbolic_hamiltonian(circuit, edges, Cv=None, V=None, Z=None,
                            var_types:dict={},
                            return_mats: bool = False,
                            return_vars: bool = False,
                            return_H_class: bool = False,
                            return_combos: bool = False,
                            collect_phase: bool = True,
                            expand_trig: bool = True):
    """
    Perform a symbolic circuit quantization for the given circuit.

    This function performs symbolic quantization of a given circuit. The quantized 
    variables are categorized as:
    - Periodic variables are represented by \hat{n} / \hat{θ}.
    - Extended variables are represented by \hat{q} / \hat{ϕ}.
    - Node variables are represented by \hat{q} / \hat{φ}.

    Parameters
    ----------
    circuit : list
        A list of element labels for the desired circuit.  
        Example: ``[["J"], ["L", "J"], ["C"]]``.
    edges : list
        A list of edge connections for the desired circuit.  
        Example: ``[(0, 1), (0, 2), (1, 2)]``.
    Cv : sym.Matrix, optional
        Coupling matrix representing the interaction between nodes in the circuit 
        and fixed voltage nodes.
    V : sym.Matrix, optional
        A matrix of fixed voltages applied in the circuit.
    Z : sym.Matrix, optional
        A change of basis matrix that transforms node variables to new variables. 
        This matrix corresponds to the Z transformation of scqubits.
        If the new variables are expressed in terms of the old, this should be the inverse.
    return_mats : bool, optional
        If True, return the capacitance and inductance matrices along with the Hamiltonian.
    return_vars : bool, optional
        If True, return the sympy variables used to construct the Hamiltonian.
    return_H_class : bool, optional
        If True, return the Hamiltonian with all coefficients removed.
    return_combos : bool, optional
        If True, return the combination of variables present in the Hamiltonian.
    collect_phase : bool, optional
        If True, skip collecting phase terms for faster execution (results in slightly less precision).

    Returns
    -------
    tuple
        A tuple containing:
        - The Hamiltonian of the circuit (sympy expression).
        - Optionally, the capacitance matrix and inductance matrix.
        - Optionally, the sympy variables used to construct the Hamiltonian.
        - Optionally, the Hamiltonian class with coefficients removed.
        - Optionally, the combinations of variables present in the Hamiltonian.
    """
    edges = utils.renumber_nodes(edges)
    n_nodes = utils.get_num_nodes(edges)
    q_vec, th_vec = gen_variables(n_nodes, Z, var_types.get("compact", []))
    

    cMat = gen_cap_mat(circuit, edges)
    lMat = gen_ind_mat(circuit, edges)
    wJT = gen_w(circuit, edges, w_elem="J").transpose()

    # Set zero applied voltage
    if Cv is None:
        Qv = sym.zeros(rows=n_nodes, cols=1)
    else:
        Qv = Cv*V

    # Obtain a transformation if none was given
    if Z is None:
        Z, var_types, _ = choose_Z(circuit, edges)
    
    # Transform C, L and Qv
    cMat = sym.transpose(Z)*cMat*Z
    lMat = sym.transpose(Z)*lMat*Z
    Qv = Z.inv()*Qv
    wJT = wJT*Z
        

    # J terms shouldn't contain anything from free modes or frozen modes
    n_nd = (len(var_types.get("free", []))+
            len(var_types.get("frozen", []))+ 
            len(var_types.get("sigma", [])))
    if not wJT[:, -n_nd:].is_zero_matrix:
        raise ValueError("Junction potential depends on nondynamical modes")
    J_terms = gen_junc_pot(circuit, edges, th_vec, Z=Z)

    ## Remove any nondynamical modes
    # First check they're actually nondynamical
    for i in range(n_nd):
        has_C = cMat[n_nodes-n_nd, n_nodes-n_nd] != 0
        has_L = lMat[n_nodes-n_nd, n_nodes-n_nd] != 0
        if has_C and has_L:
            raise ValueError("Nondynamical modes are not Nondynamical")
    # Then verify we're decoupled from them
    Zdc = sym.eye(Z.shape[0])
    for i in range(n_nd):
        cTrans = Zdc.transpose()*cMat*Zdc
        Zdc = Zdc*decoupling_transformation(cTrans, i)
        lTrans = Zdc.transpose()*cMat*Zdc
        Zdc = Zdc*decoupling_transformation(lTrans, i)
    if Zdc != sym.eye(Z.shape[0]):
        print("WARNING: Dynamical modes are coupled to defined free or frozen modes. Applying transformation to decouple")
        Z = Z*Zdc
    
    # Truncate matrices
    cMat = cMat[:-n_nd, :-n_nd]
    lMat = lMat[:-n_nd, :-n_nd]
    wJT = wJT[:, :-n_nd]
    Qv = Qv[:-n_nd, :]
    q_vec = q_vec[:-n_nd, :]
    th_vec = th_vec[:-n_nd, :]

    # Invert Capcitance Matrix
    if cMat.shape[0] == 1:
        C_inv = cMat.inv()
    else:
        # Check for the weird all 0 issue
        C_inv = sym.inv_quick(cMat)
        if C_inv == sym.zeros(rows=C_inv.shape[0],
                                cols=C_inv.shape[1]):
            C_inv = cMat.inv()
    
    # Explicitly subtract out constant terms from coupling
    C_terms = sym.Rational(1, 2)*sym.transpose(q_vec - Qv)*C_inv*(q_vec - Qv)
    C_terms += -sym.Rational(1, 2)*sym.transpose(Qv)*C_inv*Qv
    L_terms = sym.Rational(1, 2)*sym.transpose(th_vec)*lMat*th_vec

    # Combine terms and group terms in H
    H = C_terms[0] + L_terms[0] + J_terms
    H = sym.nsimplify(H)
    if expand_trig:
        H = sym.expand_trig(H)
    H, combos, combosQ = collect_H_terms(H, zero_ext=False, collect_phase = collect_phase)

    to_return = (H,)

    if return_H_class:
        to_return = to_return + (utils._remove_coeff(H, list(combosQ)+combos),)
    if return_combos:
        to_return = to_return + (list(combosQ)+combos,)
    if return_mats:
        to_return = to_return + (cMat, lMat, wJT)
    if return_vars:
        to_return = to_return + (q_vec, th_vec)
    if len(to_return) == 1:
        to_return = to_return[0]

    return to_return



if __name__ == "__main__":

    import time

    # edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    # circuit = [("L_1", "C_1"), ("L_2", "C_2"), ("L_3",), ("J_1",)]
    
    # # circuit = [("L", "C"), ("L", "C"), ("L",), ("J",)]

    # circuit= [('C_1', 'L_1'), ('C_2', 'L_2'), ('C_3', 'L_3'), ('C_4', 'L_4'), ('C_5', 'J_1', 'L_5'), ('C_6', 'J_2', 'L_6')]
    # edges= [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # circuit =  [('C_1',), ('C_2', 'J_1', 'L_1'), ('C_3', 'J_2', 'L_2'), ('C_4', 'L_3'), ('C_5', 'J_3')]
    # edges =  [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    

    # circuit =  [('C_1',), ('J_1',), ('J_2', 'L_1'), ('J_3', 'L_2'), ('C_2', 'L_3'), ('J_4', 'L_4')]
    # edges =  [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # circuit= [('J_1',), ('J_2',), ('J_3',), ('C_1', 'L_1'), ('C_2', 'L_2'), ('C_3', 'L_3')]
    # edges= [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # circuit =  [('C', 'J', 'L'), ('C', 'J', 'L'), ('C', 'J', 'L'), ('C', 'J', 'L'), ('C', 'J', 'L'), ('C', 'J', 'L')]
    # circuit = utils.add_elem_number(circuit)
    # edges =  [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    # circuit = [("L1", "C1"), ("L2", "C2"), ("L3",), ("J",)]

    edges = [(1, 2), (3, 4), (1, 3), (2, 4)]
    circuit = [("L", "C"), ("L", "C"), ("L",), ("J",)]

    # cMat = gen_cap_mat(circuit, edges)
    # lMat = gen_ind_mat(circuit, edges)
    # edges = utils.renumber_nodes(edges)
    # all_Z, var_types = gen_spaced_var_trans(circuit, edges)
    # Z0 = all_Z[0]
    # cTrans = Z0.transpose()*cMat*Z0
    # lTrans = Z0.transpose()*lMat*Z0
    # print(Z0)
    
    times = []
    from tqdm import tqdm
    for i in tqdm(range(1)):
        t0 = time.time()
        # Z = secondary_decouple(Z0, var_types, cMat, lMat, True)
        choose_Z(circuit, edges)
        # print(Z)
        tf = time.time()
        times.append(tf-t0)
    print("Max", np.max(times), "Min:", np.min(times))
    print("Mean:", np.mean(times), "+/-", np.std(times))

    # breakpoint()

    # db_path = "/Users/eweissler/Library/CloudStorage/OneDrive-UCB-O365/Circuit Enumeration/circuits_4_nodes_7_elems.db"
    # for n in range(4, 5):
    #     df = utils.get_unique_qubits(db_path, n).iloc[:]
    #     from tqdm import tqdm
    #     order = np.arange(df.shape[0])
    #     np.random.shuffle(order)
    #     import time
    #     for i in tqdm(order[:]):
    #         t0 = time.time()
    #         row = df.iloc[i]
    #         circuit = row.circuit
    #         circuit = utils.add_elem_number(circuit)
    #         Z, var_types, hash = choose_Z(circuit, row.edges)
    #         tf = time.time()
    #         if tf-t0 > 5:
    #             print("LONG CIRCUIT ------")
    #             print("circuit=",circuit)
    #             print("edges=",row.edges)