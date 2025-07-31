__doc__ = "quantize.py: contains functions used to produce symbolic hamiltonians"
__author__ = "Eli Weissler"
__version__ = "0.1.0"
__all__ = ["gen_cap_mat", "gen_ind_mat", "gen_junc_pot", "quantize_circuit"]

import itertools
import functools

from typing import Union, Sequence

import sympy as sym
import numpy as np
import networkx as nx

from sympy import collect, expand_mul, Mul, Dummy
from sympy.core.add import Add

from sircuitenum import utils


PERIODIC_CHARGE = "n"
PERIODIC_PHASE = "θ"
EXTENDED_CHARGE = "q"
EXTENDED_PHASE = "φ"
NODE_CHARGE = "q"
NODE_PHASE = "ϕ"
EXT_CHARGE = "n_g"
EXT_PHASE = "_{ext}"


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


def _indices_to_arr(dim, ind):
    arr = np.zeros((dim, 1), dtype=int)
    for i in ind:
        arr[i] = 1
    return sym.Matrix(arr)

def _islands_to_vectors(circuit, edges, links):
    edges = utils.zero_start_edges(edges)
    ind = find_islands(circuit, edges, links)
    n_nodes = utils.get_num_nodes(edges)
    vecs = []
    for i in range(len(ind)):
        vecs.append(_indices_to_arr(n_nodes, ind[i]))
    return vecs


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
    assert len(_linearly_indep_cols(sym.Matrix.hstack(*vecs1))) == len(vecs1)
    assert len(_linearly_indep_cols(sym.Matrix.hstack(*vecs2))) == len(vecs2)
    assert len(_linearly_indep_cols(sym.Matrix.hstack(*support))) == len(support)
    assert vecs1[0].shape[0] == vecs2[0].shape[0]
    if len(support) > 0:
        assert support[0].shape[0] == vecs1[0].shape[0]

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

# def _vec_space_overlap(vecs1:Union[list[sym.Matrix], sym.Matrix],
#                        vecs2:Union[list[sym.Matrix], sym.Matrix],
#                        support:Union[list[sym.Matrix], sym.Matrix] = []
#                        idx:list[int]=[],
#                        v1_recon:bool=True,
#                        return_decomp:bool=False) -> list[sym.Matrix]:
#     """
#     Calculates the overlap of two linearly independent sets of
#     vectors vecs1, vecs2 using NULL(vecs1, -vecs2)

#     Args:
#         vecs1 (Union[list[sym.Matrix], sym.Matrix]): set of column vectors
#         vecs2 (Union[list[sym.Matrix], sym.Matrix]): set of column vectors
#         idx (list[int], optional): Consider equality only in a specified set of indices.
#                                    Note: In this case the vectors are reconstructed from
#                                    the set specified by v_recon.
#         v1_recon bool: In the case of only examining equality
#                                                        for a specified set of indices, the set to
#                                                        reconstruct the full vector from. True is vecs1
#                                                        false is vecs2.
#         return_decomp (bool, optional): Return the decomposition of the overlap
#                                         vectors in each set. Defaults to False.

#     Returns:
#         list[sym.Matrix]: list of vectors that span the overlap space.
#     """
#     # Convert matrices to list
#     if isinstance(vecs1, sym.Matrix):
#         vecs1 = [vecs1[:, j] for j in range(vecs1.shape[1])]
#     if isinstance(vecs2, sym.Matrix):
#         vecs2 = [vecs2[:, j] for j in range(vecs2.shape[1])]

#     # If either one is empty, return no overlap
#     if len(vecs1) == 0 or len(vecs2) == 0:
#         if return_decomp:
#             return [], [], []
#         else:
#             return []
#     # Assert vector sets are linearly independent
#     assert len(_linearly_indep_cols(sym.Matrix.hstack(*vecs1))) == len(vecs1)
#     assert len(_linearly_indep_cols(sym.Matrix.hstack(*vecs2))) == len(vecs2)

#     # Examine all indices if none is given
#     assert vecs1[0].shape[0] == vecs2[0].shape[0]
#     if idx == []:
#         idx = list(range(vecs1[0].shape[0]))

#     # Calculate the overlap of the two vector spaces
#     divide = len(vecs1)
#     ns = sym.Matrix.hstack(*[v[idx, :] for v in vecs1],
#                            *[-v[idx, :] for v in vecs2]).nullspace()
    
#     # Gather the entries
#     in_v1 = []
#     in_v2 = []
#     for vec in ns:
#         v1_entry = vec[:divide, :]
#         v2_entry = vec[divide:, :]
#         if not(v1_entry.is_zero_matrix or v2_entry.is_zero_matrix):
#             in_v1.append(v1_entry)
#             in_v2.append(v2_entry)
    
#     # Reconstruct the overlap vectors
#     if v1_recon:
#         mat_recon = sym.Matrix.hstack(*vecs1)
#         overlap_vecs = [mat_recon*v for v in in_v1]
#     else:
#         mat_recon = sym.Matrix.hstack(*vecs2)
#         overlap_vecs = [mat_recon*v for v in in_v2]
    
#     if return_decomp:
#         return overlap_vecs, in_v1, in_v2
#     else:
#         return overlap_vecs


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


def _equiv_cols(c1, c2, shifts=None):
    if shifts is None:
        # Just the global shift
        shifts = [sym.ones(c1.shape[0], 1)]
    for sign in [1, -1]:
        # Can you make the difference using the specified shifts?
        diff = sign*c1 - c2
        if diff.is_zero_matrix:
            return True
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
    else:
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


def _unique_mag_row(X):
    # Unique magnitudes of number in each row
    mags = []
    for i in range(X.shape[0]):
        this_row = []
        for j in range(X.shape[1]):
            v = sym.Abs(X[i,j])
            if v not in this_row:
                this_row.append(v)
        mags.append(sorted(this_row))
    return mags


def _unique_mag_col(X):
    return _unique_mag_row(X.transpose())


def _unique_vals(X):
    vals = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            v = X[i,j]
            if v not in vals:
                vals.append(v)
    return vals

def _unique_col_combos(basis, n_elem, signs=[1,-1], li_vecs=[], valid=lambda v: True, shifts=None):


    # Possible combinations of signs plus 0 -- ignore the all 0 combo
    combos  = list(itertools.combinations_with_replacement([0]+signs,len(basis)))
    combos = [c for c in combos if c != (0,)*len(basis)]
    # Unique permutations of each combination
    perms = list(itertools.chain(*[list(set(itertools.permutations(c)))for c in combos]))

    # Turn the permutations into vectors
    basis_mat = sym.Matrix.hstack(*basis)
    possible_vecs = []
    for p in perms:
        vec = basis_mat*sym.Matrix(p)
        # No equivalent vectors and it's linearly independent
        # from the specified ones
        if (valid(vec) and
            len(_find_equiv_cols(vec, possible_vecs, shifts)) == 0 and
            len(_linearly_indep_cols(sym.Matrix.hstack(*li_vecs, vec))) == len(li_vecs) + 1):
            possible_vecs.append(vec)

    # Now consider all n_elem linearly independent combinations of the possible_vecs
    possible_combos = []
    for combo in itertools.combinations(possible_vecs, n_elem):
        # Is the combo linearly independent when considered with the other vecs
        li = False
        if len(li_vecs) == 0:
            li = True
        elif len(_linearly_indep_cols(sym.Matrix.hstack(*li_vecs, *combo))) == len(li_vecs) + n_elem:
            li = True
        if li:
            # Is it equivalent to a matrix we've seen yet
            M = sym.Matrix.hstack(*combo)
            if not len(_find_equiv_mats(M, possible_combos, shifts)) > 0:
                possible_combos.append(M)
    
    return possible_combos



def _nonzero_entries_str(X: Union[sym.Matrix, np.ndarray],
                         prepend_sum: bool = True) -> str:
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
    if not isinstance(X, np.ndarray):
        X = np.array(X).astype(float)

    nz_idx = []
    n_nz = 0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if j > i:
                if np.abs(X[i,j]) > 0:
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
            # print(row_vec, col_vec)
            # print(wT)
            wT_mod = row_vec[:, np.newaxis]*wT*col_vec[np.newaxis, :]
            # print(wT_mod)
            # breakpoint()
            val = np.sum(wT_mod)
            # if val > 10:
            #     breakpoint()
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


def _remove_row(M, i):
    if not M.row_del(i) is None:
        return M.row_del(i)
    else:
        return M

def _remove_col(M, i):
    if not M.col_del(i) is None:
        return M.col_del(i)
    else:
        return M


def _var_col_perms(var_types,
                   dyn_modes=["compact", "extended", "harmonic"],
                   nd_modes=["free", "frozen", "sigma"]):
    
    mode_perms = []
    for dyn_mode in dyn_modes:
        if dyn_mode in var_types:
            mode_perms.append(tuple(itertools.permutations(var_types[dyn_mode], len(var_types[dyn_mode]))))
    
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


def __sol_indep_of_vars(expr, solve_vars, bad_vars):
    """
    Determines whether a solution to the given expr
    exists that is not dependant on polynomial and
    inverse polynomial powers of bad_vars.

    Args:
        expr (sym.Add): equation to be set equal to 0
        solve_vars (list): variables to solve for
        bad_vars (list): variables you would like no dependence on

    Returns:
        list[dict]: list of possible solutions
    """
    # Solve for coeffecicients of bad_vars that all go to zero
    # Multivariable quadratic in solve vars
    
    # Put everything into one fraction
    expr = sym.together(expr)
    numer, denom = expr.as_numer_denom()
    numer = sym.expand(numer)
    # Identify unique products of system parameters
    # and substitute in dummy variables
    var_combos = _unique_products(numer, exclude=solve_vars)
    dummies = {}
    for i, vc in enumerate(var_combos):
        dummies[vc] = sym.Symbol(f"d{i+1}", real=True)
    numer = numer.subs(dummies)
    # Create a system of equations for setting all
    # the coefficients of dummy variables equal to 0
    collected = sym.collect(numer, dummies.values())
    eqs = []
    eqs += [collected.coeff(x) for x in dummies.values()]
    eqs += [collected.coeff(1/x) for x in dummies.values()]
    eqs = [x for x in eqs if x != 0]
    eqs = set(eqs)
    
    if any(v in expr.free_symbols for v in solve_vars):
        # for eq in eqs:
        test = expr
        for v in solve_vars:
            test = test.subs(v,1)
        # return _sol_indep_of_vars(expr, solve_vars, bad_vars)
        

        sol = sym.solve(eqs, solve_vars, dict=True)

        # verify the solution doesn't make the denominator 0
        sol_final = []
        for sol_i in sol:
            if sym.simplify(denom.subs(sol_i)) != 0:
                sol_final.append(sol_i)
        return sol_final
    else:
        return []
    
def _sol_indep_of_vars(expr, solve_vars, bad_vars):
    """
    Determines whether a solution to the given expr exists that is independent
    of bad_vars (in polynomial or inverse polynomial dependence).

    Args:
        expr (sym.Expr): Expression to be set to 0
        solve_vars (list): Variables to solve for
        bad_vars (list): Variables you want no dependence on

    Returns:
        list[dict]: list of solutions that do not depend on bad_vars
    """
    # Simplify to rational form
    expr = sym.together(expr, deep=True)
    numer, denom = expr.as_numer_denom()
    numer = sym.expand(numer)

    # Get unique symbolic products (excluding solve_vars)
    var_combos = _unique_products(numer, exclude=solve_vars)
    
    # Build equations by collecting coefficients of var combos
    eqs = []
    collected = sym.collect(numer, var_combos, evaluate=False)

    for v in var_combos:
        eqs.append(collected.get(v, 0))
        eqs.append(collected.get(1 / v, 0))

    eqs = [eq for eq in eqs if eq != 0]
    eqs = list(set(eqs))  # Remove duplicates

    # If solve_vars appear in the expression, try solving
    if any(v in numer.free_symbols for v in solve_vars):

        sol = sym.solve(eqs, solve_vars, dict=True)

        # Filter out any solution that causes denominator to vanish
        return [s for s in sol if sym.simplify(denom.subs(s)) != 0]
    else:
        return []




def _are_substitutions_compatible(subs_list: list[dict]):
    """
    Determines whether the given list of substitutions
    are all "compatible," meaning they can be
    simultaneously fulfilled.

    Args:
        subs_list (list[dict]): list of substitutions mapping
                                sympy symbol to its value.

    Returns:
        (bool, list[dict]): first entry is whether the substitutions are compatible
                            second entry is a refined set of substitutions that
                            achieve them all simultaneously
    """

    if isinstance(subs_list, dict):
        subs_list = [subs_list]

    # Turn dictionary into a system of equations
    # With a bunch of equals signs
    equations = []
    all_var = set()
    for subs in subs_list:
        for var, val in subs.items():
            var = sym.sympify(var)
            all_var.add(var)
            val = sym.sympify(val)
            equations.append(sym.Eq(var, val))

    sols = sym.solve(equations, all_var, dict=True)

    return len(sols)>0, sols

def _unique_products(expr: sym.Expr, exclude: list[sym.Symbol] = []):
    """
    Return a list of unique products of free symbols
    from an expanded expression, excluding specified symbols.
    """
    expr = sym.expand(expr)
    products = set()

    for term in expr.as_ordered_terms():
        # Extract multiplicative factors
        if isinstance(term, Mul):
            factors = [f for f in term.args
                       if isinstance(f, (sym.Symbol, sym.Pow)) and f not in exclude]
        elif isinstance(term, sym.Pow) and isinstance(term.base, sym.Symbol):
            factors = [term] if term not in exclude else []
        elif isinstance(term, sym.Symbol):
            factors = [term] if term not in exclude else []
        else:
            factors = []

        if factors:
            # Use frozenset to hash without sorting
            products.add(frozenset(factors))

    return [functools.reduce(lambda a, b: a * b, p, sym.S.One) for p in products]



def _find_Z_instance(Z: sym.Matrix, vals=[0,1,-1,2,-2], all_real=True, sort=True):

    to_sub = sorted(Z.free_symbols, key=str)
    det = sym.simplify(Z.det())
    best_val = ""
    best_Z = None
    assignments = itertools.combinations_with_replacement(vals, len(to_sub))
    if sort:
        assignments = sorted(assignments, key=lambda x: sum(abs(xi) for xi in x))
    for assign in assignments:
        subs = dict(zip(to_sub, assign))
        if det.subs(subs) != 0:
            Zsub = Z.subs(subs)
            if sym.im(Zsub).is_zero_matrix or not(all_real):
                return Zsub
    raise ValueError("No Valid Instance Present With Provided Values")

def find_islands(circuit: list, edges: list, links: list):
    """
    Identifies islands in the circuit separated from the rest of the
    circuit by the elements present in links

    Args:
        circuit (list): _description_
        edges (list): _description_
        bridges (list): _description_
    """
    # Remove all edges that contain only the elements in links
    remaining_edges = []
    for i in range(len(edges)):
        to_remove = []
        for elem in circuit[i]:
            if any(l in elem for l in links):
                to_remove.append(True)
            else:
                to_remove.append(False)
        if not all(to_remove):
            remaining_edges.append(edges[i])
    
    # Find the connected components in the resulting graph
    G = nx.Graph()
    G.add_nodes_from(np.unique(np.ndarray.flatten(np.array(edges))))
    G.add_edges_from(remaining_edges)
    return [tuple(c) for c in nx.connected_components(G)]


def well_spaced(wT):
    mags = _unique_mag_row(wT)
    for mag in mags:
        if len(mag) > 2:
            return False
        for m in mag:
            if m not in [0, 1]:
                return False
    return True


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

    # Dynamical and nondynamical blocks
    d_block = list(range(n_d))
    nd_block = list(range(n_d, X.shape[0]))
    return decoupling_transformation_3block(X, nd_block,
                                            d_block, nd_block)
    # coupled = []
    # for i in range(n_d, X.shape[0]):
    #     if any(X[i, n_d:]):
    #         coupled.append(i)

    # # Make the transformation to uncouple them
    # # (I 0)
    # # (M I)
    # # with M = -X_22^-1 X_21
    # Z2 = sym.eye(X.shape[0])
    # X22 = X[coupled, coupled]
    # X21 = X[coupled, :n_d]
    # M = -X22.inv()*X21
    # for i, row in enumerate(coupled):
    #     Z2[row, :n_d] = M[i, :]
    # return Z2


def decoupling_transformation_3block(X:sym.Matrix, block1: Sequence[int],
                                     block2: Sequence[int], block3: Sequence[int]):
    # Transformation is
    #     (I 0 0)
    # Z = (0 I 0)
    #     (0 M I)
    #(block1, block2, block3)
    #
    # Decouples
    #       (X11 X12 X13)        (- X12 + X13*M -)
    # Z^T * (X21 X22 X23) * Z =  (X21 + M^T*X31 - -)
    #       (X31 X32 X33)        (- - -)
    #
    #
    # By setting M = -X13^-1 * X12
    #
    # Simplifies to the two block version
    #
    # (I 0)
    # (M I)
    # with M = -X_22^-1 X_21
    #
    # when block1 = block3

    # Simplify to make sure zero entries appear
    # as zero entries

    # Variables in block n1 coupled to 
    # variables in block n2 for key [n1][n2]
    coupled = {}
    for key in ["12", "21", "13", "31"]:
        coupled[key] = []
   
    # Determine which block 1 variables are
    # coupled to block 3 variables
    for i in block1:
        for j in block3:
            if sym.simplify(X[i, j]) != 0:
                coupled["13"].append(i)
                coupled["31"].append(j)
    # block 2 and block 1
    for i in block1:
        for j in block2:
            if sym.simplify(X[i, j]) != 0:
                coupled["12"].append(i)
                coupled["21"].append(j)
    
    # Sort and keep unique entries
    for key in coupled:
        coupled[key] = sorted(set(coupled[key]))

    # Nothing to decouple
    # or not enough degrees of freedom to decouple
    if (len(coupled["12"]) == 0 or len(coupled["13"]) == 0 or
        len(coupled["21"])*len(coupled["31"]) < len(_unique_vals(X[coupled["12"],coupled["21"]]))):
        return sym.eye(X.shape[0])
    


    Z2 = sym.eye(X.shape[0])
    X13 = X[coupled["13"], coupled["31"]]
    # May be able to do it with fewer block3 variables
    X13_LI = _linearly_indep_cols(X13)
    if len(X13_LI) != X13.shape[1]:
        coupled["13"] = X13_LI
    X12 = X[coupled["13"], coupled["21"]]
    M = -X13.pinv()*X12
    for i, row in enumerate(coupled["31"]):
        for j, col in enumerate(coupled["21"]):
            Z2[row, col] = M[i, j]

    # Verify transformation
    Xtrans = Z2.transpose()*X*Z2
    for i in coupled["12"]:
        for j in coupled["21"]:
            if sym.simplify(Xtrans[i,j]) != 0:
                raise ValueError("Decoupling Transformation Unsuccessful")
    
    return sym.simplify(Z2)


def unique_compact_extended(circuit, edges, nd_mat, cMat=None, lMat=None):

    if cMat is None:
        cMat = gen_cap_mat(circuit, edges)
    
    if lMat is None:
        lMat = gen_ind_mat(circuit, edges)

    ## Compact -- J,C shunted islands
    JC_islands = _islands_to_vectors(circuit, edges, ["J", "C"])
    LC_islands = _islands_to_vectors(circuit, edges, ["L", "C"])
    sigma_vec = _islands_to_vectors(circuit, edges, [])
    C_islands = _islands_to_vectors(circuit, edges, ["C"])
    L_islands = _islands_to_vectors(circuit, edges, ["L"])
    wJ = gen_w(circuit, edges, "J")
    n_nd = nd_mat.shape[1]
    n_comp = len(_linearly_indep_cols(sym.Matrix.hstack(*JC_islands, nd_mat))) - n_nd
    n_ext = len(_linearly_indep_cols(wJ)) - n_comp
    Z_final = []

    # No compact variables
    # Consider all possible wJ psuedoinverses
    if n_comp == 0:
        for col_set in _linearly_indep_col_sets(wJ):
            Z = wJ[:, col_set].transpose().pinv()
            # Decouple columns
            for j in range(Z.shape[1]):
                Z[:, j] = decouple_column(Z[:, j], nd_mat, cMat)
                Z[:, j] = decouple_column(Z[:, j], nd_mat, lMat)
            if len(_find_equiv_mats(Z, Z_final,
                                    shifts=sigma_vec)) == 0:
                Z_final.append(Z)
        return Z_final

    # Yes compact variables
    # First identify a possible choice of correctly
    # scaled compact variables
    # Choose to work with the set of columns that
    # yields the most uncoupled junctions
    most_uncoupled = -1
    all_PI_cols = []
    best_wJpi = None
    best_in_wJpi = None
    best_coupled = None
    best_only_ext = None
    for col_set in _linearly_indep_col_sets(wJ):
        wJpi = wJ[:, col_set].transpose().pinv()

        # Identify all unique PI columns
        for j in range(wJpi.shape[1]):
            if len(_find_equiv_cols(wJpi[:, j], all_PI_cols,
                                    shifts=sigma_vec+L_islands+C_islands)) == 0:
                all_PI_cols.append(wJpi[:, j])


        comp_vars_base, _, in_wJpi, _ = _vec_space_overlap(JC_islands, wJpi,
                                                    support=LC_islands,
                                                    return_decomp=True)
        in_wJpi = sym.Matrix.hstack(*in_wJpi).transpose()
        comp_vars_base = [decouple_column(v, nd_mat, cMat) for v in comp_vars_base]

        # Decouple columns of wJpi from nd vars for building extended variables
        for j in range(wJpi.shape[1]):
            wJpi[:, j] = decouple_column(wJpi[:, j], nd_mat, cMat)
            wJpi[:, j] = decouple_column(wJpi[:, j], nd_mat, lMat)

        # junctions that don't depend on compact variables
        only_ext = list(range(wJpi.shape[1]))

        # junctions that do depend on compact variables
        coupled = []
        for i in range(in_wJpi.shape[0]):
            vi = in_wJpi[i, :]
            by_vi = []
            for j in range(len(vi)):
                # Record variables that are coupled
                # by the compact variables
                if vi[j] != 0:
                    by_vi.append(j)
                    if j in only_ext:
                        only_ext.remove(j)
                        coupled.append(j)
        
        if len(only_ext) > most_uncoupled:
            best_wJpi = wJpi
            best_in_wJpi = in_wJpi
            best_coupled = coupled
            best_only_ext = only_ext
            
    # Now enumerate the options
    ext_vec_uncoupled = []
    # Uncoupled junctions -> columns of wJpi
    for j in best_only_ext:
        ext_vec_uncoupled.append(best_wJpi[:, j])
    # Coupled variables -> +/- 1 combinations of columns of best_wJpi
    # Different choices of compact variables represent transformations
    # on only the compact subspace, so it won't change which ext
    # constraints are linearly independent
    ext_vec_coupled = []
    if len(best_coupled) > 1:
        # We need a number of variables equal
        # to remaining linearly independent
        # degrees of freedom for the specified best_wJpi columns
        n_elem = len(best_coupled) - len(_linearly_indep_cols(best_in_wJpi[:, best_coupled]))
        if n_elem > 0:
            # [best_wJpi[:,j] for j in best_coupled]
            # all_PI_cols
            ext_vec_coupled += [_unique_col_combos([best_wJpi[:,j] for j in best_coupled],
                                                n_elem, signs=[1, -1],
                                                li_vecs=comp_vars_base,
                                                valid=lambda Ze: well_spaced(wJ.transpose()*Ze),
                                                shifts=sigma_vec+L_islands+C_islands)]


    # Now consider ''Center-ing'' the compact variables
    # on different junctions to generate unique choices of
    # compact variable to combine with the extended
    wJtrans = sym.simplify(wJ.transpose()*sym.Matrix.hstack(*comp_vars_base))
    for Zc in compact_alignment_transformation(wJtrans, n_comp):
        comp_vars = sym.Matrix.hstack(*comp_vars_base)*Zc
        # Fully uncoupled case returned earlier
        if len(ext_vec_coupled) == 0:
            Z = sym.Matrix.hstack(comp_vars, *ext_vec_uncoupled)
            if len(_find_equiv_mats(Z, Z_final, shifts=sigma_vec)) == 0:
                    Z_final.append(Z)
        else:
            for ext_coupled in list(itertools.product(*ext_vec_coupled)):
                Z = sym.Matrix.hstack(comp_vars, *ext_coupled, *ext_vec_uncoupled)
                if len(_find_equiv_mats(Z, Z_final, shifts=sigma_vec)) == 0:
                    Z_final.append(Z)

    return Z_final


def unique_harmonic(circuit, edges, nd_mat, cMat=None, lMat=None):

    if cMat is None:
        cMat = gen_cap_mat(circuit, edges)
    if lMat is None:
        lMat = gen_ind_mat(circuit, edges)

    # Linearly independent incidence matrices
    wC = gen_w(circuit, edges, "C")
    wC = wC[:, _linearly_indep_cols(wC)]
    wCpi = wC.transpose().pinv()
    wL = gen_w(circuit, edges, "L")
    wL = wL[:, _linearly_indep_cols(wL)]
    wLpi = wL.transpose().pinv()


    # Shifts
    shifts=[nd_mat[:,j] for j in range(nd_mat.shape[1])]

    ## Harmonic -- L,C shunted islands
    # Are there any LC islands that have both?
    LC_islands = _islands_to_vectors(circuit, edges, ["L", "C"])
    # number of harmonic variables is the number of LC islands
    # minus the number of free/sigma variables that can be made
    # from those islands
    n_nd = nd_mat.shape[1]
    n_harm = len(_linearly_indep_cols(sym.Matrix.hstack(nd_mat, *LC_islands))) - n_nd
    if n_harm > 0:
        # Identify PI columns that correspond
        # to inductors that are involved in the
        # harmonic mode - i.e. connected to L and C
        # but not J
        harm_vec = []
        for v in _vec_space_overlap(LC_islands, wLpi, support=wL.transpose().nullspace()):
            if not (wC.transpose()*v).is_zero_matrix:
                v = decouple_column(v, nd_mat, cMat)
                v = decouple_column(v, nd_mat, lMat)
                if len(_find_equiv_cols(v, harm_vec, shifts=shifts))==0:
                    harm_vec.append(v)
    else:
        return []
    
    # Now only need one
    M = sym.Matrix.hstack(*harm_vec)
    cols = _linearly_indep_cols(M)
    return [M[:, cols]]

    # return _unique_col_combos(harm_vec, n_harm, signs=[1, -1], shifts=shifts,
    #                             li_vecs=[nd_mat[:, j] for j in range(n_nd)])
    

def H_hash(Z, var_types, cMat, lMat, wJ, equalJ=False,
           dyn_modes=["compact", "extended", "harmonic"],
           nd_modes=["free", "frozen", "sigma"], try_perms = True,
           eps=1e-10):
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

    # Numerically treat the matrices
    C, cVals = num_subs(cMat, symbol="C")
    Z, _ = num_subs(Z, symbol="C", vals_in=cVals, hermitify=False)
    L, lVals = num_subs(lMat, symbol="L")
    Z, _ = num_subs(Z, symbol="L", vals_in=lVals, hermitify=False)
    n_nodes = Z.shape[0]

    C = np.array(C).astype(float)
    L = np.array(L).astype(float)
    Z = np.array(Z).astype(float)
    wJ = np.array(wJ).astype(float)

    # Record number of modes
    ext_var = var_types.get("extended", [])
    n_ext = len(ext_var)
    comp_var = var_types.get("compact", [])
    n_comp = len(comp_var)
    harm_var = var_types.get("harmonic", [])
    n_harm = len(harm_var)

    mode_str = f"{n_comp}{n_ext}{n_harm}"
    n_nd = n_nodes - n_ext - n_comp - n_harm
    n_nl = n_ext + n_comp

    # All different ways to arrange columns
    # Interchanging like variable columns
    if try_perms:
        perms = _var_col_perms(var_types, dyn_modes, nd_modes)
    else:
        perms = [tuple(range(Z.shape[0]))]

    lowest_hash = ""
    lowest_Z = None
    for perm in perms:

        Z_perm = Z[:, perm]
    
        # Transformed capacitance and inductance matrices
        C_tilde = Z_perm.T@C@Z_perm
        L_tilde = Z_perm.T@L@Z_perm
        # Truncate to dynamical modes
        C_tilde = C_tilde[:-n_nd, :-n_nd]
        L_tilde = L_tilde[:-n_nd, :-n_nd]
        
        # Invert capacitance matrix and trim small numerical values
        try:
            C_tilde_inv = np.linalg.inv(C_tilde)
            C_tilde_inv[np.abs(C_tilde_inv)/np.abs(C_tilde_inv).max() < eps] = 0
        except:
            breakpoint()
        if np.abs(L_tilde).max() > 0:
            try:
                L_tilde[np.abs(L_tilde)/np.abs(L_tilde).max() < eps] = 0
            except:
                breakpoint()
        # Key for C, L = [n_coupled]-[nz entries of off diag]
        C_key =  _nonzero_entries_str(C_tilde_inv)
        L_key = _nonzero_entries_str(L_tilde)

        if wJ.shape[1] > 0:
            wT_tilde = wJ.T@Z_perm
            wT_tilde = wT_tilde[:, :-n_nd]
            # Put wT into cananocal ordering
            wT_tilde, _, _ = _maximize_wT(_sort_wT(wT_tilde))
            # Key for wT = [n_coupled]-[nz entries of off diag]
            w_key = _wT_key(wT_tilde, equalJ = equalJ)
        else:
            w_key = "0-"+"0"*(len(L_key)-2)

        # TODO: At the end add a base 3 wT key to encode exact nonlinear form

        # Make the hash string
        Z_hash = "_".join([mode_str, w_key, str(int(L_key[0])+int(C_key[0])), L_key, C_key])
        if lowest_hash == "" or Z_hash < lowest_hash:
            lowest_hash = Z_hash
            lowest_Z = Z_og[:, perm]

    return lowest_hash, lowest_Z


def gen_cap_mat(circuit, edges):
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

    # Generate a blank matrix
    edges = utils.zero_start_edges(edges)
    n_nodes = utils.get_num_nodes(edges)
    cap_mat = sym.Matrix(np.zeros((n_nodes, n_nodes)))

    C = {}
    CJ = {}
    for elems in circuit:
        for elem in elems:
            if elem in C or elem in CJ:
                continue
            if "C" in elem:
                C[elem] = sym.Symbol(elem, positive=True, real=True)
            elif "J" in elem:
                suffix = elem.replace("J", "")
                CJ[elem] = sym.Symbol("C_{J" + suffix + "}", positive=True, real=True)

    # Fill in capacitance values
    for edge, elems in zip(edges, circuit):
        i, j = edge
        val = 0
        for elem in elems:
            if "C" in elem:
                val += C[elem]
            elif "J" in elem:
                val += CJ[elem]

        cap_mat[i, i] += val
        cap_mat[j, j] += val
        cap_mat[i, j] += -val
        cap_mat[j, i] += -val

    return cap_mat


def gen_ind_mat(circuit, edges):
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

    # Generate a blank matrix
    edges = utils.zero_start_edges(edges)
    n_nodes = utils.get_num_nodes(edges)
    ind_mat = sym.Matrix(np.zeros((n_nodes, n_nodes)))

    L = {}
    for elems in circuit:
        for elem in elems:
            if elem in L:
                continue
            if "L" in elem:
                L[elem] = sym.Symbol(elem, positive=True, real=True)

    # Fill in inductance values
    for edge, elems in zip(edges, circuit):
        i, j = edge
        val = 0
        for elem in elems:
            if "L" in elem:
                val += 1/L[elem]
        ind_mat[i, i] += val
        ind_mat[j, j] += val
        ind_mat[i, j] += -val
        ind_mat[j, i] += -val

    return ind_mat


def gen_w(circuit: list, edges: list, w_elem: str = "J"):
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
    edges = utils.zero_start_edges(edges)
    n_nodes = utils.get_num_nodes(edges)
    n_elem = utils.count_elems_mapped(circuit)[w_elem]
    if n_elem == 0:
        return sym.Matrix([[]])
    w = sym.Matrix(np.zeros((n_nodes, n_elem), dtype=int))
    w_count = 0
    for edge, elems in zip(edges, circuit):
        for e in elems:
            if w_elem in e:
                w[min(edge), w_count] = 1
                w[max(edge), w_count] = -1
                w_count += 1
    return w


def gen_spaced_var_trans(circuit, edges, cMat=None, lMat=None):
    """
    Generates all variable transformations that
    separate periodic, extended, harmonic,
    free (no potential), frozen (no kinetic), and
    sigma (neither potential nor kinetic) variables.

    The transformation must also maintain the spacing
    of the position variable for periodic junction terms,
    extended junction terms, and harmonic inductor terms.

    Orders variables in [compact, extended, harmonic,
                            free, frozen, sigma] order

    Args:
        circuit (_type_): _description_
        edges (_type_): _description_
    """
    # match indices with array indices
    edges = utils.zero_start_edges(edges)
    n_nodes = utils.get_num_nodes(edges)

    # Incidence matrices
    wC = gen_w(circuit, edges, "C")
    wL = gen_w(circuit, edges, "L")
    wJ = gen_w(circuit, edges, "J")

    # Capacitance and susceptance matrices
    if cMat is None:
        cMat = gen_cap_mat(circuit, utils.zero_start_edges(edges))
    if lMat is None:
        lMat = gen_ind_mat(circuit, utils.zero_start_edges(edges))

    # labels of mode types
    mode_types = []
    # dynamical and nondynamical mode vectors
    dy_modes = []
    nd_modes = []
    
    ## sigma -- each set of connected components (should only be 1)
    sig_vec = _islands_to_vectors(circuit, edges, [])
    if len(sig_vec) > 1:
        print("--- WARNING: Multiple disconnected circuit segments ---")
    ## Free -- capacitively shunted islands
    C_islands = _islands_to_vectors(circuit, edges, ["C"])
    free_vec = C_islands
    ## Frozen -- inductively shunted islands
    L_islands = _islands_to_vectors(circuit, edges, ["L"])
    froz_vec = L_islands

    # Single nondynamical variable matrix -- doesn't matter
    # which you pick, but keep track
    nd_mat = sym.Matrix.hstack(*sig_vec, *froz_vec, *free_vec)
    nd_cols = _linearly_indep_cols(nd_mat)
    nd_mat = nd_mat[:, nd_cols[::-1]]
    divide = (len(sig_vec), len(sig_vec+froz_vec), len(sig_vec+free_vec+froz_vec))
    nd_types = {"free": [], "frozen": [], "sigma": []}
    count = n_nodes-1
    for i in range(len(sig_vec+free_vec+froz_vec)):
        if i in nd_cols:
            if i < divide[0]:
                nd_types["sigma"] += [count]
            elif divide[0] <= i < divide[1]:
                nd_types["frozen"] += [count]
            else:
                nd_types["free"] += [count]
            count += -1
    
    # Number of each variable dynamical type
    n_nd = nd_mat.shape[1]
    LC_islands = _islands_to_vectors(circuit, edges, ["L", "C"])
    JC_islands = _islands_to_vectors(circuit, edges, ["J", "C"])
    n_comp = len(_linearly_indep_cols(sym.Matrix.hstack(*JC_islands, nd_mat))) - n_nd
    n_harm = len(_linearly_indep_cols(sym.Matrix.hstack(*LC_islands, nd_mat))) - n_nd
    n_ext = len(_linearly_indep_cols(wJ)) - n_comp
    # print("----------------")
    # print(circuit, edges)
    # print("n_comp", n_comp, "n_harm", n_harm, "n_ext", n_ext)
    
    ## Compact -- J,C shunted islands
    u_harm = unique_harmonic(circuit, edges, nd_mat, cMat, lMat)
    u_comp_ext = unique_compact_extended(circuit, edges, nd_mat, cMat, lMat)

    # verify number of harmonic
    if n_harm == 0:
        assert len(u_harm) == 0
    else:
        try:
            assert n_harm == u_harm[0].shape[1]
        except:
            breakpoint()

    # Categorize variable types
    var_types = {}
    if n_comp > 0:
        var_types["compact"] = list(range(n_comp))
    if n_ext > 0:
        var_types["extended"] = list(range(n_comp, n_comp+n_ext))
    if n_harm > 0:
        var_types["harmonic"] = list(range(n_comp+n_ext, n_comp+n_ext+n_harm))
    for nd_mode in nd_types:
        if len(nd_types[nd_mode]) > 0:
            var_types[nd_mode] = nd_types[nd_mode]
    try:
        assert n_nd + n_comp + n_harm + n_ext == n_nodes
    except:
        breakpoint()

    # Enumerate combinations for Z
    all_Z = []
    present_modes = [x for x in (u_comp_ext, u_harm) if len(x) > 0]
    eye = sym.eye(n_nodes)
    for dyn_cols in itertools.product(*present_modes):
        Z = sym.Matrix.hstack(*dyn_cols, nd_mat)
        # Bad combination of extended and harmonic
        if Z.det() == 0:
            continue
        cTrans = Z.transpose()*cMat*Z
        lTrans = Z.transpose()*lMat*Z
        if (decoupling_transformation(cTrans, n_comp+n_ext+n_harm) != eye or
            decoupling_transformation(lTrans, n_comp+n_ext+n_harm) != eye):
            raise ValueError("Decoupling Transformation Failed")
        all_Z.append(Z)
        
    return all_Z, var_types








def secondary_decouple(Z0: sym.Matrix, var_types: dict[str, list[int]],
                       cMat: sym.Matrix, lMat: sym.Matrix,
                       return_instance: bool = False) -> sym.Matrix:
    """
    Performs a secondary decoupling transformation of
    the form
    [ I      0      0       0 ]
    [ 0      I      0       0 ]
    [ 0     Z_eh  Z_hh      0 ]
    [ 0      0      0       I ]
    and chooses the blocks Z_eh and Z_hh to maximize the
    number of nonzero off diagonal elements in the
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


    eye = sym.eye(Z0.shape[0])
    comp = var_types.get("compact", [])
    n_comp = len(comp)
    ext = var_types.get("extended", [])
    n_ext = len(ext)
    harm = var_types.get("harmonic", [])
    n_harm = len(harm)

    # No transformation to do
    if n_harm == 0:
        return eye
    
    cTrans = sym.simplify(Z0.transpose()*cMat*Z0)
    lTrans = sym.simplify(Z0.transpose()*lMat*Z0)

    ## Both Possible Secondary Transformations
    # Zhe != 0
    Z2 = sym.eye(Z0.shape[0])
    # Zh != I
    Z3 = sym.eye(Z0.shape[0])
    varsZ2 = []
    varsZ3 = []
    for i in range(Z2.shape[0]):
        for j in range(Z2.shape[1]):
            if j in ext and i in harm:
                Z2[i,j] = sym.Symbol(f"Zeh{i}{j}", real=True)
                varsZ2.append(Z2[i,j])
            elif j in harm and i in harm:
                Z3[i,j] = sym.Symbol(f"Zh{i}{j}", real=True)
                varsZ3.append(Z3[i,j])
    Z = Z2*Z3
    vars = varsZ2+varsZ3
    n_vars = len(vars)

    cTrans2 = Z.transpose()*cTrans*Z
    cTransInv2 = cTrans2[:n_comp+n_ext+n_harm,:n_comp+n_ext+n_harm].inv()
    lTrans2 = Z.transpose()*lTrans*Z

    # Determine which entries are possible to decouple
    # without depending on system parameters
    params =  [x for x in cTransInv2.free_symbols if "C" in str(x).upper()]
    params += [x for x in lTrans2.free_symbols if "L" in str(x).upper()]


    # Try to set to zero as many coupling terms as possible
    coupling_c = []
    coupling_l = []
    coupling_terms = {}
    # Identify coupling terms that have
    # Solutions Independent of circuit parameters
    det = sym.simplify(Z.det())
    for i1 in comp:
        for i2 in ext + harm:
            coupling_terms[("C", i1, i2)] = _sol_indep_of_vars(cTransInv2[i1, i2], vars, params)
    for i1 in ext + harm:
        for i2 in ext + harm:
            # Symmetric coupling
            if i2 <= i1:
                continue
            coupling_terms[("C", i1, i2)] = _sol_indep_of_vars(cTransInv2[i1, i2], vars, params)
            coupling_terms[("L", i1, i2)] = _sol_indep_of_vars(lTrans2[i1, i2], vars, params)
    # Remove empty solutions and those that make the transformation not inverible
    for k in list(coupling_terms.keys()):
        sols = coupling_terms[k]
        new_sols = []
        for sol in sols:
            # Invertible
            if sym.simplify(det.subs(sol)) != 0:
                new_sols.append(sol)
        # empty solution
        if len(new_sols) == 0:
            del coupling_terms[k]
        else:
            coupling_terms[k] = new_sols
    
    # Maximize the number of coupling terms that go to zero
    n_couple = len(coupling_terms)
    incompatible = []
    compatible = {}
    compatible_keys = []
    best_Z = eye
    best_val, _ = H_hash(best_Z, var_types, cTrans, lTrans, wJ=sym.Matrix([]))
    for nz in range(1, n_couple):
        # Pidgeonhole principle, is it possible to to select nz keys that
        # might be compatible
        if len(incompatible) > 0:
            if nz > n_couple - max(len(incompat_keys) for incompat_keys in incompatible):
                break
        for keys in itertools.combinations(coupling_terms.keys(), nz):
            # We already know a subset of the keys are incompatible
            if any(all(k in keys for k in incompat_keys) for incompat_keys in incompatible):
                continue
            # Insert previous simplifications
            all_rules = []
            did_already = []
            # Loop through in reverse order of adding
            # to get bigger simplifications
            for compat_keys in compatible_keys[::-1]:
                if all(k in keys for k in compat_keys):
                    all_rules.append(compatible[compat_keys])
                    did_already += compat_keys
            for k in keys:
                if k not in did_already:
                    all_rules.append(coupling_terms[k])
            # Enumerate combinations of rules that make the relevant entries zero
            for rules in itertools.product(*all_rules):
                # First check if the selected rules give a zero determinant -- not invertible
                this_det = det
                for sub in rules:
                    this_det = this_det.subs(sub)
                if this_det == 0:
                    incompatible.append(keys)
                    continue
                # Second, check if the selected rules can all be fulfilled simultaneously
                is_compat, compat_subs = _are_substitutions_compatible(rules)
                if is_compat:
                    # Test different substitutions that
                    # are compatible with the chosen rules
                    good_subs = []
                    for Z_sub in compat_subs:
                        Ztest = sym.simplify(Z.subs(Z_sub))
                        # Double check determinant != 0
                        if sym.simplify(det.subs(Z_sub)) == 0:
                            continue
                        Z_free = list(Ztest.free_symbols)
                        # Insert random for remaining free variables
                        # for hashing
                        Zhash = _find_Z_instance(Ztest, sort=False)
                        val, Z_perm = H_hash(Zhash, var_types, cTrans, lTrans, wJ=sym.Matrix([]))
                        if val < best_val:
                            best_val = val
                            best_Z = Ztest
                        good_subs.append(Z_sub)
                    # Save the solved compatibility (or incompatibility) for these keys
                    if len(good_subs) > 0:
                        compatible[keys] = good_subs
                        compatible_keys.append(keys)
                    else:
                        incompatible.append(keys)
                else:
                    incompatible.append(keys)

    if return_instance:
        best_Z = _find_Z_instance(best_Z, vals=[0,1])
    return best_Z


def choose_Z(circuit: list, edges: list,
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
    
    # Generate capacitance matrix, susceptance matrix, and incidence matrix
    cMat = gen_cap_mat(circuit, utils.zero_start_edges(edges))
    lMat = gen_ind_mat(circuit, utils.zero_start_edges(edges))
    wJ = gen_w(circuit, edges, w_elem="J")

    
    all_Z, var_types = gen_spaced_var_trans(circuit, edges, cMat, lMat)
    lowest_hash = ""
    lowest_Z = []
    # print(len(all_Z), "transformations", var_types)
    # print("circuit=",circuit)
    # print("edges=",edges)
    for Z in all_Z:
        if Z.det() == 0:
            breakpoint()
        # Consider secondary harmonic extended transformation
        Z2  = secondary_decouple(Z, var_types, cMat, lMat, wJ)
        val, Z_perm = H_hash(Z*Z2, var_types, cMat, lMat, wJ)
        if val < lowest_hash or lowest_hash == "":
            lowest_Z = [Z_perm]
            lowest_hash = val
        elif val == lowest_hash:
            lowest_Z.append(Z_perm)
    
    # If there are multiple with the same lowest hash
    # then see if they separate with equal parameter values
    Z_final = lowest_Z[0]
    hash_final = lowest_hash
    for Z in lowest_Z[1:]:
        Z_equal = sym.simplify(_sub_equal_LC(Z))
        val, Z_perm = H_hash(Z_equal, var_types,
                            cMat, lMat, wJ, equalJ=True)
        if val < hash_final:
            hash_final = val
            Z_final = Z

    if return_instance:
        Z_final = _find_Z_instance(Z_final, vals=[0,1])
    
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
                if vals_in == {}:
                    vals[x_str] = rand_range[0] + (rand_range[1]-rand_range[0])*np.random.random()
                # prescribed value
                else:
                    vals[x_str] = vals_in[x_str]
                C = C.subs(x, vals[x_str])

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
    edges = utils.zero_start_edges(edges)
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
        breakpoint()
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
    circuit= [('C_1', 'L_1'), ('C_2', 'L_2'), ('C_3', 'L_3'), ('C_4', 'L_4'), ('C_5', 'J_1', 'L_5'), ('C_6', 'J_2', 'L_6')]
    edges= [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    # circuit= [('J_1',), ('J_2',), ('J_3',), ('C_1', 'L_1'), ('C_2', 'L_2'), ('C_3', 'L_3')]
    # edges= [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    cMat = gen_cap_mat(circuit, edges)
    lMat = gen_ind_mat(circuit, edges)
    edges = utils.zero_start_edges(edges)
    all_Z, var_types = gen_spaced_var_trans(circuit, edges)
    Z0 = all_Z[0]
    cTrans = Z0.transpose()*cMat*Z0
    lTrans = Z0.transpose()*lMat*Z0
    print(Z0)
    
    times = []
    for i in range(10):
        t0 = time.time()
        Z = secondary_decouple(Z0, var_types, cMat, lMat, True)
        # print(Z)
        tf = time.time()
        times.append(tf-t0)

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