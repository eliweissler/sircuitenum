__doc__ = "quantize.py: contains functions used to produce symbolic hamiltonians"
__author__ = "Eli Weissler"
__version__ = "0.1.0"
__all__ = ["gen_cap_mat", "gen_ind_mat", "gen_junc_pot", "quantize_circuit"]

import itertools
import functools
import time
from copy import deepcopy

from dataclasses import dataclass, field
from typing import Union, Sequence, Iterable, Mapping, Optional, Tuple

import sympy as sym
import numpy as np
import networkx as nx

from sympy import collect, expand_mul, Mul, Dummy
from sympy.core.add import Add

from sircuitenum import utils
from sircuitenum.equationset import EquationSet, cached_solve, fully_compatible_set, maximally_compatible_set, sol_indep_of_vars, extract_denom, unique_solutions



PERIODIC_CHARGE = "n"
PERIODIC_PHASE = "θ"
EXTENDED_CHARGE = "q"
EXTENDED_PHASE = "φ"
NODE_CHARGE = "q"
NODE_PHASE = "ϕ"
EXT_CHARGE = "n_g"
EXT_PHASE = "_{ext}"

# Possible integer values
WJ_VALS = [0,1,-1,2,-2,3,-3,4,-4]


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
    """
    Are two transformation columns equivalent up to a change
    in sign and and given shifts?

    Args:
        c1 (sym.Matrix): column 1
        c2 (sym.Matrix): column 2
        shifts (list[sym.Matrix], optional): Shifts. Defaults to global 1.

    Returns:
        bool: True if equivalent up to sign and shifts. False otherwise.
    """
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
        A (sym.Matrix): Transformation matrix A
        B (sym.Matrix): Transformation matrix B
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
                if not sym.simplify(X[i,j]).is_zero:
                    nz_idx.append("1")
                    n_nz += 1
                else:
                    nz_idx.append("0")

    bin_str = "".join(nz_idx)
    if prepend_sum:
        bin_str = str(n_nz) + "-" + bin_str
    return bin_str 


def _remove_permutation_equivalent_transformations(Z_list, perms):
    """Filter transformations that are column permutations or sign-flips of each other. Considers
    subsets of columns based on variable types. Substitutes dummy variables to allow for equality
    checks even when symbolic variables differ.

    For each transformation matrix in ``Z_list``, generates column permutations
    based on perms and checks if it's equivalent (up to column shifts
    and sign flips) to any previously kept transformation.

    Parameters
    ----------
    Z_list : list[sym.Matrix]
        List of transformation matrices to filter.
    perms : list[tuple[int]]
        List of column index permutations to consider for equivalence.

    Returns
    -------
    tuple[list[sym.Matrix], list[str]]
        Filtered lists of transformation matrices and their nonzero strings.
    """
    unique_Z = []
    idx_keep = []
    for k, Z_k in enumerate(Z_list):
        Z_set = []
        to_add = True
        for perm in perms:
            Z_perm = Z_k[:, perm]
            # Make dummy variables to allow for equality check
            subs = {}
            v_count = 1
            for j in range(Z_perm.shape[1]):
                for i in range(Z_perm.shape[0]):
                    for v in Z_perm[i, j].free_symbols:
                        if v not in subs:
                            subs[v] = sym.Symbol(f"v_{v_count}", real=True)
                            v_count += 1
            Z_perm = Z_perm.subs(subs)
            
            # Is there any past transformation equivalent to this one?
            if any(any(_equal_up_to_column_shift_and_sign(Z_past, Z_perm, shifts=[]) 
                      for Z_past in Z_past_set) 
                  for Z_past_set in unique_Z):
                to_add = False
                break
            Z_set.append(Z_perm)
        
        if to_add:
            unique_Z.append(Z_set)
            idx_keep.append(k)
    
    return [Z_set[0] for Z_set in unique_Z], idx_keep


def _sort_wT(wT: Union[sym.Matrix, np.ndarray]):
    if not isinstance(wT, np.ndarray):
        wT = np.array(wT)
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


    # Sort to put rows with identical nz entries
    # next to each other
    wT = _sort_wT(wT)

    # TODO: COULD DO THIS WITH SWAP PERMUTATIONS FUNCTION
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

    base_n = len(WJ_VALS)
    add = (base_n - 1)//2

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
                    wT_perm = wT_mod[row_order, :].astype(int)
                    # Flattened matrix in base n -- for canonical ordering
                    key = "".join((wT_perm + add).flatten().astype(str))
                    if val > best_val or (val == best_val and key > best_key):
                        best_val = val
                        best_key = key
                        best_w = (wT_perm.copy(), row_vec.copy(),
                                  col_vec.copy(), row_order)

        
    return best_w[0], best_key, best_w[1:]


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


# Deterministic symbolic instantiation helper
def _find_Z_instance_deterministic(Z: sym.Matrix, var_list: list[sym.Symbol],
                                   max_tries=5, nonzero=[], return_mapping=False):
    """
    Deterministically substitute symbolic parameters in Z with
    rational values guaranteeing (if possible) a nonzero det.
    """
    
    
    ordered = sorted([v for v in Z.free_symbols if v in var_list], key=str)
    if not ordered:
        return Z
    subs = {v: sym.Rational(i+1, len(ordered)+1) for i, v in enumerate(ordered)}
    det = sym.simplify(Z.det())
    tries = 0
    while det.is_zero or any(sym.simplify(d.subs(subs))== 0 for d in nonzero):
        # Try and perturb the values a bit
        for v in subs:
            subs[v] += sym.Rational(1, len(ordered)+1)
        det = sym.simplify(Z.det().subs(subs))
        tries += 1
        if tries > max_tries:
            # Give up
            raise ValueError("Could not find non-singular instance")
    if return_mapping:
        return sym.nsimplify(Z.subs(subs), rational=True), subs
    return sym.nsimplify(Z.subs(subs), rational=True)


def _find_Z_instance(Z: Union[sym.Matrix, sym.Expr], var_list: list[sym.Expr],
                     var_types: dict, wJ: sym.Matrix, vals=WJ_VALS,
                     all_real=True, sort=True, nonzero=[]):
    
    # If no variables, just return
    if len(Z.free_symbols) == 0:
        return Z
    
    
    n_nl = len(var_types.get("compact", [])) + len(var_types.get("extended", []))

    # Sort vals by abs value, with positive first
    vals = sorted(vals, key=lambda x: (abs(x), -x))
    
    if wJ.shape[0] == 0 or n_nl == 0:
        # Just try and find a random instance
        return _find_Z_instance_deterministic(Z, var_list, max_tries=10, nonzero=nonzero)
        # return _find_Z_instance_random(Z, var_list, max_tries=10, nonzero=nonzero)

    # Variables to substitute concrete values in for
    det = sym.simplify(Z.det())

    # Identify a valid assignment that yields a non-singular Z
    # with the specified values
    # Try and identify a transformation that 
    ext = var_types.get("extended", [])
    wJT_trans = wJ.transpose()*Z
    all_eqs = []
    all_weights = []
    zero_idx = []
    idx = 0
    for i in range(wJT_trans.shape[0]):
        for j in ext:
            val = sym.simplify(wJT_trans[i,j])
            if val != 0:
                sols = []
                weights = []
                for v_possible in vals:
                    specific_sols = cached_solve([sym.Eq(val, v_possible)], [v for v in var_list if v in val.free_symbols])
                    sols += sorted(specific_sols, key=lambda x: (len(str(x)), str(x)))
                    weights += [abs(v_possible)]*len(specific_sols)
                    if specific_sols and v_possible == 0:
                        zero_idx.append(idx)
                if sols:
                    all_eqs.append(sols)
                    all_weights.append(weights)
                else:
                    raise ValueError("No Valid Instance Present With Provided Values")
                idx += 1

    s2 = fully_compatible_set(all_eqs, solve_vars=var_list, nonzero=nonzero+[det], depth_first=True)
    min_key = ""
    best_s = None
    WJ_zero = len(WJ_VALS)//2
    for s in s2:
        try:
            wJT_trans = sym.simplify((wJ.transpose()*Z).subs(s))
        except:
            breakpoint()
        # print(s)
        # print(wJT_trans)
        _, key, _ = _maximize_wT(wJT_trans[:, :n_nl])
        n_val = sum(int(x) != WJ_zero for x in key)
        key = str(n_val) + "-" + key
        if key < min_key or min_key == "":
            min_key = key
            best_s = s

    if len(s2) == 0:
        # breakpoint()
        raise ValueError("No Valid Instance Present With Provided Values")
    else:
        Zsubs = Z.subs(best_s)
        to_sub = [x for x in var_list if x in Zsubs.free_symbols]
        if len(to_sub) > 0:
            Zsubs = _find_Z_instance_deterministic(Zsubs, to_sub, max_tries=10, nonzero=nonzero)
        if Zsubs.det() == 0:
            breakpoint()
        return Zsubs


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


def decouple_column(v:sym.Matrix, nd_mat:sym.Matrix, mat:sym.Matrix):

    Z1 = sym.Matrix.hstack(v, nd_mat)
    # decouple from free modes -- this is guaranteed to not change w
    # Decouple each nondynamical mode individually
    i = nd_mat.shape[1]
    while i >= 1:
        Z1 = Z1*_decoupling_transformation((Z1).transpose()*mat*(Z1), n_d=i)    
        i -= 1
    return sym.simplify(Z1[:, 0])
    

def _decoupling_transformation(X:sym.Matrix, n_d:int):


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

def decoupling_transformation(X:sym.Matrix, n_d:Union[int, list[int]]):
    """
    Performs a transformation to decouple the specified columns
    from the rest of the system in the given X matrix.

    Args:
        X (sym.Matrix): Matrix to decouple columns from
        n_d (Union[int, list[int]]): Either the number of columns
                                     to decouple (last n_d columns)
                                     or a list of the specific column
                                     indices to decouple

    Returns:
        sym.Matrix: Transformation matrix that decouples the specified
                    columns from the rest of the system
    """

    if isinstance(n_d, int):
        return _decoupling_transformation(X, n_d=n_d)
    
    # re-order matrix so nondynamical modes are last
    n_d = sorted(n_d)
    col_order = [i for i in range(X.shape[1]) if i not in n_d] + n_d
    inv_order = np.argsort(col_order)
    X_reorder = X[:, col_order][col_order, :]
    Z2 = _decoupling_transformation(X_reorder, n_d=len(n_d))
    # Reorder back to original
    Z2 = Z2[inv_order, :][:, inv_order]
    return Z2



## TODO: MAKE THIS SYMBOLIC
def H_hash(cTransInv, lTrans, wJtTrans, EJ, var_types, try_perms = True,
           ordering_matters:bool=True, extra_nl:bool=False):
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
    
    # Record number of modes
    n_nodes = len(functools.reduce(lambda a,b: a+b, var_types.values(), []))
    ext_var = var_types.get("extended", [])
    n_ext = len(ext_var)
    comp_var = var_types.get("compact", [])
    n_comp = len(comp_var)
    harm_var = var_types.get("harmonic", [])
    n_harm = len(harm_var)
    mode_str = f"{n_comp}{n_ext}{n_harm}"
    n_nd = n_nodes - n_ext - n_comp - n_harm
    n_dyn = n_nodes - n_nd
    n_nl = n_ext + n_comp

    # Truncate to just the dynamical modes
    L_tilde = lTrans[:n_dyn, :n_dyn]
    cTransInv = cTransInv[:n_dyn, :n_dyn]
    wJ = wJtTrans[:, :n_dyn]
    
    # All different ways to arrange columns
    # Interchanging like variable columns
    if try_perms and ordering_matters:
        perms = _var_col_perms(var_types, dyn_only=True)
        perms = [p[:n_dyn] for p in perms]
    else:
        perms = [tuple(range(n_dyn))]

    lowest_hash = ""
    lowest_perm = tuple(range(n_nodes))
    WJ_zero = len(WJ_VALS)//2
    for perm in perms:
        
        L_key = _nonzero_entries_str(lTrans[:, perm])
        C_key =  _nonzero_entries_str(cTransInv[:, perm])
        w_key = _nonzero_entries_str(incidence_to_square(wJtTrans[:, perm].transpose(), EJ))
        wT_tilde, wT_key_full, _ = _maximize_wT(_sort_wT(wJtTrans[:, perm]))
        if extra_nl:
            n_val = sum(int(x) != WJ_zero for x in wT_key_full)
            w_key += "-" + str(n_val) + "-" + wT_key_full

        # If ordering doesn't matter
        if not ordering_matters:
            w_key = w_key[:w_key.find("-")]
            C_key = C_key[:C_key.find("-")]
            L_key = L_key[:L_key.find("-")]

        # Make the hash string
        Z_hash = "_".join([mode_str, w_key, str(int(L_key[0])+int(C_key[0])), L_key, C_key])
        if lowest_hash == "" or Z_hash < lowest_hash:
            lowest_hash = Z_hash
            lowest_perm = perm + tuple(range(n_dyn, n_nodes))

    return lowest_hash, lowest_perm


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
                        param_func=lambda e: "C_J" + e.replace("J", ""))
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
                       wJ: sym.Matrix = sym.Matrix([]),
                       do_ce=True, do_e=True, do_eh=True, do_hh=True) -> sym.Matrix:
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
                if ((i in comp and j in ext and do_ce) or 
                    (i in ext and j in ext and do_e) or 
                    (i in harm and j in ext and do_eh) or
                    (i in harm and j in harm and do_hh)):
                    Z[i,j] = sym.Symbol(f"{prefix}{i}{j}", real=True)
                    var_list.append(Z[i,j])
    else:
        Z = Z_in
        var_list = [x for x in Z.free_symbols if prefix in str(x)]
        
    # Sort variable list for consistent substitution
    var_list.sort(key=lambda s: s.name)

    # Transformed Capacitance and Susceptance Matrices
    n_vars = len(var_list)
    if not cMat is None:
        cTrans2 = Z.transpose()*cTrans*Z
        cTrans2_trunc = cTrans2[:n_comp+n_ext+n_harm,:n_comp+n_ext+n_harm]
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
    dets = [sym.simplify(Z.det()), cTrans2_det]
    pairs = list(itertools.product(comp + ext + harm, ext + harm))
    for (i1, i2) in pairs:
        # Symmetric coupling
        if i2 <= i1:
            continue
        c_vars = [v for v in var_list if v in cTransInv2[i1, i2].free_symbols]
        sol_c = sol_indep_of_vars(cTransInv2[i1, i2], c_vars, nonzero=dets)
        i,j = i1, i2
        if sol_c:
            coupling_c.append(sol_c)
        # L will be 0 if this is a real capacitance matrix
        l_vars = [v for v in var_list if v in lTrans2[i1, i2].free_symbols]
        sol_l = sol_indep_of_vars(lTrans2[i1, i2], l_vars, nonzero=dets)
        if sol_l:
            coupling_l.append(sol_l)
    
    n_couple_c = len(coupling_c)
    n_couple_l = len(coupling_l)
    tiebreaker_fn = lambda keys: f"{n_couple_c + n_couple_l - len(keys)}_"+ \
                                 "".join(["0" if i in keys else "1" for i in range(n_couple_c, n_couple_c+n_couple_l)]) + "_" + \
                                 "".join(["0" if i in keys else "1" for i in range(n_couple_c)])

    # If no couplings to decouple, return the general transformation
    if len(coupling_c) + len(coupling_l) == 0:
        return [Z]
    best_keys, best_subs = maximally_compatible_set(coupling_c + coupling_l, solve_vars=var_list, nonzero=dets)

    # Remove duplicate solutions and ones that are supersets of others
    # Sort by tiebreaker function
    best_nz_str = [tiebreaker_fn(k) for k in best_keys]
    nz_str_order = np.argsort(best_nz_str)
    best_subs = [best_subs[i] for i in nz_str_order]
    best_subs, best_idx = unique_solutions(best_subs, return_idx=True)
    best_Z = [Z.subs(s) for s in best_subs]
    best_nz_str = [best_nz_str[nz_str_order[i]] for i in best_idx]
    
    # Remove transformations that are column permutations or sign-flips of each other
    perms = _var_col_perms(var_types, dyn_only=True)
    perms = [p + tuple(range(n_d, Z0.shape[0])) for p in perms]
    _, idx_kept = _remove_permutation_equivalent_transformations(best_Z, perms)
    final_nz_str = [best_nz_str[i] for i in idx_kept]
    final_Z = [best_Z[i] for i in idx_kept]

    # Process transformation before returning
    if return_instance:
        final_Z = [_find_Z_instance(Z, var_list, wJ=wJ, var_types=var_types) for Z in final_Z]
    if ordering_matters:
        return final_Z[np.argmin(final_nz_str)]
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
    if utils.count_elems_mapped(circuit)["J"] > 0:
        wJ, EJ = gen_w(utils.add_elem_number(circuit), edges, w_elem="J", return_params=True)
    else:
        wJ = sym.Matrix(np.zeros((n_nodes, 1)))
        EJ = [0]
    elem_counts = utils.count_elems_mapped(circuit)
    jMat = incidence_to_square(wJ, EJ) if elem_counts["J"] > 0 else sym.zeros(n_nodes, n_nodes)

    # Step 0: Separate into different variable types and decouple nondynamical
    Z0, var_types = var_trans_basis(circuit, edges, ground_node=ground_node)
    n_dyn = len(var_types["compact"] + var_types["extended"] + var_types["harmonic"])
    n_nl = len(var_types["compact"] + var_types["extended"])

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
                                    ordering_matters=False, wJ=wJ)
            for Z2_bi in Z2_b:
                Z_tot = Z*Z2_bi
                var_list = [x for x in Z2_bi.free_symbols if "Z" in str(x)]
                Z_hash = _find_Z_instance_deterministic(Z_tot, var_list, nonzero=extract_denom(Z_tot))
                cTransInv = sym.simplify(Z_hash.transpose()*cMat*Z_hash)[:n_dyn, :n_dyn].inv()
                lTrans2 = sym.simplify(Z_hash.transpose()*lMat*Z_hash)
                wJtTrans = sym.simplify(wJ.transpose()*Z_hash)
                val, Z_perm = H_hash(cTransInv, lTrans2, wJtTrans, EJ, var_types=var_types)
                if val < lowest_hash or lowest_hash == "":
                    lowest_Z = [Z_tot]
                    lowest_hash = val
                elif val == lowest_hash:
                    lowest_Z += [Z_tot]

    # If there are multiple Z with the same lowest hash
    # then see if they separate with equal L/C values
    # or with longer H Hash that includes nonlinear terms
    Z_final = []
    hash_final = ""
    hash_full = []
    for Z in lowest_Z:

        Z_equal = sym.simplify(_sub_equal_LC(Z))
        var_list = [x for x in Z.free_symbols if "Z" in str(x)]
        # val, Z_perm = H_hash(_find_Z_instance_random(Z_equal, var_list),
        #                     var_types, cMat, lMat, wJ, equalJ=True)
        Z_hash1 = _find_Z_instance_deterministic(Z_equal, var_list, nonzero=extract_denom(Z_equal))
        cTransInv = sym.simplify(Z_hash1.transpose()*cMat*Z_hash1)[:n_dyn, :n_dyn].inv()
        lTrans2 = sym.simplify(Z_hash1.transpose()*lMat*Z_hash1)
        wJtTrans = sym.simplify(wJ.transpose()*Z_hash1)
        val, Z_perm = H_hash(cTransInv, lTrans2, wJtTrans, EJ, var_types=var_types)

        Z_hash2 = _find_Z_instance(Z, var_list, var_types=var_types, wJ=wJ, nonzero=extract_denom(Z))

        # Z_hash2 = Z_hash1
        cTransInv = sym.simplify(Z_hash2.transpose()*cMat*Z_hash2)[:n_dyn, :n_dyn].inv()
        lTrans2 = sym.simplify(Z_hash2.transpose()*lMat*Z_hash2)
        wJtTrans = sym.simplify(wJ.transpose()*Z_hash2)
        val_full, _ = H_hash(cTransInv, lTrans2, wJtTrans, EJ, var_types=var_types, extra_nl=True)
        if val < hash_final or hash_final == "":
            hash_final = val
            hash_full = [val_full]
            if return_instance:
                Z_final = [Z_hash2]
            else:
                Z_final = [Z]
        elif val == hash_final:
            if return_instance:
                Z_final.append(Z_hash2)
            else:
                Z_final.append(Z)
            hash_full.append(val_full)

    # print(hash_full)
    # Put the transformation in a canonical form
    # for the nonlinear terms
    # best_w, best_key, (row_vec, col_vec, row_order) = _maximize_wT(Z_final[0])
    if len(Z_final) > 1:
        hash_full = np.array(hash_full)
        if sum(hash_full == min(hash_full)) == 1:
            Z_final = [Z_final[np.argmin(hash_full)]]

    # Get a specific instance of the transformation
    # if return_instance:
    #     # Nonzero terms -- det is already done in find_Z_instance
    #     # so collect all the denominators
    #     Z_final_instance = []
    #     for Zf in Z_final:
    #         var_list = [x for x in Zf.free_symbols if "Z" in str(x)]
    #         Z_final_instance.append(_find_Z_instance(Zf, var_list, var_types=var_types,
    #                                                 wJ=wJ, nonzero=extract_denom(Zf)))
    #     Z_final = Z_final_instance

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


    circuit = [("J", "L"), ("C", "J"), ("C", "J", "L")]
    circuit = utils.add_elem_number(circuit)
    edges_og = ((0, 1), (0, 2), (1, 2))
    edges_all = [edges_og]
    for p in itertools.combinations([0, 1, 2], r=2):
        edges_all.append(tuple(utils.swap_nodes(edges_og, p[0], p[1])))
    for i in range(1):
        res = {}
        for edges in edges_all:
            print("--------------\nEdges", edges,"\n--------------")
            wJ = gen_w(circuit, edges, w_elem="J")
            Z, var_types, hash = choose_Z(circuit, edges, return_instance=True)
            Z = Z[0]
            assert var_types["extended"] == [0,1]
            assert var_types["sigma"] == [2]
            # print(edges, hash, _maximize_wT(wJ.transpose()*Z)[1])
            if edges not in res:
                res[edges] = set()
            res[edges].add(_maximize_wT((wJ.transpose()*Z)[:,:2])[1])
        for e in res:
            res[e] = frozenset(res[e])

        print(res)
        assert len(set(res.values())) == 1
    
    assert False
    


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