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

from sircuitenum import utils
from sircuitenum.qpackage_interface import subgraph


PERIODIC_CHARGE = "n"
PERIODIC_PHASE = "θ"
EXTENDED_CHARGE = "q"
EXTENDED_PHASE = "φ"
NODE_CHARGE = "q"
NODE_PHASE = "ϕ"
EXT_CHARGE = "n_g"
EXT_PHASE = "_{ext}"


def gen_variables(n_nodes, cob, periodic):

    Q_str = ""
    th_str = ""
    for n in range(1, n_nodes+1):
        if cob is None:
            th_str += "\hat{" + NODE_PHASE + "}_{"+str(n)+"}, "
            Q_str += "\hat{" + NODE_CHARGE + "}_{"+str(n)+"}, "
        elif n in periodic:
            th_str += "\hat{" + PERIODIC_PHASE + "}_{"+str(n)+"}, "
            Q_str += "\hat{" + PERIODIC_CHARGE + "}_{"+str(n)+"}, "
        else:
            th_str += "\hat{" + EXTENDED_PHASE + "}_{"+str(n)+"}, "
            Q_str += "\hat{" + EXTENDED_CHARGE + "}_{"+str(n)+"}, "

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
        wT = np.array(wT).astype(float)
    keys = []
    for i in range(wT.shape[0]):
        try:
            keys.append(str(sum(wT[i, :] != 0)) +"_"+"-".join(wT[i, :].nonzero()[0].astype(str)))
        except:
            breakpoint()
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
    coup_mat = wT[0][np.newaxis, :]*wT[0][:, np.newaxis]
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
        if "C" in str(x) and "J" not in str(x):
            X = X.subs(x, C)
        if "C" in str(x) and "J" in str(x):
            X = X.subs(x, Cj)
    # All L the same
    L = sym.symbols("L", positive=True, real=True)
    for x in X.free_symbols:
        if "L" in str(x):
            X = X.subs(x, L)
    return X

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

    # Variables in block n1 coupled to 
    # variables in block n2 for key [n1][n2]
    coupled = {}
    for key in ["12", "21", "13", "31"]:
        coupled[key] = []
   
    # Determine which block 1 variables are
    # coupled to block 3 variables
    for i in block1:
        for j in block3:
            if X[i, j] != 0:
                coupled["13"].append(i)
                coupled["31"].append(j)
    # block 2 and block 1
    for i in block1:
        for j in block2:
            if X[i, j] != 0:
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
            for ext_coupled in itertools.product(*ext_vec_coupled):
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

    return _unique_col_combos(harm_vec, n_harm, signs=[1, -1], shifts=shifts,
                                li_vecs=[nd_mat[:, j] for j in range(n_nd)])
    

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
    # Numerically treat the matrices
    C, _ = num_subs(cMat, symbol="C")
    L, _ = num_subs(lMat, symbol="L")
    n_nodes = Z.shape[0]

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
        C_tilde = Z_perm.transpose()*C*Z_perm
        L_tilde = Z_perm.transpose()*L*Z_perm
        # Truncate to dynamical modes
        C_tilde = np.array(C_tilde[:-n_nd, :-n_nd])
        L_tilde = np.array(L_tilde[:-n_nd, :-n_nd])
        # Invert capacitance matrix and trim small numerical values
        C_tilde_inv = np.linalg.inv(np.array(C_tilde).astype(float))
        C_tilde_inv[np.abs(C_tilde_inv)/np.abs(C_tilde_inv).max() < eps] = 0
        if not lMat.is_zero_matrix:
            L_tilde[np.abs(L_tilde)/np.abs(L_tilde).max() < eps] = 0
        # Key for C, L = [n_coupled]-[nz entries of off diag]
        C_key =  _nonzero_entries_str(C_tilde_inv)
        L_key = _nonzero_entries_str(L_tilde)

        if wJ.shape[1] > 0:
            wT_tilde = wJ.transpose()*Z_perm
            wT_tilde = wT_tilde[:, :-n_nd]
            # Put wT into cananocal ordering
            wT_tilde, _, _ = _maximize_wT(_sort_wT(wT_tilde))
            # Key for wT = [n_coupled]-[nz entries of off diag]
            w_key = _wT_key(wT_tilde, equalJ = equalJ)
            wT_full = (1+wT_tilde).astype(int).astype(str)
        else:
            w_key = "0-"+"0"*(len(L_key)-2)
            wt_full = ""

        # TODO: At the end add a base 3 wT key to encode exact nonlinear form

        # Make the hash string
        Z_hash = "_".join([mode_str, w_key, str(int(L_key[0])+int(C_key[0])), L_key, C_key])
        if lowest_hash == "" or Z_hash < lowest_hash:
            lowest_hash = Z_hash
            lowest_Z = Z_perm

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
    # u_comp = unique_compact(circuit, edges, nd_mat, cMat)
    u_harm = unique_harmonic(circuit, edges, nd_mat, cMat, lMat)
    # u_ext = unique_extended(circuit, edges, nd_mat, cMat, lMat)
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
        cTrans = Z.transpose()*cMat*Z
        lTrans = Z.transpose()*lMat*Z
        if (decoupling_transformation(cTrans, n_comp+n_ext+n_harm) != eye or
            decoupling_transformation(lTrans, n_comp+n_ext+n_harm) != eye):
            raise ValueError("Decoupling Transformation Failed")
        all_Z.append(Z)
        
    return all_Z, var_types


def secondary_transformation_harm_ext(Z0, var_types, cMat, lMat, wJ=sym.Matrix([[]]),
                                      tried=[False, False, False]):

    cTrans = Z0.transpose()*cMat*Z0
    lTrans = Z0.transpose()*lMat*Z0
    eye = sym.eye(Z0.shape[0])
    n_comp = len(var_types.get("compact", []))
    n_ext = len(var_types.get("extended", []))
    n_harm = len(var_types.get("harmonic", []))

    if n_harm == 0 or n_ext == 0:
        return eye, H_hash(Z0, var_types, cMat, lMat, wJ, try_perms=False)[0]

    # Try each of the three possible decouplings
    Zl_eh = decoupling_transformation(lTrans, n_comp+n_ext)
    Zc_eh = decoupling_transformation(cTrans, n_comp+n_ext)
    Zc_ce = decoupling_transformation_3block(cTrans,
                                             var_types.get("compact", []),
                                             var_types.get("extended", []),
                                             var_types.get("harmonic", []))
    decouple_trans = [Zl_eh, Zc_eh, Zc_ce]
    valid_trans = [Ztest != eye and len(Ztest.free_symbols) == 0
                   for Ztest in decouple_trans]
    best_Z = eye
    # Recursive case, there are valid transformations
    if any(valid_trans):
        best_hash = ""
        for i, Ztest in enumerate(decouple_trans):
            # Try each ordering of different decoupling transformations
            if valid_trans[i] and not tried[i]:
                new_tried = [x for x in tried]
                new_tried[i] = True
                Z2, hash = secondary_transformation_harm_ext(Z0*Ztest, var_types,
                                                                  cMat, lMat, wJ,
                                                             new_tried)
                if best_hash == "" or hash < best_hash:
                    best_hash = hash
                    best_Z = Ztest*Z2
    return best_Z, H_hash(Z0*best_Z, var_types, cMat, lMat, wJ, try_perms=False)[0]


def choose_Z(circuit, edges) -> tuple[sym.Matrix, str]:
    
    # Generate capacitance matrix, susceptance matrix, and incidence matrix
    cMat = gen_cap_mat(circuit, utils.zero_start_edges(edges))
    lMat = gen_ind_mat(circuit, utils.zero_start_edges(edges))
    wJ = gen_w(circuit, edges, w_elem="J")

    
    all_Z, var_types = gen_spaced_var_trans(circuit, edges, cMat, lMat)
    lowest_hash = ""
    lowest_Z = []
    for Z in all_Z:
        # Consider secondary harmonic extended transformation
        Z2, _ = secondary_transformation_harm_ext(Z, var_types,
                                                  cMat, lMat, wJ)
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
    
    return Z_final, var_types, lowest_hash

def gen_junc_pot(circuit, edges, flux_vars, cob=None, eps=1e-10) -> sym.Matrix:
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
    cob : sym.Matrix
        A change of basis matrix used to transform the node flux variables.

    Returns
    -------
    sym.Matrix
        The capacitance matrix as a symbolic matrix, representing the junction potential terms.
    """

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
            node_vec = np.zeros(n_nodes, dtype=int)
            node_vec[i] = -1
            node_vec[j] = 1
            node_vec = sym.Matrix(node_vec)
            if cob is not None:
                node_vec = sym.transpose(cob)*node_vec
            j_terms += val*sym.cos((sym.transpose(flux_vars)*node_vec)[0])

    return j_terms

def num_subs(C, vals_in = {}, symbol="C"):
    vals = {}
    for x in C.free_symbols:
        x_str = str(x)
        if symbol in x_str.upper() and "J" not in x_str.upper():
            # Random capacitance 1/(0.1 - 1.1 GHZ)
            if vals_in == {}:
                vals[x_str] = 1/(0.1 + np.random.random())
            else:
                vals[x_str] = vals_in[x_str]
        else:
            # Random capacitance 1/(10 - 20 GHZ)
            if vals_in == {}:
                vals[x_str] = 1/(10 + 10*np.random.random())
            else:
                vals[x_str] = vals_in[x_str]
        C = C.subs(x, vals[x_str])

    return C/2 + C.transpose()/2, vals


def quantize_circuit(circuit, edges, Cv=None, V=None, cob=None,
                     periodic=[], extended=[], free=[], frozen=[],
                     sigma = [], return_mats=False, return_vars=False,
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
    cob : sym.Matrix, optional
        A change of basis matrix that transforms node variables to new variables. 
        This matrix corresponds to the Z transformation of scqubits.
        If the new variables are expressed in terms of the old, this should be the inverse.
    periodic : list of int, optional
        A list of mode numbers (indexed from 1) indicating which coordinates are periodic.
    extended : list of int, optional
        A list of mode numbers (indexed from 1) indicating which coordinates are extended.
    free : list of int, optional
        A list of mode numbers (indexed from 1) indicating which coordinates are free.
    frozen : list of int, optional
        A list of mode numbers (indexed from 1) indicating which coordinates are frozen.
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

    Q_vec, th_vec = gen_variables(n_nodes, cob, periodic)
    

    C_mat = gen_cap_mat(circuit, edges)
    L_mat = gen_ind_mat(circuit, edges)

    # Set zero applied voltage
    if Cv is None:
        Qv = sym.zeros(rows=n_nodes, cols=1)

    if cob is not None:
        C_mat = sym.transpose(cob)*C_mat*cob
        L_mat = sym.transpose(cob)*L_mat*cob
        if Cv is not None:
            if V is None:
                raise ValueError("Provide Voltages for Coupling")
            Qv = sym.transpose(cob)*Cv*V
    elif Cv is not None:
        Qv = Cv*V

    # J terms shouldn't contain anything from free modes or frozen modes
    J_terms = gen_junc_pot(circuit, edges, th_vec, cob=cob)

    # Remove any marked modes
    C_mat_full = C_mat.copy()
    L_mat_full = L_mat.copy()
    # All modes to remove
    remove_modes = free + frozen + sigma
    if remove_modes:
        # Go in order to make the indexing
        # post deletion straightforward
        n_deleted = 0
        for n in range(n_nodes):
            if n+1 in remove_modes:
                # Different sympy versions do this in place
                # or not in place so branch this off into
                # helper function
                Qv = _remove_row(Qv, n-n_deleted)
                Q_vec = _remove_row(Q_vec, n-n_deleted)
                th_vec = _remove_row(th_vec, n-n_deleted)
                C_mat = _remove_row(C_mat, n-n_deleted)                
                C_mat = _remove_col(C_mat, n-n_deleted)
                L_mat = _remove_row(L_mat, n-n_deleted)
                L_mat = _remove_col(L_mat, n-n_deleted)
                n_deleted += 1
    try:
        if C_mat.shape[0] == 1:
            C_inv = C_mat.inv()
        else:
            # Check for the weird all 0 issue
            C_inv = sym.inv_quick(C_mat)
            if C_inv == sym.zeros(rows=C_inv.shape[0],
                                  cols=C_inv.shape[1]):
                C_inv = C_mat.inv()
    except Exception as exc:
        print(exc)
        print("circuit:", circuit)
        print("edges:", edges)
        print("C_mat_full:", C_mat_full)
        print("C_mat:", C_mat)
        return C_mat_full, L_mat_full

    # Explicitly subtract out constant terms from coupling
    C_terms = sym.Rational(1, 2)*sym.transpose(Q_vec - Qv)*C_inv*(Q_vec - Qv)
    C_terms += -sym.Rational(1, 2)*sym.transpose(Qv)*C_inv*Qv
    L_terms = sym.Rational(1, 2)*sym.transpose(th_vec)*L_mat*th_vec

    # Combine terms and group terms in H
    H = C_terms[0] + L_terms[0] + J_terms
    H = sym.expand(sym.nsimplify(H))
    if expand_trig:
        H = sym.expand_trig(H)
    if cob is None:
        H, combos, combosQ = utils.collect_H_terms(H, zero_ext=False,
                                  periodic_charge="n", periodic_phase="θ",
                                  extended_charge="q", extended_phase="ϕ",
                                  collect_phase = collect_phase)
    else:
        H, combos, combosQ = utils.collect_H_terms(H, zero_ext=False,
                                  periodic_charge="n", periodic_phase="θ",
                                  extended_charge="q", extended_phase="φ",
                                  collect_phase = collect_phase)

    to_return = (H,)

    if return_H_class:
        to_return = to_return + (utils._remove_coeff(H, list(combosQ)+combos),)
    if return_combos:
        to_return = to_return + (list(combosQ)+combos,)
    if return_mats:
        to_return = to_return + (C_mat, L_mat)
    if return_vars:
        to_return = to_return + (Q_vec, th_vec)
    if len(to_return) == 1:
        to_return = to_return[0]

    return to_return

