__doc__ = "quantize.py: contains functions used to produce symbolic hamiltonians"
__author__ = "Eli Weissler"
__version__ = "0.1.0"
__all__ = ["gen_cap_mat", "gen_ind_mat", "gen_junc_pot", "quantize_circuit"]

import itertools
import functools

from typing import Union

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
                       idx:list[int]=[],
                       v1_recon:bool=True,
                       return_decomp:bool=False) -> list[sym.Matrix]:
    """
    Calculates the overlap of two linearly independent sets of
    vectors vecs1, vecs2 using NULL(vecs1, -vecs2)

    Args:
        vecs1 (Union[list[sym.Matrix], sym.Matrix]): set of column vectors
        vecs2 (Union[list[sym.Matrix], sym.Matrix]): set of column vectors
        idx (list[int], optional): Consider equality only in a specified set of indices.
                                   Note: In this case the vectors are reconstructed from
                                   the set specified by v_recon.
        v1_recon bool: In the case of only examining equality
                                                       for a specified set of indices, the set to
                                                       reconstruct the full vector from. True is vecs1
                                                       false is vecs2.
        return_decomp (bool, optional): Return the decomposition of the overlap
                                        vectors in each set. Defaults to False.

    Returns:
        list[sym.Matrix]: list of vectors that span the overlap space.
    """
    # Convert matrices to list
    if isinstance(vecs1, sym.Matrix):
        vecs1 = [vecs1[:, j] for j in range(vecs1.shape[1])]
    if isinstance(vecs2, sym.Matrix):
        vecs2 = [vecs2[:, j] for j in range(vecs2.shape[1])]

    # If either one is empty, return no overlap
    if len(vecs1) == 0 or len(vecs2) == 0:
        if return_decomp:
            return [], [], []
        else:
            return []
    # Assert vector sets are linearly independent
    assert len(_linearly_indep_cols(sym.Matrix.hstack(*vecs1))) == len(vecs1)
    assert len(_linearly_indep_cols(sym.Matrix.hstack(*vecs2))) == len(vecs2)

    # Examine all indices if none is given
    assert vecs1[0].shape[0] == vecs2[0].shape[0]
    if idx == []:
        idx = list(range(vecs1[0].shape[0]))

    # Calculate the overlap of the two vector spaces
    divide = len(vecs1)
    ns = sym.Matrix.hstack(*[v[idx, :] for v in vecs1],
                           *[-v[idx, :] for v in vecs2]).nullspace()
    
    # Gather the entries
    in_v1 = []
    in_v2 = []
    for vec in ns:
        v1_entry = vec[:divide, :]
        v2_entry = vec[divide:, :]
        if not(v1_entry.is_zero_matrix or v2_entry.is_zero_matrix):
            in_v1.append(v1_entry)
            in_v2.append(v2_entry)
    
    # Reconstruct the overlap vectors
    if v1_recon:
        mat_recon = sym.Matrix.hstack(*vecs1)
        overlap_vecs = [mat_recon*v for v in in_v1]
    else:
        mat_recon = sym.Matrix.hstack(*vecs2)
        overlap_vecs = [mat_recon*v for v in in_v2]
    
    if return_decomp:
        return overlap_vecs, in_v1, in_v2
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


def _equal_up_to_column_shift_and_sign(A: sym.Matrix, B: sym.Matrix, shifts: list[sym.Matrix] = None):
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
    if shifts is None:
        # Just the global shift
        shifts = [sym.ones(A.shape[0], 1)]
    else:
        # Get linearly independent set of shifts
        li_shifts = _linearly_indep_cols(sym.Matrix.hstack(*shifts))
        shifts = [shifts[i] for i in range(len(shifts)) if i in li_shifts]
    n_shifts = len(shifts)

    rows, cols = A.shape
    for j in range(cols):
        if not _equiv_cols(A[:, j], B[:, j], shifts):
            return False
    return True


def _find_equiv_cols(c1, cList, shifts):
    return [i for i in range(len(cList)) if _equal_up_to_column_shift_and_sign(c1, cList[i], shifts)]


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


def _unique_col_combos(basis, n_elem, signs, li_vecs=[], valid=lambda v: True, shifts=None):

    basis_with_sign = []
    for v in basis:
        for sign in signs:
            basis_with_sign.append(sign*v)

    possible_vecs = []
    # Add r different basis elements together
    for r in range(1, len(basis)+1):
        # vecs to add together
        for vecs in itertools.combinations(basis_with_sign, r):
            v = functools.reduce(lambda x,y: x + y, vecs)
            # If the sum vec hasn't been considered before, add it to the set
            if valid(v):
                if len(_find_equiv_cols(v, possible_vecs, shifts)) == 0:
                    possible_vecs.append(v)
    
    # Now consider all n_elem linearly independent combinations of the possible_vecs
    possible_combos = []
    for combo in itertools.combinations(possible_vecs, n_elem):
        # Is the combo linearly independent when considered with the other vecs
        if len(_linearly_indep_cols(sym.Matrix.hstack(*li_vecs, *combo))) == n_elem + len(li_vecs):
            # Is it equivalent to a matrix we've seen yet
            M = sym.Matrix.hstack(*combo)
            if not any(_equal_up_to_column_swaps_and_shift_and_sign(M, X, shifts) for X in possible_combos):
                possible_combos.append(M)
    
    return possible_combos


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
        if not np.all(np.in1d(circuit[i], links)):
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
    all_Zc = []
    for rows in row_sets:
        # Get the appropriate transformation for each set
        Zc = wc[rows, :].pinv()
        # Verify it's well-aligned
        if compact_well_aligned(wc*Zc, n_c):
            if not any(_equal_up_to_column_swaps_and_shift_and_sign(Zc, X) for X in all_Zc):
                all_Zc.append(Zc)

    # Return all possible transformations
    all_Z = []
    for Zc in all_Zc:
        Z = sym.eye(wT.shape[1])
        Z[:n_c, :n_c] = Zc
        all_Z.append(Z)

    return all_Z


def decoupling_transformation(X:sym.Matrix, n_d:int):

    # Determine which nondynamical variables are coupled to dynamical variables
    n_nd = X.shape[0]-n_d
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


def unique_compact(comp_vec, nd_mat, sigma_vec, free_vec, cMat, wJ):

    # number of compact variables is the number of CJ islands
    # minus the number of free/sigma variables that can be made
    # from those islands
    
    # nd_mat = sym.Matrix.hstack(*sigma_vec, *free_vec)
    # nd_mat = nd_mat[:, _linearly_indep_cols(nd_mat)][:, ::-1] # keep sigma but put them last
    # Verify nondynamical variables are linearly independent
    assert len(_linearly_indep_cols(nd_mat)) == nd_mat.shape[1]
    n_nd = nd_mat.shape[1]
    n_comp = len(_linearly_indep_cols(sym.Matrix.hstack(*comp_vec, nd_mat))) - n_nd
    if n_comp == 0:
        return []

    # Initial candidates
    Z_cand = _unique_col_combos(comp_vec, n_comp, signs=[1, -1], shifts=sigma_vec+free_vec,
                                # valid = lambda Zc: well_spaced(wJ.transpose()*Zc),
                                li_vecs=[nd_mat[:, j] for j in range(n_nd)])
    Z_final = []
    # Go through and do a secondary transformation to make things well-aligned
    # and decoupled from any free modes
    for Zc in Z_cand:
        # Total
        Z1 = sym.Matrix.hstack(Zc, nd_mat)

        # make sure compact is well aligned
        wT = wJ.transpose()*(Z1)
        if not compact_well_aligned(wT, n_comp):
            Z2_possible = compact_alignment_transformation(wT, n_comp)
        else:
            Z2_possible = [sym.eye(n_comp + n_nd)]

        for Z2 in Z2_possible:
            # decouple from free modes
            Z3 = decoupling_transformation((Z1*Z2).transpose()*cMat*(Z1*Z2), n_comp)
            # Make sure we're still well-aligned
            if compact_well_aligned(wJ.transpose()*(Z1*Z2*Z3), n_comp):
                # Get the compact part of the final transformation
                # The nondynamical ones are unaffected by Z2 and Z3
                Z = (Z1*Z2*Z3)[:, :n_comp]
                if not any(_equal_up_to_column_swaps_and_shift_and_sign(Z, Zi, shifts=sigma_vec) for Zi in Z_final):
                    Z_final.append(Z)
    
    return Z_final


def unique_harmonic(harm_vec, nd_mat, sigma_vec, free_vec, frozen_vec, cMat, lMat, wL):

    # number of harmonic variables is the number of LC islands
    # minus the number of free/sigma variables that can be made
    # from those islands
    
    # Verify nondynamical variables are linearly independent
    assert len(_linearly_indep_cols(nd_mat)) == nd_mat.shape[1]
    n_nd = nd_mat.shape[1]
    n_harm = len(_linearly_indep_cols(sym.Matrix.hstack(*harm_vec, nd_mat))) - n_nd
    if n_harm == 0:
        return []

    # Initial candidates
    Z_cand = _unique_col_combos(harm_vec, n_harm, signs=[1, -1], shifts=sigma_vec+free_vec+frozen_vec,
                                valid = lambda Zh: well_spaced(wL.transpose()*Zh),
                                li_vecs=[nd_mat[:, j] for j in range(n_nd)])
    Z_final = []
    # Go through and do a secondary transformation to make things well-aligned
    # and decoupled from any free modes
    for Zh in Z_cand:
        # Total
        Z1 = sym.Matrix.hstack(Zh, nd_mat)
        
        # make sure harmonic is well spaced
        wT = wL.transpose()*(Z1)
        if not well_spaced(wT):
            # If it's not well spaced, can we make it
            # so by only scaling columns?
            mags = _unique_mag_row(wT.transpose())
            # Check if there's only 1 nonzero magnitude per column
            if all(sum(x > 0 for x in mag) < 2 for mag in mags):
                scaling = [sym.Rational(1, max(mag)) if max(mag) > 0 else 1 for mag in mags]
                Z2 = sym.diag(scaling)
            # Can't make it well spaced
            else:
                continue
        else:
            Z2 = sym.eye(n_harm + n_nd)

        # decouple from free modes and frozen modes
        Z3A = decoupling_transformation((Z1*Z2).transpose()*cMat*(Z1*Z2), n_harm)
        Z3B = decoupling_transformation((Z1*Z2*Z3A).transpose()*lMat*(Z1*Z2*Z3A), n_harm)
        Z3 = Z3A*Z3B

        # Make sure it's still well-spaced
        if well_spaced(wL.transpose()*(Z1*Z2*Z3)):
            Z = (Z1*Z2*Z3)[:, :n_harm]
            if not any(_equal_up_to_column_swaps_and_shift_and_sign(Z, Zi) for Zi in Z_final):
                Z_final.append(Z)
    
    return Z_final


def unique_extended(comp_mat, harm_mat, nd_mat, sigma_vec, free_vec, frozen_vec, cMat, lMat, wJ):

    # Select columns from linearly independent cols of wJ^-1
    # that are linearly independent with comp and harm (I thin harm is for free here)
    # ext_vec = wJ.pinv().transpose()

    # Only keep the linearly independent columns
    comp_mat = comp_mat[:, _linearly_indep_cols(comp_mat)]
    harm_mat = harm_mat[:, _linearly_indep_cols(harm_mat)]
    nd_mat = nd_mat[:, _linearly_indep_cols(nd_mat)]


    n_nd = nd_mat.shape[1]
    n_comp = comp_mat.shape[1]
    n_harm = harm_mat.shape[1]
    nd_vars = [nd_mat[:, j] for j in range(n_nd)]
    comp_vars = [comp_mat[:, j] for j in range(n_comp)]
    harm_vars = [harm_mat[:, j] for j in range(n_harm)]
    # Number of variables needing junctions minus number of compact vars
    n_ext = len(_linearly_indep_cols(wJ)) - n_comp
    assert n_ext == wJ.shape[0] - n_nd - n_comp - n_harm


    # Try all different linearly independent col combinations of wJ
    col_sets = _linearly_indep_col_sets(wJ)
    all_basis = []
    for cols in col_sets:
        # Get the columns that invert wJ -> possible basis
        Ze = wJ[:,cols].pinv().transpose()
        if not any(_equal_up_to_column_swaps_and_shift_and_sign(Ze, Z) for Z in all_basis):
            all_basis.append(Ze)
    
    # Consider each possible basis for extended
    Z_final = []
    for basis_mat in all_basis:
        basis = [basis_mat[:, j] for j in range(basis_mat.shape[1])]
        Z_cand = _unique_col_combos(basis, n_ext, signs=[1, -1], shifts=sigma_vec+free_vec+frozen_vec,
                                valid = lambda Ze: well_spaced(wJ.transpose()*Ze),
                                li_vecs=comp_vars+harm_vars+nd_vars)
        for Ze in Z_cand:
            # Total
            Z1 = sym.Matrix.hstack(Ze, nd_mat)
            
            # make sure extended is well spaced
            wT = wJ.transpose()*(Z1)
            if not well_spaced(wT):
                # If it's not well spaced, can we make it
                # so by only scaling columns?
                mags = _unique_mag_row(wJ.transpose())
                # Check if there's only 1 nonzero magnitude per column
                if all(sum(x > 0 for x in mag) < 2 for mag in mags):
                    scaling = [sym.Rational(1, max(mag)) if max(mag) > 0 else 1 for mag in mags]
                    Z2 = sym.diag(scaling)
                # Can't make it well spaced
                else:
                    continue
            else:
                Z2 = sym.eye(n_ext + n_nd)

            # decouple from free modes and frozen modes
            Z3A = decoupling_transformation((Z1*Z2).transpose()*cMat*(Z1*Z2), n_ext)
            Z3B = decoupling_transformation((Z1*Z2*Z3A).transpose()*lMat*(Z1*Z2*Z3A), n_ext)
            Z3 = Z3A*Z3B

            # Make sure it's still well-spaced
            if well_spaced(wJ.transpose()*(Z1*Z2*Z3)):
                Z = (Z1*Z2*Z3)[:, :n_ext]
                if not any(_equal_up_to_column_swaps_and_shift_and_sign(Z, Zi) for Zi in Z_final):
                    Z_final.append(Z)
            

    return Z_final



def H_hash(circuit, edges, symmetric=True, numerical=False, eps=1e-10):
    """
    Produces a string that represents in the diagonal charge basis:

    1) a = number of nonzero charge variables
    2) b = positions of nonzero inductance terms as a binary representation
    3) c = which variables are present in the nonlinear terms

    hash = a_b_c (ex: 3_511_012)

    Args:
        circuit (_type_): _description_
        edges (_type_): _description_
        symmetric (bool, optional): _description_. Defaults to True.
        numerical (bool, optional):
        eps (float, optional): 
    """

    n_nodes = utils.get_num_nodes(edges)

    # Generate the transformation that diagonalizes
    # the capcitance matrix
    if not symmetric:
        circuit = utils.add_elem_number(circuit)
    C = gen_cap_mat(circuit, edges)
    L = gen_ind_mat(circuit, edges)
    Z_diag, C_vals = diag_cap_transform(C, numerical=True)

    # Substitute values in for C
    C, _ = num_subs(C, vals_in=C_vals)

    # Transformed capacitance and inductance matrices
    C_tilde = Z_diag.transpose()*C*Z_diag
    L_tilde = Z_diag.transpose()*L*Z_diag

    ## Things for hash
    # 1) Number of nonzero terms in C
    nonzero_c = np.sum(np.abs(np.array(C_tilde.diagonal())) > eps)

    # 2) Nonzero entries in L_tilde
    #  -- pick permutation of rows that produces the
    #     largest binary number
    #  -- in case of a tie, sort alphabetically by
    #     variables present in the junction terms
    L_tilde_trunc, _ = num_subs(L_tilde[:nonzero_c, :nonzero_c], symbol="L")
    highest_val = 0
    highest_perm = None
    highest_J_str = "99999999999999"
    Q_vec, th_vec = gen_variables(n_nodes, cob=Z_diag, periodic=[])
    for perm in itertools.permutations(range(nonzero_c)):
        nonzero_L = (np.abs(sym.flatten(L_tilde_trunc[perm, perm])) > eps).astype(int)
        val = nonzero_L.dot(2**np.arange(nonzero_L.size)[::-1])
        if val >= highest_val:
            # Look at 3) Junction terms
            Z_permed = Z_diag[:, perm + tuple(range(nonzero_c, n_nodes))]
            # terms_present = gen_junc_pot(circuit, edges, th_vec, Z_permed)
            terms_present = []
            for edge, elems in zip(edges, circuit):
                if any("J" in e for e in elems):
                    # Get a vector that's the argument of the cos
                    # truncating any terms less than eps
                    n1, n2 = edge
                    node_vec = np.zeros(n_nodes, dtype=int)
                    node_vec[n1] = -1
                    node_vec[n2] = 1
                    terms = np.array(Z_permed.transpose())@node_vec
                    terms[np.abs(terms) < eps] = 0
                    for i, t in enumerate(terms):
                        if abs(t) > 0:
                            terms_present.append(str(i))
            J_str = "".join(sorted(np.unique(terms_present)))
            if val > highest_val or J_str < highest_J_str:
                highest_val = val
                highest_perm = perm
                highest_J_str = J_str
    
    # return the hash
    return str(nonzero_c) + "_" + str(highest_val) + "_" + highest_J_str




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
        if w_elem in elems:
            w[min(edge), w_count] = 1
            w[max(edge), w_count] = -1
            w_count += 1
    return w


def gen_spaced_var_trans(circuit, edges):
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

    # Linearly independent incidence matrices
    wC = gen_w(circuit, edges, "C")
    wC = wC[:, _linearly_indep_cols(wC)]
    wCpi = wC.transpose().pinv()

    wL = gen_w(circuit, edges, "L")
    wL = wL[:, _linearly_indep_cols(wL)]
    wLpi = wL.transpose().pinv()
    
    wJ = gen_w(circuit, edges, "J")
    wJ = wJ[:, _linearly_indep_cols(wJ)]
    wJpi = wJ.transpose().pinv()

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
    # free_vec = C_islands
    idx_c = [i for i in range(wCpi.shape[0]) if not wCpi[i, :].is_zero_matrix]
    free_vec = _vec_space_overlap(C_islands, wC.transpose().pinv(), idx=idx_c, v1_recon=True)
    ## Frozen -- inductively shunted islands
    L_islands = _islands_to_vectors(circuit, edges, ["L"])
    # froz_vec = L_islands
    idx_l = [i for i in range(wLpi.shape[0]) if not wLpi[i, :].is_zero_matrix]
    froz_vec = _vec_space_overlap(L_islands, wL.transpose().pinv(), idx=idx_l, v1_recon=True) 

    # Single nondynamical variable matrix -- doesn't matter
    # which you pick
    nd_mat = sym.Matrix.hstack(*sig_vec, *froz_vec, *free_vec)
    nd_mat = nd_mat[:, _linearly_indep_cols(nd_mat)[::-1]]
    
    ## Compact -- J,C shunted islands
    JC_islands = _islands_to_vectors(circuit, edges, ["J", "C"])
    if len(JC_islands) > 1:
        # Consider overlap with wJ pseudo inverse 
        # for the nodes that are connected to junctions
        idxJ = [i for i in range(wJpi.shape[0]) if not wJpi[i, :].is_zero_matrix]
        comp_vec = _vec_space_overlap(JC_islands, wJpi, idx=idxJ,
                                      v1_recon=True)
        # Verify spacing
        assert all(well_spaced(wJ.transpose()*v) for v in comp_vec)
    else:
        comp_vec = []


    #### TODO: Fix the overlap condition for when
    # there isn't something in the overlap of 
    # Maybe gram-schidt on the nullspace of C?

    ## Harmonic -- L,C shunted islands
    if wC.shape[1] > 0 and wL.shape[1] > 0:
        # Identify PI columns that correspond
        # to inductors that are involved in the
        # harmonic mode - i.e. connected to L and C
        idx_lc = [i for i in range(wCpi.shape[0]) if not (wCpi[i, :].is_zero_matrix or
                                                       wLpi[i, :].is_zero_matrix)]
        LC_ovlp = _vec_space_overlap(wCpi, wLpi, idx=idx_lc, v1_recon=False)
        if len(LC_ovlp) > 0:
            LC_islands = _islands_to_vectors(circuit, edges, ["L", "C"])
            # Verify spacing
            assert all(well_spaced(wL.transpose()*v) for v in LC_ovlp)
            idx_l = [i for i in range(wLpi.shape[0]) if not
                        sym.Matrix.hstack(*LC_ovlp)[i, :].is_zero_matrix]
            harm_vec = _vec_space_overlap(LC_islands, LC_ovlp, idx=idx_l,
                                        v1_recon=True)
            assert all(well_spaced(wL.transpose()*v) for v in harm_vec)
        else:
            harm_vec = []
    else:
        harm_vec = []
    # number of each variable kind
    n_nd = nd_mat.shape[1]
    n_comp = len(comp_vec)
    n_harm = len(harm_vec)
    n_ext = len(_linearly_indep_cols(wJ)) - n_comp
    try:
        assert n_nd + n_comp + n_harm + n_ext == max([wJ.shape[0], wC.shape[0], wL.shape[0]])
    except:
        breakpoint()

    # Generate unique choices for compact, harmonic, extended
    # Need cMat and lMat to consider decoupling from free/frozen modes
    cMat = gen_cap_mat(circuit, utils.zero_start_edges(edges))
    lMat = gen_ind_mat(circuit, utils.zero_start_edges(edges))
    if n_comp > 0:
        u_comp = unique_compact(comp_vec, nd_mat, sig_vec, free_vec, cMat, wJ)
    else:
        u_comp = []
    if n_harm > 0:
        u_harm = unique_harmonic(harm_vec, nd_mat, sig_vec, free_vec, froz_vec, cMat, lMat, wL)
    else:
        u_harm = []
    if n_ext > 0:
        u_ext = unique_extended(sym.Matrix.hstack(*u_comp), sym.Matrix.hstack(*u_harm),
                            nd_mat, sig_vec, free_vec, froz_vec, cMat, lMat, wJ)
    else:
        u_ext = []
   
    all_Z = []
    present_modes = [x for x in (u_comp, u_ext, u_harm) if len(x) > 0]
    for dyn_cols in itertools.product(*present_modes):
        all_Z.append(sym.Matrix.hstack(*dyn_cols, nd_mat))
        
    return all_Z




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

