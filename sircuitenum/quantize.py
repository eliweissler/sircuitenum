__doc__ = "quantize.py: contains functions used to produce symbolic hamiltonians"
__author__ = "Eli Weissler"
__version__ = "0.1.0"
__all__ = ["gen_cap_mat", "gen_ind_mat", "gen_junc_pot", "quantize_circuit"]

import itertools
import functools

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


def _indices_to_arr(dim, ind):
    arr = np.zeros((dim, 1), dtype=int)
    for i in ind:
        arr[i] = 1
    return sym.Matrix(arr)

def _islands_to_vectors(circuit, edges, links, start_idx=1):
    edges = utils.zero_start_edges(edges)
    ind = find_islands(circuit, edges, links)
    n_nodes = utils.get_num_nodes(edges)
    vecs = []
    for i in range(start_idx, len(ind)):
        vecs.append(_indices_to_arr(n_nodes, ind[i]))
    return vecs


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

def _equal_up_to_column_sign(A: sym.Matrix, B: sym.Matrix):
    
    if A.shape != B.shape:
        return False
    
    rows, cols = A.shape
    for j in range(cols):
        ratio = 0
        for i in range(rows):
            # Check if 0 entries match
            if A[i, j] == 0 or B[i, j] == 0:
                if A[i, j] == B[i, j]:
                    continue
                else:
                    return False
            # Check if nonzero entries are all 1 or -1
            # Set ratio using first nonzero element
            elif ratio == 0:
                ratio = A[i, j]/B[i,j]
                if not(ratio == 1 or ratio == -1):
                    return False
            # Use that one to compare to all subsequent ones
            elif ratio != A[i, j]/B[i,j]:
                return False
    
    return True


def _equal_up_to_column_shift_and_sign(A: sym.Matrix, B: sym.Matrix):

    print(A, B)
    if A.shape != B.shape:
        return False
    rows, cols = A.shape
    for j in range(cols):
        # try positive and negative version of column
        diff1 =   A[:, j] - B[:, j]
        diff2 = - A[:, j] - B[:, j]
        print(diff1, diff2)
        if not (all(diff1[i] == diff1[0] for i in range(rows)) or 
                all(diff2[i] == diff2[0] for i in range(rows))):
            return False
    return True


def compact_alignment_transformation(wT:sym.Matrix, n_c:int):

    wc = wT[:, :n_c]

    # TODO:
    # Think about row magnitudes, we want the rows with the
    # smallest magnitude
    # Shouldn't matter

    # Try all different linearly independent row combinations
    row_sets = _linearly_indep_row_sets(wc)
    all_Zc = []
    for rows in row_sets:
        # Get the appropriate transformation for each set
        Zc = wc[rows, :].pinv()
        # Verify it's well-aligned
        if compact_well_aligned(wc*Zc, n_c):
            if not (any(_equal_up_to_column_sign(Zc, X) for X in all_Zc) or 
                    any(_equal_up_to_column_shift(Zc, X) for X in all_Zc)):
                all_Zc.append(Zc)

    # Return all possible transformations
    all_Z = []
    for Zc in all_Zc:
        Z = sym.eye(wT.shape[1])
        Z[:n_c, :n_c] = Zc
        all_Z.append(Z)

    return all_Z


def compact_well_aligned(wT: sym.Matrix, n_c: int):

    # Nothing to worry about here
    if n_c == 0:
        return True
    
    wc = wT[:, :n_c]
    rows, cols = wc.shape

    # 1 periodic variable -- is there a 1 or -1 there
    if n_c == 1:
        nz_entries = [sym.Abs(wc[i,0]) for i in range(rows) if wc[i,0].is_nonzero]
        if len(nz_entries) == 0:
            return False
        elif 1 in nz_entries:
            return True
        else:
            return False
        
    # 2 or more periodic variables --
    # Does each compact variable have at least one junction that
    # depends on it alone (among compact variables)
    else:
        # Select all rows with a single 1/-1 and rest 0's
        e_rows = []
        for i in range(wc.shape[0]):
            nz_entries = [sym.Abs(wc[i,j]) for j in range(cols) if wc[i,j].is_nonzero]
            if len(nz_entries) == 0:
                continue
            if len(nz_entries) == 1 and nz_entries[0] == 1:
                e_rows.append(i)

        # Check if all the single entry rows span the
        # number of compact variables: yes->well aligned
        return len(_linearly_indep_rows(wc[e_rows, :])) == n_c




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
    # labels of mode types
    mode_types = []
    # dynamical and nondynamical mode vectors
    dy_modes = []
    nd_modes = []
    
    # sigma -- each set of connected components (should only be 1)
    sig_vec = _islands_to_vectors(circuit, edges, [], start_idx=0)
    if len(sig_vec) > 1:
        print("--- WARNING: Multiple disconnected circuit segments ---")
    # Free -- capacitively shunted islands
    free_vec = _islands_to_vectors(circuit, edges, ["C"])
    # Frozen -- inductively shunted islands
    froz_vec = _islands_to_vectors(circuit, edges, ["L"])
    
    # Compact -- J,C shunted islands
    comp_vec = _islands_to_vectors(circuit, edges, ["J", "C"])
    # Harmonic -- L,C shunted islands
    harm_vec = _islands_to_vectors(circuit, edges, ["L", "C"])

    # Extended -> must fill out the junction connectivity
    # matrix. Minimize coupling, or maybe project to construct
    # elements orthogonal to the harmonic vectors?
    # All the compact variables will already be orthogonal from each other.
    w_inv = sym.transpose(gen_w(circuit, edges, "J").pinv())
    jj_basis = comp_vec + [w_inv.col(i) for i in range(w_inv.shape[1]-1)]
    # jj_basis = sym.GramSchmidt(jj_basis)

    return sym.Matrix.hstack(*jj_basis, *(harm_vec+free_vec+froz_vec+sig_vec))
    

def gen_cap_mat(circuit, edges):
    """
    Generates a capacitance matrix using Sympy for the given circuit.
    Energy C = Q_vec @ ind_mat @ Q_vec

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



def diag_cap_transform(C, numerical=False, eps = 1e-10):
    """
    Generates a transformation into the
    basis where the capacitance matrix is diagonal

    C = P^-1 D P -> D = P C P^-1

    If P is constructed as orthonormal, then

    D = P C P^T

    -> Z = P^T

    Args:
        C (sym.Matrix): capacitance matrix

    Returns:
        Z transformation where node_vars = Z@new_vars
    """

    # Plug in random numbers if numerical
    if numerical:
        C, C_vals = num_subs(C)
        

    # Re-order to put zeros last
    evecs, evals = C.diagonalize()
    evecs = sym.re(evecs)
    evals = sym.re(evals.diagonal())
    nonzero_ind = []
    zero_ind = []
    for i, ev in enumerate(evals):
        if numerical:
            if ev < eps:
                zero_ind.append(i)
            else:
                nonzero_ind.append(i)
        else:
            if ev == 0:
                zero_ind.append(i)
            else:
                nonzero_ind.append(i)

    order = nonzero_ind + zero_ind

    evecs = evecs[:, order]


    # Make columns unit length
    P_ortho = sym.zeros(*evecs.shape)
    for col in range(evecs.shape[1]):
        P_ortho[:, col] = evecs[:, col]/evecs[:, col].norm()

    # clip if numerical
    if numerical:
        P_ortho = np.array(P_ortho)
        P_ortho[np.abs(P_ortho) < eps] = 0
        P_ortho = sym.Matrix(P_ortho)

    if numerical:
        return P_ortho, C_vals
    else:
        return P_ortho


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
                Qv = remove_row_(Qv, n-n_deleted)
                Q_vec = remove_row_(Q_vec, n-n_deleted)
                th_vec = remove_row_(th_vec, n-n_deleted)
                C_mat = remove_row_(C_mat, n-n_deleted)                
                C_mat = remove_col_(C_mat, n-n_deleted)
                L_mat = remove_row_(L_mat, n-n_deleted)
                L_mat = remove_col_(L_mat, n-n_deleted)
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

def remove_row_(M, i):
    if not M.row_del(i) is None:
        return M.row_del(i)
    else:
        return M

def remove_col_(M, i):
    if not M.col_del(i) is None:
        return M.col_del(i)
    else:
        return M