"qpackage_interface.py: converts circuits to analysis packages"
__author__ = "Mohit Bhat, Eli Weissler"
__version__ = "0.1.0"
__status__ = "Development"
__all__ = ['single_edge_loop_kiting',
           'find_loops',
           'inductive_subgraph',
           'to_SQcircuit',
           'to_SCqubits',
           'to_CircuitQ',
           'to_Qucat']

# -------------------------------------------------------------------
# Import Statements
# -------------------------------------------------------------------

import numpy as np
import networkx as nx

import SQcircuit as sq
import scqubits as scq
from typing import Union

import sircuitenum.utils as utils

# -------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------


def single_edge_loop_kiting(circuit, edges):
    """ expands edges which contain loops by splitting inductors and
    adding nodes. Done since networkx doesn't calcuate loops for multigraphs

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [("J",),("L", "J"), ("C",)]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]

    Returns:
        copies of circuit/edges where every edge that contained both a J
        and an L have been expanded into two L's with an intermediate node
    """

    # Make copies
    circuit_out = []
    edges_out = []

    # Highest index node present
    max_node = utils.get_num_nodes(edges)-1

    for element, edge in zip(circuit, edges):

        # If we have a junction and an inductor, then there's a loop
        # Make it understandable by loop finding
        # By streching the inductor into two, adding an extra node
        if "J" in element and "L" in element:

            # Put in the edge without inductors
            circuit_out.append(tuple([x for x in element if x != "L"]))
            edges_out.append(edge)

            # Make the inductors off to the side
            circuit_out.append(("L",))
            edges_out.append((edge[0], max_node + 1))
            circuit_out.append(("L",))
            edges_out.append((max_node + 1, edge[1]))

            # Keep track of how many nodes you've added
            max_node += 1

        # If there's not both a junction and inductor
        # then keep the edge the same
        else:
            circuit_out.append(element)
            edges_out.append(edge)

    return circuit_out, edges_out


def find_loops(circuit, edges, ind_elem=["J", "L"]):
    """ Provides a list of loops

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [("J",),("L", "J"), ("C",)]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        ind_elem (list): symbols that define inductive elements.
                        Default is ind_elem = ["J", "L"]

    Returns:
        loop_lst (list): a list of loops in the circuit
        circuit (list): a list of element labels for the desired circuit
                        e.g. [("J",),("L", "J"), ("C",)]
        edges (list): a list of edge connections for the desired circuit
    """

    # save min mode number for recovering afterwards
    min_node = min(min(x) for x in edges)

    # Renumber to start from 0
    edges = utils.zero_start_edges(edges)

    # Expand single edge loops
    max_node_og = utils.get_num_nodes(edges)-1
    circuit_temp, edges_temp = single_edge_loop_kiting(circuit, edges)

    # Make a graph that represents only inductive edges
    ind_edges = inductive_subgraph(circuit_temp, edges_temp, ind_elem)
    G = nx.from_edgelist(ind_edges)

    # Find loops in the inductive subgraph
    # And filter out any edges that we added
    loop_lst = [tuple(sorted([x for x in c if x <= max_node_og]))
                for c in nx.cycle_basis(nx.Graph(G))]
    
    # Add min node number to recover the original numbering
    loop_lst = [tuple([n + min_node for n in nodes]) for nodes in loop_lst]

    return loop_lst


def inductive_subgraph(circuit, edges, ind_elem=["J", "L"]):
    """Returns a list of edges that contain an inductive element

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        ind_elem (list): symbols that define inductive elements.
                        Default is ind_elem = ["J", "L"]
    """

    return [edges[i] for i in range(len(edges))
            if np.any(np.in1d(circuit[i], ind_elem))]


def add_explicit_ground_node(circuit: list, edges: list, params: dict, ecg: float = 20,
                             rand_amp=0.0):
    """
    Takes in a circuit + edges combo and returns a modified
    version with an explicit ground node added (as node 0)

    If the 0 node was present before, adds 1 to each node

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [("J"),("L", "J"), ("C")]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        params (dict): dictionary with entries C, L, J, CJ,
                    which represent the paramaters for the circuit elements.
                    Additionally entries of C_units, L_units, J_units,
                    and CJ_units.
        ecg (float): EC for capacitors coupling to ground in GHz
    
    Returns:
        Modified version of circuit, edges with capacitive coupling
        to a ground node added
    """
    # Get unique node values
    edges_og = edges[:]
    edges = utils.zero_start_edges(edges)
    node_vals = []
    for n1, n2 in edges:
        if n1 not in node_vals:
            node_vals.append(n1)
        if n2 not in node_vals:
            node_vals.append(n2)
    node_vals = sorted(node_vals)
    n_nodes = len(node_vals)
    new_circuit = circuit + [("C",)]*n_nodes
    new_edges = [(e[0] + 1, e[1] + 1) for e in edges] + [(0, n+1) for n in node_vals]

    # Modify params for old elements
    # to reflect new labeling
    new_params = {}
    n_edges_og = len(edges_og)
    for i in range(n_edges_og):
        edge_og = edges_og[i]
        edge = new_edges[i]
        for elem in circuit[i]:
            new_params[(edge, elem)] = params[(edge_og, elem)]
            if elem == "J" and (edge_og, "CJ") in params:
                new_params[(edge, "CJ")] = params[(edge_og, "CJ")]
    
    # Add the capacitive connections to ground
    for i in range(n_edges_og, n_edges_og + n_nodes):
        edge = new_edges[i]
        elem = new_circuit[i][0]
        new_params[(edge, elem)] = (ecg*np.random.normal(1, rand_amp), "GHz")

    return new_circuit, new_edges, new_params


def swap_nodes(edges: list, na: int, nb: int):
    new_edges = []
    for (n0, n1) in edges:
        # Swap na and nb
        if n0 == nb:
            n0 = na
        elif n0 == na:
            n0 = nb
        if n1 == nb:
            n1 = na
        elif n1 == na:
            n1 = nb
        new_edges.append((n0, n1))
    return new_edges

def to_SQcircuit(circuit: list, edges: list,
                 trunc_num: Union[int, list] = 50, **kwargs):
    """Converts circuit from list of labels and edges to a
    SQcircuit formatted circuit network

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        trunc_num (int or list): truncation number for each mode
        params (dict): dictionary with entries C, L, J, CJ,
                    which represent the paramaters for the circuit elements.
                    Additionally entries of C_units, L_units, J_units,
                    and CJ_units. Inputting nothing uses the default
                    parameter values/units from utils.ELEM_DICT.

    Returns:
        converted_circuit (SQcircuit.Circuit): returns the input circuit
                                               converted to SQcircuit.
    """

    params = kwargs.get("params", utils.gen_param_dict(circuit, edges,
                                                       utils.ELEM_DICT,
                                                       rand_amp=kwargs.get("rand_amp", 0.00)))
    flux_dist = kwargs.get("flux_dist", "junctions")

    # ground node is node = 0
    ground_node = kwargs.get("ground_node", None)
    if ground_node is None:
        circuit, edges, params = add_explicit_ground_node(circuit, edges, params)
    elif ground_node != 0:
        edges = swap_nodes(edges, 0, ground_node)
        new_params = {}
        for key in params:
            edge, elem = key
            new_edge = swap_nodes([edge], 0, ground_node)[0]
            new_params[(new_edge, elem)] = params[(edge, elem)]
        params = new_params

    loops = find_loops(circuit, edges)
    loop_defs = {}
    # Map inductive cycle basis to loops
    for lp in loops:
        loop_defs[lp] = sq.Loop(id_str=str(lp))

    # Build sqcircuit dictionary that maps edges
    # to a list of element objects, which have
    # their loops set
    circuit_dict = {}
    for elems, edge in zip(circuit, edges):

        # Record all the loops for this edge
        loops_pres_J = []
        loops_pres_L = []
        for lp in loops:
            # If the edge is part of the loop
            if edge[0] in lp and edge[1] in lp: 
                # Either J or L in the branch
                if "J" in elems and "L" not in elems:
                    loops_pres_J.append(loop_defs[lp])
                elif "J" not in elems and "L" in elems:
                    loops_pres_L.append(loop_defs[lp])
                # Both J and L in the branch
                else:
                    # two-node loop -- Flux between J & L
                    if len(lp)==2:
                        loops_pres_J.append(loop_defs[lp])
                        loops_pres_L.append(loop_defs[lp])
                    # > 2 node loop -- Assign flux to JJ
                    else:
                        loops_pres_J.append(loop_defs[lp])
                        # loops_pres_L.append(loop_defs[lp])
        
        # print("edge:", edge)
        # print('loops_pres J:', loops_pres_J)
        # print("loops_pres L:",loops_pres_L)
        # Add all the elements
        circuit_dict[edge] = []
        for elem in elems:
            val, units = params[(edge, elem)]
            units = "GHz"

            if elem == "C":
                id_str = "C_" + "".join([str(x) for x in edge])
                circuit_dict[edge].append(sq.Capacitor(val, units,
                                                    id_str=id_str,
                                                    ))

            elif elem == "L":
                id_str = "L_" + "".join([str(x) for x in edge])
                circuit_dict[edge].append(sq.Inductor(val, units,
                                                    id_str=id_str,
                                                    loops=loops_pres_L))



            elif elem == "J":
                id_str = "J_" + "".join([str(x) for x in edge])
                if (edge, "CJ") in params:
                    val2, units2 = params[(edge, "CJ")]
                    if val2 > 0:
                        j_c = sq.Capacitor(val2, units2, id_str="C"+id_str)
                        circuit_dict[edge].append(sq.Junction(val, units,
                                                            id_str=id_str,
                                                            loops=loops_pres_J,
                                                            cap=j_c))
                    else:
                        circuit_dict[edge].append(sq.Junction(val, units,
                                                            id_str=id_str,
                                                            loops=loops_pres_J)
                                                )
                else:
                    circuit_dict[edge].append(sq.Junction(val, units,
                                                        id_str=id_str,
                                                        loops=loops_pres_J))
            else:
                raise ValueError("Unknown circuit compenent present.\
                                Must be either C, J, or L")



    sqC = sq.Circuit(circuit_dict, flux_dist=flux_dist)

    # Convert truncation num to list
    if isinstance(trunc_num, int):
        trunc_num = [trunc_num]*sqC.n
    
    if sqC.n > len(trunc_num):
        # print("Warning, too few trunc nums given, filling in with max value (and adding 10)", max(trunc_num))
        trunc_num = [x for x in trunc_num] + [np.max(trunc_num)]*(sqC.n-len(trunc_num))
    elif sqC.n < len(trunc_num):
        trunc_num = [np.max(trunc_num)]*sqC.n

    sqC.set_trunc_nums([x for x in trunc_num])

    return sqC


def to_SCqubits(circuit: list, edges: list,
                trunc_num: Union[int, list] = 50,
                cutoff: Union[int, list] = 101,
                **kwargs):
    """Converts circuit from list of labels and edges to a
    SCqubits formatted circuit network

    ## NOTE: ONLY SUPPORTS VALUES IN GHz

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [("J",),("L", "J"), ("C",)]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        trunc_num (int or list): Number of eigenstates to consider for each
                                 mode in a composite circuit.
        https://scqubits.readthedocs.io/en/latest/guide/ipynb/custom_circuit_hd.html
        cutoff (int or list): Number of points to use in the underlying
                              position space for each mode.

        params (dict): dictionary with entries C, L, J, CJ,
                    which represent the paramaters for the circuit elements.
                    Additionally entries of C_units, L_units, J_units,
                    and CJ_units. Inputting nothing uses the default
                    parameter values/units from utils.ELEM_DICT.

    Returns:
        converted_circuit (scqubits.Circuit): returns the input circuit
                                               converted to scqubits.
    """
    params = kwargs.get("params", utils.gen_param_dict(circuit, edges,
                                                       utils.ELEM_DICT))
    sym_cir = kwargs.get("sym_cir", False)
    initiate_sym_calc = kwargs.get("initiate_sym_calc", True)

    # ground node is node = 0
    ground_node = kwargs.get("ground_node", None)
    if ground_node is None:
        edges = utils.zero_start_edges(edges)
        edges = [(n1 + 1, n2 + 1) for (n1, n2) in edges]
        new_params = {}
        for key in params:
            edge, elem = key
            new_edge = (edge[0] + 1, edge[1] + 1)
            new_params[(new_edge, elem)] = params[(edge, elem)]
        params = new_params
    elif ground_node != 0:
        edges = swap_nodes(edges, 0, ground_node)
        for key in params:
            edge, elem = key
            new_edge = swap_nodes([edge], 0, ground_node)[0]
            new_params[(new_edge, elem)] = params[(edge, elem)]
        params = new_params

    basis_completion = kwargs.get("basis_completion", "heuristic")

    # Build scqubits circuit yaml string
    circuit_yaml = "branches:"
    for elems, edge in zip(circuit, edges):
        edge_str = "_".join([str(x) for x in edge])
        # Add all the elements
        for elem in elems:
            val = f"{elem}_{edge_str} = "
            if elem == "J":
                e_str = "JJ"
                val1, _ = params[(edge), elem]
                if (edge, "CJ") in params:
                    val2, _ = params[(edge), "CJ"]
                    if val2 > 0:
                        val += f"{val1}, {val2}"
                    else:
                        val += f"{val1}, 1000.0"
                else:
                    val += f"{val1}, 1000.0"
            else:
                e_str = elem
                val += f"{params[(edge), elem][0]}"

            circuit_yaml += "\n"
            circuit_yaml += f'- ["{e_str}", {edge[0]}, {edge[1]}, {val}]'
    if sym_cir:
        return scq.SymbolicCircuit.from_yaml(circuit_yaml, from_file=False,
                                             basis_completion=basis_completion,
                                             initiate_sym_calc=initiate_sym_calc)
    else:
        conv = scq.Circuit(circuit_yaml, from_file=False,
                           basis_completion=basis_completion,
                           generate_noise_methods=True)

    # Set cutoff and count number of modes
    n_nodes = utils.get_num_nodes(edges)
    n_modes = 0
    if not isinstance(cutoff, list):
        if n_nodes > 2:
            cutoff = [cutoff]*(n_nodes - 1)
        else:
            cutoff = [cutoff]
    for mode_type in ['periodic', 'extended']:
        if mode_type == "periodic":
            mode_str = "n"
        elif mode_type == "extended":
            mode_str = "ext"
        for mode in conv.var_categories[mode_type]:
            n_modes += 1
            exec(f"conv.cutoff_{mode_str}_{mode}={cutoff[mode-1]}")

    # Set truncation
    if n_modes > 1:
        hier = [[x] for x in np.arange(n_modes) + 1]
        if not isinstance(trunc_num, list):
            trunc_num = [trunc_num]*(n_modes)
        conv.configure(system_hierarchy=hier,
                       subsystem_trunc_dims=trunc_num)

    return conv
