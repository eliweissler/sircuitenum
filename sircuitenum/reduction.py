#!/usr/bin/env python
"""circuit_reduction.py: Contains functions to reduce the number
 of duplicate circuits present"""
__author__ = "Mohit Bhat, Eli Weissler"
__version__ = "0.1.0"
__status__ = "Development"

# -------------------------------------------------------------------
# Import Statements
# -------------------------------------------------------------------


import numpy as np
import pandas as pd
import networkx as nx

from sircuitenum import utils


def colors_match(n1_attrib, n2_attrib):
    '''returns False if either no color or if the colors do not match'''
    try:
        return n1_attrib['color'] == n2_attrib['color']
    except KeyError:
        return False


def convert_circuit_to_component_graph(circuit: list, edges: list,
                                       ground_nodes: list = [],
                                       ground_color: int = -1,
                                       comp_map: dict = None):
    """Encodes a circuit as a colored component graph -- see
    Enumeration of Architectures with Perfect Matchings
    Herber, Guo, Allison.

    Assumes that all circuit elements are two port-simple
    devices (i.e. symmetric)

    Assumes component type isomorphism. This
    means that different copies of the same component 
    are considered identical.

    Automatically removes any edges placed between labeled 
    ground nodes.

    Practically differs from port graphs in that each node
    and device is represented by a single graph vertex, as
    opposed to each port being a different vertex.

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        ground_nodes (list): optionally, a list of nodes that are grounded
                             e.g. [0, 1]
        ground_color int: color for ground nodes, default is -1
        comp_map (dict): dictionary that maps components to colors for
                         the colored graph. So it's consistent between
                         different circuits.

    Returns:
        nx.Graph representation of the port graph
    """
    if comp_map is None:
        comp_map = utils.ENUM_PARAMS["EDGE_COLOR_DICT"]

    # Count the degree of each vertex to include as a component
    edge_array = np.array(edges)
    vert, counts = np.unique(edge_array, return_counts=True)

    # Max degree is equal to number of vertices - 1 (fully connected)
    # Always have this for consistency of coloring
    max_deg = len(vert)-1

    # Build the graph
    # One vertex per node
    v_labels = {}
    G = nx.Graph()
    for i in range(len(vert)):
        if vert[i] in ground_nodes:
            color = ground_color
            label = f'GND'
        else:
            color = counts[i]
            label = f'v{vert[i]}'

        # In the case of multiple grounds
        if label not in G.nodes:
            G.add_node(label, color=color)
        v_labels[vert[i]] = label

    # Keep track of which copy of each device you're on
    device_counts = {}
    for d in comp_map:
        device_counts[tuple(sorted(d))] = 0

    for i, c in enumerate(circuit):
        comps = tuple(sorted(c))
        # Get the device count
        elem = ''.join(comps)
        copy = device_counts[comps]


        # Add edges to connected components
        # ignore edge if both are in ground nodes
        ext = edges[i]
        if ground_nodes:
            if ext[0] in ground_nodes and ext[1] in ground_nodes:
                continue
        
        # Add a node for the device
        G.add_node(f"{elem}{copy}", color=max_deg+1+comp_map[comps])

        edges_to_add = []
        for p in range(len(ext)):
            v = ext[p]
            edges_to_add += [(f"{elem}{copy}", v_labels[v])]
        G.add_edges_from(edges_to_add)

        # Iterate device count
        device_counts[comps] += 1

    return G


def convert_circuit_to_port_graph(circuit: list, edges: list,
                                  ground_nodes: list = [],
                                  ground_color: int = -1,
                                  comp_map: dict = None):
    """Encodes a circuit as a colored port graph -- see
    Enumeration of Architectures with Perfect Matchings
    Herber, Guo, Allison.

    Assumes that all circuit elements are two port-simple
    devices (i.e. symmetric)

    Assumes port type and component type isomorphism. This
    means that ports within a device and different copies
    of the same component are considered identical.

    Automatically removes any edges placed between ground nodes.

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        ground_nodes (list): optionally, a list of nodes that are grounded
                             e.g. [0, 1]
        ground_color int: color for ground nodes, default is -1
        comp_map (dict): dictionary that maps components to colors for
                         the colored graph. So it's consistent between
                         different circuits.

    Returns:
        nx.Graph representation of the port graph
    """
    if comp_map is None:
        comp_map = utils.ENUM_PARAMS["EDGE_COLOR_DICT"]

    # Count the degree of each vertex to include as a component
    edge_array = np.array(edges)
    vert, counts = np.unique(edge_array, return_counts=True)

    # Max degree is equal to number of vertices - 1 (fully connected)
    # Always have this for consistency of coloring
    max_deg = len(vert)-1

    # Build the graph
    G = nx.Graph()
    v_labels = {}
    for i in range(len(vert)):
        n_ports = counts[i]
        v_labels[i] = [""]*n_ports
        # Add a port for each connection
        for p in range(n_ports):
            if vert[i] in ground_nodes:
                color = ground_color
                label = f'v{vert[i]}_p{p}_GND'
            else:
                color = counts[i]
                label = f'v{vert[i]}_p{p}'
            G.add_node(label, color=color)
            v_labels[vert[i]][p] = label

        # Add the internal connections
        edges_to_add = []
        for p0 in range(counts[i]):
            for p1 in range(counts[i]):
                if p0 != p1 and p0 < p1:
                    edges_to_add += [(v_labels[vert[i]][p0],
                                      v_labels[vert[i]][p1])]
        G.add_edges_from(edges_to_add)

    # Add internal ground connections if there are multiple ground nodes
    if len(ground_nodes) > 1:
        edges_to_add = []
        # Identify two different ground nodes
        for v1 in ground_nodes:
            i1 = np.where(vert == v1)[0][0]
            for v2 in ground_nodes:
                # v1 < v2 so we don't repeat
                if v1 != v2 and v1 < v2:
                    i2 = np.where(vert == v2)[0][0]
                    for p1 in range(counts[i1]):
                        for p2 in range(counts[i2]):
                            label1 = f'v{v1}_p{p1}_GND'
                            label2 = f'v{v2}_p{p2}_GND'
                            edges_to_add += [(label1, label2)]
        G.add_edges_from(edges_to_add)



    # Keep track of how many ports are taken on each node
    ports_taken = np.zeros(len(vert), dtype=int)

    # Keep track of which copy of each device you're on
    device_counts = {}
    for d in comp_map:
        device_counts[tuple(sorted(d))] = 0

    for i, c in enumerate(circuit):

        comps = tuple(sorted(c))
        # Get the device count
        elem = ''.join(comps)
        copy = device_counts[comps]

        # external connections -- ignore if it is between two ground nodes
        ext = edges[i]
        if ground_nodes:
            if ext[0] in ground_nodes and ext[1] in ground_nodes:
                continue

        # Add two ports for each device
        G.add_node(f"{elem}{copy}_p0", color=max_deg+1+comp_map[comps])
        G.add_node(f"{elem}{copy}_p1", color=max_deg+1+comp_map[comps])

        # Add edges -- internal connection and external
        # Have first edge be from port 0
        # Have second edge be from port 1
        # internal
        edges_to_add = [(f"{elem}{copy}_p0", f"{elem}{copy}_p1")]
        # external
        for p in range(len(ext)):
            v = ext[p]
            edges_to_add += [(f"{elem}{copy}_p{p}", v_labels[v][ports_taken[v]])]
            ports_taken[v] += 1
        G.add_edges_from(edges_to_add)

        # Iterate device count
        device_counts[comps] += 1

    return G


def isomorphic_circuit_in_set(circuit: list, edges: list, c_set: list,
                              e_set=None, return_index=False):
    """Helper function to see if a circuit that is isomprphic
    to the given circuit
    (list/tuple of tuples) is in a set of circuits
    (list of list/tuple of tuples)

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [("J"),("L", "J"), ("C")]
        edges (list): a list of edge connections for the circuit
                        (assumed edges for everything in the set
                        if no e_set is given)
                        e.g. [(0,1), (0,2), (1,2)]
        c_set (list of lists): list of circuit-like elements
        e_set (list of lists): list of edges for the circuit list. If none
                               is given then assumes edges argument is the edge
        return_index (bool): return index of isomorphic circuit, returns
                             nan if it's not present

    Returns:
        True if circuit is present in c_set, False if it isn't
    """
    port_graph = convert_circuit_to_port_graph(circuit, edges)
    for i, c2 in enumerate(c_set):
        c2_edges = edges
        if e_set is not None:
            c2_edges = e_set[i]
        port_graph_2 = convert_circuit_to_port_graph(c2, c2_edges)
        if nx.is_isomorphic(port_graph, port_graph_2, node_match=colors_match):
            if return_index:
                return i
            else:
                return True
    if return_index:
        return np.nan
    else:
        return False


def mark_non_isomorphic_set(df: pd.DataFrame, **kwargs):
    """Reduces a set of circuits to contain only those
    whose port graphs are not isomorphic to each other.

    Args:
        df (pd.DataFrame): Dataframe where each row represents a
                           specific circuit. Assumes that every
                           entry has the same number of nodes,
                           and comes from the same basegraph
        to_consider (np.array, optional): logical array that marks
                                rows to consider. For use
                                when some have already been
                                eliminated for other reasons.
                                Defaults to considering all.

    Returns:
        Nothing, fills in the 'in_non_iso_set' and 'equiv_circuit'
        columns of df
    """
    to_consider = kwargs.get("to_consider", np.ones(df.shape[0], dtype=bool))

    # See if there's anything to consider
    if np.all(np.logical_not(to_consider)):
        return

    # Determine starting point -- first entry to consider
    i0 = np.argmax(to_consider)

    in_non_iso_set = df['in_non_iso_set'].values
    equiv_circuit = df['equiv_circuit'].values

    # The first graph is always unique
    unique_graphs = [(convert_circuit_to_component_graph(df.iloc[i0]['circuit'],
                                                    df.iloc[i0]['edges']),
                      df.iloc[i0]['unique_key'])]

    in_non_iso_set[i0] = 1
    equiv_circuit[i0] = ""

    # Compare to each previously found unique graph
    # If it's not isomorphic to any of them
    # Then add it to the list
    # If it is, then mark which graph it is isomorphic to
    for i in range(i0+1, df.shape[0]):
        if to_consider[i]:
            # Compare port graph to all entries in unique set
            row = df.iloc[i]
            # g_new = convert_circuit_to_port_graph(
            #     row['circuit'], row['edges'])
            g_new = convert_circuit_to_component_graph(
                row['circuit'], row['edges'])
            iso_flag = False
            for (g, uid) in unique_graphs:
                if nx.is_isomorphic(g, g_new, node_match=colors_match):
                    iso_flag = True
                    equiv_circuit[i] = uid
                    break
            # Is unique - add to unique set
            if not iso_flag:
                unique_graphs.append((g_new, row['unique_key']))
                in_non_iso_set[i] = 1
                equiv_circuit[i] = ""
            # Is not unique
            else:
                in_non_iso_set[i] = 0

    df['in_non_iso_set'] = in_non_iso_set
    df['equiv_circuit'] = equiv_circuit


def remove_series_elems(circuit: list, edges: list,
                        to_reduce: list = ["L", "C"]):
    """
    Reduces the size of the given circuit by eliminating
    linear components that are in series.

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        to_reduce (list, optional): circuit elements to reduce.
                                    Defaults to linear elements ['L','C']

    Returns:
        True if the circuit cannot be reduced
        False if the circuit can be reduced
    """

    num_nodes = utils.get_num_nodes(edges)

    # Base case: 2 nodes
    if num_nodes == 2:
        return circuit, edges

    node_representation = utils.circuit_node_representation(
        circuit, edges)

    # Check each node to see if there are only
    # two of the same linear element connect to it
    for node in range(num_nodes):

        # Record how many of each component is at this node
        # and how many components total are there
        n_present = {}
        total_present = 0
        for component, node_repr in node_representation.items():
            n_present[component] = node_repr[node]
            total_present += n_present[component]

        # Can't reduce if we have more than two components
        # connected to the node
        if total_present == 2:
            for component in to_reduce:

                # Recursive case:
                # Reduce if both components connected to
                # a node are the same linear element
                if n_present[component] == 2:

                    # Remove this node and insert
                    # a direct edge in its place
                    new_c = []
                    new_e = []
                    to_connect = []
                    for i in range(len(edges)):
                        edge = list(edges[i])
                        if node in edge:
                            edge.remove(node)
                            to_connect.append(edge[0])

                        else:
                            new_e.append(edges[i])
                            new_c.append(circuit[i])
                    new_e.append(tuple(sorted(to_connect)))
                    new_c.append((component,))

                    # Re-number nodes if you removed
                    # not the max number
                    new_e = utils.renumber_nodes(new_e)

                    # Combine any redundant edges
                    new_c, new_e = utils.combine_redundant_edges(new_c, new_e)

                    return remove_series_elems(new_c, new_e)

    # Base case -- nothing to reduce
    return circuit, edges


def full_reduction(df: pd.DataFrame):
    """Performs the full reduction procedure:
    1) Removes circuits that have isolated series linear elements
    2) Removes circuits that have no jj's
    3) Creates a set of circuits whose port-graphs are non-isomorphic


    Args:
        df (pd.Dataframe): dataframe that contains the

    Returns:
        nothing, fills in 'no_series', 'has_nl', 'in_non_isomorphic_set',
        and 'equivalent_circuit' columns of df.
    """

    # Mark series circuits
    eq_circuits = df.apply(lambda row: remove_series_elems(row['circuit'],
                                                           row['edges']),
                           axis=1)
    no_series = np.array([utils.get_num_nodes(eq_circuits.iloc[i][1]) ==
                          utils.get_num_nodes(df['edges'].iloc[i])
                          for i in range(df.shape[0])])
    df['no_series'] = no_series.astype(int)

    # Mark no jj/no qps circuits
    has_nl = df.apply(lambda row: utils.ENUM_PARAMS["filter"](row['circuit']), axis=1).values
    df['filter'] = has_nl.astype(int)

    # Create non-isomorphic set of yes-jj, no-series circuits
    mark_non_isomorphic_set(df, to_consider=np.logical_and(no_series, has_nl))

    # Create non-isomorphic set of no-jj, no-series circuits
    mark_non_isomorphic_set(df,
                            to_consider=np.logical_and(no_series,
                                                       np.logical_not(has_nl)))


def full_reduction_by_group(df: pd.DataFrame):
    """Performs the full reduction procedure,
    iterating over graph index and edge counts unique
    values for efficiency

    Args:
        df (pd.DataFrame): dataframe that contains the circuits

    Returns:
        dataframe with reduced circuit set
    """

    reduced = []

    # Iterate through graph index and edge_counts
    by_basegraph = df.groupby("graph_index")
    for graph_index in by_basegraph.indices:
        subset1 = by_basegraph.get_group(graph_index)
        print(subset1)
        by_edge_counts = subset1.groupby("edge_counts")
        for edge_counts in by_edge_counts.indices:
            subset2 = by_edge_counts.get_group(edge_counts).copy()
            full_reduction(subset2)
            reduced.append(subset2)

    return pd.concat(reduced)
