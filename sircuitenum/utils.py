__doc__ = "utils.py: Contains utilities used in other files in the package"
__author__ = "Eli Weissler, Mohit Bhat"
__version__ = "0.1.0"
__all__ = ['get_circuit_data_batch', 'find_circuit_in_db', "get_equiv_circuits", "get_equiv_circuits_uid", "graph_index_to_edges", "edges_to_graph_index"]

import itertools
import functools
import multiprocessing
from typing import Union
from pathlib import Path
from time import sleep

import sqlite3
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

from func_timeout import func_timeout, FunctionTimedOut

# Set ENUM_PARAMS at end of file
global ENUM_PARAMS
ENUM_PARAMS = {}

ELEM_DICT = {
    'C': {'default_unit': 'GHz', 'default_value': 0.2},
    'L': {'default_unit': 'GHz', 'default_value': 1.0},
    'J': {'default_unit': 'GHz', 'default_value': 5.0},
    'CJ': {'default_unit': 'GHz', 'default_value': 20.0}
}

DOWNLOAD_PATH = Path(__file__).parent.parent

# Dictionary to store loaded basegraphs, so
# you don't have to load them from storage
# every time
LOADED_BASEGRAPHS = {}


def _basegraph_cache_key(n_nodes: int, planar: bool = False, regular: bool = False) -> str:
    """Build a cache key that is unique for a graph family."""
    return f"n{int(n_nodes)}_p{int(planar)}_r{int(regular)}"


def _basegraph_sort_key(graph: nx.Graph):
    """Deterministic, unique sort key for a base graph within a node count."""
    g6 = nx.to_graph6_bytes(graph, header=False).decode("ascii").strip()
    return (graph.number_of_edges(), g6)


def _edges_from_graph6(basegraph_g6: str):
    """Decode graph6 text and return the graph edge list."""
    return list(nx.from_graph6_bytes(basegraph_g6.strip().encode("ascii")).edges)


def graph_index_to_edges(graph_index: int, n_nodes: int):
    """
    Returns a list of edges [(from, to), (from, to)]
    for the specified base graph

    Args:
        graph_index (int): base graph number
        n_nodes (int): number of nodes in the base graph


    Returns:
        list of len 2 tuples where each tuple represents
        the starting and ending nodes for an edge in the graph
        [(from, to), (from, to),...]
    """
    return list(get_basegraphs(n_nodes)[graph_index].edges)


def edges_to_graph_index(edges: list, return_mapping: bool = False) -> int:
    """
    Matches a given set of edges to an isomorphic base graph.

    This function finds a base graph that is isomorphic 
    to the input edge set.

    If none is found, returns -1

    Parameters
    ----------
    edges : list of tuple of int
        A list of edge connections representing the desired circuit.  
        Example: ``[(0,1), (0,2), (1,2)]``.
    return_mapping : bool, optional
        If `True`, returns the mapping of edges to the base graph.  
        Defaults to `False`.

    Returns
    -------
    int
        The index of the graph matching the given edges within  
        the set of graphs with the same number of nodes.
    dict, optional
        If `return_mapping=True`, also returns a dictionary  
        mapping edges to the base graph.
    """
    # Graph object to use in comparison
    G1 = nx.Graph()
    G1.add_edges_from(edges)

    n_nodes = get_num_nodes(edges)
    n_edges = len(edges)
    possible_graphs = get_basegraphs(n_nodes)
    for i, G2 in enumerate(possible_graphs):
        if G2.number_of_edges() == n_edges:
            GM = nx.isomorphism.GraphMatcher(G1, G2)
            if GM.is_isomorphic():
                if return_mapping:
                    return i, GM.mapping
                return i

    return -1


def encoding_to_components(circuit_raw: str, char_mapping: dict = None):
    """Maps the raw circuit encoding to a list of lists of elements
    e.g. 261 -> [["J"], ["C", "J", "L"], ["L"]]

    Args:
        circuit_raw (str): string that represents base n number
                           where each character maps to a combination
                           of circuit componenets
        char_mapping (dict, optional): mapping from characters to
                                       circuit components.
                                       Defaults to CHAR_TO_COMBINATION.

    Returns:
        list of lists that represent the circuit elements along an edge:
        e.g. [["J"], ["C", "J", "L"], ["L"]]
    """
    if char_mapping is None:
        char_mapping = ENUM_PARAMS["CHAR_TO_COMBINATION"]
    return [char_mapping[str(e)] for e in circuit_raw]


def components_to_encoding(circuit: list, elem_mapping: dict = None):
    """Maps the list of circuit components to the database encoding
    e.g. [["J"], ["C", "J", "L"], ["L"]] -> 261

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        elem_mapping (dict, optional): mapping from circuit
                                       components to characters.
                                       Defaults to COMBINATION_TO_CHAR.

    Returns:
        list of lists that represent the circuit elements along an edge:
        e.g. [["J"], ["C", "J", "L"], ["L"]]
    """
    if elem_mapping is None:
        elem_mapping = ENUM_PARAMS["COMBINATION_TO_CHAR"]
    return "".join([elem_mapping[tuple(comps)] for comps in circuit])


def convert_loaded_df(df: pd.DataFrame, n_nodes: int, char_mapping: dict = None):
    """Load the edges/circuit element labels for a freshly-loaded df


    Args:
        df (pd.Dataframe): dataframe of circuits
        n_nodes (int): number of nodes in the circuits
        char_mapping (dict, optional): mapping from circuit
                                       components to characters.
                                       Defaults to CHAR_TO_COMBINATION.

    Returns:
        Nothing, modifies the dataframe
    """
    if char_mapping is None:
        char_mapping = ENUM_PARAMS["CHAR_TO_COMBINATION"]
    # Prefer the persisted basegraph graph6 when available.
    # This keeps edge reconstruction stable even if graph indexing changes.
    if 'basegraph_g6' in df.columns:
        edges = []
        for graph_index, basegraph_g6 in zip(df.graph_index.values,
                                             df.basegraph_g6.values):
            if basegraph_g6 is not None and str(basegraph_g6) != "":
                edges.append(_edges_from_graph6(str(basegraph_g6)))
            else:
                edges.append(graph_index_to_edges(int(graph_index), n_nodes))
        df['edges'] = edges
    else:
        # Get the edges
        df['edges'] = [graph_index_to_edges(int(i), n_nodes)
                       for i in df.graph_index.values]
    df['circuit_encoding'] = df.circuit.values.copy()
    df['circuit'] = [encoding_to_components(c, char_mapping=char_mapping)
                     for c in df.circuit.values]


def get_basegraphs(n_nodes: int, planar: bool = False, regular: bool = False):
    """
    Loads the base graphs for a specific number of nodes

    Args:
        n_nodes (int): number of nodes in the graph
        planar (bool): if True, only load planar graphs
        regular (bool): if True, only load regular graphs
    """
    cache_key = _basegraph_cache_key(n_nodes, planar=planar, regular=regular)
    # Return if it has already been loaded
    if cache_key in LOADED_BASEGRAPHS:
        return LOADED_BASEGRAPHS[cache_key]
    if int(n_nodes) > 8 and (not planar) and (not regular):
        raise ValueError("Only basegraphs up to 8 nodes are included in generality. See https://users.cecs.anu.edu.au/~bdm/data/graphs.html for larger sets of graphs, or select planar/regular.")
    elif n_nodes == 10 and not regular:
        raise ValueError("Only regular graphs are included for 10 nodes. See https://users.cecs.anu.edu.au/~bdm/data/graphs.html for larger sets of graphs, or select regular.")
    # Load it if it hasn't been loaded
    if cache_key not in LOADED_BASEGRAPHS:
        fname =  f"graph{n_nodes}c"
        if planar:
            fname = "planar_" + fname
        if regular:
            fname = fname + "_regular"
        fname = fname + ".g6"
        f = Path(DOWNLOAD_PATH, 'sircuitenum', 'graphs', fname)
        all_graphs = nx.read_graph6(f)
        # Fix two vertex case so it always returns a list
        if n_nodes == 2 or isinstance(all_graphs, nx.Graph):
            all_graphs = [all_graphs]
        # Sort first by number of edges and then by graph6 encoding.
        # The secondary key makes ties deterministic across reloads.
        all_graphs = sorted(all_graphs, key=_basegraph_sort_key)
        LOADED_BASEGRAPHS[cache_key] = all_graphs

    # Return if it has already been loaded
    return LOADED_BASEGRAPHS[cache_key]


def count_elems(circuit: list, base: int):
    """
    Counts the total number of each element
    label in the circuit, for use with the unmapped
    integer labels

    Args:
        circuit (list of str): a list of element labels for the desired circuit
                                (i.e., ['0','2','5','1'])
        base (int): The number of possible edges. By default this is 7:
                        (i.e., J, C, I, JI, CI, JC, JCI)

    Returns:
        list of length base, where each entry is the number of the
        element found at that index of ENUM_PARAMS["CHAR_LIST"]
    """
    counts = [0]*base
    for part in circuit:
        counts[ENUM_PARAMS["CHAR_LIST"].index(part)] += 1
    return counts


def count_elems_mapped(circuit: list, **kwargs):
    """
    Counts the total number of each mapped circuit
    element in the circuit

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        possible_elems (list): list of possible elements, default
                               is the unique set in CHAR_TO_COMBINATION

    Returns:
        dict: each entry is element -> number, i.e. "J" -> 2
    """
    possible_elems = kwargs.get("possible_elems", list_single_elems())
    counts = {}
    for elem in possible_elems:
        counts[elem] = 0

    for elems in circuit:
        for elem in possible_elems:
            for device in elems:
                if elem in device:
                    counts[elem] += 1

    return counts


def add_elem_number(circuit: list, **kwargs):
    """
    Returns a new circuit list where elements
    are numbered

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        possible_elems (list): list of possible elements, default
                               is the unique set in CHAR_TO_COMBINATION

    Returns:
        list: each element is followed by a number starting at 1 e.g. "C2" or "L1"
    """
    possible_elems = kwargs.get("possible_elems", list_single_elems())
    circuit_new = []

    counts = {}
    for elem in possible_elems:
        counts[elem] = 0

    for elems in circuit:
        elems_new = []
        for elem in elems:
            elem = [c for c in elem if c in possible_elems][0]
            counts[elem] += 1
            elems_new.append(elem+"_"+str(counts[elem]))
        circuit_new.append(tuple(elems_new))

    return circuit_new


def circuit_entry_dict(circuit: list, graph_index: int, n_nodes: int,
                       circuit_num: int, base: int,
                       basegraph_g6: str = None):
    """Creates a dictionary that can serve as a row of a dataframe of
    circuits, or can be used to write an individual row to a database

    Args:
        circuit (list of str): a list of element labels for the desired circuit
                                (i.e., ['0','2','5','1'])
        graph index (int): the index of the graph for the written circuit
                                within the file for the number of nodes
        n_nodes (int): Number of nodes in circuit
        circuit_num (int): n-th circuit generated from the basegraph, to make
                           a unique key.
        base (int): The number of possible edges. By default this is 7:
                        (i.e., J, C, I, JI, CI, JC, JCI)
        basegraph_g6 (str): graph6 encoding of the basegraph used
                    for this circuit row.

    Returns:
        dictionary with circuit, graph_index, edge_counts, n_nodes
    """
    c_dict = {}
    c_dict['circuit'] = "".join(circuit)
    c_dict['graph_index'] = graph_index
    c_dict['unique_key'] = f"n{n_nodes}_g{graph_index}_c{circuit_num}"
    c_dict['in_non_iso_set'] = 0
    c_dict['no_series'] = 0
    c_dict['filter'] = 0
    c_dict['equiv_circuit'] = ""
    if basegraph_g6 is not None:
        c_dict['basegraph_g6'] = str(basegraph_g6)

    counts = [str(c) for c in count_elems(circuit, base)]
    c_dict['edge_counts'] = ",".join(counts)
    c_dict['n_nodes'] = n_nodes
    c_dict['base'] = base
    return c_dict


def gen_param_dict(circuit, edges, vals=ELEM_DICT, rand_amp=0, min_val=1e-06):
    """
    Generates a dictionary of parameters for use with
    the circuit conversion functions. Sets all components
    to the same values.

    Maps (edge, elem) to (value, unit):

    i.e., ((0,1), "J") -> (5.0, "GHz")

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [("J",),("L", "J"), ("C",)]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        vals (dict of dicts): Dictionary with entries for each circuit
                                element. Shows default values

    Returns:
        dict: (edge, elem) -> (val, unit)
    """
    param_dict = {}
    for elems, edge in zip(circuit, edges):
        for elem in elems:
            key = (edge, elem)
            val = vals[elem]['default_value']
            if rand_amp > 0 and val > 0:
                val = max(min_val, val*np.random.normal(1, rand_amp))
            param_dict[key] = (val, vals[elem]['default_unit'])

            # Junction capacitance
            key = (edge, "CJ")
            if elem == "J" and "CJ" in vals:
                val = vals["CJ"]['default_value']
                if rand_amp > 0 and val > 0:
                    max(min_val, val*np.random.normal(1, rand_amp))
                param_dict[key] = (val, vals["CJ"]['default_unit'])

    return param_dict


def convert_circuit_to_graph(circuit: list, edges: list, **kwargs):
    """
    Encodes a circuit as a simple, undirected nx graph with labels
    on the edges for the circuit element, unit, and value

    Args:
        circuit (list of str): a list of elements for the desired circuit
                               (i.e., [[['C'],['C'],['L'],['C','J']])
        edges (list of tuples of ints): a list of edge connections for the
                                        desired circuit
                                        (i.e., [(0,1),(1,2),(2,3),(3,0)])
        params (dict): dictionary with entries C, L, J, CJ,
                    which represent the paramaters for the circuit elements.
                    Additionally entries of C_units, L_units, J_units,
                    and CJ_units. Inputting nothing uses the default parameter
                    values/units from utils.ELEM_DICT.

    """

    params = kwargs.get("params", gen_param_dict(circuit, edges, ELEM_DICT))

    circuit_graph = nx.MultiGraph()
    for elems, edge in zip(circuit, edges):
        for elem in elems:
            value, unit = params[(edge, elem)]
            circuit_graph.add_edge(edge[0], edge[1], element=elem,
                                   unit=unit,
                                   value=value)
            # Junction capacitance
            if elem == "J":
                if (edge, "CJ") in params:
                    value, unit = params[(edge, "CJ")]
                    if value > 0:
                        circuit_graph.add_edge(edge[0], edge[1],
                                               element="CJ",
                                               unit=unit,
                                               value=value)
    return circuit_graph


def circuit_degree(circuit: list, edges: list):
    """
    Counts the number of elements connected to each node

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]

    Returns:
       list of how many elements are connected to each node
       e.g. [3, 2, 3]
    """
    node_repr = circuit_node_representation(circuit, edges)
    return list(sum([np.array(x) for x in node_repr.values()]))


def jj_present(circuit: list):
    """
    Simple function that returns true if there
    is at least one JJ in the circuit and false if there isn't

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
    """

    for edge in circuit:
        for device in edge:
            if device == "J":
                return True
    return False


def qps_present(circuit: list):
    """
    Simple function that returns true if there
    is at least one qps in the circuit and false if there isn't

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["Q"]]
    """

    for edge in circuit:
        for device in edge:
            if device == "Q":
                return True
    return False


def circuit_node_representation(circuit: list, edges: list):
    """
    Converts a circuit into its "node representation"
    that shows how many of each component are connected
    to each node.

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]

    Returns:
        dictionary that maps component label to how many are
        connected to each node: e.g. {'J': [0,0,1,2,0]}
    """

    # Extract number of nodes from edge list
    n_nodes = get_num_nodes(edges)

    # Dictionary that maps component to a list that says
    # how many of that component connect to a given node
    # i.e. 'J': [0,0,1,2,0]
    component_counts = {}
    for comp in list_single_elems():
        component_counts[comp] = [0] * n_nodes

    # Go through each component code in the circuit
    # and loop through the circuit element that it entails
    # and add counts to the appropriate nodes
    for components, edge in zip(circuit, edges):
        for comp in components:
            component_counts[comp][edge[0]] += 1
            component_counts[comp][edge[1]] += 1

    return component_counts


def list_single_elems():
    """
    Simple function to list all single characters in CHAR_TO_COMBINATION

    Returns:
        list[str]: list of characters
    """
    return list(np.unique(np.concatenate(list(ENUM_PARAMS["CHAR_TO_COMBINATION"].values()))))


def get_num_nodes(edges: list):
    """
    Simple function that returns the number of unique nodes
    in edges
    """
    return np.unique(np.concatenate(edges)).size


def swap_nodes(edges: list, na: int, nb: int):
    """
    Swaps all instances of node na with nb and vice versa

    Args:
        edges (list): A list of edge connections for the desired circuit
                       e.g. [(0,1), (0,2), (1,2)]
        na (int): The first node to swap
        nb (int): The second node to swap

    Returns:
        list: A new list of edges with the nodes swapped
    """
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


def renumber_nodes(edges: list, return_map=False):
    """
    Renumbers nodes so that there is a continuous range
    of integers between 0 and the max number

    Args:
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]

    Returns:
        new version of edges with nodes relabeled so that the max
        number present is equal to the number of nodes + 1
    """
    new_edges = edges[:]
    nodes = np.unique(np.concatenate(new_edges))
    relabel_map = {}
    if nodes[-1] != nodes.shape[0]-1:
        for i in range(len(nodes)):
            relabel_map[nodes[i]] = i
        for i in range(len(new_edges)):
            edge = new_edges[i]
            new_edges[i] = tuple([relabel_map[x] for x in edge])
    else:
        relabel_map = {i: i for i in range(len(nodes))}

    if return_map:
        return new_edges, relabel_map
    else:
        return new_edges


def combine_redundant_edges(circuit: list, edges: list):
    """
    Combines edges that are between the same two nodes

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]

    Returns:
        New version of circuit/edges with any redundant edges combined.
        If multiple edges have the same element, then a single

    """
    edge_dict = {}
    for i in range(len(edges)):
        edge = tuple(sorted(edges[i]))
        comps = circuit[i]
        if edge in edge_dict:
            edge_dict[edge] = edge_dict[edge] + comps
        else:
            edge_dict[edge] = comps
    new_edges = list(edge_dict.keys())
    new_circuit = [tuple(sorted(set(edge_dict[x]))) for x in new_edges]

    return new_circuit, new_edges


def circuit_in_set(circuit: list, c_set: list):
    """Helper function to see if a particular circuit
    (list/tuple of tuples) is in a set of circuits
    (list of list/tuple of tuples)

    Args:
        cir (list): a list of element labels for the desired circuit
                        e.g. [("J"),("L", "J"), ("C")]
        c_set (list of lists): list of cir-like elements

    Returns:
        True if cir is present in c_set, False if it isn't
    """
    for c2 in c_set:
        if len(circuit) == len(c2):
            if all(circuit[i] == c2[i] for i in range(len(circuit))):
                return True
    return False


###############################################################################
# I/O Functions for Circuit Database
###############################################################################


def write_df(file: str, df: pd.DataFrame, n_nodes: int, overwrite=False, table_name: str = None):
    """
    Writes the given dataframe to a database file. Appends it if the
    table is already there.

    Args:
        file (str, optional): Database file to write to.
        df (pd.Dataframe): dataframe that represents the circuit entries
        n_nodes (int, optional): number of nodes in the circuit. Defaults to 7.
        overwrite (bool, optional): overwrite the table or
                                    append to it if it exists

    Returns:
        None, writes the dataframe to the database

    """

    to_write = df.copy()

    # drop list columns to save circuit back into saving format
    del to_write['edges']

    # Rename circuit encoding column
    to_write['circuit'] = to_write['circuit_encoding']
    del to_write['circuit_encoding']
    if_exists = "append"
    if overwrite:
        if_exists = "replace"

    with sqlite3.connect(file) as con:
        if table_name is None:
            table_name = 'CIRCUITS_' + str(n_nodes) + '_NODES'
        to_write.to_sql(table_name, con, if_exists=if_exists, index=False)
        con.commit()


def update_db_from_df(file: str, df: pd.DataFrame,
                      to_update: list,
                      str_cols: list = [],
                      float_cols: list = [],
                      uids: list = [], parallel_safe: bool = True,
                      table_name: str = None):
    """
    Updates the given columns listed in to_update
    for entries within df.

    Assumes all columns not listed as str cols or float cols are integers.

    Args:
        file (str, optional): Database file to write to.
        df (pd.Dataframe): dataframe that represents the circuit entries
        to_update (list): columns to update
        str_cols (list): columns that are string valued
        float_cols (list): columns that are float valued
        uids (list): list of individual uids to update
        parallel_safe (bool): if True, uses a method that is safe
                              for parallel writing to the database

    Returns:
        None, writes the dataframe info to the database

    """
    if len(uids) == 0:
        uids = list(df.unique_key.values)




    # Group updates by table (n_nodes)
    updates_by_table = {}
    
    for uid in uids:
        row = df.loc[uid]
        n_nodes = row['n_nodes']
        
        if n_nodes not in updates_by_table:
            updates_by_table[n_nodes] = []
        
        values = []
        for col in to_update:
            val = row[col]
            if col not in str_cols and col not in float_cols:
                values.append(int(val))
            elif col in str_cols:
                values.append(str(val).replace("'", ""))
            else:
                values.append(float(val))
        values.append(row['unique_key'])  # WHERE clause value
        updates_by_table[n_nodes].append(tuple(values))

    with sqlite3.connect(file, timeout=5000) as con:
        cur = con.cursor()
        
        cur.execute("PRAGMA busy_timeout = 30000")
        if parallel_safe:
            cur.execute("PRAGMA synchronous = NORMAL")
        else:
            cur.execute("PRAGMA cache_size = -64000")
         
        for n_nodes, batch_values in updates_by_table.items():
            set_clause = ", ".join(f"{col} = ?" for col in to_update)
            if table_name is None:
                table_name = 'CIRCUITS_' + str(n_nodes) + '_NODES'
            sql = f"UPDATE {table_name} SET {set_clause} WHERE unique_key = ?"
            
            written = False
            while not written:
                try:
                    cur.executemany(sql, batch_values)
                    written = True
                except sqlite3.OperationalError as exc:
                    sleep(np.abs(np.random.random()))
        
        con.commit()


def delete_circuit_data(file: str, n_nodes: int, indices: Union[list, str]):
    """
    Deletes the specified graphs (num nodes/indices) from the database file

    Args:
        file (str, optional): path to the databse file.
        n_nodes (int): number of nodes for the graph
        indices (list or str): unique key (or list of keys) of the graph(s)
                                to be deleted

    Returns:
        None, just modifies the database
    """

    # Convert individual entry for batch use
    if isinstance(indices, str):
        indices = [indices]

    connection_obj = sqlite3.connect(file)
    cursor_obj = connection_obj.cursor()
    table_name = 'CIRCUITS_' + str(n_nodes) + '_NODES'
    for index in indices:
        cursor_obj.execute('''DELETE FROM {table} WHERE unique_key = '{index}';
                       '''.format(table=table_name, index=str(index)))
    connection_obj.commit()
    connection_obj.close()

    return


def get_circuit_data(file: str, unique_key: str, char_mapping: dict = None):
    """ gets circuit data from database

    Args:
        n_nodes (int): The number of nodes in the circuit
        unique_key (str): Unique Idenitifier of the circuit
        file (str): path to the database to get circuit from
        char_mapping (dict, optional): mapping from character to list of
                                       circuit elements

    Returns:
        circuit (list) : a list of element labels for the desired circuit
                         (i.e., ['0','2','5','1'])
        edges (list of tuples of ints): a list of edge connections for the
                                        desired circuit
                                        (i.e., [(0,1),(1,2),(2,3),(3,0)])
    """
    if char_mapping is None:
        char_mapping = ENUM_PARAMS["CHAR_TO_COMBINATION"]
    # Parse uid to get number of nodes
    n_nodes = unique_key[unique_key.find("n") + 1:unique_key.find("_")]

    # Fetch entry from database
    connection_obj = sqlite3.connect(file, uri=True)
    cursor_obj = connection_obj.cursor()
    table_name = 'CIRCUITS_' + str(n_nodes) + '_NODES'
    query_str = f"SELECT * FROM {table_name} WHERE unique_key = '{unique_key}'"
    cursor_obj.execute(query_str)
    output = cursor_obj.fetchone()
    columns = [x[0] for x in cursor_obj.description]
    connection_obj.commit()
    connection_obj.close()

    if output is None:
        raise ValueError(f"No circuit found for unique_key '{unique_key}'")

    row = dict(zip(columns, output))

    # Map the edges and circuit component info
    if 'basegraph_g6' in row and row['basegraph_g6'] not in [None, ""]:
        edges = _edges_from_graph6(str(row['basegraph_g6']))
    else:
        edges = graph_index_to_edges(int(row['graph_index']), n_nodes)
    circuit = encoding_to_components(row['circuit'], char_mapping=char_mapping)

    return circuit, edges


def get_circuit_data_batch(db_file: str, n_nodes: int,
                           char_mapping: dict = None,
                           filter_str: str = '',
                           unique_keys:list[str] = [],
                           table_name: str = '') -> pd.DataFrame:
    """
    Retrieve all circuits from the database for a specified number of nodes, 
    with optional filtering criteria.

    Parameters
    ----------
    db_file : str, optional
        Path to the SQLite database file. Defaults to ``"circuits.db"``.
    n_nodes : int
        Number of nodes in the circuit.
    char_mapping : dict, optional
        A mapping from characters to lists of circuit elements.
    filters : str, optional
        SQL filter statement for refining the query.  
        Example: ``"WHERE circuit_index = 100"``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row represents a circuit matching the query.
        If present in the database table, the ``basegraph_g6`` column is
        preserved and used to reconstruct edges during loading.
    """

    if char_mapping is None:
        char_mapping = ENUM_PARAMS["CHAR_TO_COMBINATION"]
    if table_name == '':
        table_name = 'CIRCUITS_' + str(n_nodes) + '_NODES'
    with sqlite3.connect(db_file, timeout=5000) as con:
        if filter_str != '' and unique_keys != []:
            raise ValueError("Provide either filter string or list of keys")
        elif filter_str == '' and unique_keys == []:
            query = f"SELECT * FROM {table_name}"
        elif filter_str != "":
            query = f"SELECT * FROM {table_name} {filter_str}"
        else:
            unique_keys = [f" '{k}'" for k in unique_keys]
            query = f"SELECT * FROM {table_name} WHERE unique_key in ({','.join(unique_keys)})"
        df = pd.read_sql_query(query, con)

    convert_loaded_df(df, n_nodes, char_mapping)

    # Make a useful index if it's there
    if 'unique_key' in df.columns:
        df.index = df['unique_key']

    # Convert int to bool columns
    int_to_bool = ["no_series", "filter", "in_non_iso_set"]
    for col in int_to_bool:
        df[col] = df[col].astype(bool)

    return df


def get_unique_qubits(db_file: str, n_nodes: str):
    """
    Loads all entries corresponding to unique qubits
    from the specified file for the specified number
    of nodes

    Args:
        db_file (str): sqlite db_file to look in.
                                Defaults to "circuits.db"
        n_nodes (int): number of nodes in the circuit

    Returns:
        pd.DataFrame: set of unique qubit circuits
    """
    filter_str = "WHERE in_non_iso_set = 1 AND "
    filter_str += "filter = 1 AND no_series = 1"
    return get_circuit_data_batch(db_file, n_nodes, filter_str=filter_str)


def get_equiv_circuits_uid(db_file: str, unique_key: str) -> pd.DataFrame:
    """
    Finds all circuits in the database that match a given unique key  
    or are equivalent to it.

    This function searches for circuits in an SQLite database that either:
    - Have the specified `unique_key`, or  
    - Are considered equivalent circuits by component-like isomorphism.

    Parameters
    ----------
    db_file : str
        Path to the SQLite database file.  
        Defaults to `"circuits.db"`.
    unique_key : str
        The unique identifier for the circuit.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing all matching circuits.
    """
    tables = list_all_tables(db_file)
    entries = []
    filt_str = f"WHERE equiv_circuit LIKE '{unique_key}'\
                 OR unique_key LIKE '{unique_key}'"
    for tbl in tables:
        n_nodes = int([n for n in tbl if n.isdigit()][0])
        entries.append(get_circuit_data_batch(db_file, n_nodes,
                                              filter_str=filt_str))

    return pd.concat(entries).sort_values(by="equiv_circuit")


def get_equiv_circuits(db_file: str, circuit: list, edges: list) -> Union[pd.DataFrame, None]:
    """
    Finds all circuits in the database that are equivalent to a given circuit.  

    Parameters
    ----------
    db_file : str
        Path to the SQLite database file.  
        Defaults to `"circuits.db"`.
    circuit : list
        A list of element labels defining the desired circuit.  
        Example: `[["J"], ["L", "J"], ["C"]]`
    edges : list
        A list of edge connections defining how circuit elements are connected.  
        Example: `[(0,1), (0,2), (1,2)]`

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame containing all equivalent circuits found in the database.  
        Returns `None` if no equivalent circuits are found.
    """

    entry = find_circuit_in_db(db_file, circuit, edges)
    if entry.shape[0] > 1:
        raise ValueError("Getting too many circuits")
    elif entry.empty:
        return None
    else:
        entry = entry.iloc[0]

    if entry["in_non_iso_set"]:
        uid = entry["unique_key"]
    elif entry["equiv_circuit"] != "not found":
        uid = entry["equiv_circuit"]
    else:
        return [entry]

    return get_equiv_circuits_uid(db_file, uid)


def find_circuit_in_db(db_file: str, circuit: list, edges: list):
    """
    Finds the database entry for a given circuit/edges combination

    Args:
        db_file (str): sqlite db_file to look in.
                                Defaults to "circuits.db"
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
    """

    n_nodes = get_num_nodes(edges)
    graph_index, mapping = edges_to_graph_index(edges, return_mapping=True)
    # Re-order circuit to match basegraph edges
    new_edge_order = graph_index_to_edges(graph_index, n_nodes)
    circuit_in_order = [None]*len(circuit)
    for i in range(len(edges)):
        n0, n1 = edges[i]
        if (mapping[n0], mapping[n1]) in new_edge_order:
            new_i = new_edge_order.index((mapping[n0], mapping[n1]))
        else:
            new_i = new_edge_order.index((mapping[n1], mapping[n0]))            
        circuit_in_order[new_i] = circuit[i]
    encoding = components_to_encoding(circuit_in_order)
    filters = f"WHERE circuit LIKE '{encoding}' AND\
                graph_index = '{graph_index}'"
    
    return get_circuit_data_batch(db_file, n_nodes, filter_str=filters)



def write_circuit(cursor_obj, c_dict: dict, to_commit: bool = False):
    """Appends an individual circuit to a database

    Args:
        cursor_obj: sqllite cursor object pointing to the desired database
        c_dict: dictionary that represents a circuit entry
        to_commit: commit the database (i.e., save changes)
    """
    table = f"CIRCUITS_{c_dict['n_nodes']}_NODES"
    sql_fields = ["circuit", "graph_index", "edge_counts",
                  "unique_key", "n_nodes", "base",
                  "no_series", "filter", "in_non_iso_set",
                  "equiv_circuit"]
    if "basegraph_g6" in c_dict:
        sql_fields.append("basegraph_g6")
    columns = ", ".join(sql_fields)
    placeholders = ", ".join(["?"]*len(sql_fields))
    values = [c_dict[field] for field in sql_fields]
    cursor_obj.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
        values,
    )

    if to_commit:
        cursor_obj.connection.commit()


def list_all_tables(db_file: str):
    """
    Lists all the tables in the database file

    Args:
        db_file (str): file to examine
    """
    with sqlite3.connect(db_file, uri=True) as connection_obj:
        cursor_obj = connection_obj.cursor()
        tables = cursor_obj.execute("SELECT name FROM sqlite_master\
                                WHERE type='table'").fetchall()
    return [x[0] for x in tables]


def list_all_columns(db_file: str, table_name: str):
    """
    Lists all the tables in the database file

    Args:
        db_file (str): file to examine
        table_name (str): table to get 
    """
    with sqlite3.connect(db_file, uri=True) as connection_obj:
        cursor_obj = connection_obj.cursor()
        info = cursor_obj.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        cols = [x[1] for x in info]

    return cols


def run_with_timeout(func, args=(), kwargs=None, timeout=1):
    """
    Runs the specified function with a timeout

    Args:
        func (function): function to run
        args (tuple, optional): arguments to the function. Defaults to ().
        kwargs (_type_, optional): kwarguments to the function. Defaults to None.
        timeout (int, optional): timeout in minutes. Defaults to 1.
    Raises:
        KI: KeyboardInterrupt if interrupted by user
    Returns:
        None if didn't return, else the function output
    """
    if kwargs is None:
        kwargs = {}
    try:
        return func_timeout(60*timeout, func, args, kwargs)
    except FunctionTimedOut:
        return None
    except KeyboardInterrupt as KI:
        raise KI


def set_enum_params(char_to_combo = {'0': ('C',),
                                     '1': ('J',),
                                     '2': ('L',),
                                     '3': ('C', 'L'),
                                     '4': ('J', 'L'),
                                     '5': ('C', 'J'),
                                     '6': ('C', 'J', 'L')},
                    filter = None):
    if filter is None:
        filter = jj_present
    ENUM_PARAMS["filter"] = filter
    ENUM_PARAMS["CHAR_TO_COMBINATION"] = char_to_combo
    ENUM_PARAMS["COMBINATION_TO_CHAR"] = {}
    ENUM_PARAMS["EDGE_COLOR_DICT"] = {}
    ENUM_PARAMS["CHAR_LIST"] = []
    for i, c in enumerate(ENUM_PARAMS["CHAR_TO_COMBINATION"].keys()):
        ENUM_PARAMS["CHAR_LIST"].append(c)
        ENUM_PARAMS["COMBINATION_TO_CHAR"][ENUM_PARAMS["CHAR_TO_COMBINATION"][c]] = c
        ENUM_PARAMS["EDGE_COLOR_DICT"][ENUM_PARAMS["CHAR_TO_COMBINATION"][c]] = i


# set parameters
if len(ENUM_PARAMS) == 0:
    set_enum_params()