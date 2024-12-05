import sqlite3
import itertools
import functools

from typing import Union
import numpy as np
from pathlib import Path
import networkx as nx
import pandas as pd

from tqdm import tqdm
from time import sleep

import sympy as sym
from sympy import collect, expand_mul, Mul, Dummy
from sympy.core.add import Add
from sympy.core.symbol import Symbol

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


def edges_to_graph_index(edges: list, return_mapping: bool = False):
    """
    Matches a set of edges to a basegraph that's isomorphic to it

    Args:
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        return_mapping (bool): return the mapping from edges to the basegraph
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

    raise ValueError("Error: No Isomorphic Graph Found")


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
    # Get the edges
    df['edges'] = [graph_index_to_edges(int(i), n_nodes)
                   for i in df.graph_index.values]
    df['circuit_encoding'] = df.circuit.values.copy()
    df['circuit'] = [encoding_to_components(c, char_mapping=char_mapping)
                     for c in df.circuit.values]


def get_basegraphs(n_nodes: int):
    """
    Loads the base graphs for a specific number of nodes

    Args:
        n_nodes (int): number of nodes in the graph
    """
    if int(n_nodes) > 6:
        raise ValueError("Only basegraphs up to 6 nodes are included. See https://users.cecs.anu.edu.au/~bdm/data/graphs.html for larger sets of graphs.")
    # Load it if it hasn't been loaded
    if str(n_nodes) not in LOADED_BASEGRAPHS:
        f = Path(DOWNLOAD_PATH, 'sircuitenum', 'graphs', f"graph{n_nodes}c.g6")
        all_graphs = nx.read_graph6(f)
        # Fix two vertex case so it always returns a list
        if n_nodes == 2:
            all_graphs = [all_graphs]
        LOADED_BASEGRAPHS[str(n_nodes)] = all_graphs

    # Return if it has already been loaded
    return LOADED_BASEGRAPHS[str(n_nodes)]


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
        for elem in elems:
            counts[elem] += 1

    return counts


def add_elem_number(circuit: list, **kwargs):
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
    circuit_new = []

    counts = {}
    for elem in possible_elems:
        counts[elem] = 0

    for elems in circuit:
        elems_new = []
        for elem in elems:
            counts[elem] += 1
            elems_new.append(elem+"_"+str(counts[elem]))
        circuit_new.append(tuple(elems_new))

    return circuit_new


def circuit_entry_dict(circuit: list, graph_index: int, n_nodes: int,
                       circuit_num: int, base: int):
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
       e.g. [1, 2, 1]
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


def renumber_nodes(edges: list):
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
    if nodes[-1] != nodes.shape[0]-1:
        relabel_map = {}
        for i in range(len(nodes)):
            relabel_map[nodes[i]] = i
        for i in range(len(new_edges)):
            edge = new_edges[i]
            new_edges[i] = tuple([relabel_map[x] for x in edge])

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


def write_df(file: str, df: pd.DataFrame, n_nodes: int, overwrite=False):
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
        to_write.to_sql(f"CIRCUITS_{n_nodes}_NODES",
                        con, if_exists=if_exists, index=False)


def update_db_from_df(file: str, df: pd.DataFrame,
                      to_update: list,
                      str_cols: list = [],
                      float_cols: list = []):
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

    Returns:
        None, writes the dataframe info to the database

    """

    n_fields = len(to_update)

    with sqlite3.connect(file, timeout=5000) as con:
        cur = con.cursor()
        # sql_str = ""
        for _, row in df.iterrows():
            n_nodes = row['n_nodes']
            sql_str = f"UPDATE CIRCUITS_{n_nodes}_NODES SET "
            for i, col in enumerate(to_update):
                val = row[col]
                if col not in str_cols and col not in float_cols:
                    val = int(val)
                elif col in str_cols:
                    val = str(val).replace("'", "")
                else:
                    val = float(val)
                if i < n_fields - 1:
                    sql_str += f"{col} = '{val}', "
                else:
                    sql_str += f"{col} = '{val}' "
                    sql_str += f"WHERE unique_key = '{row['unique_key']}';\n"
        
            written = False
            while not written:
                try:
                    cur.executescript(sql_str)
                    written = True
                except sqlite3.OperationalError:
                    # Database is locked, wait random amount
                    # of time and try again
                    print("Write Conflict")
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
    connection_obj.commit()
    connection_obj.close()

    # Map the edges and circuit component info
    edges = graph_index_to_edges(int(output[1]), n_nodes)
    circuit = encoding_to_components(output[0], char_mapping=char_mapping)

    return circuit, edges


def get_circuit_data_batch(db_file: str, n_nodes: int,
                           char_mapping: dict = None,
                           filter_str: str = ''):
    """
    Returns all the circuits present in the database for the specified
    number of nodes, and any other filter statements given.

    Args:
        db_file (str, optional): sqlite db_file to look in.
                                Defaults to "circuits.db"
        n_nodes (int): number of nodes in the circuit
        char_mapping (dict, optional): mapping from character to
                                       list of circuit elements
        filters (str, optional): SQL filter statement
                                (i.e. WHERE circuit_index = 100).

    Returns:
        pandas dataframe containing each circuit as a row
    """
    if char_mapping is None:
        char_mapping = ENUM_PARAMS["CHAR_TO_COMBINATION"]
    table_name = 'CIRCUITS_' + str(n_nodes) + '_NODES'
    connection_obj = sqlite3.connect(db_file, timeout=5000)
    query = "SELECT * FROM {table} {filter_str}".format(
        table=table_name, filter_str=filter_str)

    df = pd.read_sql_query(query, connection_obj)

    connection_obj.commit()
    connection_obj.close()

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


def get_equiv_circuits_uid(db_file: str, unique_key: str):
    """
    Finds all circuits in the database with either the
    given unique key, or with it as the equiv circuit

    Args:
        db_file (str): sqlite db_file to look in.
                                Defaults to "circuits.db"
        unique_key (str): unique identifier for the circuit
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


def get_equiv_circuits(db_file: str, circuit: list, edges: list):
    """
    Finds all circuits equivalent to the one provided
    that are present in the database.
    Returns None if none are found.

    Args:
        db_file (str): sqlite db_file to look in.
                                Defaults to "circuits.db"
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
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
    sql_str = f"INSERT INTO {table} VALUES ("
    sql_fields = ["circuit", "graph_index", "edge_counts",
                  "unique_key", "n_nodes", "base",
                  "no_series", "filter", "in_non_iso_set",
                  "equiv_circuit"]
    n_fields = len(sql_fields)
    for i, field in enumerate(sql_fields):
        if i < n_fields - 1:
            sql_str += f"'{c_dict[field]}', "
        else:
            sql_str += f"'{c_dict[field]}')"
    cursor_obj.execute(sql_str)

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

def collect_H_terms(H: Add, zero_ext: bool = True, 
                    periodic_charge="n", periodic_phase="θ",
                    extended_charge="q", extended_phase="φ",
                    ext_charge: str = "ng", ext_flux: str = "_{ext}",
                    no_coeff: bool = False, collect_phase: bool = True) -> Add:
    """
    Groups terms in the Hamiltonian.

    Default settings are for our operator convention --
    not Scqubits (q -> Q and \varphi -> \theta).

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
        indices = np.unique([str(x)[-1] for x in terms])
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

    H = collect(H, list(combosQ.keys()) + combos, func=sym.ratsimp)
    if collect_phase:
        H = collect(H, combos_trig)

    if no_coeff:
        H = remove_coeff_(H, list(combosQ.keys()) + combos + combos_trig)

    return H, combos+combos_trig, combosQ


def remove_coeff_(H, all_combos):
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


def zero_start_edges(edges):
    """
    Helper function to convert a list of edges from 1
    indexing to 0 indexing of the nodes

    Args:
        edges (list of tuples of ints): a list of edge connections for the
                                        desired circuit
                                        (i.e., [(0,1),(1,2),(2,3),(3,0)])

    Returns:
         list[tuple[int]]: edges modified to have zero as the lowest
                           index
    """
    min_node = min([min(edge) for edge in edges])
    if min_node > 0:
        edges = [(edge[0] - min_node, edge[1] - min_node) for edge in edges]
    return edges


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