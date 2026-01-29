__doc__ = "enumeration.py: Contains the core functionality for enumerating circuits"
__author__ = "Eli Weissler, Mohit Bhat"
__version__ = "0.1.0"
__all__ = ["generate_all_circuits", "generate_graphs_node", "trim_graph_node", "gen_hamiltonian", "find_equiv_cir_series", "find_unique_ground_placements", "num_possible_circuits"]

import sqlite3
import itertools
import functools
import traceback
import contextlib
from pathlib import Path
from typing import Union
from multiprocessing import Pool
from multiprocessing import set_start_method
try:
    set_start_method("fork")
except:
    print("Multiprocessing fork not available on your system.\
           More than one worker is not supported for enumeration \
           with custom elements.")

import sympy as sym
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from sircuitenum import utils
from sircuitenum import reduction as red
from sircuitenum import qpackage_interface as pi
from sircuitenum import quantize
from sircuitenum.singular_interface import initialize_singular

# -------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------
def num_possible_circuits(base: int, n_nodes: int, quiet: bool = True) -> int:
    """
    Estimate the number of possible circuits for a given number of edges and vertices.

    This function calculates the number of possible circuits for a graph with `n_nodes` vertices 
    and `base` possible edge types. The estimate may be an overestimation.

    Parameters
    ----------
    base : int, optional
        The number of possible edge types. Defaults to ``7``, corresponding to:
        ``J, C, L, JL, CL, JC, JCL``.
    n_nodes : int
        The number of vertices (nodes) in the graph.
    quiet : bool, optional
        If ``False``, prints the estimated number of circuits. Defaults to ``True``.

    Returns
    -------
    int
        The estimated number of possible circuits.
    """
    all_graphs = utils.get_basegraphs(n_nodes)
    n_circuits = 0
    for graph in all_graphs:
        n_circuits += base**len(graph.edges)
    if not quiet:
        print("With " + str(base) + " elements and " + str(n_nodes) +
              " nodes there are " + str(n_circuits) + " possible circuits")
    return n_circuits


def generate_for_specific_graph(base: int, graph: nx.Graph,
                                graph_index: int,
                                cursor_obj=None,
                                return_vals: bool = False):
    """Generates all circuits derived from a given graph

    Args:
        base (int): The number of possible edges. By default this is 7:
                        (i.e., J, C, I, JI, CI, JC, JCI)
        graph (nx Graph) : base graph to generate circuits for
        graph index (int): the index of the graph for the written
                           circuit within the file for the number of nodes
        n_nodes (int): Number of nodes in circuit
        cursor_obj: sqllite cursor object pointing to the desired database.
        return_vals (bool): return the circuits as a dataframe
    """

    n_nodes = len(graph.nodes)

    if cursor_obj is None and return_vals is False:
        raise ValueError("Graphs are generating but neither \
                          being returned nor saved")

    edges = graph.edges
    n_edges = len(edges)
    if return_vals:
        data = []
    
    num_configs = base**n_edges
    for i, circuit in enumerate(itertools.product(utils.ENUM_PARAMS["CHAR_LIST"][:base], repeat=n_edges)):
        c_dict = utils.circuit_entry_dict(circuit, graph_index, n_nodes, i, base)
        # Commit for the last one in the set
        if cursor_obj is not None:
            utils.write_circuit(cursor_obj, c_dict,
                                to_commit=i == (num_configs-1))
        if return_vals:
            data.append(c_dict)

    if return_vals:
        return pd.DataFrame(data)


def delete_table(db_file: str, n_nodes: int):
    """Deletes table in sql database

    Args:
        n_nodes (int): Number of nodes for table
        db_file (str): sql database to delete table from
    """
    connection_obj = sqlite3.connect(db_file)
    cursor_obj = connection_obj.cursor()
    table_name = 'CIRCUITS_' + str(n_nodes) + '_NODES'
    cursor_obj.execute("DROP TABLE IF EXISTS {table}".format(table=table_name))
    connection_obj.commit()
    connection_obj.close()
    return


def find_unique_ground_placements(circuit: list, edges: list) -> tuple[int]:
    """
    Uses component graph isomorphism to determine the unique
    ground node placements for a given circuit.

    Parameters
    ----------
    circuit : list of list of str
        A list representing the elements of the desired circuit.  
        Example: ``[["J"], ["L", "J"], ["C"]]``.
    edges : list of tuple of int
        A list of edge connections for the desired circuit.  
        Example: ``[(0,1), (0,2), (1,2)]``.

    Returns
    -------
    tuple of int
        A tuple containing integers representing the unique ground node placements
        for the given circuit.
    """
    unique_nodes = []
    unique_graphs = []
    for gnd in range(utils.get_num_nodes(edges)):
        test = red.convert_circuit_to_component_graph(circuit, edges, ground_nodes=[gnd])
        isomorphic_in_set = False
        for ref in unique_graphs:
            if nx.is_isomorphic(test, ref, node_match=red.colors_match):
                isomorphic_in_set = True
                break
        if not isomorphic_in_set:
            unique_graphs.append(test)
            unique_nodes.append(gnd)
    return tuple(unique_nodes)


def expand_ground_node(df: pd.DataFrame):
    """
    Create new entries in the dataframe
    for unique placements of ground
    nodes

    Args:
        df (pd.DataFrame): circuit dataframe

    Returns:
        pd.DataFrame: dataframe with an entry for each ground
                      node placement.
    """
    new_df = []
    df["ground_node"] = -1
    for i in tqdm(range(df.shape[0])):
        row = df.iloc[[i]].copy()
        circuit, edges = row["circuit"].iloc[0], row["edges"].iloc[0]
        for gnd in find_unique_ground_placements(circuit, edges):
            new_row = row.copy()
            new_row["ground_node"] = gnd
            new_df.append(new_row)
    return pd.concat(new_df)


def has_dangling_edges(circuit: list, edges: list):
    """
    Determines whether a circuit has a dangling edge, i.e.
    a single branch through which current cannot flow

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
    Returns:
        True if there is a dangling edge, false if not
    """

    deg = utils.circuit_degree(circuit, edges)
    if all(d > 1 for d in deg):
        return False
    else:
        return True


def remove_dangling_edges(df: pd.DataFrame):
    """
    Removes edges that cannot have current flowing
    through them after placing a ground node

    Args:
        df (pd.DataFrame): circuit dataframe

    Returns:
        pd.DataFrame: dataframe with all circuits that
                      have dangling edges removed.
    """
    ind_to_keep = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        if not has_dangling_edges(row["circuit"], row["edges"]):
            ind_to_keep.append(i)
    return df.iloc[ind_to_keep].copy()


def find_equiv_cir_series(db_file: str, circuit: list, edges: list) -> str:
    """
    Searches the database for circuits that are equivalent to the given one,
    up to a reduction of series linear circuit elements.

    Parameters
    ----------
    db_file : str
        Path to the SQLite database file that has been preprocessed for the given number of nodes.
    circuit : list of list of str
        A list representing the circuit elements.  
        Example: ``[["J"], ["L", "J"], ["C"]]``.
    edges : list of tuple of int
        A list of edge connections that define the circuit's connectivity.  
        Example: ``[(0,1), (0,2), (1,2)]``.

    Returns
    -------
    str
        The unique key of the equivalent circuit found in the non-isomorphic set.  
        Returns "" if no equivalent circuit is found.
    """

    # What does it look like with series elems removed
    c2, e2 = red.linear_star_mesh(circuit, edges)
    equiv = utils.find_circuit_in_db(db_file, c2, e2)
    if equiv.empty:
        return ""
    # Return the equivalent circuit
    if equiv.iloc[0]['equiv_circuit'] == "":
        return equiv.iloc[0]['unique_key']
    else:
        return equiv.iloc[0]['equiv_circuit']


def generate_graphs_node(db_file: str, n_nodes: int,
                         base: int, return_vals: bool = False) -> Union[pd.DataFrame, None]:
    """
    Generate circuits for all graphs with a given number of nodes and store them in an SQL database.

    This function generates circuits for all possible graphs with `n_nodes` nodes and 
    stores them in a table within the specified SQL database. The table is labeled as 
    ``CIRCUITS_<n_nodes>_NODES``.

    Parameters
    ----------
    n_nodes : int
        The number of nodes for which circuits will be generated and stored.
    base : int, optional
        The number of possible edge types. Defaults to ``7``, corresponding to:
        ``J, C, L, JL, CL, JC, JCL``.
    db_file : str
        Path to the SQL database file where the circuits will be stored.
    return_vals : bool, optional
        If ``True``, returns the generated circuits as a Pandas DataFrame. Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame or None
        If `return_vals` is ``True``, returns a DataFrame containing the generated circuits. 
        Otherwise, returns ``None``.
    """


    # Initialize table
    if db_file is not None:
        if Path(db_file).exists():
            delete_table(db_file, n_nodes)
        table_name = 'CIRCUITS_' + str(n_nodes) + '_NODES'
        connection_obj = sqlite3.connect(db_file)
        cursor_obj = connection_obj.cursor()
        sql_str = f"CREATE TABLE {table_name} (circuit, graph_index int, edge_counts, \
            unique_key, n_nodes int, base int, no_series int, \
            filter int, in_non_iso_set int, \
            equiv_circuit, "
        sql_str += "PRIMARY KEY(unique_key))"
        cursor_obj.execute(sql_str)
        connection_obj.commit()
    else:
        cursor_obj = None

    all_graphs = utils.get_basegraphs(n_nodes)
    data = []
    for graph_index, G in tqdm(enumerate(all_graphs), total=len(all_graphs)):
        data.append(generate_for_specific_graph(base, G,
                                                graph_index,
                                                cursor_obj,
                                                return_vals))

    if cursor_obj is not None:
        connection_obj.close()

    if return_vals:
        return pd.concat(data)


def trim_graph_node(db_file: str, n_nodes: int,
                    base: int = None,
                    n_workers: int = 1) -> None:
    """
    Mark circuits in the database based on Josephson junctions, series linear components, 
    and non-isomorphism.

    This function updates the database to indicate whether each circuit contains 
    Josephson junctions (JJs), series linear components, and belongs to a non-isomorphic 
    set of circuits. If a circuit is not in the non-isomorphic set, an equivalent 
    circuit that is in the set is recorded. 

    All three conditions must be met for inclusion in the final set.

    Parameters
    ----------
    db_file : str
        Path to the SQL database file where circuits are stored.
    n_nodes : int
        The number of nodes to consider.
    base : int, optional
        The number of possible edge types. Defaults to ``7``, corresponding to:
        ``J, C, L, JL, CL, JC, JCL``.
    n_workers : int, optional
        The number of workers to use for processing. Defaults to ``1``.
    """
    if base is None:
        base = len(utils.ENUM_PARAMS["CHAR_TO_COMBINATION"])

    # Get the max number of edges
    # from fully connected graph
    all_graphs = utils.get_basegraphs(n_nodes)
    n_edges_in_graph = [len(g.edges) for g in all_graphs]
    n_graphs = len(all_graphs)
    max_edges = max(n_edges_in_graph)

    # Loop through all possible numbers of each component
    # For all unique base graphs and create non-isomorphic
    # Sets within these slices
    print("Trimming graphs with no jj's, linear elements in series",
          "and reducing isomorphic graphs...")
    with sqlite3.connect(db_file) as con:
        cur = con.cursor()
        table_name = 'CIRCUITS_' + str(n_nodes) + '_NODES'
        sql_query = f"SELECT DISTINCT edge_counts FROM {table_name}"
        counts_to_consider = [x[0] for x in cur.execute(sql_query).fetchall()]

    args = []
    for counts_str in counts_to_consider:
        n_edges = sum(int(x) for x in counts_str.split(","))
        for graph_index in range(n_graphs):
            # Skip entries without the right number of edges
            # in edge counts
            if n_edges != n_edges_in_graph[graph_index]:
                continue
            else:
                filter_str = f"WHERE edge_counts = '{counts_str}'\
                               AND graph_index = {graph_index}"
                args.append((filter_str, db_file, n_nodes,
                             utils.ENUM_PARAMS["CHAR_TO_COMBINATION"]))

    # Shuffle to spread out longer cases for more accurate time
    # estimates and better parallel performance
    np.random.shuffle(args)
    if n_workers > 1:
        pool = Pool(processes=n_workers)
        for _ in tqdm(pool.imap_unordered(_reduce_individual_set, args),
                      total=sum(1 for _ in args)):
            pass
    else:
        for arg_set in tqdm(args):
            _reduce_individual_set(arg_set)


def _reduce_individual_set(args: tuple):
    """
    Parallel helper function for calling full reduction
    on groups defined by the specified sql filter string.
    Intended to split by number of each circuit element.

    Args:
        args (tuple): filter_str, db_file, n_nodes, mapping

    Raises:
        ValueError: when an empty df is encountered
    
    Returns:
        None, updates dataframe specified by db_file
    """
    # print(args)

    filter_str = args[0]
    db_file = args[1]
    n_nodes = args[2]
    mapping = args[3]

    df = utils.get_circuit_data_batch(db_file, n_nodes,
                                      char_mapping=mapping,
                                      filter_str=filter_str)
    if df.empty:
        print('-------------------------------')
        print(utils.get_circuit_data_batch(db_file, n_nodes))
        print("Filter String:", filter_str)
        raise ValueError("Empty Dataframe when there shouldn't be")

    # Mark up the set
    red.full_reduction(df)

    # Find equivalent circuits for the series reduced circuits
    equiv_cir = df['equiv_circuit'].values
    yes_series = np.logical_not(df['no_series'].values)
    for i in range(df.shape[0]):
        if yes_series[i]:
            row = df.iloc[i]
            equiv_cir[i] = find_equiv_cir_series(db_file,
                                                 row['circuit'],
                                                 row['edges']
                                                 )

    # Update the table
    to_update = ["no_series", "filter",
                 "in_non_iso_set", "equiv_circuit"]
    str_cols = ["equiv_circuit"]
    utils.update_db_from_df(db_file, df, to_update, str_cols)


def _gen_ham_class_row(args):
    """
    Helper function to generate the Hamiltonian for the given uid
    in the given db file.

    Args:
        uid (str): circuit unique key
        db_file (str): database file

    Raises:
        ValueError: Error with circuit database
        kbi: Keyboard interrupt
    """
    uid, db_file, eq_params, update = args

    # Load the graphs with the specified edges counts and graph index
    filter_str = f"WHERE unique_key LIKE '{uid}'"
    n_nodes = int(uid[1])
    df = utils.get_circuit_data_batch(db_file, n_nodes,
                                      filter_str=filter_str)

    if df.shape[0] > 1:
        breakpoint()
        raise ValueError("Multiple Circuits on Unique Key")
    
    # Choose the transformation
    ## Different Circuit Paramter values
    entry = df.iloc[0]
    if eq_params:
        circuit, edges = entry.circuit, entry.edges
    else:
        circuit, edges = utils.add_elem_number(entry.circuit), entry.edges
    try:
        Z, var_types, h_class = quantize.choose_Z(circuit, edges)
        wJT_key = h_class.split("_")[1].split("-")[-1]
    except TimeoutError as timeout:
        print("[TIMEOUT]")
        print(traceback.format_exc())
        print("circuit =", circuit)
        print("edges =", edges)
        h_class = "UNDEFINED"
        wJT_key = "UNDEFINED"
    except KeyboardInterrupt as kbi:
        raise kbi
    except Exception as exc:
        print("-------------------------------------------")
        print("Unable to Generate Hamiltonian for:", uid)
        print(traceback.format_exc())
        print(exc)
        print("circuit =", circuit)
        print("edges =", edges)
        print("-------------------------------------------")
        h_class = "UNDEFINED"
        wJT_key = "UNDEFINED"


    # Set values    
    if eq_params:
        to_update = ["H_class_sym", "wJT_sym"]
        df.at[uid, "wJT_sym"] = wJT_key
        df.at[uid, "H_class_sym"] = h_class
        str_cols=["H_class_sym","wJT_sym"]
    else:
        to_update = ["n_compact", "n_extended", "n_harmonic",
                "n_free", "n_frozen", "n_sigma", "H_class", "wJT"]
        df.at[uid, "H_class"] = h_class
        df.at[uid, "wJT"] = wJT_key
        df.at[uid, "n_compact"] = len(var_types.get("compact", []))
        df.at[uid, "n_extended"] = len(var_types.get("extended", []))
        df.at[uid, "n_harmonic"] = len(var_types.get("harmonic", []))
        df.at[uid, "n_free"] = len(var_types.get("free", []))
        df.at[uid, "n_frozen"] = len(var_types.get("frozen", []))
        df.at[uid, "n_sigma"] = len(var_types.get("sigma", []))
        str_cols=["H_class","wJT"]

    # Update value in database
    if update:
        utils.update_db_from_df(db_file, df, to_update, str_cols=str_cols)
    return df, to_update, str_cols


def add_hamiltonian_classes(db_file: str, n_nodes: int,
                              n_workers: int = 4, resume: bool = False,
                              eq_params=False, save_every=100):
    """
    Constructs a variable transformation and identifies the hamiltonian
    class for each circuit in the database

    Args:
        db_file (str): database file
        n_nodes (int): number of nodes to add for
        n_workers (int): parallelize the Hamiltonian generation to this many
                         processes.
        resume (bool, optional): whether to resume a previously started run.
                                 this only grabs rows that don't have Hamiltonians
                                 yet.
        eq_params (bool, optional): whether to set circuit parameters to be equal

    Raises:
        ValueError: if multiple circuits with the same unique key exist

    Returns:
        None
    """

    with sqlite3.connect(db_file) as con:
        
        # Add new columns if not resuming
        cur = con.cursor()
        table_name = 'CIRCUITS_' + str(n_nodes) + '_NODES'
        if not resume:
            columns = utils.list_all_columns(db_file, table_name)
            new_cols = ["n_compact", "n_extended", "n_harmonic",
                            "n_free", "n_frozen", "n_sigma",
                            "H_class", "wJT",  "H_class_sym", "wJT_sym"]
            for col in new_cols:
                if col not in columns:
                    sql_str = f"ALTER TABLE {table_name}\n"
                    sql_str += f"ADD {col}"
                    cur.execute(sql_str)
                    con.commit()
               
        # If we're resuming filter out those without H_class made
        sql_query = f"SELECT DISTINCT unique_key\
                    FROM {table_name}\
                    WHERE in_non_iso_set LIKE 1\
                    AND filter LIKE 1"
        if resume:
            if eq_params:
                sql_query += " AND (H_class_sym is null OR H_class_sym = 'UNDEFINED')"
            else:
                sql_query += " AND (H_class is null OR H_class = 'UNDEFINED')"
        unique_keys = [x[0] for x in cur.execute(sql_query).fetchall()]
    n_to_do = len(unique_keys)
    # Randmize order because difficult ones tend to be near each other
    # This will give more accurate time estimates and spread workers better
    np.random.shuffle(unique_keys)

    # Go through all the circuits and update rows with info
    args = zip(unique_keys, itertools.repeat(db_file, n_to_do), itertools.repeat(eq_params, n_to_do), itertools.repeat(False, n_to_do))
    df_update = []
    count = 0
    if n_workers > 1:
        pool = Pool(processes=n_workers, initializer=initialize_singular)
        for entry, to_update, str_cols in tqdm(pool.imap_unordered(_gen_ham_class_row, args),
                          total=n_to_do):
            count += 1
            df_update.append(entry)
            if count % save_every == 0 or count == n_to_do:
                combined_df = pd.concat(df_update)
                utils.update_db_from_df(db_file, combined_df, to_update, str_cols=str_cols)
                df_update = []
    else:
        for arg_set in tqdm(args, total=n_to_do):
            entry, to_update, str_cols = _gen_ham_class_row((arg_set[0], arg_set[1], arg_set[2], arg_set[3]))
            count += 1
            df_update.append(entry)
            if count % save_every == 0 or count == n_to_do:
                combined_df = pd.concat(df_update)
                utils.update_db_from_df(db_file, combined_df, to_update, str_cols=str_cols)
                df_update = []



def generate_and_trim(n_nodes: int, db_file: str = "circuits.db",
                      base: int = None,
                      n_workers: int = 1, resume: bool = False):
    """ Generates circuits for all graphs for a given number of nodes
        Then trims identical circuits from database.
        Stores circuits in sql database

    Args:
        n_nodes (int): Number of nodes for table
        db_file (str): sql database to store data in
        base (int): The number of possible edges. By default this is 7:
                        (i.e., J, C, I, JI, CI, JC, JCI)
        n_workers (int): The number of workers to use. Default 1.
        resume (bool): Resuming a run or not
    """
    if base is None:
        base = len(utils.ENUM_PARAMS["CHAR_TO_COMBINATION"])
    
    # Check if Hamiltonians have started or not
    if resume:
        table_name = f'CIRCUITS_{n_nodes}_NODES'
        columns = utils.list_all_columns(db_file, table_name)
        H_started = "H_class" in columns
        if H_started:
            print("---------------------------------------")
            print("Resuming at Hamiltonian Class Phase")
            print("---------------------------------------")

    # Pre-Hamiltonian Steps are Fast
    if (not resume) or (not H_started):
        print("----------------------------------------")
        print('Starting generating ' + str(n_nodes) + ' node circuits.')
        generate_graphs_node(db_file, n_nodes, base)
        print("Circuits Generated for " +
            str(n_nodes) + " node circuits.")
        print("Now Trimming.")
        trim_graph_node(db_file=db_file, n_nodes=n_nodes, base=base,
                        n_workers=n_workers)
        print("Finished trimming " + str(n_nodes) + " node circuits.")

    if (not resume) or H_started:
        # Hamiltonian is the slow part
        print("Appending Hamiltonian Classes to " + str(n_nodes) + " node circuits.")
        add_hamiltonian_classes(db_file=db_file, n_nodes=n_nodes,
                                n_workers=n_workers, resume=resume, eq_params=False)
        print("Appending Hamiltonian Classes (equal params) to " + str(n_nodes) + " node circuits.")
        add_hamiltonian_classes(db_file=db_file, n_nodes=n_nodes,
                                n_workers=n_workers, resume=resume, eq_params=True)


    return True


def generate_all_circuits(db_file: str = "circuits.db",
                        n_nodes_start: int = 2,
                        n_nodes_stop: int = 4,
                        base: int = None,
                        n_workers: int = 1,
                        resume: bool = False,
                        quiet: bool = True) -> None:
    """
    Generate all circuits with node counts between `n_nodes_start` and `n_nodes_stop`.  

    This function generates circuits with varying numbers of nodes, removes duplicate 
    circuits, and stores both the full set and the deduplicated set in an SQL database.

    Parameters
    ----------
    file : str
        Path to the SQL database file where the generated circuits will be stored.
    n_nodes_start : int
        Minimum number of nodes to generate circuits for.
    n_nodes_stop : int
        Maximum number of nodes to generate circuits for.
    base : int, optional
        The number of possible edge types. Defaults to ``7``, corresponding to:
        ``J, C, L, JL, CL, JC, JCL``.
    n_workers : int, optional
        The number of workers to use for circuit generation. Defaults to ``1``.
    """
    if base is None:
        base = len(utils.ENUM_PARAMS["CHAR_TO_COMBINATION"])

    if not quiet:
        print("---------------------------------------")
        print("---------------------------------------")
        print("Starting Circuit Enumeration")
        print("db_file:", db_file)
        print("n_nodes_start:", n_nodes_start)
        print("n_nodes_stop:", n_nodes_stop)
        print("base:", base)
        print("n_workers:", n_workers)
        print("resume:", resume)
        print("---------------------------------------")
        print("---------------------------------------")

    # Determine number of nodes to start at
    if resume:
        tables = utils.list_all_tables(db_file)
        if f'CIRCUITS_{n_nodes_stop}_NODES' in tables:
            n_nodes_start = n_nodes_stop
        else: 
            for n in range(n_nodes_start, n_nodes_stop+1):
                if f'CIRCUITS_{n}_NODES' not in tables:
                    n_nodes_start = n - 1
        if not quiet:
            print("---------------------------------------")
            print("Resuming enumeration at", n_nodes_start, "nodes")
            print("---------------------------------------")

    for n in range(n_nodes_start, n_nodes_stop+1):
        if not quiet:
            tqdm.__init__ = functools.partialmethod(tqdm.__init__, disable=False)
            generate_and_trim(n, db_file=db_file, base=base,
                              n_workers=n_workers, resume=resume)
        else:
            with contextlib.redirect_stdout(None):
                tqdm.__init__ = functools.partialmethod(tqdm.__init__, disable=True)
                generate_and_trim(n, db_file=db_file, base=base,
                              n_workers=n_workers, resume=resume)
