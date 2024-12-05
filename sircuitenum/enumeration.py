#!/usr/bin/env python
from __future__ import print_function
"edge_enumerate.py: Contains functions to enumerated edges of quantum circuits"
__author__ = "Mohit Bhat, Eli Weissler"
__version__ = "0.1.0"
__status__ = "Development"

# -------------------------------------------------------------------
# Import Statements
# -------------------------------------------------------------------

import sqlite3
import itertools
import functools
import traceback
import contextlib

import sympy as sym
from sympy.parsing.latex import parse_latex

import networkx as nx
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
from multiprocessing import Pool
from multiprocessing import set_start_method
try:
    set_start_method("fork")
except:
    print("Multiprocessing fork not available on your system.\
           More than one worker is not supported for enumeration \
           with custom elements.")

from sircuitenum import utils
from sircuitenum import reduction as red
from sircuitenum import qpackage_interface as pi
from sircuitenum import quantize

# -------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------


def num_possible_circuits(base: int, n_nodes: int, quiet: bool = True):
    """ Calculates the number of possible circuits for a given number
    # of edges and vertices, may overestimate.

    Args:
        base (int): The number of possible edges. By default this is 7:
                        (i.e., J, C, I, JI, CI, JC, JCI)
        n_nodes (int): the number of vertices in a graph.
        quiet (bool): print the number of circuits or not

    Returns:
        n_circuits (int): a list of networkx graphs
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


def find_unique_ground_placements(circuit: list, edges: list):
    """
    Uses component graph isomorphism to determine the unique
    ground node placements for a given circuit.

    Assumes edges is a continous list from 0 to max number

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]

    Returns:
        tuple of integers representing unique ground node
        placements
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


def find_equiv_cir_series(db_file: str, circuit: list, edges: list):
    """
    Searches the database for circuits that are equivalent
    to the one given, up to a reduction of series linear
    circuit elements

    Args:
        db_file (str): sql database file that's already been completed
                       for the previous number of nodes.
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]

    Returns:
        unique key of the equivalent circuit that is in the
        non isomorphic set
    """

    # What does it look like with series elems removed
    c2, e2 = red.remove_series_elems(circuit, edges)
    equiv = utils.find_circuit_in_db(db_file, c2, e2)
    if equiv.empty:
        return "not found"
    # Return the equivalent circuit
    if equiv.iloc[0]['equiv_circuit'] == "":
        return equiv.iloc[0]['unique_key']
    else:
        return equiv.iloc[0]['equiv_circuit']


def generate_graphs_node(db_file: str, n_nodes: int,
                         base: int, return_vals: bool = False):
    """ Generates circuits for all graphs for a given number of nodes
        Stores circuits in table in sql database for the number of nodes
        Table labeled: 'CIRCUITS_' + str(n_nodes) + '_NODES'

    Args:
        n_nodes (int): Number of nodes for table
        base (int): The number of possible edges. By default this is 7:
                        (i.e., J, C, I, JI, CI, JC, JCI)
        db_file (str): sql database to store data in
        return_vals (bool): return the values in a dataframe or not
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
                    n_workers: int = 1):
    """
    Marks the circuits in the database as having
    jj-s, series linear components, and being in a
    non-isomorphic set of circuits. If the circuit
    is not in the non-isomorphic set, an equivalent
    one that is in the set is recorded.

    All three must be true for the desired final set.

    Args:
        db_file (str): path to database to trim
        n_nodes (int): Number of nodes to consider
        base (int): The number of possible edges. By default this is 7:
                        (i.e., J, C, I, JI, CI, JC, JCI)
        n_workers (int): The number of workers to use. Default 1.
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
        for _ in tqdm(pool.imap_unordered(reduce_individual_set_, args),
                      total=sum(1 for _ in args)):
            pass
    else:
        for arg_set in tqdm(args):
            reduce_individual_set_(arg_set)


def reduce_individual_set_(args: tuple):
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


def add_hamiltonians_to_table(db_file: str, n_nodes: int,
                              n_workers: int = 4, resume: bool = False):
    """
    Adds hamiltonians to the specified db file

    Args:
        db_file (str): database file
        n_nodes (int): number of nodes to add for
        n_workers (int): parallelize the Hamiltonian generation to this many
                         processes.
        resume (bool, optional): whether to resume a previously started run.
                                 this only grabs rows that don't have Hamiltonians
                                 yet.

    Raises:
        ValueError: if multiple circuits with the same unique key exist

    Returns:
        None
    """

    # Add new columns if not resuming
    with sqlite3.connect(db_file) as con:
        cur = con.cursor()
        table_name = 'CIRCUITS_' + str(n_nodes) + '_NODES'
        if not resume:
            new_cols = ["n_periodic", "n_extended", "n_harmonic",
                        "periodic", "extended", "harmonic"]
            new_cols += gen_func_combos_(n_nodes-1).keys()
            new_cols += [x+"_sym" for x in new_cols]
            new_cols = ["H", "H_sym", "coord_transform",
                        "H_class", "H_class_sym", "nonlinearity_counts",
                        "nonlinearity_counts_sym",
                        "H_group", "H_group_sym"] + new_cols
            for col in new_cols:
                sql_str = f"ALTER TABLE {table_name}\n"
                if "n_" in col or "cos" in col or "sin" in col:
                    sql_str += f"ADD {col} int DEFAULT 0"
                else:
                    sql_str += f"ADD {col}"
                cur.execute(sql_str)
                con.commit()

        sql_query = f"SELECT DISTINCT unique_key\
                      FROM {table_name}\
                      WHERE in_non_iso_set LIKE 1\
                      AND filter LIKE 1"
        unique_keys_all = [x[0] for x in cur.execute(sql_query).fetchall()]
        n_total = len(unique_keys_all)
        # If we're resuming filter out those without H_class made
        if resume:
            sql_query += " AND H_class is null"
            unique_keys = [x[0] for x in cur.execute(sql_query).fetchall()]
        else:
            unique_keys = unique_keys_all

    # Randmize order because difficult ones tend to be near each other
    # This will give more accurate time estimates and spread parallel better
    np.random.shuffle(unique_keys)

    # Go through all the circuits and update rows with info
    args = list(zip(unique_keys, [db_file]*len(unique_keys)))
    if n_workers > 1:
        pool = Pool(processes=n_workers)
        for _ in tqdm(pool.imap_unordered(timed_out_, args),
                          total=n_total, initial=n_total-len(unique_keys)):
            pass
    else:
        for arg_set in tqdm(args):
            gen_ham_row_(arg_set[0], arg_set[1])

# Sometimes it doesn't work and hangs :(
# 60 Minute Timeout
def timed_out_(args):
    timeout_min = 60
    try:
        return func_timeout(60*timeout_min, gen_ham_row_, args)
    except FunctionTimedOut:
        print(f"Could not complete {args[0]} ({timeout_min} min timout)")
    except Exception as e:
        raise e


def gen_ham_row_(uid: str, db_file: str):
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

    # Load the graphs with the specified edges counts and graph index
    filter_str = f"WHERE unique_key LIKE '{uid}'"
    n_nodes = int(uid[1])
    df = utils.get_circuit_data_batch(db_file, n_nodes,
                                      filter_str=filter_str)

    if df.shape[0] > 1:
        raise ValueError("Multiple Circuits on Unique Key")
    entry = df.iloc[0]

    # Generate the Hamiltonian
    try:
        H, trans, H_class, all_combos = gen_hamiltonian(entry.circuit,
                                                        entry.edges,
                                                        symmetric=False,
                                                        return_combos=True)
        H_str = refine_latex(sym.latex(H))
        info = categorize_hamiltonian(H)

        # Symmetrize the Hamiltonian
        C = sym.Symbol("C", positive=True, real=True)
        EJ = sym.Symbol("E_{J}", positive=True, real=True)
        CJ = sym.Symbol("C_{J}", positive=True, real=True)
        L = sym.Symbol("L", positive=True, real=True)
        H_sym = H.copy()
        for s in H.free_symbols:
            if "C_" in str(s) and "J" not in str(s):
                H_sym = H_sym.subs(s, C)
            elif "C_" in str(s) and "J" in str(s):
                H_sym = H_sym.subs(s, CJ)
            elif "L_" in str(s):
                H_sym = H_sym.subs(s, L)
            elif "E_{J" in str(s):
                H_sym = H_sym.subs(s, EJ)

        # Zero out terms to get the H_class
        H_class_sym = utils.remove_coeff_(H_sym, all_combos)
        H_sym_str = refine_latex(sym.latex(H_sym))
        info_sym = categorize_hamiltonian(H_sym)
    except KeyboardInterrupt as kbi:
        raise kbi
    except Exception as exc:
        print("-------------------------------------------")
        print("Unable to Generate Hamiltonian for:", uid)
        print(traceback.format_exc())
        print(exc)
        print("-------------------------------------------")
        return

    # Set values
    to_update = ["H", "H_sym", "coord_transform", "H_class", "H_class_sym",
                 "nonlinearity_counts", "nonlinearity_counts_sym",
                 "H_group", "H_group_sym"]
    df.at[uid, "H"] = H_str
    df.at[uid, "H_sym"] = H_sym_str
    df.at[uid, "coord_transform"] = str(trans)
    df.at[uid, "H_class"] = refine_latex(sym.latex(H_class))
    df.at[uid, "H_class_sym"] = refine_latex(sym.latex(H_class_sym))
    for col in info:
        if col in df.columns:
            df.at[uid, col] = info[col]
            to_update.append(col)
    for col in info_sym:
        if col+"_sym" in df.columns:
            df.at[uid, col+"_sym"] = info_sym[col]
            to_update.append(col+"_sym")

    # Add nonlinearity counts
    nonlinearity_cols = [x for x in df.columns if "sin_" in x or "cos_" in x]
    nonlinearity_cols_sym = [x for x in nonlinearity_cols if "_sym" in x]
    nonlinearity_cols = [x for x in nonlinearity_cols if "_sym" not in x]
    nonlinearity_counts = "".join([str(int(x)) for x in
                                   df[nonlinearity_cols].loc[uid].values])
    nonlinearity_counts_sym = "".join([str(int(x)) for x in
                                       df[nonlinearity_cols_sym].loc[uid].values])
    df.at[uid, "nonlinearity_counts"] = nonlinearity_counts
    df.at[uid, "nonlinearity_counts_sym"] = nonlinearity_counts_sym

    # Update value in database
    utils.update_db_from_df(db_file, df, to_update,
                            str_cols=["H", "H_sym", "periodic",
                                      "extended", "harmonic",
                                      "coord_transform",
                                      "periodic_sym", "extended_sym",
                                      "harmonic_sym",
                                      "H_class", "H_class_sym",
                                      "nonlinearity_counts",
                                      "nonlinearity_counts_sym",
                                      "H_group", "H_group_sym"])


def gen_hamiltonian(circuit: list, edges: list, symmetric: bool = False,
                    cob: sym.Matrix = None, var_class: dict = None,
                    return_combos: bool = False, basis_completion: str = "heuristic"):
    """
    Generate a Sympy Hamiltonian for the specified circuit.
    Uses scqubits to come up with an appropriate variable transformation.

    NOTE: External fluxes/charges are not supported right now

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [["J"],["L", "J"], ["C"]]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        symmetric (bool, optional): Whether to set all capacitances,
                                    inductances, and Josephson energies equal.
                                    Risks losing terms. Defaults to False.
        cob (sym.Matrix, optional): Optionally give a variable transformation
                                    instead of using the Z transformation matrix
                                    from scqubits.
        var_class (dict, optional): If you give a variable transformation, also
                                    give a dictionary like scqubits var_categories
                                    with keys "free", "frozen", "periodic", and
                                    "extended" that classify the variables.
        return_combos (bool, optional): optionally return the combination of
                                        variables present
        basis_completion (str, optional): basis completion option for scqubits

    Returns:
        (Sympy Add, np.array, Sympy Add):
                   1) Symbolic Hamiltonian where periodic modes are labeled by
                      n and extended variables are labeled by q.
                   2) Coordinate transformation matrix (new in terms of node
                      variables)
                   3) Hamiltonian "class" that has all constants removed.
    """

    elems = {
            'C': {'default_unit': 'GHz', 'default_value': 0.2},
            'L': {'default_unit': 'GHz', 'default_value': 1.0},
            'J': {'default_unit': 'GHz', 'default_value': 15.0},
            'CJ': {'default_unit': 'GHz', 'default_value': 500.0}
            }
    params = utils.gen_param_dict(circuit, edges, elems)

    if not symmetric:
        # Set random values to avoid unintentionally
        # deleting terms
        for edge, comp in params:
            if comp in ["C", "L"]:
                param_range = (0.1, 1)
            else:
                param_range = (1, 20)
            params[(edge, comp)] = (np.random.uniform(*param_range), "GHz")

    if cob is None:
        # Use scQubits to get a transformation Matrix
        obj = pi.to_SCqubits(circuit, edges, params=params, sym_cir=True,
                             initiate_sym_calc=False,
                             basis_completion=basis_completion)

        # Get symbolic Hamiltonian and add final free mode
        # as a given
        cob, var_class = obj.variable_transformation_matrix()
        var_class["free"] += [utils.get_num_nodes(edges)]

    elif var_class is None:
        raise ValueError("Must include variable classification with cob matrix")

    # Un-symmetrize the edges if that's what's requested
    if not symmetric:
        new_circuit = []
        counts = {"J": 0, "L": 0, "C": 0}
        for elems in circuit:
            new_elems = []
            for elem in elems:
                new_elems.append(f"{elem}_{counts[elem] + 1}")
                counts[elem] += 1
            new_circuit.append(new_elems)
        circuit = new_circuit

    H, H_class, all_combos = quantize.quantize_circuit(circuit, edges,
                                                       cob=sym.Matrix(cob),
                                                       **var_class,
                                                       return_H_class=True,
                                                       return_combos=True,
                                                       collect_phase=True)
    to_return = (H, cob, H_class)
    if return_combos:
        to_return = to_return + (all_combos,)
    return to_return


def assign_H_groups(db_file: str, n_nodes: int,
                    n_workers: int = 1, resume: bool = False) -> None:
    """
    Assigns Hamiltonians in the database into groups
    based on the functional form of the linear and
    nonlinear parts of their hamiltonians
    
    Args:
        db_file (str): path to database file
        n_nodes (int): number of nodes to examine
        n_workers (int, optional): Number of workers to use. Defaults to 1.
    """
    # Figure out where to start if resuming
    if resume:
        table_name = 'CIRCUITS_' + str(n_nodes) + '_NODES'
        columns = utils.list_all_columns(db_file, table_name)
        H_group_started = "H_group" in columns
        H_group_sym_started = "H_group_sym" in columns
    else:
        H_group_started = False
        H_group_sym_started = False

    # Get the unique nonlinearity counts strings
    with sqlite3.connect(db_file) as con:
        cur = con.cursor()
        table_name = 'CIRCUITS_' + str(n_nodes) + '_NODES'
        sql_str = f"SELECT DISTINCT nonlinearity_counts \
                    FROM {table_name}\
                    WHERE in_non_iso_set LIKE 1\
                    AND filter LIKE 1"
        unique_counts = [x for x in cur.execute(sql_str).fetchall()]
        n_counts = len(unique_counts)
        sql_str_sym = f"SELECT DISTINCT nonlinearity_counts_sym \
                    FROM {table_name}\
                    WHERE in_non_iso_set LIKE 1\
                    AND filter LIKE 1"
        unique_counts_sym = [x for x in cur.execute(sql_str_sym).fetchall()]
        n_counts_sym = len(unique_counts)
        if resume:
            if H_group_sym_started:
                sql_str_sym = sql_str[:]
                sql_str_sym += " AND H_group_sym is null"
                unique_counts_sym = [x for x in cur.execute(sql_str_sym).fetchall()]
            elif H_group_started:
                sql_str += " AND H_group is null"
                unique_counts = [x for x in cur.execute(sql_str).fetchall()]
    
    print("Total Groups:", n_counts)

    # Filter out none values from circuits that timed out in 
    # quantization
    unique_counts = [x[0] for x in unique_counts if x is not None]
    unique_counts_sym = [x[0] for x in unique_counts_sym if x is not None]

    # Shuffle for accurate runtime estimates
    np.random.shuffle(unique_counts)
    np.random.shuffle(unique_counts_sym)

    # Make the pool if we're parallel
    if n_workers > 1:
        pool = Pool(processes=n_workers)

    # Do non-symmetric first
    # args are db_file, n_nodes, nl_cnt, symmetric, mapping
    n_entries = len(unique_counts)
    if not H_group_sym_started:
        print("Full Hamiltonians...")
        args = list(zip([db_file]*n_entries, [n_nodes]*n_entries,
                        unique_counts, [False]*n_entries,
                        [utils.ENUM_PARAMS["CHAR_TO_COMBINATION"]]*n_entries))
        if n_workers > 1:
            for _ in tqdm(pool.imap_unordered(unique_hams_for_count_, args),
                        total=n_counts, initial=n_counts-n_entries):
                pass
        else:
            for arg_set in args:
                unique_hams_for_count_(arg_set)

    # Now do symmetric
    print("Symmetric Hamiltonians...")
    n_entries = len(unique_counts_sym)
    args = list(zip([db_file]*n_entries, [n_nodes]*n_entries,
                    unique_counts_sym, [True]*n_entries,
                    [utils.ENUM_PARAMS["CHAR_TO_COMBINATION"]]*n_entries))
    if n_workers > 1:
        for _ in tqdm(pool.imap_unordered(unique_hams_for_count_, args),
                      total=n_counts_sym, initial=n_counts_sym-n_entries):
            pass
    else:
        for arg_set in args:
            unique_hams_for_count_(arg_set)


def unique_hams(hams: list[str], group_base: str = "", normalize_sign: bool = True):
    """
    Identifies Hamiltonians in a set that differ
    by relabelling variables. Assigns each entry to
    a group.

    Args:
        hams (list[str]): list of Hamiltonians, loaded from the database
        group_base (str, optional): optional prefix for group name. Defaults to "".
        normalize_sign (bool, optional): Whether to make all terms positive in H_class.
                                         Defaults to True.
    
    Returns:
        list[str]: list of unique hamiltonian strings
        list[str]: group labels for each entry
    """

    reduced = []
    groups = []
    # Examine every row in the set
    group_n = 1
    for l_str in hams:

        is_dup = False

        # Make the string parsable by sympy
        l_str = l_str.replace("\\hat{" + quantize.EXTENDED_CHARGE + "}", "Q")
        l_str = l_str.replace("\\hat{" + quantize.PERIODIC_CHARGE + "}", "n")
        l_str = l_str.replace("\\hat{" + quantize.EXTENDED_PHASE + "}", "F")
        l_str = l_str.replace("\\hat{" + quantize.PERIODIC_PHASE + "}", "p")
        
        H_base = parse_latex(l_str)

        # Get the Phase and Charge terms
        q_list = [q for q in H_base.free_symbols if "Q" in str(q)]
        n_list = [q for q in H_base.free_symbols if "n" in str(q)]
        f_list = [q for q in H_base.free_symbols if "F" in str(q)]
        p_list = [q for q in H_base.free_symbols if "p" in str(q)]

        # Alternative A, B, C terms
        q_list_alt = np.array([sym.Symbol(f"Q_{chr(ord('A') + int(n))}")
                                for n in range(len(q_list))])
        n_list_alt = np.array([sym.Symbol(f"n_{chr(ord('A') + int(n))}")
                                for n in range(len(n_list))])
        f_list_alt = np.array([sym.Symbol(f"F_{chr(ord('A') + int(n))}")
                                for n in range(len(f_list))])
        p_list_alt = np.array([sym.Symbol(f"p_{chr(ord('A') + int(n))}")
                                for n in range(len(p_list))])

        # Possible assignments of 1, 2, 3 -> A, B, C
        q_ass = itertools.permutations(range(len(q_list)), len(q_list))
        n_ass = itertools.permutations(range(len(n_list)), len(n_list))
        f_ass = itertools.permutations(range(len(f_list)), len(f_list))
        p_ass = itertools.permutations(range(len(p_list)), len(p_list))

        # Try every permutation of A, B, C -> 1, 2, 3
        for combo in itertools.product(q_ass, n_ass, f_ass, p_ass):

            # Make it a list for indexing
            combo = [list(x) for x in combo]

            # Test out the specific permutation
            H_test = H_base.copy()

            # Replace 1, 2, 3 with A, B, C
            if combo[0]:
                for x1, x2 in zip(q_list, q_list_alt[combo[0]]):
                    H_test = H_test.subs(x1, x2)
            if combo[1]:
                for x1, x2 in zip(n_list, n_list_alt[combo[1]]):
                    H_test = H_test.subs(x1, x2)
            if combo[2]:
                for x1, x2 in zip(f_list, f_list_alt[combo[2]]):
                    H_test = H_test.subs(x1, x2)
            if combo[3]:
                for x1, x2 in zip(p_list, p_list_alt[combo[3]]):
                    H_test = H_test.subs(x1, x2)

            # Check if this permutation is in the reduced set already
            for i, H_ref in enumerate(reduced):

                if H_ref is None:
                    continue

                if H_test - H_ref == 0:
                    is_dup = True
                    dup_group = groups[i]
                    break
            if is_dup:
                break
        if is_dup:
            reduced.append(None)
            groups.append(dup_group)
        if not is_dup:
            reduced.append(H_test)
            groups.append(group_base + f"_{group_n}")
            group_n += 1

    return reduced, groups


def unique_hams_in_df(df: pd.DataFrame, symmetric: bool, normalize_sign: bool = True):
    """
    Marks unique Hamiltonian classes by creating the H_group column. Meant to be
    provided a dataframe containing values for a single nonlinearity counts.

    Catches entries with the same H_class and
    those that differ by renumbering variables.

    Args:
        df (pd.DataFrame): dataframe containing circuit entries for a single
                           value of nonlinearity_counts
        symmetric (bool): whether to examine the "_sym" columns or not.
        normalize_sign (bool, optional): Whether to make all terms positive in H_class.
                                         Defaults to True.

    Returns:
        group_col_name: name of column added to input df
    """

    if symmetric:
        l_str_vec = df["H_class_sym"].values
        nl_cnt = df["nonlinearity_counts_sym"].iloc[0]

    else:
        l_str_vec = df["H_class"].values
        nl_cnt = df["nonlinearity_counts"].iloc[0]
        
    if normalize_sign:
        l_str_vec = [x.replace("-", "+") for x in l_str_vec]


    # Catch the obviously same ones, i.e.
    # the H_str is the exact same
    unique_str, index, inv = np.unique(l_str_vec, return_index=True,
                                       return_inverse=True)

    # Catch the ones with variables labeled differently
    reduced, groups = unique_hams(unique_str, group_base=nl_cnt, normalize_sign=normalize_sign)

    groups = np.array(groups)
    group_col = groups[inv]
    if symmetric:
        group_col_name = "H_group_sym"
        df[group_col_name] = group_col
    else:
        group_col_name = "H_group"
        df[group_col_name] = group_col

    return group_col_name


def unique_hams_for_count_(args):

    db_file, n_nodes, nl_cnt, symmetric, mapping = args

    if symmetric:
        col_name = "nonlinearity_counts_sym"
    else:
        col_name = "nonlinearity_counts"

    filter_str = f"WHERE {col_name} LIKE '{nl_cnt}'"

    df = utils.get_circuit_data_batch(db_file, n_nodes,
                                      char_mapping=mapping,
                                      filter_str=filter_str)
    # if df.shape[0] == 0:
    #     raise ValueError("No entries found for nl count")

    group_col_name = unique_hams_in_df(df, symmetric)
    
    utils.update_db_from_df(db_file, df,
                            to_update=[group_col_name],
                            str_cols=[group_col_name]
                            )

    return


def gen_func_combos_(n_modes: int) -> dict:
    """
    Helper function that generates all combinations of sin/cos
    given a maximum power

    Args:
        n_modes (int): Number of modes in the circuit (i.e. max power)

    Returns:
        dict: dictionary with combos as keys and 0 as values
    """
    info = {}
    for n in range(1, n_modes+1):
        for type_combo in itertools.product(["p", "e"], repeat=n):
            for func_combo in itertools.product(["cos", "sin"], repeat=n):
                # Count the functions present
                counts = {}
                for combo in zip(func_combo, type_combo):
                    if combo in counts:
                        counts[combo] += 1
                    else:
                        counts[combo] = 1
                # Add the field in the dictionary
                combos = []
                for combo in counts:
                    combos += [combo]*counts[combo]
                # Sort alphabetically for consistency
                order = np.sort(["_".join(x) for x in combos])
                info_str = "_".join(order)
                info[info_str] = 0
    return info


def categorize_hamiltonian(H: sym.core.Add):
    """
    Categorizes a Hamiltonian according to the nonlinearities
    present.

    Assumes frozen and free modes have already been removed.

    Args:
        H (sympy.core.Add): sympy Hamiltonian, generated
                            from gen_hamiltonian

    Returns:
        info: Dictionary counting the nonlinearities present.
              Considers every possibility of cos/sin and 
              extended/periodic variables. Should be 4 choices
              with one modes, 10 choices with two modes,
              and 20 with three modes.
    """

    # Expand H to make searching easier
    H_test = sym.expand(H)

    # List of variable types
    theta_list = [th for th in H.free_symbols
                  if (quantize.PERIODIC_PHASE in str(th) or
                      quantize.EXTENDED_PHASE in str(th) or
                      quantize.NODE_PHASE in str(th))
                  and quantize.EXT_PHASE not in str(th)]

    # Information about the hamiltonian
    n_modes = len(theta_list)
    info = {"n_modes": n_modes,
            "periodic": [],
            "extended": [],
            "harmonic": []}

    # Categorize Modes:
    types = {}
    funcs = set()
    [[funcs.add(f) for f in x.atoms(sym.Function)]
        for x in H_test.atoms(sym.Mul)]
    for th in theta_list:
        mode_num = "".join([x for x in str(th) if x.isnumeric()])
        if quantize.PERIODIC_PHASE in str(th):
            info["periodic"].append(mode_num)
            types[str(th)] = "p"
        elif sym.cos(th) in funcs or sym.sin(th) in funcs:
            info["extended"].append(mode_num)
            types[str(th)] = "e"
        else:
            info["harmonic"].append(mode_num)
            types[str(th)] = "h"
    
    # Sort mode list
    for var_type in ["periodic", "extended", "harmonic"]:
        info[var_type] = sorted(info[var_type])

    # Add counts
    info["n_periodic"] = len(info["periodic"])
    info["n_extended"] = len(info["extended"])
    info["n_harmonic"] = len(info["harmonic"])

    # Products of sin/cos up to n_modes
    info.update(gen_func_combos_(n_modes))

    # Count the nonlinear terms
    funcs = set(functools.reduce(lambda x, y: x*y, list(x.atoms(sym.Function)))
                for x in H_test.atoms(sym.Mul) if len(x.atoms(sym.Function)) > 0)
    # n = number of nonlinear terms
    for n in range(1, n_modes+1):
        # th_combo = list of variables
        for th_combo in itertools.product(theta_list, repeat=n):
            # Variable types
            th_types = [types[str(th)] for th in th_combo]
            # All different combinations of sin's and cos
            # of the two thetas
            for bar in range(n+1):
                term = 1
                # Cos terms
                for i in range(bar):
                    term *= sym.cos(th_combo[i])
                # Sin Terms
                for i in range(bar, n):
                    term *= sym.sin(th_combo[i])
                # Check if term was present
                if term in funcs:
                    info_str = ["_".join(x) for x in
                                zip(["cos"]*bar, th_types[:bar])]
                    info_str += ["_".join(x) for x in
                                 zip(["sin"]*(n-bar), th_types[bar:])]
                    info_str = "_".join(np.sort(info_str))
                    info[info_str] += 1
                    funcs.remove(term)

    return info


def refine_latex(latex_str):
    """
    Adds hats to operators, and removes cdots before
    parenthesis

    Args:
        latex_str (str): string of the latex math

    Returns:
        str: copy of the latex_str with the modifications done
    """
    latex_str = latex_str.replace(r"\cdot \left(", r"\left(")
    return latex_str


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
        H_group_started = "H_group" in columns
        H_group_sym_started = "H_group_sym" in columns
        
        if H_started and not H_group_started:
            print("---------------------------------------")
            print("Resuming at Hamiltonian Phase")
            print("---------------------------------------")
        elif H_started and H_group_started and not H_group_sym_started:
            print("---------------------------------------")
            print("Resuming at Full H Group")
            print("---------------------------------------")
        elif H_started and H_group_started and H_group_sym_started:
            print("---------------------------------------")
            print("Resuming at Symmetric H Group")
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

    # Hamiltonian is the slow part
    if (not resume) or (not H_group_started):
        print("Appending Hamiltonians to " + str(n_nodes) + " node circuits.")
        add_hamiltonians_to_table(db_file=db_file, n_nodes=n_nodes,
                                n_workers=n_workers, resume=resume)

    # print("Categorizing Linear Portion of Hamiltonians for " + str(n_nodes) + " node circuits.")
    # assign_H_groups(db_file=db_file, n_nodes=n_nodes, n_workers=n_workers,
    #                 resume=resume)
    
    # print("Categorizing Non-Linear Portion of Hamiltonians for " + str(n_nodes) + " node circuits.")
    # Max 10 workers because this is fast and db conflicts
    assign_H_groups(db_file=db_file, n_nodes=n_nodes, n_workers=n_workers,
                    resume=False)
    return True


def generate_all_circuits(db_file: str = "circuits.db",
                        n_nodes_start: int = 2,
                        n_nodes_stop: int = 4,
                        base: int = None,
                        n_workers: int = 1,
                        resume: bool = False,
                        quiet: bool = True):
    """ Generates all circuits with numbers of nodes between
        `n_nodes_start` and `n_nodes_stop`, then removes identical
        circuits the generated circuits.

        The circuits with and without the identcal elemements removed
        are saved in sql database.

    Args:
        file (str): sql database file to store data in
        n_nodes_start (int) : Min number of nodes to generate circuits for.
        n_nodes_stop (int) : Max number of nodes to generate circuits for.
        base (int): The number of possible edges. By default this is 7:
                        (i.e., J, C, I, JI, CI, JC, JCI)
        n_workers (int): The number of workers to use. Default 1.
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
