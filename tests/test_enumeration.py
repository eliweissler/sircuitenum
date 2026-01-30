import os
import itertools
import sqlite3
from pathlib import Path
import pytest 


import sympy as sy
import numpy as np
import pandas as pd

from sircuitenum import enumeration as enum
from sircuitenum import utils
from sircuitenum import reduction as red

import numpy.random
numpy.random.seed(7)  # seed random number generation for all calls to rand_ops


ALL_CONNECTED_3 = [[utils.ENUM_PARAMS["CHAR_TO_COMBINATION"][c]
                    for c in np.base_repr(i, 3).zfill(3)] for i in range(27)]
NON_ISOMORPHIC_3 = [
                  (("L",), ("C",), ("J",)),
                  (("L",), ("J",), ("J",)),
                  (("C",), ("J",), ("J",)),
                  (("J",), ("J",), ("J",))
                 ]
NON_SERIES_3 = list(itertools.permutations([("L",), ("C",), ("J",)], 3))
NON_SERIES_3 += [
                (("L",), ("J",), ("J",)),
                (("J",), ("L",), ("J",)),
                (("J",), ("J",), ("L",))
              ]
NON_SERIES_3 += [
                (("C",), ("J",), ("J",)),
                (("J",), ("C",), ("J",)),
                (("J",), ("J",), ("C",))
              ]
NON_SERIES_3 += [
                (("J",), ("J",), ("J",))
              ]


TEMP_FILE = "temp.db"
TEMP_FILE2 = "temp2.db"

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Clean up temp database files before and after each test"""
    # Setup - clean before test
    for f in [TEMP_FILE, TEMP_FILE2]:
        cleanup_db(f)
    
    yield  # Run the test
    
    # Teardown - clean after test
    for f in [TEMP_FILE, TEMP_FILE2]:
        cleanup_db(f)

def cleanup_db(filename):
    for suffix in ["", "-wal", "-shm"]:
        f = Path(str(filename) + suffix)
        if f.exists():
            try:
                os.remove(f)
            except:
                pass


def test_num_possible_circuits():

    assert enum.num_possible_circuits(3, 2) == 3
    assert enum.num_possible_circuits(3, 3) == 36
    assert enum.num_possible_circuits(7, 3) == 392


def test_generate_for_specific_graph():

    # Most simple two node graph
    G = utils.get_basegraphs(2)[0]
    df = enum.generate_for_specific_graph(7, G, 0, return_vals=True)
    exp_circuits = ['0', '1', '2', '3', '4', '5', '6']
    assert [x for x in df['circuit'].values] == exp_circuits

    # Fully connected three node with no parallel stuff
    G = utils.get_basegraphs(3)[1]
    df = enum.generate_for_specific_graph(3, G, 1, return_vals=True)
    exp_circuits = ["".join([utils.ENUM_PARAMS["COMBINATION_TO_CHAR"][combo]
                             for combo in circuit])
                    for circuit in ALL_CONNECTED_3]
    assert [x for x in df['circuit'].values] == exp_circuits

    # Four nodes
    n_trials = 1000
    graph_index = 3
    n_nodes = 4
    base = 7
    G = utils.get_basegraphs(n_nodes)[graph_index]
    df = enum.generate_for_specific_graph(base, G, graph_index,
                                          return_vals=True)
    n_edges = len(G.edges)
    choices = [np.base_repr(x, base) for x in range(base)]
    for i in range(n_trials):
        random_circuit = [x for x in np.random.choice(choices, size=n_edges)]
        assert utils.circuit_in_set(random_circuit, df['circuit'].values)

    # Five nodes
    n_trials = 10000
    graph_index = 8
    n_nodes = 5
    base = 7
    G = utils.get_basegraphs(n_nodes)[graph_index]
    df = enum.generate_for_specific_graph(base, G, graph_index,
                                          return_vals=True)
    n_edges = len(G.edges)
    choices = [np.base_repr(x, base) for x in range(base)]
    for i in range(n_trials):
        random_circuit = [x for x in np.random.choice(choices, size=n_edges)]
        assert utils.circuit_in_set(random_circuit, df['circuit'].values)


def test_delete_table():

    if Path(TEMP_FILE).exists():
        cleanup_db(TEMP_FILE)

    # Generate 2 node circuits
    enum.generate_graphs_node(TEMP_FILE, 2, 7)
    # Should be one table in the file
    t1 = utils.list_all_tables(TEMP_FILE)
    enum.delete_table(TEMP_FILE, 2)
    # Should be zero tables in the file
    t2 = utils.list_all_tables(TEMP_FILE)

    assert len(t1) == 1
    assert len(t2) == 0

    cleanup_db(TEMP_FILE)


def test_find_uniuqe_ground_placements():
    
    edges = [(0, 1)]
    circuit = [("L",)]
    gnds = enum.find_unique_ground_placements(circuit, edges)
    assert gnds == (0,)

    edges = [(0, 2), (2, 1), (0, 1)]
    circuit = [("L",), ("L",), ("L",)]
    gnds = enum.find_unique_ground_placements(circuit, edges)
    assert gnds == (0,)

    edges = [(0, 2), (2, 1), (0, 1)]
    circuit = [("L",), ("J",), ("J",)]
    gnds = enum.find_unique_ground_placements(circuit, edges)
    assert gnds == (0, 1)

    edges = [(0, 2), (2, 1), (0, 1)]
    circuit = [("L",), ("J",), ("C",)]
    gnds = enum.find_unique_ground_placements(circuit, edges)
    assert gnds == (0, 1, 2)

    edges = [(0, 1), (1, 2)]
    circuit = [("L",), ("J",)]
    gnds = enum.find_unique_ground_placements(circuit, edges)
    assert gnds == (0, 1, 2)


def test_expand_ground_node():

      df = pd.DataFrame({"edges": [[(0, 2), (2, 1), (0, 1)]],
                         "circuit": [[("L",), ("J",), ("J",)]]})
      new_df = enum.expand_ground_node(df)
      assert new_df.shape[0] == 2
      assert 0 in new_df["ground_node"].values
      assert 1 in new_df["ground_node"].values


def test_has_dangling_edges():

    edges = [(0, 1)]
    circuit = [("L",)]
    assert enum.has_dangling_edges(circuit, edges)

    edges = [(0, 1)]
    circuit = [("L", "C")]
    assert not enum.has_dangling_edges(circuit, edges)

    edges = [(0, 2), (2, 1), (0, 1)]
    circuit = [("L",), ("L",), ("L",)]
    assert not enum.has_dangling_edges(circuit, edges)

    edges = [(0, 1), (1, 2), (2, 3)]
    circuit = [("L",), ("J",), ("J",)]
    assert enum.has_dangling_edges(circuit, edges)

def test_remove_dangling_edges():

    df = pd.DataFrame({"edges": [[(0, 2), (2, 1), (0, 1), (2, 3)]],
                        "circuit": [[("L",), ("J",), ("J",), ("C",)]]})
    new_df = enum.remove_dangling_edges(df)
    assert new_df.shape[0] == 0

    df = pd.DataFrame({"edges": [[(0, 2), (2, 1), (0, 1)]],
                        "circuit": [[("L",), ("J",), ("J",)]]})
    new_df = enum.remove_dangling_edges(df)
    assert new_df.shape[0] == 1


def test_find_equiv_cir_series():

    if Path(TEMP_FILE).exists():
        cleanup_db(TEMP_FILE)

    # Generate all the 2/3 node circuits
    enum.generate_all_circuits(TEMP_FILE, 2, 3, base=7, n_workers=1, quiet=False)

    # Find the equivalent circuits for ones that would
    # be reduced
    edges = [(0, 2), (2, 1), (0, 1)]
    circuit = [("L",), ("L",), ("L",)]
    uid = enum.find_equiv_cir_series(TEMP_FILE, circuit, edges)
    c, e = red.linear_star_mesh(circuit, edges)
    c2, e2 = utils.get_circuit_data(TEMP_FILE, uid)
    assert red.isomorphic_circuit_in_set(c, e, [c2])

    edges = [(0, 2), (2, 1), (0, 1)]
    circuit = [("C",), ("L",), ("L",)]
    uid = enum.find_equiv_cir_series(TEMP_FILE, circuit, edges)
    c, e = red.linear_star_mesh(circuit, edges)
    c2, e2 = utils.get_circuit_data(TEMP_FILE, uid)
    assert red.isomorphic_circuit_in_set(c, e, [c2])

    edges = [(0, 2), (2, 1), (0, 1)]
    circuit = [("C",), ("C",), ("J",)]
    uid = enum.find_equiv_cir_series(TEMP_FILE, circuit, edges)
    c, e = red.linear_star_mesh(circuit, edges)
    c2, e2 = utils.get_circuit_data(TEMP_FILE, uid)
    assert red.isomorphic_circuit_in_set(c, e, [c2])

    edges = [(0, 1), (1, 2), (2, 3)]
    circuit = [("C",), ("C",), ("J",)]
    uid = enum.find_equiv_cir_series(TEMP_FILE, circuit, edges)
    c, e = red.linear_star_mesh(circuit, edges)
    c2, e2 = utils.get_circuit_data(TEMP_FILE, uid)
    assert red.isomorphic_circuit_in_set(c, e, [c2])

    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    circuit = [("C",), ("C",), ("J",), ("L",)]
    uid = enum.find_equiv_cir_series(TEMP_FILE, circuit, edges)
    c, e = red.linear_star_mesh(circuit, edges)
    c2, e2 = utils.get_circuit_data(TEMP_FILE, uid)
    assert red.isomorphic_circuit_in_set(c, e, [c2])

    cleanup_db(TEMP_FILE)


def test_generate_graphs_node():

    # Most simple two node graph
    G = utils.get_basegraphs(2)[0]
    df = enum.generate_graphs_node(None, 2, 7, return_vals=True)
    exp_circuits = ['0', '1', '2', '3', '4', '5', '6']
    assert [x for x in df['circuit'].values] == exp_circuits

    # Three nodes
    n_trials = 100
    n_nodes = 3
    base = 7
    df = enum.generate_graphs_node(None, n_nodes, base, return_vals=True)
    grouped = df.groupby("graph_index")
    for graph_index, G in enumerate(utils.get_basegraphs(n_nodes)):
        subset = grouped.get_group(graph_index)
        n_edges = len(G.edges)
        choices = [np.base_repr(x, base) for x in range(base)]
        for i in range(n_trials):
            random_circuit = [x for x in
                              np.random.choice(choices, size=n_edges)]
            assert utils.circuit_in_set(random_circuit,
                                        subset['circuit'].values)

    # Four nodes
    n_trials = 100
    n_nodes = 4
    base = 7
    df = enum.generate_graphs_node(None, n_nodes, base, return_vals=True)
    grouped = df.groupby("graph_index")
    for graph_index, G in enumerate(utils.get_basegraphs(n_nodes)):
        subset = grouped.get_group(graph_index)
        n_edges = len(G.edges)
        choices = [np.base_repr(x, base) for x in range(base)]
        for i in range(n_trials):
            random_circuit = [x for x in
                              np.random.choice(choices, size=n_edges)]
            assert utils.circuit_in_set(random_circuit,
                                        subset['circuit'].values)


def test__reduce_individual_set():

    # Generate all the 2/3 node circuits
    enum.generate_graphs_node(TEMP_FILE, 2, base=7)
    enum.generate_graphs_node(TEMP_FILE, 3, base=7)

    # CJL Delta
    filter_str = f"WHERE edge_counts LIKE '1,1,1,0,0,0,0' AND graph_index LIKE 1"
    args = (filter_str, TEMP_FILE, 3, utils.ENUM_PARAMS["CHAR_TO_COMBINATION"], True)
    enum._reduce_individual_set(args)
    df = utils.get_circuit_data_batch(TEMP_FILE, 3, char_mapping=utils.ENUM_PARAMS["CHAR_TO_COMBINATION"], filter_str=filter_str)
    assert df["in_non_iso_set"].sum() == 1

    # CLL Delta
    filter_str = f"WHERE edge_counts LIKE '1,0,2,0,0,0,0' AND graph_index LIKE 1"
    args = (filter_str, TEMP_FILE, 3, utils.ENUM_PARAMS["CHAR_TO_COMBINATION"], True)
    enum._reduce_individual_set(args)
    df = utils.get_circuit_data_batch(TEMP_FILE, 3, char_mapping=utils.ENUM_PARAMS["CHAR_TO_COMBINATION"], filter_str=filter_str)
    assert df["in_non_iso_set"].sum() == 0

    cleanup_db(TEMP_FILE)


def test_trim_graph_node():


    # Serial/Parallel
    for n_workers in [1, 4]:

        # Generate all the 2/3 node circuits
        enum.generate_graphs_node(TEMP_FILE, 2, base=7)
        enum.generate_graphs_node(TEMP_FILE, 3, base=7)
        
        enum.trim_graph_node(TEMP_FILE, 3, base = 7, n_workers = n_workers)

        # CJJ Delta
        filter_str = f"WHERE edge_counts LIKE '1,2,0,0,0,0,0' AND graph_index LIKE 1"
        df = utils.get_circuit_data_batch(TEMP_FILE, 3, char_mapping=utils.ENUM_PARAMS["CHAR_TO_COMBINATION"], filter_str=filter_str)
        assert df["in_non_iso_set"].sum() == 1

        # CJL Delta
        filter_str = f"WHERE edge_counts LIKE '1,1,1,0,0,0,0' AND graph_index LIKE 1"
        df = utils.get_circuit_data_batch(TEMP_FILE, 3, char_mapping=utils.ENUM_PARAMS["CHAR_TO_COMBINATION"], filter_str=filter_str)
        assert df["in_non_iso_set"].sum() == 1

        # CC line
        filter_str = f"WHERE edge_counts LIKE '2,0,0,0,0,0,0' AND graph_index LIKE 0"
        df = utils.get_circuit_data_batch(TEMP_FILE, 3, char_mapping=utils.ENUM_PARAMS["CHAR_TO_COMBINATION"], filter_str=filter_str)
        assert df["in_non_iso_set"].sum() == 0

        cleanup_db(TEMP_FILE)


def test__gen_ham_class_row():

    # Generate all the 2 node circuits
    enum.generate_graphs_node(TEMP_FILE, 2, base=7)

    # Fluxonium
    filter_str = f"WHERE edge_counts LIKE '0,0,0,0,1,0,0' AND graph_index LIKE 0"
    df = utils.get_circuit_data_batch(TEMP_FILE, 2, char_mapping=utils.ENUM_PARAMS["CHAR_TO_COMBINATION"], filter_str=filter_str)
    table_name = 'CIRCUITS_' + str(2) + '_NODES'
    uid = df.iloc[0]["unique_key"]

    # Add cols
    new_cols = ["n_compact", "n_extended", "n_harmonic",
                "n_free", "n_frozen", "n_sigma",
                "H_class", "wJT",  "H_class_sym", "wJT_sym"]
    con = sqlite3.connect(TEMP_FILE)
    cur = con.cursor()
    for col in new_cols:
        sql_str = f"ALTER TABLE {table_name}\n"
        if "n_" in col or "cos" in col or "sin" in col:
            sql_str += f"ADD {col} int DEFAULT 0"
        else:
            sql_str += f"ADD {col}"
        cur.execute(sql_str)
    con.commit()
    con.close()

    enum._gen_ham_class_row((uid, TEMP_FILE, False, True))
    enum._gen_ham_class_row((uid, TEMP_FILE, True, True))

    # Test stuff is right
    df = utils.get_circuit_data_batch(TEMP_FILE, 2, char_mapping=utils.ENUM_PARAMS["CHAR_TO_COMBINATION"], filter_str=filter_str)
    entry = df.iloc[0]
    assert entry["n_compact"] == 0
    assert entry["n_extended"] == 1
    assert entry["n_harmonic"] == 0
    assert entry["n_free"] == 0
    assert entry["n_frozen"] == 0
    assert entry["n_sigma"] == 1
    assert entry["H_class"] == '010_0--1-1-5_0_0-_0-'
    assert entry["H_class_sym"] == '010_0--1-1-5_0_0-_0-'
    assert entry["wJT"] == "5"
    assert entry["wJT_sym"] == "5"


    cleanup_db(TEMP_FILE)


def df_equality_check(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Helper function that tests whether every entry of
    every row of two dataframes are equal

    Args:
        df1 (pd.DataFrame): dataframe 1 to compare
        df2 (pd.DataFrame): dataframe 2 to compare
    """
    assert df1.shape[0] == df2.shape[0]
    for i in range(df1.shape[0]):
        for k in df1.columns:
            if k in df2.columns:
                v1 = df1.iloc[i][k]
                v2 = df2.iloc[i][k]
                if isinstance(v1, list):
                    assert len(v1) == len(v2)
                    assert all(x in v2 for x in v1)
                    assert all(x in v1 for x in v2)
                else:
                    assert v1 == v2


def test_add_hamiltonian_classes():
    
    cleanup_db(TEMP_FILE)
    cleanup_db(TEMP_FILE2)

    res =     ['011_0-0-1-1-5_0_0-0_0-0', '011_0-0-1-1-5_1_0-0_1-1', '011_0-0-1-1-5_1_1-1_0-0', '011_0-0-1-1-5_2_1-1_1-1', '020_0-0-2-1-5445_0_0-0_0-0', '020_0-0-2-1-5445_1_0-0_1-1', '020_0-0-2-1-5445_1_1-1_0-0', '020_0-0-2-1-5445_2_1-1_1-1', '020_1-1-4-1-544555_1_0-0_1-1', '020_1-1-4-1-544555_2_1-1_1-1', '101_0-0-1-1-5_0_0-0_0-0', '101_0-0-1-1-5_1_0-0_1-1', '110_0-0-2-1-5445_0_0-0_0-0', '110_0-0-2-1-5445_1_0-0_1-1', '110_1-1-3-1-5455_1_0-0_1-1', '110_1-1-4-1-544555_1_0-0_1-1', '200_0-0-2-1-5445_0_0-0_0-0', '200_0-0-2-1-5445_1_0-0_1-1', '200_1-1-4-1-544555_1_0-0_1-1']
    res_sym = ['011_0-0-1-1-5_0_0-0_0-0', '011_0-0-1-1-5_1_0-0_1-1', '020_0-0-2-1-5445_0_0-0_0-0', '020_0-0-2-1-5445_1_0-0_1-1', '020_0-0-2-1-5445_1_1-1_0-0', '020_0-0-2-1-5445_2_1-1_1-1', '020_1-1-4-1-544555_1_0-0_1-1', '020_1-1-6-2-465553_0_0-0_0-0', '101_0-0-1-1-5_0_0-0_0-0', '101_0-0-1-1-5_1_0-0_1-1', '110_0-0-2-1-5445_0_0-0_0-0', '110_0-0-2-1-5445_1_0-0_1-1', '110_1-1-3-1-5455_1_0-0_1-1', '110_1-1-4-1-544555_1_0-0_1-1', '110_1-1-4-1-5553_0_0-0_0-0', '110_1-1-6-2-465553_0_0-0_0-0', '200_0-0-2-1-5445_0_0-0_0-0', '200_0-0-2-1-5445_1_0-0_1-1', '200_1-1-4-1-544555_1_0-0_1-1']
    # Generate all the 3 node circuits and check that
    # there's 19 H classes
    print("--- not parallel ---")
    enum.generate_all_circuits(TEMP_FILE2, 2, 3, base=7, n_workers=1, quiet=False)
    print("--- yes parallel ---")
    enum.generate_all_circuits(TEMP_FILE, 2, 3, base=7, n_workers=4, quiet=False)
    # assert False
    df = utils.get_unique_qubits(TEMP_FILE, 3)
    df2 = utils.get_circuit_data_batch(TEMP_FILE2, 3)
    for i, row in df2.iterrows():
        comp = utils.find_circuit_in_db(TEMP_FILE,
                                      row['circuit'],
                                      row['edges']).iloc[0]
        if comp["H_class"] != row["H_class"]:
            print("mismatch found (H_class):")
            print("ref:", row['circuit'], row['H_class'])
            print("test:", comp['circuit'], comp['H_class'])
            # breakpoint()
            assert False
        if comp["H_class_sym"] != row["H_class_sym"]:
            print("mismatch found (H_class_sym):")
            print("ref:", row['circuit'], row['H_class_sym'])
            print("test:", comp['circuit'], comp['H_class_sym'])
            # breakpoint()
            assert False
    classes = sorted(df.H_class.unique())
    sym_classes = sorted(df.H_class_sym.unique())
    diff = [x for x in classes if x not in sym_classes]
    diff2 = [x for x in sym_classes if x not in classes]


    assert len(classes) == 19
    assert len(sym_classes) == 19
    for x in res:
        assert x in classes
    for x in res_sym:
        assert x in sym_classes
    

    exp_diff_sym_nonsym = ['020_1-1-6-2-465553_0_0-0_0-0', '110_1-1-4-1-5553_0_0-0_0-0', '110_1-1-6-2-465553_0_0-0_0-0']
    for x in exp_diff_sym_nonsym:
        assert x in sym_classes and x not in classes
    exp_diff_nonsym_sym = ['011_0-0-1-1-5_1_1-1_0-0', '011_0-0-1-1-5_2_1-1_1-1', '020_1-1-4-1-544555_2_1-1_1-1']
    for x in exp_diff_nonsym_sym:
        assert x in classes and x not in sym_classes

    cleanup_db(TEMP_FILE)
    cleanup_db(TEMP_FILE2)
    
    enum.generate_all_circuits(TEMP_FILE, 2, 3, base=5, n_workers=1)
    df = utils.get_unique_qubits(TEMP_FILE, 3)
    assert df.H_class.unique().size == 19
<<<<<<< HEAD
    assert df.H_class_sym.unique().size == 17
    cleanup_db(TEMP_FILE)
=======
    assert df.H_class_sym.unique().size == 16
    os.remove(TEMP_FILE)
>>>>>>> 3be7a2183f647ffb306c522a69587f579db2c8cd


def test_generate_all_circuits():

    if Path(TEMP_FILE).exists():
        cleanup_db(TEMP_FILE)

    # Generate all the 2, 3 node circuits
    enum.generate_all_circuits(TEMP_FILE, 2, 3, base=3, quiet=False)

    # Test the 2 nodes I/O
    df_untrimmed = utils.get_circuit_data_batch(TEMP_FILE, n_nodes=2)
    df_trimmed = utils.get_unique_qubits(TEMP_FILE, n_nodes=2)

    df_untrimmed_good = enum.generate_graphs_node(None, 2, 3, True)
    utils.convert_loaded_df(df_untrimmed_good, n_nodes=2)
    red.full_reduction(df_untrimmed_good)
    unique_qubits = np.logical_and(np.logical_and(
        df_untrimmed_good['in_non_iso_set'],
        df_untrimmed_good['filter']),
        df_untrimmed_good['no_series'])
    df_trimmed_good = df_untrimmed_good[unique_qubits]
    cols_to_compare = df_trimmed_good.columns
    # Sort the dataframes to ensure consistent ordering
    df_untrimmed = df_untrimmed.sort_index()
    df_untrimmed_good = df_untrimmed_good.sort_values(by="unique_key")
    df_trimmed = df_trimmed.sort_index()
    df_trimmed_good = df_trimmed_good.sort_values(by="unique_key")
    df_equality_check(df_untrimmed[cols_to_compare], df_untrimmed_good[cols_to_compare])
    df_equality_check(df_trimmed[cols_to_compare], df_trimmed_good[cols_to_compare])

    # Test the 3 nodes I/0
    df_untrimmed = utils.get_circuit_data_batch(TEMP_FILE, n_nodes=3)
    df_trimmed = utils.get_unique_qubits(TEMP_FILE, n_nodes=3)

    df_untrimmed_good = enum.generate_graphs_node(None, 3, 3, True)
    utils.convert_loaded_df(df_untrimmed_good, n_nodes=3)
    red.full_reduction(df_untrimmed_good)
    unique_qubits = np.logical_and(np.logical_and(
        df_untrimmed_good['in_non_iso_set'],
        df_untrimmed_good['filter']),
        df_untrimmed_good['no_series'])
    df_trimmed_good = df_untrimmed_good[unique_qubits]

    # Find equivalent circuits for the series reduced circuits
    equiv_cir = df_untrimmed_good['equiv_circuit'].values
    yes_series = np.logical_not(df_untrimmed_good['no_series'].values)
    for i in range(df_untrimmed_good.shape[0]):
        if yes_series[i]:
            row = df_untrimmed_good.iloc[i]
            equiv_cir[i] = enum.find_equiv_cir_series(TEMP_FILE,
                                                      row['circuit'],
                                                      row['edges']
                                                      )

    df_trimmed_good = df_untrimmed_good[unique_qubits]
    cols_to_compare = df_trimmed_good.columns
    # Sort the dataframes to ensure consistent ordering
    df_untrimmed = df_untrimmed.sort_index()
    df_untrimmed_good = df_untrimmed_good.sort_values(by="unique_key")
    df_trimmed = df_trimmed.sort_index()
    df_trimmed_good = df_trimmed_good.sort_values(by="unique_key")
    df_equality_check(df_untrimmed[cols_to_compare], df_untrimmed_good[cols_to_compare])
    df_equality_check(df_trimmed[cols_to_compare], df_trimmed_good[cols_to_compare])


    # Test the accuracy
    df2 = utils.get_unique_qubits(TEMP_FILE, n_nodes=2)
    df3_og = utils.get_unique_qubits(TEMP_FILE, n_nodes=3)
    df3 = df3_og[df3_og['graph_index'] == 1]

    assert df2.shape[0] == 1
    assert df3.shape[0] == len(NON_ISOMORPHIC_3)

    edges = [(0, 1), (1, 2), (2, 0)]
    for c in NON_ISOMORPHIC_3:
        assert red.isomorphic_circuit_in_set(c, edges, df3.circuit.values)

    cleanup_db(TEMP_FILE)

    # Compare parallel vs. not parallel generation for 3 nodes
    df3 = df3_og.copy()
    df3.index = np.arange(df3.shape[0])
    df3 = df3.sort_values(by="unique_key")
    enum.generate_all_circuits(TEMP_FILE, 2, 3, base=3, n_workers=4)
    comp = utils.get_unique_qubits(TEMP_FILE, n_nodes=3)
    comp.index = np.arange(comp.shape[0])
    comp = comp.sort_values(by="unique_key")
    df_equality_check(df3, comp)


    cleanup_db(TEMP_FILE)


def test_qps_enum():

    import itertools
    char_map = {}
    c = 97
    for n in range(1, 5):  
        for comb in itertools.combinations(["C", "J", "L", "Q"], n):
            char_map[chr(c)] = comb
            c += 1
    utils.set_enum_params(char_map, lambda x: utils.jj_present(x) or utils.qps_present(x))

    n = 2
    enum.generate_graphs_node(TEMP_FILE, n_nodes=n, base=15)
    enum.trim_graph_node(TEMP_FILE, n_nodes=n, base=15, n_workers=1)
    df = utils.get_circuit_data_batch(TEMP_FILE, n_nodes=n)
    assert df.shape[0] == 15
    assert df["filter"].sum() == 12

    n = 3
    enum.generate_graphs_node(TEMP_FILE, n_nodes=n, base=15)
    enum.trim_graph_node(TEMP_FILE, n_nodes=n, base=15, n_workers=1)
    enum.trim_graph_node(TEMP_FILE, n_nodes=n, base=15, n_workers=2)
    df = utils.get_circuit_data_batch(TEMP_FILE, n_nodes=n)

    # reset enum params
    utils.set_enum_params()

    cleanup_db(TEMP_FILE)


if __name__ == "__main__":
    # test_num_possible_circuits()
    # test_generate_for_specific_graph()
    # test_delete_table()
    # test_find_uniuqe_ground_placements()
    # test_expand_ground_node()
    # test_has_dangling_edges()
    # test_remove_dangling_edges()
    # test_find_equiv_cir_series()
    # test_generate_graphs_node()
    # test__reduce_individual_set()
    # test_trim_graph_node()
    # test__gen_ham_class_row()
    # test_add_hamiltonian_classes()
    test_generate_all_circuits()
    # test_qps_enum()

