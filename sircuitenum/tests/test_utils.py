import os
import itertools
import sqlite3

import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path

from sircuitenum import utils
from sircuitenum.tests.test_qpackage_interface import TEST_CIRCUITS

import numpy.random
numpy.random.seed(7)  # seed random number generation for all calls to rand_ops


ALL_CONNECTED_3_RAW = [[c for c in np.base_repr(i, 3).zfill(3)]
                       for i in range(27)]
ALL_CONNECTED_3 = [[utils.ENUM_PARAMS["CHAR_TO_COMBINATION"][c] for c in x]
                   for x in ALL_CONNECTED_3_RAW]
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

TEST_C = ["0", "1", "2"]
TEST_MAPPED_C = utils.encoding_to_components(TEST_C)
TEST_GI = 1
TEST_NN = 3
TEST_B = 7
TEST_CN = 16
TEST_ENTRY = utils.circuit_entry_dict(TEST_C, TEST_GI, TEST_NN,
                                      TEST_CN, TEST_B)

TEMP_FILE = 'temp.db'


def set_enum_params():

    test = {"0": ('C', 'L'), "G": ('Q')}
    utils.set_enum_params(test)

    assert utils.ENUM_PARAMS["CHAR_TO_COMBINATION"] == test
    assert utils.ENUM_PARAMS["COMBINATION_TO_CHAR"][('C', 'L')] == "0"
    assert utils.ENUM_PARAMS["COMBINATION_TO_CHAR"][('Q')] == "G"
    assert utils.ENUM_PARAMS["EDGE_COLOR_DICT"][('C', 'L')] == 0
    assert utils.ENUM_PARAMS["EDGE_COLOR_DICT"][('Q')] == 1

    utils.set_enum_params()
    default  = {'0': ('C',),
                '1': ('J',),
                '2': ('L',),
                '3': ('C', 'L'),
                '4': ('J', 'L'),
                '5': ('C', 'J'),
                '6': ('C', 'J', 'L')}
    assert utils.ENUM_PARAMS["CHAR_TO_COMBINATION"] == default
     


def test_get_basegraphs():

    # Most simple two node graph
    G = utils.get_basegraphs(2)[0]
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1)])
    assert nx.is_isomorphic(G, G2)

    # test three node graphs
    G = utils.get_basegraphs(3)[0]
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 2)])
    assert nx.is_isomorphic(G, G2)

    G = utils.get_basegraphs(3)[1]
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 2), (0, 2)])
    assert nx.is_isomorphic(G, G2)

    # and a four node one too
    G = utils.get_basegraphs(4)[1]
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 2), (2, 3)])
    assert nx.is_isomorphic(G, G2)


def test_graph_index_to_edges():

    # test done assuming get_basegraphs is working
    G = utils.get_basegraphs(2)[0]
    g = utils.graph_index_to_edges(0, 2)
    assert all(x in G.edges for x in g)

    G = utils.get_basegraphs(3)[0]
    g = utils.graph_index_to_edges(0, 3)
    assert all(x in G.edges for x in g)

    G = utils.get_basegraphs(3)[1]
    g = utils.graph_index_to_edges(1, 3)
    assert all(x in G.edges for x in g)

    G = utils.get_basegraphs(4)[0]
    g = utils.graph_index_to_edges(0, 4)
    assert all(x in G.edges for x in g)

    G = utils.get_basegraphs(5)[3]
    g = utils.graph_index_to_edges(3, 5)
    assert all(x in G.edges for x in g)


def test_edges_to_graph_index():

    # 2 Nodes
    edges = [(0, 1)]
    gi = utils.edges_to_graph_index(edges)
    assert gi == 0
    edges = [(1, 0)]
    gi = utils.edges_to_graph_index(edges, return_mapping=True)
    assert gi[0] == 0, gi[1] == {0:1, 1:0}

    # 3 Nodes
    edges = [(0, 2), (2, 1)]
    gi = utils.edges_to_graph_index(edges)
    assert gi == 0
    edges = [(0, 2), (2, 1), (0, 1)]
    gi = utils.edges_to_graph_index(edges)
    assert gi == 1

    # 4 Nodes
    edges = [(0, 2), (1, 0), (3, 0)]
    gi = utils.edges_to_graph_index(edges)
    assert gi == 0
    edges = [(0, 2), (2, 1), (3, 1)]
    gi = utils.edges_to_graph_index(edges)
    assert gi == 1
    edges = [(0, 1), (1, 2), (2, 0), (1, 3)]
    gi = utils.edges_to_graph_index(edges)
    assert gi == 2
    edges = [(0, 1), (1, 3), (2, 3), (0, 2)]
    gi = utils.edges_to_graph_index(edges, return_mapping=True)
    assert gi[0] == 3
    edges = [(0, 3), (1, 3), (2, 1), (0, 2)]
    gi = utils.edges_to_graph_index(edges, return_mapping=True)
    assert gi[0] == 3, gi[1] == {0:0, 1:3, 2:2, 3:1}
    edges = [(0, 1), (1, 2), (2, 0), (1, 3), (2, 3)]
    gi = utils.edges_to_graph_index(edges)
    assert gi == 4
    edges = [(0, 1), (1, 2), (2, 0), (1, 3), (2, 3), (3, 0)]
    gi = utils.edges_to_graph_index(edges)
    assert gi == 5


def test_renumber_nodes():

    # Test a few obvious cases
    edges = [(0, 1), (1, 2), (2, 0)]
    e2 = utils.renumber_nodes(edges)
    assert e2 == edges

    edges = [(1, 3), (1, 2)]
    e2 = utils.renumber_nodes(edges)
    assert e2 == [(0, 2), (0, 1)]

    edges = [(0, 1), (1, 2), (6, 0)]
    e2 = utils.renumber_nodes(edges)
    assert e2 == [(0, 1), (1, 2), (3, 0)]

    # Some larger ones -- test the 2,3/3,4 connection
    edges = [(0, 1), (1, 2), (2, 3), (2, 4), (3, 4)]
    e2 = utils.renumber_nodes(edges)
    assert e2 == edges

    edges = [(1, 4), (1, 2), (2, 5), (2, 4), (2, 4)]
    e2 = utils.renumber_nodes(edges)
    assert e2 == [(0, 2), (0, 1), (1, 3), (1, 2), (1, 2)]


def test_combine_redundant_edges():

    # Test a few obvious cases
    edges = [(0, 1), (1, 2), (2, 0)]
    circuit = [("L",), ("L",), ("L",)]
    c2, e2 = utils.combine_redundant_edges(circuit, edges)
    assert e2 == [(0, 1), (1, 2), (0, 2)]
    assert c2 == circuit

    edges = [(0, 1), (1, 0)]
    circuit = [("L",), ("L",)]
    c2, e2 = utils.combine_redundant_edges(circuit, edges)
    assert e2 == [(0, 1)]
    assert c2 == [("L",)]

    edges = [(0, 1), (0, 1), (2, 0)]
    circuit = [("C",), ("C",), ("L",)]
    c2, e2 = utils.combine_redundant_edges(circuit, edges)
    assert e2 == [(0, 1), (0, 2)]
    assert c2 == [("C",), ("L",)]

    # Some larger ones -- test the 2,3/3,4 connection
    edges = [(0, 1), (1, 0), (2, 1), (2, 1), (2, 3)]
    circuit = [("C",), ("J",), ("C", "J"), ("L", "J"), ("L",)]
    c2, e2 = utils.combine_redundant_edges(circuit, edges)
    assert e2 == [(0, 1), (1, 2), (2, 3)]
    assert c2 == [("C", "J"), ("C", "J", "L"), ("L",)]


def test_encoding_to_components():

    components = utils.encoding_to_components("261")
    assert components == [("L",), ("C", "J", "L"), ("J",)]

    components = utils.encoding_to_components("243")
    assert components == [("L",), ("J", "L"), ("C", "L")]

    components = utils.encoding_to_components("2435")
    assert components == [("L",), ("J", "L"), ("C", "L"), ("C", "J")]


def test_components_to_encoding():


    encoding = utils.components_to_encoding([("L",), ("C", "J", "L"), ("J",)])
    assert encoding == "261"

    encoding = utils.components_to_encoding([("L",), ("C", "J"), ("C", "L")])
    assert encoding == "253"

    encoding = utils.components_to_encoding([("L",), ("C", "J"), ("C", "L"),
                                             ("J", "L")])
    assert encoding == "2534"


def test_count_elems():

    assert utils.count_elems(['0', '2', '5', '1'], 7) == [1, 1, 1, 0, 0, 1, 0]

    assert utils.count_elems(['4', '2', '2', '1'], 7) == [0, 1, 2, 0, 1, 0, 0]

    assert utils.count_elems(['4', '4', '4'], 7) == [0, 0, 0, 0, 3, 0, 0]


def test_count_elems_mapped():

    circuit = [("C",)]
    counts = utils.count_elems_mapped(circuit)
    assert len(counts) == 3
    assert counts["C"] == 1
    assert counts["J"] == 0
    assert counts["L"] == 0

    circuit = [("C",), ("C", "J"), ("C", "L"),  ("C", "J", "L"),
               ("C", "L"), ("C",)]
    counts = utils.count_elems_mapped(circuit)
    assert len(counts) == 3
    assert counts["C"] == 6
    assert counts["J"] == 2
    assert counts["L"] == 3

def test_list_single_elems():

    ans = utils.list_single_elems()
    assert "C" in ans
    assert "L" in ans
    assert "J" in ans
    assert len(ans) == 3

def test_get_num_nodes():

    assert utils.get_num_nodes([(0, 1)]) == 2

    assert utils.get_num_nodes([(0, 1), (1, 2)]) == 3

    assert utils.get_num_nodes([(0, 1), (1, 2), (2, 3)]) == 4

    assert utils.get_num_nodes([(0, 1), (0, 2), (1, 2), (1, 3)]) == 4

    assert utils.get_num_nodes([(0, 1), (45, 2), (1, 2), (1, 3)]) == 5


def test_circuit_in_set():

    assert utils.circuit_in_set(
        [("L",), ("J",), ("J",)], NON_ISOMORPHIC_3) is True

    assert utils.circuit_in_set(
        [("C",), ("J",), ("J",)], NON_ISOMORPHIC_3) is True

    assert utils.circuit_in_set(
        [("L",), ("L",), ("L",)], NON_ISOMORPHIC_3) is False

    assert utils.circuit_in_set(
        [("L",), ("C",), ("L",)], NON_ISOMORPHIC_3) is False


def test_gen_param_dict():

    elem_dict = {
                'C': {'default_unit': 'GHz', 'default_value': 0.2},
                'L': {'default_unit': 'GHz', 'default_value': 1.0},
                'J': {'default_unit': 'GHz', 'default_value': 5.0},
                'CJ': {'default_unit': 'GHz', 'default_value': 20.0}
                }

    # Go through test circuits
    for i in range(len(TEST_CIRCUITS)):
        edges, circuit = TEST_CIRCUITS[i][0], TEST_CIRCUITS[i][1]
        param_dict = utils.gen_param_dict(circuit, edges,
                                          vals=elem_dict)
        counts = utils.count_elems_mapped(circuit)
        for components, edge in zip(circuit, edges):
            # Make sure component value/unit setting
            # worked right
            for comp in components:
                counts[comp] -= 1
                v, u = param_dict[(edge, comp)]
                assert v == elem_dict[comp]['default_value']
                assert u == elem_dict[comp]['default_unit']
                if comp == "J":
                    v2, u2 = param_dict[(edge, "CJ")]
                    assert v2 == elem_dict["CJ"]['default_value']
                    assert u2 == elem_dict["CJ"]['default_unit']

        # Make sure total counts are right
        assert all(np.array(list(counts.values())) == 0)

    elem_dict = {
            'C': {'default_unit': 'GHz', 'default_value': 0.2},
            'L': {'default_unit': 'GHz', 'default_value': 1.0},
            'J': {'default_unit': 'GHz', 'default_value': 5.0}
            }

    # Go through test circuits
    for i in range(len(TEST_CIRCUITS)):
        edges, circuit = TEST_CIRCUITS[i][0], TEST_CIRCUITS[i][1]
        param_dict = utils.gen_param_dict(circuit, edges,
                                          vals=elem_dict)
        counts = utils.count_elems_mapped(circuit)
        for components, edge in zip(circuit, edges):
            # Make sure component value/unit setting
            # worked right
            for comp in components:
                counts[comp] -= 1
                v, u = param_dict[(edge, comp)]
                assert v == elem_dict[comp]['default_value']
                assert u == elem_dict[comp]['default_unit']
                if comp == "J":
                    assert (edge, "CJ") not in param_dict

        # Make sure total counts are right
        assert all(np.array(list(counts.values())) == 0)


def test_convert_circuit_to_graph():

    vals = {
            'C': {'default_unit': 'GHz', 'default_value': 0.2},
            'L': {'default_unit': 'GHz', 'default_value': 1.0},
            'J': {'default_unit': 'GHz', 'default_value': 5.0},
            'CJ': {'default_unit': 'GHz', 'default_value': 0.0}
             }

    G = nx.MultiGraph()
    G.add_edges_from([(0, 1, 0), (1, 2, 0), (1, 2, 1), (2, 3, 0), (2, 3, 1)])
    c = [["J"], ["C", "J"], ["C", "L"]]
    e = [(0, 1), (1, 2), (2, 3)]
    params = utils.gen_param_dict(c, e, vals)
    assert utils.convert_circuit_to_graph(c, e, params=params).edges == G.edges

    G = nx.MultiGraph()
    G.add_edges_from([(0, 1, 0), (0, 1, 1), (1, 2, 0),
                      (1, 4, 0), (2, 3, 0), (2, 3, 1)])
    c = [["J", "L"], ["C"], ["C", "J"], ["L"]]
    e = [(0, 1), (1, 2), (2, 3), (1, 4)]
    params = utils.gen_param_dict(c, e, vals)
    assert utils.convert_circuit_to_graph(c, e, params=params).edges == G.edges

    G = nx.MultiGraph()
    G.add_edges_from([(0, 1, 0), (1, 2, 0), (1, 2, 1), (1, 2, 2)])
    c = [["C"], ["C", "J", "L"]]
    e = [(0, 1), (1, 2)]
    params = utils.gen_param_dict(c, e, vals)
    assert utils.convert_circuit_to_graph(c, e, params=params).edges == G.edges

    # Test junction capacitance
    vals = {
            'C': {'default_unit': 'GHz', 'default_value': 0.2},
            'L': {'default_unit': 'GHz', 'default_value': 1.0},
            'J': {'default_unit': 'GHz', 'default_value': 5.0},
            'CJ': {'default_unit': 'GHz', 'default_value': 4.0}
             }
    params = utils.gen_param_dict(c, e, vals)
    G = nx.MultiGraph()
    G.add_edges_from([(0, 1, 0), (0, 1, 1)])
    c = [["J"]]
    e = [(0, 1)]
    params = utils.gen_param_dict(c, e, vals)
    assert utils.convert_circuit_to_graph(c, e, params=params).edges == G.edges


def test_circuit_degree():
    
    c = [["J"], ["C", "J"], ["C", "L"]]
    e = [(0, 1), (1, 2), (2, 3)]
    ans = [1, 3, 4, 2]
    assert utils.circuit_degree(c, e) == ans


def test_circuit_node_representation():

    c = [["J"], ["C", "J"], ["C", "L"]]
    e = [(0, 1), (1, 2), (2, 3)]
    ans = {'C': [0, 1, 2, 1], 'J': [1, 2, 1, 0], 'L': [0, 0, 1, 1]}
    assert utils.circuit_node_representation(c, e) == ans

    c = [["J", "L"], ["C"], ["C", "J"], ["L"]]
    e = [(0, 1), (1, 2), (2, 3), (1, 4)]
    ans = {'C': [0, 1, 2, 1, 0], 'J': [1, 1, 1, 1, 0], 'L': [1, 2, 0, 0, 1]}
    assert utils.circuit_node_representation(c, e) == ans

    c = [["C"], ["C", "J", "L"]]
    e = [(0, 1), (1, 2)]
    ans = {'C': [1, 2, 1], 'J': [0, 1, 1], 'L': [0, 1, 1]}

    assert utils.circuit_node_representation(c, e) == ans

def test_jj_present():

    assert utils.jj_present([("J", "L"), ("J", "C"), ("C", "L", "J")])
    assert utils.jj_present([("C", "L"), ("C", "C"), ("C", "L", "L")]) is False
    assert utils.jj_present([("C",), ("L",), ("J",)])
    assert utils.jj_present([("C",), ("L",), ("Q",)]) is False


def test_qps_present():

    assert utils.qps_present([("J", "L"), ("J", "C"), ("C", "Q", "J")])
    assert utils.qps_present([("C", "L"), ("C", "C"), ("C", "L", "L")]) is False
    assert utils.qps_present([("C",), ("L",), ("J",)]) is False
    assert utils.qps_present([("C",), ("L",), ("Q",)])

def test_circuit_entry_dict():

    circuit = ["0", "1", "2"]
    graph_index = 0
    n_nodes = 3
    base = 7
    circuit_num = 16
    entry = utils.circuit_entry_dict(circuit, graph_index, n_nodes,
                                     circuit_num, base)
    assert entry['circuit'] == "".join(circuit)
    assert entry['graph_index'] == graph_index
    assert entry['unique_key'] == f"n{n_nodes}_g{graph_index}_c{circuit_num}"
    assert entry['edge_counts'] == "1,1,1,0,0,0,0"
    assert entry['n_nodes'] == n_nodes
    assert entry['base'] == base


def test_convert_loaded_df():

    df = pd.DataFrame([TEST_ENTRY])
    utils.convert_loaded_df(df, 3)
    assert df['circuit'].iloc[0] == [('C',), ('J',), ('L',)]
    assert df['circuit_encoding'].iloc[0] == '012'
    assert df['edges'].iloc[0] == [(0, 1), (0, 2), (1, 2)]


def test_write_df():

    if Path(TEMP_FILE).exists():
        os.remove(TEMP_FILE)

    df = write_test_df()
    df2 = utils.get_circuit_data_batch(TEMP_FILE, 3)

    assert df.shape == df2.shape
    write_test_df(overwrite=False)

    df2 = utils.get_circuit_data_batch(TEMP_FILE, 3)
    assert 2*df.shape[0] == df2.shape[0]
    assert df.shape[1] == df2.shape[1]

    write_test_df(overwrite=True)

    df2 = utils.get_circuit_data_batch(TEMP_FILE, 3)

    assert df.shape[0] == df2.shape[0]
    assert df.shape[1] == df2.shape[1]

    os.remove(TEMP_FILE)


def test_update_db_from_df():

    if Path(TEMP_FILE).exists():
        os.remove(TEMP_FILE)

    to_update = ["no_series", "filter",
                 "in_non_iso_set", "equiv_circuit"]
    str_cols = ["equiv_circuit"]

    df = write_test_df()
    df2 = utils.get_circuit_data_batch(TEMP_FILE, 3)
    df2['in_non_iso_set'] = 1
    utils.update_db_from_df(TEMP_FILE, df2, to_update, str_cols)

    df3 = utils.get_circuit_data_batch(TEMP_FILE, 3)
    assert np.all(df3['in_non_iso_set'] == 1)
    assert np.all(df['in_non_iso_set'] == 0)

    os.remove(TEMP_FILE)


def test_delete_circuit_data():

    df = write_test_df(TEMP_FILE, overwrite=True)

    uid = 'n3_g1_c21'
    utils.delete_circuit_data(TEMP_FILE, 3, uid)

    df2 = utils.get_circuit_data_batch(TEMP_FILE, 3)

    assert df2.shape[0] == df.shape[0]-1
    assert uid not in df2.unique_key

    df = write_test_df(TEMP_FILE, overwrite=True)

    n_delete = 10
    uid = sorted([f'n3_g1_c{str(i).zfill(2)}'
                  for i in np.random.choice(np.arange(df.shape[0]),
                                            size=n_delete, replace=False)])
    utils.delete_circuit_data(TEMP_FILE, 3, uid)

    df2 = utils.get_circuit_data_batch(TEMP_FILE, 3)

    assert df2.shape[0] == df.shape[0]-n_delete
    assert all(u not in df2.unique_key for u in uid)

    os.remove(TEMP_FILE)


def test_get_circuit_data():

    con, cur = write_test_circuit(TEMP_FILE)
    circuit, edges = utils.get_circuit_data(TEMP_FILE,
                                            TEST_ENTRY['unique_key'])

    assert circuit == TEST_MAPPED_C
    test_edges = [(0, 1), (1, 2), (2, 0)]
    assert len(edges) == len(test_edges)
    assert all(utils.circuit_in_set(x, test_edges) or
               utils.circuit_in_set(x[::-1], test_edges) for x in edges)
    assert all(utils.circuit_in_set(x, edges) or
               utils.circuit_in_set(x[::-1], edges) for x in test_edges)

    # Try loading and re-writing with batch function to make sure
    # it doesn't break anything
    df = utils.get_circuit_data_batch(TEMP_FILE, 3)
    utils.write_df(TEMP_FILE, df, 3, overwrite=True)

    circuit, edges = utils.get_circuit_data(TEMP_FILE,
                                            TEST_ENTRY['unique_key'])

    assert circuit == TEST_MAPPED_C
    test_edges = [(0, 1), (1, 2), (2, 0)]
    assert len(edges) == len(test_edges)
    assert all(utils.circuit_in_set(x, test_edges) or
               utils.circuit_in_set(x[::-1], test_edges) for x in edges)
    assert all(utils.circuit_in_set(x, edges) or
               utils.circuit_in_set(x[::-1], edges) for x in test_edges)

    # Load elements of the full dataframe individually
    df = write_test_df(TEMP_FILE, overwrite=True)
    for i, row in df.iterrows():
        circuit, edges = utils.get_circuit_data(TEMP_FILE,
                                                row['unique_key'])
        test_edges = row['edges']
        test_circuit = row['circuit']
        assert circuit == test_circuit
        assert len(edges) == len(test_edges)
        assert all(utils.circuit_in_set(x, test_edges) or
                   utils.circuit_in_set(x[::-1], test_edges) for x in edges)
        assert all(utils.circuit_in_set(x, edges) or
                   utils.circuit_in_set(x[::-1], edges) for x in test_edges)

    # Cleanup
    con.close()
    os.remove(TEMP_FILE)


def test_get_circuit_data_batch():

    if Path(TEMP_FILE).exists():
        os.remove(TEMP_FILE)

    df = write_test_df()
    df2 = utils.get_circuit_data_batch(TEMP_FILE, 3)

    assert df.shape == df2.shape
    for i in range(df2.shape[0]):
        for k in df2.columns:
            v1 = df.iloc[i][k]
            v2 = df2.iloc[i][k]
        if isinstance(v1, list):
            assert len(v1) == len(v2)
            assert all(x in v2 for x in v1)
            assert all(x in v1 for x in v2)
        else:
            assert v1 == v2

    os.remove(TEMP_FILE)


def test_get_unique_qubits():

    if Path(TEMP_FILE).exists():
        os.remove(TEMP_FILE)
    to_update = ["no_series", "filter",
                 "in_non_iso_set", "equiv_circuit"]
    str_cols = ["equiv_circuit"]
   
    write_test_df()
    df2 = utils.get_circuit_data_batch(TEMP_FILE, 3)
    winner = df2.iloc[3]['unique_key']
    df2.at[winner, 'in_non_iso_set'] = True
    df2.at[winner, 'no_series'] = True
    df2.at[winner, 'filter'] = True
    utils.update_db_from_df(TEMP_FILE, df2, to_update, str_cols)

    df3 = utils.get_unique_qubits(TEMP_FILE, 3)
    assert df3.shape[0] == True
    assert df3.iloc[0]['unique_key'] == winner

    os.remove(TEMP_FILE)


def test_get_equiv_circuits_uid():

    if Path(TEMP_FILE).exists():
        os.remove(TEMP_FILE)

    write_test_df()
    df2 = utils.get_circuit_data_batch(TEMP_FILE, 3)
    winner = df2.iloc[3]['unique_key']
    other = df2.iloc[1]['unique_key']
    df2.at[winner, 'in_non_iso_set'] = True
    df2.at[winner, 'no_series'] = True
    df2.at[winner, 'filter'] = True
    df2.at[other, 'equiv_circuit'] = winner
    to_update = ["no_series", "filter",
                 "in_non_iso_set", "equiv_circuit"]
    str_cols = ["equiv_circuit"]

    utils.update_db_from_df(TEMP_FILE, df2, to_update, str_cols)

    df3 = utils.get_equiv_circuits_uid(TEMP_FILE, winner)
    assert df3.shape[0] == 2
    assert df3.iloc[0]['unique_key'] == winner
    assert df3.iloc[1]['equiv_circuit'] == winner

    os.remove(TEMP_FILE)


def test_get_equiv_circuits():

    if Path(TEMP_FILE).exists():
        os.remove(TEMP_FILE)

    write_test_df()
    df2 = utils.get_circuit_data_batch(TEMP_FILE, 3)
    winner = df2.iloc[3]
    uid = winner['unique_key']
    other = df2.iloc[1]['unique_key']
    df2.at[uid, 'in_non_iso_set'] = True
    df2.at[uid, 'no_series'] = True
    df2.at[uid, 'filter'] = True
    df2.at[other, 'equiv_circuit'] = uid
    to_update = ["no_series", "filter",
                 "in_non_iso_set", "equiv_circuit"]
    str_cols = ["equiv_circuit"]
    utils.update_db_from_df(TEMP_FILE, df2, to_update, str_cols)

    df3 = utils.get_equiv_circuits(TEMP_FILE, winner.circuit,
                                   winner.edges)
    assert df3.shape[0] == 2
    assert df3.iloc[0]['unique_key'] == uid
    assert df3.iloc[1]['equiv_circuit'] == uid

    os.remove(TEMP_FILE)


def test_find_circuit_in_db():

    if Path(TEMP_FILE).exists():
        os.remove(TEMP_FILE)

    df = write_test_df()
    winner = df.iloc[3]
    df2 = utils.find_circuit_in_db(TEMP_FILE, winner.circuit,
                                   winner.edges)
    
    assert df2.shape[0] == 1
    assert df2.iloc[0]['unique_key'] == winner['unique_key']

    os.remove(TEMP_FILE)


def test_write_circuit():

    con, cur = write_test_circuit(TEMP_FILE)
    df = pd.read_sql_query("select * from CIRCUITS_3_NODES",
                           con=con)
    for k in df.columns:
        assert df.iloc[0][k] == TEST_ENTRY[k]

    # Cleanup
    con.close()
    os.remove(TEMP_FILE)

    # Write multiple rows
    con, cur = write_test_circuit(TEMP_FILE, reps=10)
    df = pd.read_sql_query("select * from CIRCUITS_3_NODES",
                           con=con)

    for i in range(df.shape[0]):
        for k in df.columns:
            assert df.iloc[i][k] == TEST_ENTRY[k]

    # Cleanup
    con.close()
    os.remove(TEMP_FILE)


def test_list_all_tables():

    if Path(TEMP_FILE).exists():
        os.remove(TEMP_FILE)
    write_test_circuit(TEMP_FILE)
    assert utils.list_all_tables(TEMP_FILE) == ["CIRCUITS_3_NODES"]
    os.remove(TEMP_FILE)


def write_test_circuit(fname: str = TEMP_FILE, reps: int = 1):
    """
    Helper function that writes a single test circuit into
    a temporary database individually for testing

    Args:
        fname (str, optional): file to write to
        reps (int, optional): number of times to write it, for
                              debugging deleting

    Returns:
        connection and cursor objects
    """

    if Path(fname).exists():
        os.remove(fname)

    con = sqlite3.connect(fname)
    cur = con.cursor()
    table_name = "CIRCUITS_3_NODES"
    cur.execute(
           "CREATE TABLE {table} (circuit, graph_index int, edge_counts, \
            unique_key, n_nodes int, base int, no_series int, \
            filter int, in_non_iso_set int, \
            equiv_circuit)".format(table=table_name))

    for i in range(reps):
        utils.write_circuit(cur, TEST_ENTRY, True)

    return con, cur


def write_test_df(fname: str = TEMP_FILE, overwrite: bool = False):
    """
    Helper function that writes a dataframe of circuits into a temporary
    file for testing

    Args:
        fname (str, optional): file to write to
        overwrite (bool, optional): whether to overwrite or duplicate

    Returns:
        dataframe that was written
    """
    # Make test dataframe
    df = pd.DataFrame({"circuit": ALL_CONNECTED_3})
    df['circuit_encoding'] = ["".join(x) for x in ALL_CONNECTED_3_RAW]

    df['graph_index'] = 1
    edges = [utils.graph_index_to_edges(1, 3)]*df.shape[0]
    df['edges'] = edges
    df['n_nodes'] = 3
    df['base'] = 7
    df['unique_key'] = [f"n3_g1_c{str(i).zfill(2)}"
                        for i in range(df.shape[0])]
    df['no_series'] = 0
    df['filter'] = 0
    df['in_non_iso_set'] = 0
    df['equiv_circuit'] = ""

    utils.write_df(TEMP_FILE, df, 3, overwrite=overwrite)

    return df


if __name__ == "__main__":
    # test_gen_param_dict()
    # test_circuit_degree()
    test_find_circuit_in_db()
