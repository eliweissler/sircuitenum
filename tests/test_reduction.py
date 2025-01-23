import itertools

import numpy as np
import pandas as pd

from sircuitenum import reduction as red
from sircuitenum import enumeration as enum
from sircuitenum import utils


ALL_CONNECTED_3 = [[utils.ENUM_PARAMS["CHAR_TO_COMBINATION"][c] for c in
                    np.base_repr(i, 3).zfill(3)] for i in range(27)]
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



def test_convert_circuit_to_component_graph():

    # Simple example
    edges = [(0, 1)]
    circuit = [("C",)]
    G = red.convert_circuit_to_component_graph(circuit, edges)
    exp_nodes = ["v0", "v1", "C0"]
    exp_edges = [("v0", "C0"), ("C0", "v1")]
    assert len(G.nodes) == len(exp_nodes)
    assert len(G.edges) == len(exp_edges)
    assert all(x in G.nodes for x in exp_nodes)
    assert all(x in G.edges for x in exp_edges)

    # Simple example with ground
    edges = [(0, 1)]
    circuit = [("C",)]
    G = red.convert_circuit_to_component_graph(circuit, edges, ground_nodes = [0])
    exp_nodes = ["GND", "v1", "C0"]
    exp_edges = [("GND", "C0"), ("C0", "v1")]
    assert len(G.nodes) == len(exp_nodes)
    assert len(G.edges) == len(exp_edges)
    assert all(x in G.nodes for x in exp_nodes)
    assert all(x in G.edges for x in exp_edges)

    # More complicated example
    edges = [(0, 1), (1, 2), (2, 0)]
    circuit = [("C",), ("L",), ("C", "L", "J")]
    G = red.convert_circuit_to_component_graph(circuit, edges)
    exp_nodes = ['v0', 'v1', 'v2',
                 'C0', 'L0', 'CJL0']
    exp_edges = [
                 ('v0', 'C0'), ('v0', 'CJL0'),
                 ('v1', 'C0'), ('v1', 'L0'),
                 ('v2', 'L0'), ('v2', 'CJL0')
                ]
    assert len(G.nodes) == len(exp_nodes)
    assert len(G.edges) == len(exp_edges)
    assert all(x in G.nodes for x in exp_nodes)
    assert all(x in G.edges for x in exp_edges)


    # More complicated example with grounding
    edges = [(0, 1), (1, 2), (2, 0)]
    circuit = [("C",), ("L",), ("C", "L", "J")]
    G = red.convert_circuit_to_component_graph(circuit, edges, ground_nodes = [0,2])
    exp_nodes = ['GND', 'v1', 'C0', 'L0']
    exp_edges = [
                 ('GND', 'C0'),
                 ('v1', 'C0'), ('v1', 'L0'),
                 ('GND', 'L0')
                ]
    assert len(G.nodes) == len(exp_nodes)
    assert len(G.edges) == len(exp_edges)
    assert all(x in G.nodes for x in exp_nodes)
    assert all(x in G.edges for x in exp_edges)


def test_convert_circuit_to_port_graph():

    # Simple example
    edges = [(0, 1)]
    circuit = [("C",)]
    G = red.convert_circuit_to_port_graph(circuit, edges)
    exp_nodes = ["v0_p0", "v1_p0", "C0_p0", "C0_p1"]
    exp_edges = [("v0_p0", "C0_p0"), ("C0_p0", "C0_p1"), ("C0_p1", "v1_p0")]
    assert len(G.nodes) == len(exp_nodes)
    assert len(G.edges) == len(exp_edges)
    assert all(x in G.nodes for x in exp_nodes)
    assert all(x in G.edges for x in exp_edges)

    # Simple example with ground
    edges = [(0, 1)]
    circuit = [("C",)]
    G = red.convert_circuit_to_port_graph(circuit, edges, ground_nodes=[0])
    exp_nodes = ["v0_p0_GND", "v1_p0", "C0_p0", "C0_p1"]
    exp_edges = [("v0_p0_GND", "C0_p0"), ("C0_p0", "C0_p1"), ("C0_p1", "v1_p0")]
    assert len(G.nodes) == len(exp_nodes)
    assert len(G.edges) == len(exp_edges)
    assert all(x in G.nodes for x in exp_nodes)
    assert all(x in G.edges for x in exp_edges)

    # More complicated example
    edges = [(0, 1), (1, 2), (2, 0)]
    circuit = [("C",), ("L",), ("C", "L", "J")]
    G = red.convert_circuit_to_port_graph(circuit, edges)
    exp_nodes = ['v0_p0', 'v0_p1', 'v1_p0', 'v1_p1', 'v2_p0',
                 'v2_p1', 'C0_p0', 'C0_p1', 'L0_p0', 'L0_p1',
                 'CJL0_p0', 'CJL0_p1']
    exp_edges = [
                 ('v0_p0', 'v0_p1'), ('v0_p0', 'C0_p0'), ('v0_p1', 'CJL0_p1'),
                 ('v1_p0', 'v1_p1'), ('v1_p0', 'C0_p1'), ('v1_p1', 'L0_p0'),
                 ('v2_p0', 'v2_p1'), ('v2_p0', 'L0_p1'), ('v2_p1', 'CJL0_p0'),
                 ('C0_p0', 'C0_p1'), ('L0_p0', 'L0_p1'), ('CJL0_p0', 'CJL0_p1')
                ]
    assert len(G.nodes) == len(exp_nodes)
    assert len(G.edges) == len(exp_edges)
    assert all(x in G.nodes for x in exp_nodes)
    assert all(x in G.edges for x in exp_edges)


    # More complicated example with ground
    edges = [(0, 1), (1, 2), (2, 0)]
    circuit = [("C",), ("L",), ("C", "L", "J")]
    G = red.convert_circuit_to_port_graph(circuit, edges, ground_nodes=[0,2])
    exp_nodes = ['v0_p0_GND', 'v0_p1_GND', 'v1_p0', 'v1_p1', 'v2_p0_GND',
                 'v2_p1_GND', 'C0_p0', 'C0_p1', 'L0_p0', 'L0_p1']
    exp_edges = [
                 ('v0_p0_GND', 'C0_p0'),
                 ('v1_p0', 'v1_p1'), ('v1_p0', 'C0_p1'), ('v1_p1', 'L0_p0'),
                 ('v2_p0_GND', 'L0_p1'),
                 ('C0_p0', 'C0_p1'), ('L0_p0', 'L0_p1')
                ]
    gnd_ports = ('v0_p0_GND', 'v0_p1_GND') + ('v2_p0_GND', 'v2_p1_GND')
    for p1 in gnd_ports:
        for p2 in gnd_ports:
            if p1 < p2:
                exp_edges.append((p1, p2))
    assert len(G.nodes) == len(exp_nodes)
    assert len(G.edges) == len(exp_edges)
    assert all(x in G.nodes for x in exp_nodes)
    assert all(x in G.edges for x in exp_edges)


def test_remove_series_elems():

    # Test a few obvious cases
    edges = [(0, 1), (1, 2), (2, 0)]
    circuit = [("L",), ("L",), ("L",)]
    c2, e2 = red.remove_series_elems(circuit, edges)
    assert e2 == [(0, 1)]
    assert c2 == [("L",)]

    edges = [(0, 1), (1, 2)]
    circuit = [("C",), ("L",)]
    c2, e2 = red.remove_series_elems(circuit, edges)
    assert e2 == edges
    assert c2 == circuit

    edges = [(0, 1), (1, 2), (2, 0)]
    circuit = [("C",), ("C",), ("L",)]
    c2, e2 = red.remove_series_elems(circuit, edges)
    assert e2 == [(0, 1)]
    assert c2 == [("C", "L")]

    # Some larger ones -- test the 2,3/3,4 connection
    edges = [(0, 1), (1, 2), (2, 3), (2, 4), (3, 4)]
    circuit = [("C",), ("J",), ("C", "J"), ("L", "J"), ("L",)]
    c2, e2 = red.remove_series_elems(circuit, edges)
    assert e2 == edges
    assert c2 == circuit

    edges = [(0, 1), (1, 2), (2, 3), (2, 4), (3, 4)]
    circuit = [("C",), ("J",), ("L",), ("J", "L"), ("L",)]
    c2, e2 = red.remove_series_elems(circuit, edges)
    assert e2 == [(0, 1), (1, 2), (2, 3)]
    assert c2 == [("C",), ("J",), ("J", "L")]

    edges = [(0, 1), (1, 2), (2, 3), (2, 4), (3, 4)]
    circuit = [("C",), ("J",), ("C", "J"), ("J", "L"), ("J", "L")]
    c2, e2 = red.remove_series_elems(circuit, edges)
    assert e2 == edges
    assert c2 == circuit

    # Test the full set of fully connected three nodes
    edges = [(0, 1), (1, 2), (2, 0)]

    n_series = 0
    n_no_series = 0
    for c in ALL_CONNECTED_3:
        c2, e2 = red.remove_series_elems(c, edges)
        if utils.circuit_in_set(c, NON_SERIES_3):
            assert utils.get_num_nodes(e2) == 3
            n_no_series += 1
        else:
            assert utils.get_num_nodes(e2) == 2
            n_series += 1
    assert n_no_series == len(NON_SERIES_3)
    assert n_series == len(ALL_CONNECTED_3) - len(NON_SERIES_3)


def test_mark_non_isomorphic_set():

    # Testing starting point/to_consider
    edges = [[(0, 1), (1, 2), (2, 0)]]*6
    circuit = list(itertools.permutations([("C",), ("L",), ("J",)]))
    df = pd.DataFrame({"edges": edges, "circuit": circuit,
                       "in_non_iso_set": False, "equiv_circuit": "",
                       "unique_key": np.arange(len(circuit))})
    to_consider = np.ones(df.shape[0], dtype=bool)
    to_consider[0:2] = False
    to_consider[-1] = False
    red.mark_non_isomorphic_set(df, to_consider=to_consider)
    filt = df[df['in_non_iso_set'] == 1]
    bad = df[df['in_non_iso_set'] == 0]
    assert filt.iloc[0]['unique_key'] == 2
    assert np.sum(bad['equiv_circuit'] != "") == 2

    # Case 1:
    edges = [[(0, 1), (1, 2), (2, 0)]]*6
    circuit = list(itertools.permutations([("C",), ("L",), ("J",)]))
    df = pd.DataFrame({"edges": edges, "circuit": circuit,
                       "in_non_iso_set": False, "equiv_circuit": "",
                       "unique_key": np.arange(len(circuit))})
    red.mark_non_isomorphic_set(df)
    filt = df[df['in_non_iso_set'] == 1]
    bad = df[df['in_non_iso_set'] == 0]
    assert filt.shape[0] == 1
    assert filt.edges.iloc[0] == [(0, 1), (1, 2), (2, 0)]
    assert utils.circuit_in_set(filt.circuit.iloc[0], circuit)
    assert all([x in filt['unique_key']
                for x in bad['equiv_circuit'].values])

    # Case 2:
    edges = [[(0, 1), (1, 2), (2, 0)]]*6
    circuit = list(itertools.permutations([("L", "J"),
                                           ("C", "J"),
                                           ("C", "L", "J")]))
    df = pd.DataFrame({"edges": edges, "circuit": circuit,
                       "in_non_iso_set": False, "equiv_circuit": "",
                       "unique_key": np.arange(len(circuit))})
    red.mark_non_isomorphic_set(df)
    filt = df[df['in_non_iso_set'] == 1]
    bad = df[df['in_non_iso_set'] == 0]
    assert filt.shape[0] == 1
    assert filt.edges.iloc[0] == [(0, 1), (1, 2), (2, 0)]
    assert utils.circuit_in_set(filt.circuit.iloc[0], circuit)
    assert all([x in filt['unique_key']
                for x in bad['equiv_circuit'].values])

    # Case 3:
    edges = [[(0, 1), (1, 2), (2, 0)]]*6
    circuit = list(itertools.permutations([("J", "L"),
                                           ("J", "C"),
                                           ("C", "L", "J")]))
    df = pd.DataFrame({"edges": edges, "circuit": circuit,
                       "in_non_iso_set": False, "equiv_circuit": "",
                       "unique_key": np.arange(len(circuit))})
    red.mark_non_isomorphic_set(df)
    filt = df[df['in_non_iso_set'] == 1]
    bad = df[df['in_non_iso_set'] == 0]
    assert filt.shape[0] == 1
    assert filt.edges.iloc[0] == [(0, 1), (1, 2), (2, 0)]
    assert utils.circuit_in_set(filt.circuit.iloc[0], circuit)
    assert all([x in filt['unique_key']
                for x in bad['equiv_circuit'].values])

    # Test the full set of fully connected three nodes
    edges = [[(0, 1), (1, 2), (2, 0)]]*len(NON_SERIES_3)
    df = pd.DataFrame({"edges": edges, "circuit": NON_SERIES_3,
                       "in_non_iso_set": False, "equiv_circuit": "",
                       "unique_key": np.arange(len(NON_SERIES_3))})
    red.mark_non_isomorphic_set(df)
    filt = df[df['in_non_iso_set'] == 1]
    bad = df[df['in_non_iso_set'] == 0]
    assert filt.shape[0] == len(NON_ISOMORPHIC_3)
    assert all(utils.circuit_in_set(c, NON_ISOMORPHIC_3)
               for c in filt.circuit.values)
    assert all([x in filt['unique_key']
                for x in bad['equiv_circuit'].values])


def test_full_reduction():

    # Test some obvious cases
    edges = [[(0, 1), (1, 2), (2, 0)]]*6
    circuit = list(itertools.permutations([("C",), ("L",), ("J",)]))
    df = pd.DataFrame({"edges": edges, "circuit": circuit,
                       "in_non_iso_set": False, "equiv_circuit": "",
                       "unique_key": np.arange(len(circuit)),
                       "graph_index": 1, 'filter': False,
                       "no_series": False})
    red.full_reduction(df)
    filt = df[df['in_non_iso_set'] == 1]
    bad = df[df['in_non_iso_set'] == 0]
    assert filt.shape[0] == 1
    assert filt.edges.iloc[0] == [(0, 1), (1, 2), (2, 0)]
    assert utils.circuit_in_set(filt.circuit.iloc[0], circuit)
    assert all([x in filt['unique_key']
                for x in bad['equiv_circuit'].values])

    # Test the full set of fully connected three nodes
    edges = [[(0, 1), (1, 2), (2, 0)]]*len(ALL_CONNECTED_3)
    df = pd.DataFrame({"edges": edges, "circuit": ALL_CONNECTED_3,
                       "in_non_iso_set": False, "equiv_circuit": "",
                       "unique_key": np.arange(len(ALL_CONNECTED_3)),
                       "graph_index": 1, 'filter': False,
                       "no_series": False})
    red.full_reduction(df)
    filt = df[df['in_non_iso_set'] == 1]
    bad = df[df['in_non_iso_set'] == 0]

    # Make sure final set is right
    assert filt.shape[0] == len(NON_ISOMORPHIC_3)
    assert all(red.isomorphic_circuit_in_set(c, edges[0], NON_ISOMORPHIC_3)
               for c in filt.circuit.values)
    assert all(red.isomorphic_circuit_in_set(c, edges[0], filt.circuit.values)
               for c in NON_ISOMORPHIC_3)
    # Make sure equivalent circuits are right
    for i, row in df.iterrows():
        if row['equiv_circuit'] != "" and row['no_series']:
            c1 = row['circuit']
            e1 = row['edges']
            c2 = df.iloc[row['equiv_circuit']]['circuit']
            assert df.iloc[row['equiv_circuit']]['in_non_iso_set']
            assert red.isomorphic_circuit_in_set(c1, e1, [c2])

    # Test a small set of 4 node Y circuits
    edges = [[(0, 1), (0, 2), (0, 3)]]*4 + [[(0, 1), (1, 2), (0, 3)]]
    circuit = [[("L",), ("L",), ("L",)],
               [("L",), ("L",), ("C",)],
               [("L",), ("C",), ("J",)],
               [("L",), ("L",), ("L",)],
               [("L",), ("L",), ("J",)]]
    df = pd.DataFrame({"edges": edges, "circuit": circuit,
                       "in_non_iso_set": False, "equiv_circuit": "",
                       "unique_key": np.arange(len(circuit)),
                       "graph_index": 1, 'filter': False,
                       "no_series": False})
    red.full_reduction(df)
    filt = df[df['in_non_iso_set'] == 1]
    bad = df[df['in_non_iso_set'] == 0]

    assert all(df['filter'] == np.array([False, False, True,
                                         False, True]))
    assert all(df['no_series'] == np.array([True, True, True,
                                            True, False]))
    assert all(df['in_non_iso_set'] == np.array([True, True, True,
                                                 False, False]))
    assert filt.shape[0] == 3
    # Make sure final set is right
    assert all(red.isomorphic_circuit_in_set(c, edges[0], [circuit[0],
                                                           circuit[1],
                                                           circuit[2]])
               for c in filt.circuit.values)
    # Make sure equivalent circuits are right
    for i, row in df.iterrows():
        if row['equiv_circuit'] != "" and row['no_series']:
            c1 = row['circuit']
            e1 = row['edges']
            c2 = df.iloc[row['equiv_circuit']]['circuit']
            assert df.iloc[row['equiv_circuit']]['in_non_iso_set']
            assert red.isomorphic_circuit_in_set(c1, e1, [c2])


def test_full_reduction_by_group():

    # Test the four node cycle circuit
    graph_index = 3
    G = utils.get_basegraphs(4)[graph_index]
    all_circuits = enum.generate_for_specific_graph(7, G, graph_index,
                                                    return_vals=True)
    utils.convert_loaded_df(all_circuits, 4)
    df = red.full_reduction_by_group(all_circuits)
    filt = df[(df['in_non_iso_set'] == 1) &
              (df['filter'] == 1) &
              (df['no_series'] == 1)]
    filt2 = df[(df['in_non_iso_set'] == 1) &
               np.logical_not(df['filter'] == 1) &
               (df['no_series'] == 1)]
    filt3 = df[np.logical_not(df['no_series'] == 1)]

    # A few hand picked examples -- make sure they're there/not
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    circuit = [("C",), ("L",), ("J",), ("L",)]
    assert red.isomorphic_circuit_in_set(circuit, edges, filt.circuit.values,
                                         filt.edges.values)

    # No JJ
    circuit = [("C",), ("L",), ("C",), ("L",)]
    assert red.isomorphic_circuit_in_set(circuit, edges, filt.circuit.values,
                                         filt.edges.values) is False
    assert red.isomorphic_circuit_in_set(circuit, edges, filt2.circuit.values,
                                         filt2.edges.values)
    circuit = [("C",), ("C", "L",), ("C", "L"), ("L",)]
    assert red.isomorphic_circuit_in_set(circuit, edges, filt.circuit.values,
                                         filt.edges.values) is False
    assert red.isomorphic_circuit_in_set(circuit, edges, filt2.circuit.values,
                                         filt2.edges.values)

    # Consecutive linear components
    circuit = [("C",), ("J",), ("L",), ("L",)]
    assert red.isomorphic_circuit_in_set(circuit, edges, filt.circuit.values,
                                         filt.edges.values) is False
    assert red.isomorphic_circuit_in_set(circuit, edges, filt3.circuit.values,
                                         filt3.edges.values)
    circuit = [("C",), ("J",), ("L",), ("J", "L")]
    assert red.isomorphic_circuit_in_set(circuit, edges, filt.circuit.values,
                                         filt.edges.values)
    assert red.isomorphic_circuit_in_set(circuit, edges, filt3.circuit.values,
                                         filt3.edges.values) is False
    circuit = [("C",), ("J",), ("L",), ("C", "L")]
    assert red.isomorphic_circuit_in_set(circuit, edges, filt.circuit.values,
                                         filt.edges.values)
    assert red.isomorphic_circuit_in_set(circuit, edges, filt3.circuit.values,
                                         filt3.edges.values) is False

    # Set that should be isomorphic, only one should be there
    circuit_set = []
    circuit_set.append([("J",), ("J",), ("J",), ("C",)])
    circuit_set.append([("J",), ("J",), ("C",), ("J",)])
    circuit_set.append([("J",), ("C",), ("J",), ("J",)])
    circuit_set.append([("C",), ("J",), ("J",), ("J",)])
    assert sum(utils.circuit_in_set(c, filt.circuit.values)
               for c in circuit_set) == 1
    assert red.isomorphic_circuit_in_set(circuit, edges,
                                         filt.circuit.values,
                                         filt.edges.values)


def test_isomorphic_circuit_in_set():

    # Test some obvious cases
    edges = [(0, 1), (1, 2), (2, 0)]
    circuit = list(itertools.permutations([("C",), ("L",), ("J",)]))
    test_c = circuit[0]
    assert red.isomorphic_circuit_in_set(test_c, edges, circuit)
    assert red.isomorphic_circuit_in_set([("C",), ("J",), ("J",)],
                                         edges, circuit) is False
    assert red.isomorphic_circuit_in_set(test_c, edges,
                                         circuit,
                                         [edges]*len(circuit))
    assert red.isomorphic_circuit_in_set(test_c, edges,
                                         circuit,
                                         [edges]*len(circuit),
                                         return_index=True) == 0
    assert red.isomorphic_circuit_in_set([("C",), ("J",), ("J",)],
                                         edges, circuit,
                                         [edges]*len(circuit)) is False

    assert np.isnan(red.isomorphic_circuit_in_set([("C",), ("J",), ("J",)],
                                                  edges, circuit,
                                                  [edges]*len(circuit),
                                                  return_index=True))


if __name__ == "__main__":
    test_convert_circuit_to_component_graph()
    test_convert_circuit_to_port_graph()
