import numpy as np
import scqubits as scq

from sircuitenum import qpackage_interface as pi
from sircuitenum import utils

# Make some test circuits
TEST_CIRCUITS = [
    [[(0, 1)], [("L",)]],
    [[(0, 1)], [("L", "J")]],
    [[(0, 1)], [("C", "J", "L")]],
    [[(0, 1), (1, 2), (2, 0)], [("L",), ("J",), ("J",)]],
    [[(0, 1), (1, 2), (2, 3), (2, 4), (3, 4), (4, 0)],
     [("C",), ("C", "J"), ("C", "L"), ("C", "J", "L"),
      ("C", "L"), ("C",)]],
    [[(0, 1), (1, 2), (2, 3), (3, 0),
      (0, 4), (1, 4), (2, 4), (3, 4)],
     [("L",), ("C", "L"), ("L",), ("C", "J", "L"),
      ("L",), ("L",), ("C",), ("J",)]],
    [[(0, 1)], [("C",)]]
    ]


def test_single_edge_loop_knitting():

    # Go through test circuits
    edges, circuit = TEST_CIRCUITS[0][0], TEST_CIRCUITS[0][1]
    c2, e2 = pi.single_edge_loop_kiting(circuit, edges)
    assert c2 == circuit
    assert e2 == edges

    edges, circuit = TEST_CIRCUITS[1][0], TEST_CIRCUITS[1][1]
    c2, e2 = pi.single_edge_loop_kiting(circuit, edges)
    assert c2 == [("J",), ("L",), ("L",)]
    assert e2 == [(0, 1), (0, 2), (2, 1)]

    edges, circuit = TEST_CIRCUITS[2][0], TEST_CIRCUITS[2][1]
    c2, e2 = pi.single_edge_loop_kiting(circuit, edges)
    assert c2 == [("C", "J"), ("L",), ("L",)]
    assert e2 == [(0, 1), (0, 2), (2, 1)]

    edges, circuit = TEST_CIRCUITS[3][0], TEST_CIRCUITS[3][1]
    c2, e2 = pi.single_edge_loop_kiting(circuit, edges)
    assert c2 == circuit
    assert e2 == edges

    edges, circuit = TEST_CIRCUITS[4][0], TEST_CIRCUITS[4][1]
    c2, e2 = pi.single_edge_loop_kiting(circuit, edges)
    assert c2 == [("C",), ("C", "J"), ("C", "L"), ("C", "J"),
                  ("L",), ("L",), ("C", "L"), ("C",)]
    assert e2 == [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5),
                  (5, 4), (3, 4), (4, 0)]

    edges, circuit = TEST_CIRCUITS[5][0], TEST_CIRCUITS[5][1]
    c2, e2 = pi.single_edge_loop_kiting(circuit, edges)
    assert c2 == [("L",), ("C", "L"), ("L",),
                  ("C", "J"), ("L",), ("L",),
                  ("L",), ("L",), ("C",), ("J",)]
    assert e2 == [(0, 1), (1, 2), (2, 3), (3, 0),
                  (3, 5), (5, 0), (0, 4),
                  (1, 4), (2, 4), (3, 4)]

    edges, circuit = TEST_CIRCUITS[6][0], TEST_CIRCUITS[6][1]
    c2, e2 = pi.single_edge_loop_kiting(circuit, edges)
    assert c2 == circuit
    assert e2 == edges


def test_inductive_subgraph():

    # Go through test circuits
    edges, circuit = TEST_CIRCUITS[0][0], TEST_CIRCUITS[0][1]
    e2 = pi.inductive_subgraph(circuit, edges, ind_elem=["J", "L"])
    assert e2 == [(0, 1)]

    edges, circuit = TEST_CIRCUITS[1][0], TEST_CIRCUITS[1][1]
    e2 = pi.inductive_subgraph(circuit, edges, ind_elem=["J", "L"])
    assert e2 == [(0, 1)]

    edges, circuit = TEST_CIRCUITS[2][0], TEST_CIRCUITS[2][1]
    e2 = pi.inductive_subgraph(circuit, edges, ind_elem=["J", "L"])
    assert e2 == [(0, 1)]

    edges, circuit = TEST_CIRCUITS[3][0], TEST_CIRCUITS[3][1]
    e2 = pi.inductive_subgraph(circuit, edges, ind_elem=["J", "L"])
    assert e2 == [(0, 1), (1, 2), (2, 0)]

    edges, circuit = TEST_CIRCUITS[4][0], TEST_CIRCUITS[4][1]
    e2 = pi.inductive_subgraph(circuit, edges, ind_elem=["J", "L"])
    assert e2 == [(1, 2), (2, 3), (2, 4), (3, 4)]

    edges, circuit = TEST_CIRCUITS[5][0], TEST_CIRCUITS[5][1]
    e2 = pi.inductive_subgraph(circuit, edges, ind_elem=["J", "L"])
    assert e2 == [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (3, 4)]

    edges, circuit = TEST_CIRCUITS[6][0], TEST_CIRCUITS[6][1]
    e2 = pi.inductive_subgraph(circuit, edges, ind_elem=["J", "L"])
    assert e2 == []


def test_find_loops():

    # Go through test circuits
    edges, circuit = TEST_CIRCUITS[0][0], TEST_CIRCUITS[0][1]
    loops = pi.find_loops(circuit, edges, ind_elem=["J", "L"])
    assert loops == []

    edges, circuit = TEST_CIRCUITS[1][0], TEST_CIRCUITS[1][1]
    loops = pi.find_loops(circuit, edges, ind_elem=["J", "L"])
    assert loops == [(0, 1)]

    edges, circuit = TEST_CIRCUITS[2][0], TEST_CIRCUITS[2][1]
    loops = pi.find_loops(circuit, edges, ind_elem=["J", "L"])
    assert loops == [(0, 1)]

    edges, circuit = TEST_CIRCUITS[3][0], TEST_CIRCUITS[3][1]
    loops = pi.find_loops(circuit, edges, ind_elem=["J", "L"])
    assert loops == [(0, 1, 2)]

    edges, circuit = TEST_CIRCUITS[4][0], TEST_CIRCUITS[4][1]
    loops = pi.find_loops(circuit, edges, ind_elem=["J", "L"])
    ans = [(2, 3, 4), (2, 4)]
    assert len(loops) == len(ans)
    for loop in ans:
        assert loop in loops

    edges, circuit = TEST_CIRCUITS[5][0], TEST_CIRCUITS[5][1]
    loops = pi.find_loops(circuit, edges, ind_elem=["J", "L"])
    ans = [(0, 1, 4), (0, 3), (0, 3, 4)]
    one_of = [(1, 2, 3, 4), (0, 1, 2, 3)]
    assert len(loops) == len(ans) + 1
    for lp in ans:
        assert lp in loops
    count = 0
    for lp in one_of:
        if lp in loops:
            count += 1
    assert count == 1

    edges, circuit = TEST_CIRCUITS[6][0], TEST_CIRCUITS[6][1]
    loops = pi.find_loops(circuit, edges, ind_elem=["J", "L"])
    assert loops == []

    # Zero pi
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (0, 3), (1, 2)]
    circuit = [("J",), ("L",), ("J",), ("L",), ("C",), ("C",)]
    loops = pi.find_loops(circuit, edges, ind_elem=["J", "L"])
    assert loops == [(0, 1, 2, 3)]

    # non zero numbering
    circuit = [('C',"J"), ("L", "J"), ('J', "C")]
    edges = [(1, 2), (1, 3), (2, 3)]
    loops = pi.find_loops(circuit, edges, ind_elem=["J", "L"])
    ans = [(1, 2, 3), (1, 3)]
    assert len(loops) == len(ans)
    for lp in ans:
        assert lp in loops

def test_add_explicit_ground_node():
    
    elems = {
            'C': {'default_unit': 'GHz', 'default_value': 0.2},
            'L': {'default_unit': 'GHz', 'default_value': 1.0},
            'J': {'default_unit': 'GHz', 'default_value': 15.0},
            'CJ': {'default_unit': 'GHz', 'default_value': 3.0}
        }

    # transmon like
    edges = [(0, 1)]
    circuit = [("C", "J")]
    params=utils.gen_param_dict(circuit, edges, elems)
    ecg = 1000
    new_circuit, new_edges, new_params = pi.add_explicit_ground_node(circuit, edges, params,
                                                                     ecg = ecg)
    assert new_edges == [(1, 2), (0, 1), (0, 2)]
    assert new_circuit == [("C", "J"), ("C",), ("C",)]
    for edge, elem in params:
        assert params[(edge, elem)] == new_params[((edge[0]+1, edge[1]+1), elem)]
    for n in range(1, utils.get_num_nodes(edges)+1):
        assert new_params[((0, n), "C")] == (ecg, "GHz")

    # Delta Circuit
    circuit = [('C',), ('J', 'L'), ('C', 'J')]
    edges = [(1, 2), (0, 2), (0, 1)]
    params = utils.gen_param_dict(circuit, edges, elems)
    new_circuit, new_edges, new_params = pi.add_explicit_ground_node(circuit, edges, params,
                                                                     ecg = ecg)
    assert new_circuit == [('C',), ('J', 'L'), ('C', 'J'), ('C',), ('C',), ('C',)]
    assert new_edges == [(2, 3), (1, 3), (1, 2), (0, 1), (0, 2), (0, 3)]
    for edge, elem in params:
        assert params[(edge, elem)] == new_params[((edge[0]+1, edge[1]+1), elem)]
    for n in range(1, utils.get_num_nodes(edges)+1):
        assert new_params[((0, n), "C")] == (ecg, "GHz")


def test_swap_nodes():

    edges = [(0, 1)]
    new_edges = pi.swap_nodes(edges, 0, 1)
    assert new_edges == [(1, 0)]

    
    edges = [(0, 1), (1, 2), (2, 0)]
    new_edges = pi.swap_nodes(edges, 2, 1)
    assert new_edges == [(0, 2), (2, 1), (1, 0)]



def test_to_SQcircuit():

    # See if we get the right frequency
    # and anharmonicity for a Transmon
    elems = {
            'C': {'default_unit': 'GHz', 'default_value': 0.2},
            'L': {'default_unit': 'GHz', 'default_value': 1.0},
            'J': {'default_unit': 'GHz', 'default_value': 15.0},
            'CJ': {'default_unit': 'GHz', 'default_value': 0}
        }
    edges = [(0, 1)]
    circuit = [("C", "J")]
    obj = pi.to_SQcircuit(circuit, edges,
                          params=utils.gen_param_dict(circuit, edges, elems),
                          ground_node=0)
    ev, es = obj.diag(3)
    w01 = ev[1] - ev[0]
    w12 = ev[2] - ev[1]
    Ec = elems['C']['default_value']
    Ej = elems['J']['default_value']
    zeta = np.sqrt(2*Ec/Ej)
    good_freq = np.sqrt(8*Ec*Ej) - Ec*(1+9*(2**-2)*zeta)
    good_anharm = Ec*(1+9*(2**-4)*zeta)
    assert abs(1 - abs(w01/good_freq)) <= 0.025
    assert abs(1 - abs(w12 - w01)/good_anharm) <= 0.025

    # Fluxonium comparing to my numerical
    # simulation
    # E0 = 0 GHz/h
    # E1 = 0.34 GHz/h
    # E2 = 4.35 GHz/h
    # EJ = 5 GHz/h
    # Ec = El = 1 GHz/h
    elems['C']['default_value'] = 1
    elems['J']['default_value'] = 5
    edges = [(0, 1)]
    circuit = [("C", "J", "L")]

    obj = pi.to_SQcircuit(circuit, edges,
                          params=utils.gen_param_dict(circuit, edges, elems),
                          ground_node=0)
    obj.loops[0].set_flux(0.5)
    ev, es = obj.diag(3)
    ev = ev - ev[0]
    good_vals = [0, 0.34, 4.35]
    for i in range(1, 3):
        assert 1 - abs(ev[i]/good_vals[i]) <= 0.025

    # Try zero pi much more complicated example
    # From https://docs.sqcircuit.org/examples/zeropi_qubit.html
    elems["C"]["default_value"] = 0.15
    elems["CJ"]["default_value"] = 10.0
    elems["J"]["default_value"] = 5.0
    elems["L"]["default_value"] = 0.13
    edges = [(0, 1), (1, 3), (2, 3), (0, 2), (0, 3), (1, 2)]
    circuit = [("J",), ("L",), ("J",), ("L",), ("C",), ("C",)]
    obj = pi.to_SQcircuit(circuit, edges, [35, 6],
                          params=utils.gen_param_dict(circuit, edges, elems),
                          ground_node=0)
    obj.loops[0].set_flux(0.5)
    ev, es = obj.diag(5)
    ev = ev - ev[0]
    good_vals = [0., 0.02479347, 1.28957426, 1.58410963, 2.18419302]
    for i in range(1, 5):
        assert 1 - abs(ev[i]/good_vals[i]) <= 0.025


def test_to_SCqubits():

    # See if we get the right frequency
    # and anharmonicity for a Transmon
    elems = {
            'C': {'default_unit': 'GHz', 'default_value': 0.2},
            'L': {'default_unit': 'GHz', 'default_value': 1.0},
            'J': {'default_unit': 'GHz', 'default_value': 15.0},
            'CJ': {'default_unit': 'GHz', 'default_value': 500.0}
            }
    edges = [(0, 1)]
    circuit = [("C", "J")]
    obj = pi.to_SCqubits(circuit, edges,
                         params=utils.gen_param_dict(circuit, edges, elems))
    ev, es = obj.eigensys(evals_count=3)
    w01 = ev[1] - ev[0]
    w12 = ev[2] - ev[1]
    Ec = elems['C']['default_value']
    Ej = elems['J']['default_value']
    zeta = np.sqrt(2*Ec/Ej)
    good_freq = np.sqrt(8*Ec*Ej) - Ec*(1+9*(2**-2)*zeta)
    good_anharm = Ec*(1+9*(2**-4)*zeta)

    assert abs(1 - abs(w01/good_freq)) <= 0.025
    assert abs(1 - abs(w12 - w01)/good_anharm) <= 0.025

    # Fluxonium comparing to my numerical
    # simulation
    # E0 = 0 GHz/h
    # E1 = 0.34 GHz/h
    # E2 = 4.35 GHz/h
    # EJ = 5 GHz/h
    # Ec = El = 1 GHz/h
    elems['C']['default_value'] = 1
    elems['J']['default_value'] = 5
    edges = [(0, 1)]
    circuit = [("C", "J", "L")]

    obj = pi.to_SCqubits(circuit, edges,
                         params=utils.gen_param_dict(circuit, edges, elems))
    obj.Φ1 = 0.5
    ev, es = obj.eigensys(evals_count=3)
    ev = ev - ev[0]
    good_vals = [0, 0.34, 4.35]
    for i in range(1, 3):
        assert 1 - abs(ev[i]/good_vals[i]) <= 0.025

    # Try zero pi much more complicated example
    # From https://docs.sqcircuit.org/examples/zeropi_qubit.html
    elems["C"]["default_value"] = 0.15
    elems["CJ"]["default_value"] = 10.0
    elems["J"]["default_value"] = 5.0
    elems["L"]["default_value"] = 0.13
    edges = [(0, 1), (1, 3), (2, 3), (0, 2), (0, 3), (1, 2)]
    circuit = [("J",), ("L",), ("J",), ("L",), ("C",), ("C",)]
    obj = pi.to_SCqubits(circuit, edges, 10,
                         params=utils.gen_param_dict(circuit, edges, elems))
    obj.Φ1 = 0.5
    system_hierarchy = [[1, 3], [2]]
    scq.truncation_template(system_hierarchy)
    obj.configure(system_hierarchy=system_hierarchy,
                  subsystem_trunc_dims=[35, 6])
    ev = obj.subsystems[0].eigenvals()
    ev = ev - ev[0]
    good_vals = [0., 0.02479347, 1.28957426, 1.58410963, 2.18419302]
    for i in range(1, 5):
        assert 1 - abs(ev[i]/good_vals[i]) <= 0.025

    # Case that broke with the truncation mode setting
    # This is just to test there is no error
    circuit, edges = ([('C',), ('J',)], [(0, 2), (1, 2)])
    obj = pi.to_SCqubits(circuit, edges, 10,
                         params=utils.gen_param_dict(circuit, edges, elems))


if __name__ == "__main__":

    # test_to_SQcircuit()
    # test_to_SCqubits()
    # test_to_CircuitQ()
    # test_to_Qucat()
    # test_swap_nodes()
    test_find_loops()
