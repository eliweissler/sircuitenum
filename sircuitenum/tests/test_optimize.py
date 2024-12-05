import  os
num_cores = "1"
os.environ["OPENBLAS_NUM_THREADS"] = num_cores
os.environ["OMP_NUM_THREADS"] = num_cores
os.environ["MKL_NUM_THREADS"] = num_cores

import scqubits as scq
import SQcircuit as sq
import numpy as np


from sircuitenum import optimize as optim
from sircuitenum import utils
from sircuitenum import qpackage_interface as qpi




def test_make_sq():
    
    # Try zero pi much more complicated example
    # From https://docs.sqcircuit.org/examples/zeropi_qubit.html
    edges = [(0, 1), (1, 3), (2, 3), (0, 2), (0, 3), (1, 2)]
    circuit = [("J",), ("L",), ("J",), ("L",), ("C",), ("C",)]
    EC = 0.15
    EL = 0.13
    EJ = 5.0
    CJ = 10.0
    obj = optim.make_sq(circuit, edges, param_sets=[EJ, EL, EJ, EL, EC, EC, 0.5, 0], cj=CJ,
                        ground_node=0)
    ev, es = obj.diag(5)
    ev = ev - ev[0]
    good_vals = [0., 0.02479347, 1.28957426, 1.58410963, 2.18419302]
    for i in range(1, 5):
        assert 1 - abs(ev[i]/good_vals[i]) <= 0.025


def test_calc_decay_rates_sq():

    # From https://docs.sqcircuit.org/examples/fluxonium.html#Decoherence-Rates
    # define the circuit elements
    loop1 = sq.Loop()
    C = sq.Capacitor(3.6, 'GHz',Q=1e6 ,error=10)
    L = sq.Inductor(0.46,'GHz',Q=500e6 ,loops=[loop1])
    JJ = sq.Junction(10.2,'GHz',cap =C , A=1e-7, x=3e-06, loops=[loop1])

    # define the circuit
    elements = {
    (0, 1): [L, JJ]
    }

    cr = sq.Circuit(elements, flux_dist='all')
    cr.set_trunc_nums([60])
    n_eig = 5

    phiExt = np.array([0.4])

    decays = {'capacitive':np.zeros_like(phiExt),
                'inductive':np.zeros_like(phiExt),
                'cc':np.zeros_like(phiExt),
                'quasiparticle':np.zeros_like(phiExt)}

    spec = np.zeros((n_eig, len(phiExt)))

    for i, phi in enumerate(phiExt):
        loop1.set_flux(phi)
        spec[:, i], _ = cr.diag(n_eig)
        for dec_type in decays:
            decays[dec_type][i]=cr.dec_rate(dec_type=dec_type, states=(1,0))

    rates = optim.calc_decay_rates_sq(cr)
    assert rates["depolarization"]["capacitive"] == decays["capacitive"]
    assert rates["depolarization"]["inductive"] == decays["inductive"]
    assert rates["depolarization"]["quasiparticle"] == decays["quasiparticle"]
    assert rates["dephasing"]["cc"] == decays["cc"]




def test_make_sc():

    # Try zero pi more complicated example
    # From https://docs.sqcircuit.org/examples/zeropi_qubit.html
    edges = [(0, 1), (1, 3), (2, 3), (0, 2), (0, 3), (1, 2)]
    circuit = [("J",), ("L",), ("J",), ("L",), ("C",), ("C",)]
    EC = 0.15
    EL = 0.13
    EJ = 5.0
    CJ = 10.0
    obj = optim.make_sc(circuit, edges, param_sets=[EJ, EL, EJ, EL, EC, EC, 0.5, 0], cj=CJ,
                        ground_node=None)
    system_hierarchy = [[1, 3], [2]]
    obj.configure(system_hierarchy=system_hierarchy,
                  subsystem_trunc_dims=[35, 6])
    ev = obj.subsystems[0].eigenvals()
    ev = ev - ev[0]
    good_vals = [0., 0.02479347, 1.28957426, 1.58410963, 2.18419302]
    for i in range(1, 5):
        assert 1 - abs(ev[i]/good_vals[i]) <= 0.025

    

def test_calc_decay_rates_sc():

    # from https://scqubits.readthedocs.io/en/v4.0/guide/circuit/ipynb/custom_circuit_coherence_estimates.html
    snail_yaml = """
                branches:
                - [JJ, 0, 1, 90, EC1=0.2]
                - [JJ, 1, 2, 90, EC3=0.2]
                - [JJ, 2, 3, 90, 0.2]
                - [JJ, 3, 0, 15, 0.2]
                - [C, 3, 0, 0.3]
                """
    circ = scq.Circuit(snail_yaml, from_file=False, use_dynamic_flux_grouping=True, generate_noise_methods=True)

    rates = optim.calc_decay_rates_sc(circ)

    assert abs(1e-09*rates["tphi"]["1_over_f_cc"] - 1/2308488.4887882355 < 1e-06)
    
    # t1_cap = circ.t1_capacitive(2,1, total=False, T=0.100, Q_cap=1e6)
    # tphi_1_over_f_cc = circ.tphi_1_over_f_cc()

    

def test_decoherence_time():

    # from https://scqubits.readthedocs.io/en/v4.0/guide/circuit/ipynb/custom_circuit_coherence_estimates.html
    snail_yaml = """
                branches:
                - [JJ, 0, 1, 90, EC1=0.2]
                - [JJ, 1, 2, 90, EC3=0.2]
                - [JJ, 2, 3, 90, 0.2]
                - [JJ, 3, 0, 15, 0.2]
                - [C, 3, 0, 0.3]
                """
    circ = scq.Circuit(snail_yaml, from_file=False, use_dynamic_flux_grouping=True, generate_noise_methods=True)

    rates = optim.calc_decay_rates_sc(circ)

    tphi = optim.decoherence_time(rates, t_phi_channels=["1_over_f_cc"])[1]

    assert abs(tphi*1e09 - 2308488.4887882355 < 1e-06)

     # From https://docs.sqcircuit.org/examples/fluxonium.html#Decoherence-Rates
    # define the circuit elements
    loop1 = sq.Loop()
    C = sq.Capacitor(3.6, 'GHz',Q=1e6 ,error=10)
    L = sq.Inductor(0.46,'GHz',Q=500e6 ,loops=[loop1])
    JJ = sq.Junction(10.2,'GHz',cap =C , A=1e-7, x=3e-06, loops=[loop1])

    # define the circuit
    elements = {
    (0, 1): [L, JJ]
    }

    cr = sq.Circuit(elements, flux_dist='all')
    cr.set_trunc_nums([60])
    n_eig = 5

    phiExt = np.array([0.4])

    decays = {'capacitive':np.zeros_like(phiExt),
                'inductive':np.zeros_like(phiExt),
                'cc':np.zeros_like(phiExt),
                'quasiparticle':np.zeros_like(phiExt)}

    spec = np.zeros((n_eig, len(phiExt)))

    for i, phi in enumerate(phiExt):
        loop1.set_flux(phi)
        spec[:, i], _ = cr.diag(n_eig)
        for dec_type in decays:
            decays[dec_type][i]=cr.dec_rate(dec_type=dec_type, states=(1,0))

    rates = optim.calc_decay_rates_sq(cr)
    times = optim.decoherence_time(rates)

    t1 = 1/(sum(rates["depolarization"].values()))
    tphi = 1/(sum(rates["dephasing"].values()))
    t2 = (1/(1/(2*t1)+1/(tphi)))

    assert abs(times[0] - t1) < 1e-12
    assert abs(times[1] - tphi) < 1e-12
    assert abs(times[2] - t2) < 1e-12


def test_get_gate_time():
    
    eps = 1e-04

    # Limiting cases
    assert abs(optim.get_gate_time(4, 20)*1e09 - 0.997356*5) < eps
    assert abs(optim.get_gate_time(1, 1)*1e09 - 59.841342*5) < eps
    assert abs(optim.get_gate_time(0.00001, 1)*1e09 - 59.841342*5) < eps

    # delta = 0.1
    assert abs(optim.get_gate_time(5, 5.1)*1e09 - 3.9894228*5) < eps

def test_get_anharmonicity():
    
    vals = [0, 0.34, 4.35]
    assert np.abs(optim.get_anharmonicity(vals) - ((vals[2] - vals[1]) - (vals[1] - vals[0]))) < 1e-09

def test_get_ngate_mc():

    # Optimized point
    # circuit = [('J',), ('L',), ('C', 'L')]
    # edges = [(0, 1), (0, 2), (1, 2)]
    # params = [28.83436338, 0.10343485, 1.36334654, 0.05167763, 0.0]
    # # trunc_num = [150, 80]
    # trunc_num = [65, 95]

    # ground_node = None
    # offset_integer = True
    # cj = 10.0
    # ntrial = 1
    # amp_param = 0
    # amp_offset = 0
    # return_std = False
    # workers = 1
    # package = "sq"

    # args = [circuit, edges, ground_node, trunc_num, offset_integer, cj,
    #         ntrial, amp_param, amp_offset, return_std, workers, package]

    # ngate = optim.get_ngate_mc(params, *args)

    # Optimized fluxonium
    circuit = [('C', 'J', 'L')]
    edges = [(0, 1)]
    params = [2.62935827, 2.85382311, 0.05129426, 0.0]
    trunc_num = [130]
    ground_node = None
    offset_integer = True
    cj = 10.0
    ntrial = 1
    amp_param = 0
    amp_offset = 0
    return_std = False
    workers = 1
    package = "sq"
    args = [circuit, edges, ground_node, trunc_num, offset_integer, cj,
            ntrial, amp_param, amp_offset, return_std, workers, package]
    ngate = optim.get_ngate_mc(params, *args)
    assert abs(ngate + 194858) < 1

    package = "sc"
    args = [circuit, edges, ground_node, trunc_num, offset_integer, cj,
            ntrial, amp_param, amp_offset, return_std, workers, package]
    ngate = optim.get_ngate_mc(params, *args)
    assert abs(ngate + 182410) < 1

    ntrial = 100
    workers = min(int(os.cpu_count()/2), ntrial)
    amp_param = 0.025
    amp_offset = 0
    return_std = True
    args = [circuit, edges, ground_node, trunc_num, offset_integer, cj,
            ntrial, amp_param, amp_offset, return_std, workers, package]
    ngate, ngate_std = optim.get_ngate_mc(params, *args)
    # assert abs(ngate/186765.5264 - 1) < 0.05
    # assert abs(ngate_std/6039.851178 - 1) < 0.1


def test_gen_param_dict_anyq():
    
    # Optimized fluxonium
    circuit = [('C', 'J', 'L')]
    edges = [(0, 1)]
    params = [2.62935827, 2.85382311, 0.05129426, 0.0]
    param_dict = optim.gen_param_dict_anyq(circuit, edges, params, cj=10.0)

    param_dict[((0,1), "C")] = params[0]
    param_dict[((0,1), "CJ")] = 10.0
    param_dict[((0,1), "J")] = params[1]
    param_dict[((0,1), "L")] = params[2]



def test_gen_param_range_anyq():
    
    # Optimized fluxonium
    circuit = [('C', 'J', 'L'), ("J",), ("J","C")]
    edges = [(1, 2), (0, 1), (2,0)]
    ground_node = 0
    offset_integer = False
    mapping = {'C': (0.05, 10.0), 'L': (0.05, 5.0), 'J': (1, 30)}
    ranges = optim.gen_param_range_anyq(circuit, edges, ground_node, offset_integer, mapping)

    assert ranges[0] == mapping["C"]
    assert ranges[1] == mapping["J"]
    assert ranges[2] == mapping["L"]
    assert ranges[3] == mapping["J"]
    assert ranges[4] == mapping["J"]
    assert ranges[5] == mapping["C"]
    assert ranges[6] == (0,1)
    assert ranges[7] == (0,1)

    offset_integer = True
    ranges = optim.gen_param_range_anyq(circuit, edges, ground_node, offset_integer, mapping)
    assert ranges[0] == mapping["C"]
    assert ranges[1] == mapping["J"]
    assert ranges[2] == mapping["L"]
    assert ranges[3] == mapping["J"]
    assert ranges[4] == mapping["J"]
    assert ranges[5] == mapping["C"]
    assert ranges[6] == (min(optim.INT_OFFSETS_FLUX.keys()), max(optim.INT_OFFSETS_FLUX.keys()))
    assert ranges[7] == (min(optim.INT_OFFSETS_CHARGE.keys()), max(optim.INT_OFFSETS_CHARGE.keys()))
    


def test_is_charge_mode():

    circuit = [('C', 'J', 'L')]
    edges = [(0, 1)]
    params = [2.62935827, 2.85382311, 0.05129426, 0.0]
    cir = optim.make_sq(circuit, edges, param_sets=params, cj=10.0, ground_node=0)
    assert(optim.is_charge_mode(cir, 1) == False)

    circuit = [('C', 'J')]
    edges = [(0, 1)]
    params = [2.62935827, 2.85382311, 0.0]
    cir = optim.make_sq(circuit, edges, param_sets=params, cj=10.0, ground_node=0)
    assert(optim.is_charge_mode(cir, 1) == True)


def test_pick_truncation_sq():

    circuit = [('J','C'), ('J', 'L', 'C')]
    edges = [(0, 2), (1, 2)]
    params = [7.412802517160099, 1.1, 4.879383407260387, 0.6606415539313616, 0.9, 20.74202294168071, 21.613559736082763, 23.0, 0.2092362500235012, 0.7460265564772862]
    cir = optim.make_sq(circuit, edges, params, ground_node=0)

    truncs = optim.pick_truncation_sq(cir, thresh=1e-06, neig=5, default_flux=50,
                                 default_charge=20, increment_flux=5,
                                 increment_charge=5)
    assert truncs == [50, 30]
    

def test_pick_truncation_sc():

    circuit = [('J','C'), ('J', 'L', 'C')]
    edges = [(0, 2), (1, 2)]
    params = [7.412802517160099, 1.1, 4.879383407260387, 0.6606415539313616, 0.9, 20.74202294168071, 21.613559736082763, 23.0, 0.2092362500235012, 0.7460265564772862]
    cir = optim.make_sc(circuit, edges, params, ground_node=0)


    truncs = optim.pick_truncation_sc(cir, thresh=1e-06, neig=5, default_flux=50,
                                 default_charge=20, increment_flux=5,
                                 increment_charge=5)
    assert truncs == [20, 50]

def test_sweep_helper():

    # [vals, n_eig, circuit, edges, ground_node, trunc_num,
    #  offset_integer, cj, extras, just_spec, package] = args

    # Optimized fluxonium
    circuit = [('C', 'J', 'L')]
    edges = [(0, 1)]
    params = [2.62935827, 2.85382311, 0.05129426, 0.0]
    vals = [[0, p] for p in params]
    trunc_num = [130]
    ground_node = None
    offset_integer = True
    cj = 10.0
    ntrial = 1
    amp_param = 0
    amp_offset = 0
    return_std = False
    workers = 1
    n_eig = 5


    package = "sq"
    just_spec = False
    args = [vals, n_eig, circuit, edges, ground_node, trunc_num,
            offset_integer, cj, {}, just_spec, package]
    idx, res = optim.sweep_helper_(args)
    assert abs(res["ngate"] - 194858) < 1

    package = "sq"
    just_spec = True
    args = [vals, n_eig, circuit, edges, ground_node, trunc_num,
            offset_integer, cj, {}, just_spec, package]
    idx, res2 = optim.sweep_helper_(args)
    assert "spec" in res2 and "ngate" not in res2
    assert np.max(np.abs(res2["spec"] - res["spec"])) < 1e-06


    package = "sc"
    just_spec = False
    trunc_num = 300
    args = [vals,  n_eig, circuit, edges, ground_node, trunc_num,
            offset_integer, cj, {}, just_spec, package]
    idx, res = optim.sweep_helper_(args)
    assert abs(res["ngate"] - 182410) < 1
    # Test sc vs. sq difference
    # Not the exact same because of ground capacitance
    assert np.max(1-np.abs((res2["spec"] - res2["spec"][0]+0.01)/(res["spec"] - res["spec"][0]+0.01))) < 0.05


    package = "sc"
    just_spec = True
    args = [vals, n_eig, circuit, edges, ground_node, trunc_num,
            offset_integer, cj, {}, just_spec, package]
    idx, res2 = optim.sweep_helper_(args)
    assert "spec" in res2 and "ngate" not in res2
    assert np.max(np.abs(res2["spec"] - res["spec"])) < 1e-06




def test_sweep_params():

    import  os
    num_cores = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = num_cores
    os.environ["OMP_NUM_THREADS"] = num_cores
    os.environ["MKL_NUM_THREADS"] = num_cores
    
    circuit = [('C', 'J', 'L')]
    edges = [(0, 1)]
    ground_node = 0
    ej = np.linspace(1,10,10)
    el = np.linspace(1,10,10)
    trunc_num = 150
    params = [0.25, ej, el, 0.499999]
    # workers = min(int(os.cpu_count()/2), ej.size*el.size)
    workers = 1
    n_eig = 5
    package = "sc"
    res_sc = optim.sweep_params(circuit, edges, params,
                                ground_node, workers,
                                n_eig, trunc_num=trunc_num,
                                package=package)
    package = "sq"
    res_sq = optim.sweep_params(circuit, edges, params,
                                ground_node, workers,
                                n_eig, trunc_num=trunc_num,
                                package=package)

    # Compare spectrum out    
    diff = (res_sq["spec"]-res_sq["spec"][:, :, [0]]+0.001)/(res_sc["spec"]-res_sc["spec"][:, :, [0]]+0.001)
    assert np.max(np.abs(1-diff)) < 0.01

    # Auto pick trunc number
    trunc_num = None
    params = [0.25, ej, el, 0.499999]
    # workers = min(int(os.cpu_count()/2), ej.size*el.size)
    workers = 1
    n_eig = 5
    package = "sc"
    res_sc = optim.sweep_params(circuit, edges, params,
                                ground_node, workers,
                                n_eig, trunc_num=trunc_num,
                                package=package)
    package = "sq"
    res_sq = optim.sweep_params(circuit, edges, params,
                                ground_node, workers,
                                n_eig, trunc_num=trunc_num,
                                package=package)

    # Compare spectrum out    
    diff = (res_sq["spec"]-res_sq["spec"][:, :, [0]]+0.001)/(res_sc["spec"]-res_sc["spec"][:, :, [0]]+0.001)
    assert np.max(np.abs(1-diff)) < 0.01



    

def test_optimize_diff_evol():

    import  os
    num_cores = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = num_cores
    os.environ["OMP_NUM_THREADS"] = num_cores
    os.environ["MKL_NUM_THREADS"] = num_cores

    circuit = [('C', 'J', 'L')]
    edges = [(0, 1)]
    ground_node = 0
    ej = np.linspace(1,10,10)
    el = np.linspace(1,10,10)
    trunc_num = 150
    params = [0.25, ej, el, 0.499999]
    # workers = min(int(os.cpu_count()/2), ej.size*el.size)
    workers=1
    package = "sq"
    offset_integer=True
    trials = [1, 100]
    ranges = [(0.1, 10), (1, 30), (0.05, 0.5), (0,3)]
    res_sq = optim.optimize_diff_evol(circuit, edges, ground_node,
                             ranges, offset_integer, trials,
                             workers=workers, package=package)
    package = "sc"
    res_sc = optim.optimize_diff_evol(circuit, edges, ground_node,
                             ranges, offset_integer, trials,
                             workers=workers, package=package)
    


if __name__ == "__main__":

    test_sweep_params()
    test_optimize_diff_evol()