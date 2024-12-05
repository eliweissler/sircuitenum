# To prevent each process from doing parallel linear algebra under the hood
import  os
num_cores = "1"
os.environ["OPENBLAS_NUM_THREADS"] = num_cores
os.environ["OMP_NUM_THREADS"] = num_cores
os.environ["MKL_NUM_THREADS"] = num_cores

import warnings
warnings.filterwarnings("ignore", message="differential_evolution")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero")
import traceback
import itertools
import matplotlib

from typing import Union
from tqdm import tqdm
from multiprocessing import Pool
from func_timeout import func_timeout, FunctionTimedOut


import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import SQcircuit as sq
import scqubits as scq
scq.settings.T1_DEFAULT_WARNING=False

from sircuitenum import utils
from sircuitenum import qpackage_interface as qpi
from sircuitenum import enumeration as enum

EPS_C = 1e-4
INT_OFFSETS_CHARGE = {0: 0.0+EPS_C, 1: 0.25+EPS_C, 2: 0.5+EPS_C, 3: 0.75+EPS_C}
EPS_F = 1e-6
INT_OFFSETS_FLUX = {0: 0.0+EPS_F, 1: 0.25+EPS_F, 2: 0.5+EPS_F, 3: 0.75+EPS_F}
DECAYS_SQ = {  'depolarization':
                    ['capacitive',
                     'inductive',
                     'quasiparticle'],
                'dephasing':
                    ['cc',
                     'flux',
                     'charge']
             }
DECAYS_SC = {  't1':
                    ['capacitive',
                     'flux_bias_line'],
                'tphi':
                    ['1_over_f_cc',
                     '1_over_f_flux',
                     '1_over_f_ng']
             }


def make_sq(circuit: list, edges: list, param_sets: list, ground_node: int = 0,
             offset_integer: bool = False, trunc_num: Union[int, list] = 50,
             cj: float = 10.0):
    """
    Constructs an SQcircuit circuit from the given circuit, edges,
    and parameter values

    (ECJ = 10 GHz, ECG = 20 GHz)

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [("J",),("L", "J"), ("C",)]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        param_sets (list): list of parameter values in GHz. Values
                           are given in the order they appear in circuit.
        ground_node (int, optional): Ground node. If None is given, then
                                     adds small capacitive coupling for each
                                     node to gorund. Defaults to 0.
        offset_integer (bool, optional): whether to restrict the offsets,
                                         values according to dict INT_OFFSETS
        trunc_num (int or list, optional): truncation number for each mode
        cj (float): junction capacitance in GHz. Pass 0 to ignore it.

    Returns:
        SQcircuit circuit object
    """

    # Generate the circuit object
    params = gen_param_dict_anyq(circuit, edges, param_sets, cj=cj)
    sqc = qpi.to_SQcircuit(circuit, edges, params=params,
                           ground_node=ground_node,
                           trunc_num=trunc_num)

    # Set the extermal fluxes/charges
    nelems = sum(utils.count_elems_mapped(circuit).values())
    i = nelems
    for loop in sqc.loops:
        if offset_integer:
            loop.set_flux(INT_OFFSETS_FLUX[int(param_sets[i])])
        else:
            loop.set_flux(param_sets[i])
        i += 1
    for mode in range(1, utils.get_num_nodes(edges)):
        if is_charge_mode(sqc, mode):
            if offset_integer:
                sqc.set_charge_offset(mode, INT_OFFSETS_CHARGE[int(param_sets[i])])
            else:
                sqc.set_charge_offset(mode, param_sets[i])
            i += 1
    return sqc


def calc_decay_rates_sq(cr, decay_types = DECAYS_SQ):
    """Uses sqcircuit to calculate the decay rates for the indicated decay types

    Args:
        cr (sqcircuit circuit): circuit you are interested in
        decay_types (dict, optional): dictionary like DECAYS_SQ at the top fo the file. Defaults to DECAYS_SQ.

    Returns:
        All rates are in units of 1/s
        dictionary {'depolarization':
                    {'capacitive': rate,
                     'inductive': rate,
                     'quasiparticle': rate},
                    'dephasing':
                    {'cc': rate,
                     'flux': rate,
                     'charge': rate}
                     }
    """
    
    # Get raw rates
    decay_rates = {}
    for dec in decay_types:
        decay_rates[dec] = {}
        for dec_type in decay_types[dec]:
            decay_rates[dec][dec_type] = cr.dec_rate(dec_type=dec_type, states=(1,0))

    return decay_rates

def make_sc(circuit: list, edges: list, param_sets: list, ground_node: int = 0,
             offset_integer: bool = False, trunc_num: Union[int, list] = 50,
             cj: float = 10.0):
    """
    Constructs an SCqubits circuit from the given circuit, edges,
    and parameter values

    (by default ECJ = 10 GHz, ECG = 20 GHz)

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [("J",),("L", "J"), ("C",)]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        param_sets (list): list of parameter values in GHz. Values
                           are given in the order they appear in circuit.
        ground_node (int, optional): Ground node. If None is given, then
                                     adds small capacitive coupling for each
                                     node to gorund. Defaults to 0.
        offset_integer (bool, optional): whether to restrict the offsets,
                                         values according to dict INT_OFFSETS
        trunc_num (int or list, optional): truncation number for each mode
        cj (float): junction capacitance in GHz. Pass 0 to ignore it.

    Returns:
        SQcircuit circuit object
    """

    # Generate the circuit object
    params = gen_param_dict_anyq(circuit, edges, param_sets, cj=cj)
    sc = qpi.to_SCqubits(circuit, edges, params=params,
                           ground_node=ground_node,
                           trunc_num=trunc_num)

    # Set the extermal fluxes/charges
    nelems = sum(utils.count_elems_mapped(circuit).values())
    i = nelems
    for Phi in sorted([str(x) for x in sc.external_fluxes]):
        if offset_integer:
            exec(f"sc.{Phi} = {INT_OFFSETS_FLUX[int(param_sets[i])]}")
        else:
            exec(f"sc.{Phi} = {param_sets[i]}")
        i += 1
    for ng in sorted([str(x) for x in sc.offset_charges]):
        if offset_integer:
            exec(f"sc.{ng} = {INT_OFFSETS_CHARGE[int(param_sets[i])]}")
        else:
            exec(f"sc.{ng} = {param_sets[i]}")
        i += 1
    return sc


def calc_decay_rates_sc(cr, decay_types = DECAYS_SC):
    """Uses scqubits to calculate the decay rates for the indicated decay types

    Args:
        cr (scqubits circuit): circuit you are interested in
        decay_types (dict, optional): dictionary like DECAYS_SQ at the top fo the file. Defaults to DECAYS_SQ.

    Returns:
        All rates are in units of 1/s
        dictionary {'t1':
                    ['capacitive',
                     'flux_bias_line'],
                    'tphi':
                    ['1_over_f_cc',
                     '1_over_f_flux',
                     '1_over_f_ng']
             }
    """
    # Get raw rates
    decay_rates = {}
    for dec in decay_types:
        decay_rates[dec] = {}
        for dec_type in decay_types[dec]:
            ## TODO: Make temperature match SQ
            try:
                decay_rates[dec][dec_type] = 1/(1e-09*eval(f"cr.{dec}_{dec_type}(total=True)"))
            except RuntimeError:
                continue

    return decay_rates


def get_gate_time(omega0:float, delta_omega:float, max_power:float=0.2*2*np.pi):
    """
    Estimates the gate time for a three level system with specified
    parameters, according to appendix A.

    Args:
        omega0 (float): qubit frequency E1 - E0 in GHz
        delta_omega (float): frequency of the E2 - E1 transition in GHz
        max_power (float, optional): Maximum drive strength in GHz.
                                     Defaults to 0.200.
    
    Returns:
        gate time estimate as half width of Gaussian pulse in s
    """
    # Turn into angular units
    omega0 = omega0*2*np.pi
    delta_omega = delta_omega*2*np.pi

    # Direct transitions
    delta = min(abs(omega0-delta_omega), abs(delta_omega))
    Vmax = min(max_power, omega0/10)
    if Vmax == 0:
        Vmax = 1e-09
    if delta/2 <= Vmax:
        tau_direct = np.sqrt(np.pi/2)/(delta/2)
    elif Vmax > 0:
        tau_direct = np.sqrt(np.pi/2)/Vmax
    
    # Raman transition
    tau_raman = 60*np.sqrt(np.pi/2)/max_power

    return 5*min(tau_direct, tau_raman)*1e-09


def decoherence_time(decay_rates, t_1_channels = [],
                                  t_phi_channels = []):
    """Turns the dictionary of decay rates into a t1, t_phi, and t2
    given a list of channels for each error type.

    Args:
        decay_rates (dict): Dictionary of decay rates in units of 1/s,
                            returned by calc_decay_rates.
        t_1_channels (list, optional): list of deplarization/t1 channels to consider.
                                       If none are given, uses all channels in decay_rates.
        t_phi_channels (list, optional): list of dephasing/tphi channels to consider.
                                       If none are given, uses all channels in decay_rates.

    Returns:
        t_1, t_phi, t_2 in units of s
    """

    if "depolarization" in decay_rates:
        dec_names = {"t1": "depolarization",
                     "tphi": "dephasing"}
    else:
        dec_names = {"t1": "t1",
                     "tphi": "tphi"}
    if len(t_1_channels) == 0:
        t_1_channels = decay_rates[dec_names["t1"]].keys()
    if len(t_phi_channels) == 0:
        t_phi_channels = decay_rates[dec_names["tphi"]].keys()
        
    # Calculate t1
    t_1_rate = 0
    for dec_type in t_1_channels:
        rate = decay_rates[dec_names['t1']][dec_type]
        if np.isfinite(rate):
            t_1_rate += rate
    t_1 = 1/t_1_rate

    # Calculate tphi
    t_phi_rate = 0
    for dec_type in t_phi_channels:
        rate = decay_rates[dec_names['tphi']][dec_type]
        if np.isfinite(rate):
            t_phi_rate += rate
    t_phi = 1/t_phi_rate    

    # Calculate t2
    t_2 = 1/(1/(2*t_1)+1/t_phi)

    return t_1, t_phi, t_2


def get_anharmonicity(spec):
    """
    Returns the anharmonicity in units of 2*pi*hbar GHz
    E_12 - E_10 = E_2 - 2*E_1 + E_0

    Args:
        spec (np.array): energy spectrum in GHz

    Returns:
        Anharmonicity in units of GHz
    """
    return spec[2] - 2*spec[1] + spec[0]


def get_ngate_mc(param_set: list, *args, **kwargs):
    """
    Objective function to be minimized for optimization.
    Returns -ngates performed by the circuit.

    Returns the average (and optionally std) of evaluations
    sampled from a gaussian with std amp, centered on
    the value inside of param_sets
    
    Args:
        param_set (list): list of parameter values in GHz. Values
                           are given in the order they appear in circuit.
        args (list): extra arguments [circuit, edges, ground_node, trunc_num,
                                      offset_integer, cj, ntrial, amp_param, 
                                      amp_offset, return_std, workers]

    Returns:
        float: -ngates done by the circuit, or -ngates, std of samples
    """
    [ntrial, amp_elem, amp_off, return_std, workers, package] = args[-6:]

    return_median = kwargs.get("return_median", False)

    # Run ntrial times, going amp on either side of params
    nelems = sum(utils.count_elems_mapped(args[0]).values())
    ngates = []
    swp_args = []
    for i in range(ntrial):
        # Generate random values and evaluate
        new_param_set = [x*np.random.normal(1, amp_elem) for x in param_set[:nelems]]
        new_param_set += [x*np.random.normal(1, amp_off) for x in param_set[nelems:]]
        swp_args.append((tuple(enumerate(new_param_set)),) + (3,) + args[:-6] + ({},) + (False,) + (package,))
    if workers == 1:
        for i in range(ntrial):
            res = sweep_helper_(swp_args[i])[1]
            ngates.append(res["ngate"])
    else:
        pool = Pool(processes=workers)
        for res in pool.imap_unordered(sweep_helper_, swp_args):
            ngates.append(res[1]["ngate"])

    if not (return_std or return_median):
        return -np.mean(ngates)
    else:
        to_return = (np.mean(ngates),)
        if return_median:
            to_return = to_return + (np.median(ngates),)
        if return_std:
            to_return = to_return + (np.std(ngates),)
        return to_return


def gen_param_dict_anyq(circuit: list, edges: list, param_sets: list,
                        cj: float = 10.0):
    """
    Generates a dictionary of parameter values for use in
    qpackage interface.

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [("J",),("L", "J"), ("C",)]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        param_sets (list): list of parameter values in GHz. Values
                           are given in the order they appear in circuit.
        cj (float): junction capacitance in GHz. Pass 0 to ignore it.

    Returns:
        dict: parameter values dictionary
    """

    param_dict = {}
    idx = 0
    for elems, edge in zip(circuit, edges):
        for elem in elems:
            key = (edge, elem)
            param_dict[key] = (param_sets[idx],'GHz')
            idx+=1
            # Junction capacitance
            if elem == "J" and cj > 0:
                key = (edge, "CJ")
                param_dict[key] = (cj, 'GHz')
    return param_dict


def gen_param_range_anyq(circuit: list, edges: list, ground_node: int, offset_integer: bool = False,
                         mapping: dict = {'C': (0.05, 10.0), 'L': (0.05, 5.0), 'J': (1, 30)},
                         package: str = "sq"):
    """
    Generates parameter ranges according to mapping,
    for use in bounded optimizations

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [("J",),("L", "J"), ("C",)]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        ground_node (int): Ground node. If None is given, then
                                     adds small capacitive coupling for each
                                     node to gorund. Defaults to 0.
        mapping (dict, optional): Ranges for the different components.
                                  Defaults to {'C': (0.1, 1), 'L': (0.1, 1), 'J': (3, 20)}.

    Returns:
        tuple: tuple of parameter ranges, for input to scipy optimization
    """
    # Circuit Elements
    param_range = []
    [[param_range.append(mapping[item]) for item in items] for items in circuit]

    if package == "sq":
        # Offsets
        cir = qpi.to_SQcircuit(circuit, edges, ground_node=ground_node, rand_amp=0.25)
        # Flux
        for loop in cir.loops:
            if offset_integer:
                param_range.append((min(INT_OFFSETS_FLUX.keys()), max(INT_OFFSETS_FLUX.keys())))
            else:
                param_range.append((0, 1))
        # Charge
        for mode in range(1, utils.get_num_nodes(edges)):
            if is_charge_mode(cir, mode):
                if offset_integer:
                    param_range.append((min(INT_OFFSETS_CHARGE.keys()), max(INT_OFFSETS_CHARGE.keys())))
                else:
                    param_range.append((0, 1))
    elif package == "sc":
        scq = qpi.to_SCqubits(circuit, edges, ground_node=ground_node, rand_amp=0.25)
        for _ in scq.external_fluxes:
            if offset_integer:
                param_range.append((min(INT_OFFSETS_FLUX.keys()), max(INT_OFFSETS_FLUX.keys())))
            else:
                param_range.append((0, 1))
        for _ in scq.offset_charges:
            if offset_integer:
                param_range.append((min(INT_OFFSETS_CHARGE.keys()), max(INT_OFFSETS_CHARGE.keys())))
            else:
                param_range.append((0, 1))

    return tuple(param_range)


def is_charge_mode(sqc: sq.Circuit, mode: int):
    """
    Helper function that returns whether the given mode
    (index from 1) is a charge mode or not

    Args:
        sqc (sq.Circuit): SQcircuit Circuit object
        mode (int): mode number (index from 1)

    Returns:
        Bool: whether the given mode is a charge mode
    """
    return mode - 1 in sqc.charge_islands


def pick_truncation_sq(cir: sq.Circuit, thresh: float = 1e-06, neig: int = 5,
                        default_flux: int = 30, default_charge: int = 10,
                        increment_flux: int = 5, increment_charge: int = 5):
    if isinstance(cir, tuple) or isinstance(cir, list):
        params = cir[2]
        cir = make_sq(*cir)
    else:
        params = None
    
    nmodes = cir.n
    truncs = [default_charge if is_charge_mode(cir, i) else
              default_flux for i in range(1, nmodes+1)]
    increments = [increment_charge if is_charge_mode(cir, i) else
                  increment_flux for i in range(1, nmodes+1)]
    cir.set_trunc_nums(truncs)
    cir.diag(5)
    diff = np.inf*np.ones(nmodes)
    def update_diff(cir, truncs, diff, i):
        old = cir.efreqs.copy()
        cir.set_trunc_nums(truncs)
        cir.diag(neig)
        diff[i] = np.max(np.abs(cir.efreqs-old)/np.abs(cir.efreqs))

    incremented = np.ones(nmodes, dtype=bool)
    while np.any(incremented):
        for i in range(nmodes):
            incremented[i] = False
            truncs[i] += increments[i]
            update_diff(cir, truncs, diff, i)
            while(diff[i] > thresh):
                truncs[i] += increments[i]
                update_diff(cir, truncs, diff, i)
                incremented[i] = True
            truncs[i] -= increments[i]
    
    if params is None:
        return truncs
    else:
        return params, truncs


def pick_truncation_sc(cir: scq.Circuit, thresh: float = 1e-06, neig: int = 5,
                        default_flux: int = 30, default_charge: int = 10,
                        increment_flux: int = 5, increment_charge: int = 5,
                        default_cutoff_flux = 300, default_cutoff_charge = 150):
    if isinstance(cir, tuple):
        params = cir[2]
        cir = make_sc(*cir)
    else:
        params = None
    
    nmodes = len(cir.var_categories["periodic"]+cir.var_categories["extended"])
    truncs = []
    increments = []
    system_hierarchy = []
    for i in range(1, nmodes+1):
        if i in cir.var_categories["periodic"]:
            truncs.append(default_charge)
            increments.append(increment_charge)
            exec(f"cir.cutoff_n_{i}={default_cutoff_charge}")
        elif i in cir.var_categories["extended"]:
            truncs.append(default_flux)
            increments.append(increment_flux)
            exec(f"cir.cutoff_ext_{i}={default_cutoff_flux}")
        system_hierarchy.append([i])

    evals = cir.eigenvals(5)
    diff = np.inf*np.ones(nmodes)
    def update_diff(cir, old, truncs, diff, i):
        cir.configure(system_hierarchy=system_hierarchy,
                      subsystem_trunc_dims=truncs)
        new = cir.eigenvals(neig)
        diff[i] = np.max(np.abs(new-old)/np.abs(new))
        return new

    incremented = np.ones(nmodes, dtype=bool)
    while np.any(incremented):
        for i in range(nmodes):
            incremented[i] = False
            truncs[i] += increments[i]
            evals = update_diff(cir, evals, truncs, diff, i)
            while(diff[i] > thresh):
                truncs[i] += increments[i]
                evals = update_diff(cir, evals, truncs, diff, i)
                incremented[i] = True
            truncs[i] -= increments[i]
    
    if params is None:
        return truncs
    else:
        return params, truncs

# 5 min timeout to help speed things up
def timed_out_(*args, **kwargs):
    timeout_min = 60
    try:
        return func_timeout(60*timeout_min, get_ngate_mc, args, kwargs)
    except FunctionTimedOut:
        print(f"Could not complete {args[0]} ({timeout_min} min timout)")
        return 0
    except KeyboardInterrupt as KI:
        raise KI
    except:
        return 0

def sweep_helper_(args, **kwargs):
    
    [vals, n_eig, circuit, edges, ground_node, trunc_num,
     offset_integer, cj, extras, just_spec, package] = args

    idx = tuple([x[0] for x in vals])
    param_set = tuple([x[1] for x in vals])
    to_return = {}

    if package == "sq":
        # try:
        cir = make_sq(circuit, edges, param_set, ground_node, offset_integer,
                    trunc_num=trunc_num, cj=cj)
        # except:
        #     return idx, {"ngate": 0}
        cir.diag(n_eig)
        to_return["spec"] = cir.efreqs
        if just_spec:
            return idx, to_return
        rates = calc_decay_rates_sq(cir)
    elif package == "sc":
        # try:
        cir = make_sc(circuit, edges, param_set, ground_node, offset_integer,
                    trunc_num=trunc_num, cj=cj)
        # except:
        #     return idx, {"ngate": 0}
        to_return["spec"] = cir.eigenvals(n_eig)
        if just_spec:
            return idx, to_return
        rates = calc_decay_rates_sc(cir)
    else:
        raise ValueError("Only package = sc or sq is supported")
    

    # Calculate anharmonicity/rates
    spec = to_return["spec"]
    alpha = get_anharmonicity(spec)
    t1, tphi, t2 = decoherence_time(rates)

    # Save anharmoniciry and gate times
    gate_time = get_gate_time(spec[1]-spec[0], spec[2]-spec[1])

    to_return["alpha"] = alpha
    to_return["rates"] = rates
    to_return["t1"] = t1
    to_return["t2"] = t2
    to_return["tphi"] = tphi
    to_return["tg"] = gate_time
    to_return["ngate"] = t2/gate_time


    # Extras
    for field in extras:
        to_return[field] = extras[field](cir)

    return idx, to_return


def sweep_params(circuit: list, edges: list, params: list, 
                 ground_node: int = 0, workers: int = 4, n_eig: int = 5,
                 extras: dict = {}, trunc_num: Union[int, list] = -1,
                 cj: float = 10.0, just_spec: bool = False, quiet: bool = False,
                 package: str = "sq"):
    """General function to perform paramater sweeps on quantum circuits
    using SQcircuit.

    Args:
        circuit (list): a list of element labels for the desired circuit
                        e.g. [("J",),("L", "J"), ("C",)]
        edges (list): a list of edge connections for the desired circuit
                        e.g. [(0,1), (0,2), (1,2)]
        params (list): list of parameter values in GHz. Float entries are fixed,
                       while iterable fields are swept over.
        ground_node (int, optional): Ground node. If None is given, then
                                     adds small capacitive coupling for each
                                     node to gorund. Defaults to 0.
        workers (int, optional): Number of workers to use in parallel evalutation.
        n_eig (int, optional): Number of eigenvalues to compute and save.
        extras: (dict, optional): Optional fields to compute, providing a function
                                  that takes in an SQcircuit circuit objects.
                                  extras[str] = func for scalar
                                  extras[str] = (dims, func) for non-scalar
        trunc_num (int or list, optional): truncation number for each mode
        cj (float): junction capacitance in GHz. Pass 0 to ignore it.
        just_spec (bool): Only calculate the energy spectrum to save time.
        quiet (bool): whether to print out messages or not
        package (str): which package to use, sqcircuit "sq" or scqubits "sc"


    Returns:
        dict: Dictionary containing many arrays that have shapes
              to match the dimensions of the params input
                - eigenvalues (last dimension is n_eig),
                - rates (dictionary that maps to arrays of decay rates), 
                - gate_time 
                - anharmonicity (alpha)
                - t1, tphi, tg, tgate, ngate
    """

    # Calculate the return array size
    # From parameter inputs
    arr_size = []
    ranges = []
    for p in params:
        if isinstance(p, float) or isinstance(p, int):
            arr_size.append(1)
            ranges.append(np.array([p]))
        else:
            ranges.append(p)
            arr_size.append(len(p))

    # Construct results dictionary
    results = {}
    results["rates"] = {}
    if package == "sq":
        decays = DECAYS_SQ
    elif package == "sc":
        decays = DECAYS_SC
    else:
        raise ValueError("only sq and sc are supported for package input")
    for dec in decays:
        results["rates"][dec] = {}
        for dec_type in decays[dec]:
            results["rates"][dec][dec_type] = np.zeros(arr_size)

    results["spec"] = np.zeros(arr_size + [n_eig])
    fields = ["spec", "alpha", "t1", "t2", "tphi", "tg", "ngate"]
    for field in fields[1:]:
        results[field] = np.zeros(arr_size)
    for field in extras:
        if isinstance(extras[field], tuple):
            results[field] = np.zeros(arr_size + extras[field][0])
            extras[field] = extras[field][1]
        else:
            results[field] = np.zeros(arr_size)
    if just_spec:
        fields = ["spec"]
    fields = fields + list(extras.keys())

    # Estimate truncation number
    if trunc_num == -1 or trunc_num is None:
        mean_params = [np.mean(x) for x in ranges]
        trunc_args = (circuit, edges, mean_params, ground_node, False, 10, cj)
        if package == "sq":
            trunc_num = pick_truncation_sq(trunc_args, thresh=1e-09)[1]
        elif package == "sc":
            trunc_num = pick_truncation_sc(trunc_args, thresh=1e-09)[1]

    if not quiet:
        print("----------------------------")
        print("Sweeping Circuit", circuit, edges, f"({workers} workers)")
        print("Ground Node:", ground_node)
        print("Truncation Numbers:", trunc_num)
        print("Ranges:")
        count = 0
        for i in range(len(circuit)):
            print(" edge:", edges[i])
            for j, elem in enumerate(circuit[i]):
                if ranges[count].size > 1:
                    print(" ", elem, (ranges[count][0], ranges[count][-1], ranges[count].size))
                else:
                    print(" ", elem, ranges[count][0])
                count += 1
        print(" offsets:")
        for i in range(count, len(ranges)):
            if len(ranges[i]) > 1:
                print(" ", (ranges[i][0], ranges[i][-1], ranges[i].size))
            else:
                print(" ", ranges[i][0])
        print("----------------------------")               
    

    # Parameter values to iterate over
    vals = itertools.product(*[enumerate(x) for x in ranges])
    vals = [(v,) + (n_eig, circuit, edges, ground_node,
                    trunc_num, False, cj, extras, just_spec, package) for v in vals]
    if workers > 1:
        pool = Pool(processes=workers)
        for idx, res in tqdm(pool.imap_unordered(sweep_helper_, vals),
                        total=sum(1 for _ in vals)):
            # Save values
            for field in fields:
                results[field][idx] = res[field]
            if "rates" in res:
                for dec in res["rates"]:
                    for dec_type in res["rates"][dec]:
                        results["rates"][dec][dec_type][idx] = res["rates"][dec][dec_type]
    else:
        for v in tqdm(vals):
            idx, res = sweep_helper_(v)
            # Save value
            for field in fields:
                results[field][idx] = res[field]
            if "rates" in res:
                for dec in res["rates"]:
                    for dec_type in res["rates"][dec]:
                        results["rates"][dec][dec_type][idx] = res["rates"][dec][dec_type]

    # Squeeze 1 length dimensions
    for field in fields:
        results[field] = np.squeeze(results[field])
    if "rates" in res:
        for dec in results["rates"]:
            for dec_type in results["rates"][dec]:
                results["rates"][dec][dec_type] = np.squeeze(results["rates"][dec][dec_type])

    return results


def callback_function(xk, convergence):
    print(f"Current best solution: {xk}")
    print(f"Convergence value: {convergence}")
    print("---------------------------------------")

def optimize_diff_evol(circuit: list, edges: list, ground_node: int,
                       ranges: list = None, offset_integer: bool = False,
                       trials: list = [1, 100],
                       amps: dict = {"elem": [0, 0.025], "offset": [0, 1e-06]},
                       trunc_num: Union[int, list] = -1,
                       cj: float = 10.0, quiet: bool = False,
                       package: str = "sq", trunc_est_n: int = 200,
                       **kwargs):
    
    kwargs = dict({'disp': 'True', 'popsize': 20,
                   "callback": callback_function, "polish": False,
                   "workers": 1, "tol": 0.1, "init": "halton",
                   "maxiter": 1000}, **kwargs)
    if quiet:
        kwargs["disp"] = False
        kwargs["callback"] = None

    # Auto make ranges if none is given
    if ranges is None:
        ranges = gen_param_range_anyq(circuit, edges, ground_node, offset_integer)
    kwargs["bounds"] = ranges

    # Determine integrality of variables
    integrality = np.zeros(len(ranges), dtype=bool)
    if offset_integer:
        nelems = sum(utils.count_elems_mapped(circuit).values())
        for i in range(nelems, integrality.size):
            integrality[i] = True
    kwargs["integrality"] = integrality

    # Dictionary with results
    to_return = {}
    
    # Estimate truncation number if none is given
    if trunc_num == -1:
        random_params = [[np.exp(np.random.random()*(np.log(ranges[i][1])-np.log(ranges[i][0]))+np.log(ranges[i][0])) 
                              if not integrality[i] else np.random.randint(*ranges[i]) for i in range(len(ranges))]
                              for i in range(trunc_est_n)]
        args = [(circuit, edges, p, ground_node, offset_integer, 10, cj) for p in random_params]
        # (circuit, edges, random_params, ground_node, offset_integer, trunc_num, cj)
        trunc_vals = []
        param_vals = []
        if not quiet:
            print("Picking Truncation Number with", trunc_est_n, "randomly chosen points")
        # Choose truncation function
        if package == "sq":
            trunc_func = pick_truncation_sq
        elif package == "sc":
            trunc_func = pick_truncation_sc
        # Test trruncation for randomly sampled points
        if kwargs["workers"] > 1:
            pool = Pool(processes=kwargs["workers"])
            for p, res in pool.imap_unordered(trunc_func, args):
                trunc_vals.append(res)
                param_vals.append(p)
        else:
            for i in range(trunc_est_n):
                p, res = trunc_func(args[i])
                trunc_vals.append(res)
                param_vals.append(p)
        trunc_vals = np.array(trunc_vals)
        param_vals = np.array(param_vals)
        to_return["max_trunc"] = list(np.max(trunc_vals, axis=0))
        trunc_num = np.round(np.percentile(trunc_vals, 90, axis=0)).astype(int)
        to_return["trunc_num"] = list(trunc_num)
        if not quiet:
            print("Maximum Cutoffs:", to_return["max_trunc"])
            print("Chosen Values (90th Percentile)", trunc_num)

    if not quiet:
        print("----------------------------")
        print("Optimizing", circuit, edges, "using differential evolution", f"({kwargs['workers']} workers)")
        print("Ground node:", ground_node)
        print("Truncation numbers:", trunc_num)
        print("Offset integer:", offset_integer)
        print("Ranges:")
        count = 0
        for i in range(len(circuit)):
            print(" edge:", edges[i])
            for j, elem in enumerate(circuit[i]):
                if len(ranges[count]) > 1:
                    print(" ", elem, (ranges[count][0], ranges[count][-1]))
                else:
                    print(" ", elem, ranges[count][0])
                count += 1
        print(" offsets (charge then flux):", ranges[count:])
        print("MC Settings: (Optimize, Eval)", "trials -", trials, "amps -", amps)
        print("DE args:", kwargs)
        print("----------------------------")               

    # Run differential evolution
    # [circuit, edges, ground_node, trunc_num, offset_integer, cj,
    #  ntrial, amp_elem, amp_off, return_std]
    args = [circuit, edges, ground_node, trunc_num, offset_integer, cj,
            trials[0], amps["elem"][0], amps["offset"][0], False, 1, package]
    res = sp.optimize.differential_evolution(timed_out_,
                                             args=args, **kwargs)

    if not quiet:
        print("Finished optimization\n", res)
        print("----------------------------")      

    to_return["ngate"] = -res.fun
    cir = make_sq(circuit, edges, res.x, ground_node=ground_node)
    to_return["param_best"] = res.x

    # Run final evaluation
    # args = [circuit, edges, ground_node, trunc_num, offset_integer, cj,
    #         ntrial, amp_param, amp_offset, return_std, workers, package]
    args2 = [circuit, edges, ground_node, trunc_num, False, cj,
             trials[1], amps["elem"][1], amps["offset"][1], True,
             kwargs["workers"], package]
    to_return["ngate_mean"] , to_return["ngate_std"] = get_ngate_mc(res.x, *args2)

    if not quiet:
        print("Finished evaluation")
        print("mean (+/- std):", int(to_return["ngate_mean"]),
              "+/-", int(to_return["ngate_std"]))
        print("----------------------------")

    return to_return