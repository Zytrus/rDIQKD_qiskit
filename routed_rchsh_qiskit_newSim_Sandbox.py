# Routed DIQKD rCHSH simulation — QuTiP version
from pathlib import Path
import numpy as np
import random
import json, ast
from collections import defaultdict
import datetime

import chaospy
import ncpol2sdpa as ncp
from ncpol2sdpa import SdpRelaxation 
from sympy.physics.quantum.dagger import Dagger
from scipy.optimize import minimize_scalar
from scipy.stats import chi2_contingency
from math import log2, log, exp
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error, NoiseModel


def bell_source():
    """Prepare |Phi+> = (|00> + |11>)/sqrt(2)."""
    qc = QuantumCircuit(q, c)
    qc.h(q[0])
    qc.cx(q[0], q[1])
    return qc

def apply_noise(v=0.99, q1_scale=1.0, q2_scale=1.0):
    """
    Baseline depolarizing noise on gates.
    v: 'visibility-like' knob; we set p_base = 1 - v.
    p1_scale, p2_scale: allow tuning 1q vs 2q gate noise separately.
    """
    p_base = 1.0 - v
    error_1q = depolarizing_error(p_base * q1_scale, 1)
    error_2q = depolarizing_error(p_base * q2_scale, 2)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1','u2','u3','x','h','rz','ry'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    return noise_model

def apply_long_path_noise(qc, travel_qubit, p_long: float):
    """Apply noise for long path."""
    error_long = depolarizing_error(p_long, 1)
    qc.append(error_long.to_instruction(), [travel_qubit])

def apply_alice_rotation(qc, x: int):
    """Alice: x=0 -> Z, x=1 -> X."""
    if x == 1:
        qc.h(q[0])
    elif x != 0:
        raise ValueError("Alice x must be 0 or 1")
    # qc.draw('mpl')
    # plt.show()
    

def apply_bob_sp_rotation(qc, z: int):
    """
    Bob_SP has z in {0,1}.
    Assumption: z=0 -> (Z+X)/sqrt(2), z=1 -> (Z-X)/sqrt(2).
    """
    if z == 0:
        qc.ry(-np.pi/4, q[1])
    elif z == 1:
        qc.ry(+np.pi/4, q[1])
    else:
        raise ValueError("Bob_SP z must be 0 or 1")
    # qc.draw('mpl')
    # plt.show()

def apply_bob_lp_rotation(qc, y: int):
    """
    Bob_LP has y in {0,1,2}.
    Assumption: y=0 -> Z, y=1 -> (Z+X)/sqrt(2), y=2 -> (Z-X)/sqrt(2).
    """
    if y == 0:
        pass
    elif y == 1:
        qc.ry(-np.pi/4, q[1])
    elif y == 2:
        qc.ry(+np.pi/4, q[1])
    else:
        raise ValueError("Bob_LP y must be 0, 1, or 2")
    # qc.draw('mpl')
    # plt.show()


def routed_round_circuit(x: int, s: int, z: int | None, y: int | None, p_long: float):
    """
    s=0 -> short path, use Bob_SP input z
    s=1 -> long path,  use Bob_LP input y and apply extra noise p_long
    """
    qc = bell_source()

    # qc.draw('mpl')
    # plt.show()
    # Route-dependent channel effect
    if s == 1:
        apply_long_path_noise(qc, q[1], p_long)

    # Measurement basis rotations
    apply_alice_rotation(qc, x)

    if s == 0:
        if z is None:
            raise ValueError("Need z when s=0 (Bob_SP round).")
        apply_bob_sp_rotation(qc, z)
    elif s == 1:
        if y is None:
            raise ValueError("Need y when s=1 (Bob_LP round).")
        apply_bob_lp_rotation(qc, y)
    else:
        raise ValueError("Route bit s must be 0 or 1.")

    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])
    # qc.draw('mpl')
    # plt.show()
    return qc

def run_circuits_get_probs(circuits: list[QuantumCircuit], shots: int, backend: AerSimulator, seed=1234) -> list[dict[str, int]]:
    """Run circuits and get counts. Make conditional probability table with them shots.

    Args:
        circuits (list[QuantumCircuit]): List of circuits to run.
        shots (int): Number of shots per circuit.
        backend: Qiskit backend to use.
        noise_model: Optional noise model to apply.
        seed (int, optional): Seed for reproducibility. Defaults to 1234.

    Returns:
        list[dict[str, int]]: List of counts dictionaries for each circuit.
    """
    
    transpiled = transpile(circuits, backend=backend, seed_transpiler=seed)
    job = backend.run(transpiled, shots=shots, seed_simulator=seed)
    results = job.result()
    
    probs_list = []
    for i in range(len(circuits)):
        counts = results.get_counts(i)  # dict: bitstring -> count
        total = sum(counts.values())

        # Convert each bitstring into (a,b)
        # For 2 classical bits, Qiskit returns strings like "00", "01", etc.
        # Convention: rightmost char is c[0], next is c[1] (little-endian classical order).
        # So:
        #   key = "... b a" for two bits.
        probs = {(0,0): 0.0, (0,1): 0.0, (1,0): 0.0, (1,1): 0.0}

        for bitstr, ct in counts.items():
            # Keep only last 2 bits in case of stray registers
            bitstr2 = bitstr.replace(" ", "")[-2:]

            a = int(bitstr2[-1])  # rightmost is c[0]
            b = int(bitstr2[-2])  # next is c[1]
            probs[(a,b)] += ct / total

        probs_list.append(probs)

    return probs_list

def accumulate_tables(probs_list, x_list, s_list, z_list, y_list):
    """
    Takes per-circuit probs and setting lists and produces aggregated conditional distributions.
    Returns:
      P_sp: dict (x,z) -> dict (a,b) -> p
      P_lp: dict (x,y) -> dict (a,b) -> p
      N_sp: dict (x,z) -> count of rounds
      N_lp: dict (x,y) -> count of rounds
    Aggregation is done by averaging per-round distributions (equivalently summing counts if shots fixed).
    """
    sum_sp = defaultdict(lambda: defaultdict(float))
    sum_lp = defaultdict(lambda: defaultdict(float))
    N_sp = defaultdict(int)
    N_lp = defaultdict(int)

    for i, probs in enumerate(probs_list):
        x = x_list[i]
        s = s_list[i]

        if s == 0:
            z = z_list[i]
            key = (x, z)
            for ab, p in probs.items():
                sum_sp[key][ab] += p
            N_sp[key] += 1
        else:
            y = y_list[i]
            key = (x, y)
            for ab, p in probs.items():
                sum_lp[key][ab] += p
            N_lp[key] += 1

    # Normalize (average across rounds)
    P_sp = {}
    for key, abdict in sum_sp.items():
        n = N_sp[key]
        P_sp[key] = {ab: abdict.get(ab, 0.0)/n for ab in [(0,0),(0,1),(1,0),(1,1)]}

    P_lp = {}
    for key, abdict in sum_lp.items():
        n = N_lp[key]
        P_lp[key] = {ab: abdict.get(ab, 0.0)/n for ab in [(0,0),(0,1),(1,0),(1,1)]}

    return P_sp, P_lp, N_sp, N_lp

def correlator_E(P_ab):
    """
    P_ab is dict (a,b)->p(a,b). Returns E = <A*B>.
    with A,B in {+1,-1} using 0->+1, 1->-1.
    """
    E = 0.0
    for (a,b), p in P_ab.items():
        E += ((-1)**(a+b)) * p
    return E

def chsh_from_table(P, b0, b1):
    """
    P: dict keyed by (x,b) -> P_ab distribution
    b0, b1: the two Bob-setting labels to use for CHSH (e.g. z=0,z=1 or y=1,y=2)
    Assumes Alice x in {0,1}.
    """
    E00 = correlator_E(P[(0, b0)])
    E01 = correlator_E(P[(0, b1)])
    E10 = correlator_E(P[(1, b0)])
    E11 = correlator_E(P[(1, b1)])
    return E00 + E01 + E10 - E11

def dict_to_matrix(p_dict: dict) -> np.ndarray:
    """Convert dict (a,b)->p to 2D numpy array.

    Args:
        p_dict (dict): Dictionary with keys (a,b) and float probabilities.

    Returns:
        np.ndarray: 2D array with shape (2,2) of probabilities.
    """
    p_matrix = np.zeros((2,2), dtype=float)
    for (a,b), p in p_dict.items():
        p_matrix[a,b] = p
    return p_matrix

def apply_losses_with_noclick(p: np.ndarray, eta_l: float, eta_r: float, keep_no_click: bool = True) -> np.ndarray:
    """Applies Detection Efficiency losses, depending on LP or SP.

    Args:
        p (np.ndarray): Probability matrix after applying POVM measurements to rho.
        eta_l (float): Detection Efficiency for the Long Path
        eta_r (float): Detection Efficiency for the Short Path
        keep_no_click (bool, optional): Bool to decide whether we keep no-click events.
    Raises:
        ValueError: If binning is enabled for only one side, it raises an error.

    Returns:
        np.ndarray: Probabilities with consideration to the detection efficiencies (both LP and SP)
    """
    k, m = p.shape
    if keep_no_click:
        # out is 3x3 loss matrix, depending on whether either A or B clicks/noclicks
        out = np.zeros((k + 1, m + 1), dtype=float)
        out[:k,:m] = eta_l*eta_r*p
        out[:k,m]  = eta_l*(1-eta_r)*p.sum(axis=1)
        out[k,:m]  = (1-eta_l)*eta_r*p.sum(axis=0)
        out[k,m]   = (1-eta_l)*(1-eta_r)
        out /= out.sum()
        # print(out.sum()) Always almost 1, just for accuracy
        return out
    else:
        # If we don't keep no-click events, we must bin both sides, could be expanded to only one side if needed
        # if not (bin_left and bin_right): 
        #     raise ValueError("Set bin_left=bin_right=True to bin ∅ into index 0.")
        out = eta_l*eta_r*p.copy()
        out[:,0] += eta_l*(1-eta_r)*p.sum(axis=1)
        out[0,:] += (1-eta_l)*eta_r*p.sum(axis=0)
        out[0,0] += (1-eta_l)*(1-eta_r)
        out /= out.sum()
        return out

def H_shannon(dist):
    """dist is iterable of probabilities. Returns Shannon entropy in bits."""
    h = 0.0
    for p in dist:
        if p > 0:
            h -= p * log2(p)
    return h

def marginals_from_Pab(P_ab):
    P_a = {0:0.0, 1:0.0}
    P_b = {0:0.0, 1:0.0}
    for (a,b), p in P_ab.items():
        P_a[a] += p
        P_b[b] += p
    return P_a, P_b

def H_A_given_B(P_ab):
    """H(A|B) = H(A,B) - H(B)."""
    P_a, P_b = marginals_from_Pab(P_ab)
    H_AB = H_shannon(P_ab.values())
    H_B  = H_shannon(P_b.values())
    return H_AB - H_B


# According to appendix A of paper
def generate_quadrature(gaussian_order):
    abscissas, weights = chaospy.quadrature.radau(gaussian_order, chaospy.Uniform(0, 1), 1)
    abscissas = abscissas[0]
    return abscissas, weights


# ----------------- SDP Code -----------------
def remove_commutative_constraints(sdp):
    """
    Removes commutative constraints from the SDP relaxation.
    """
    new_subs = {}
    Alice_obs = ["A"+str(i) for i in range(0, len(A_config))]
    BobL_obs = ["B"+str(i) for i in range(2, len(B_config))]
    BobS_obs = ["B"+str(i) for i in range(0, 2)]
    remove_long = False
    # Remove all commutative constraints between Alice and BobL operators and Eve's operators
    for key, val in sdp.substitutions.items():
        inside = False
        for a in Alice_obs:
            for b in BobS_obs:
                if a+"*"+b in str(key) or b+"*"+a in str(key):
                    inside = True
            if remove_long:
                for b in BobL_obs:
                    if a+"*"+b in str(key) or b+"*"+a in str(key):
                        inside = True
        if not inside:
            new_subs[key] = val
                
    sdp.substitutions = new_subs

def compute_entropy(SDP: SdpRelaxation, q, momentequality_constraints, Z, P):
    """
    Computes lower bound on H(A|X=0,E) using the fast (but less tight) method

        SDP   --   sdp relaxation object
        q     --   probability of bitflip
    """
    
    ck = 0.0        # kth coefficient
    ent = 0.0        # lower bound on H(A|X=0,E)

    # We can also decide whether to perform the final optimization in the sequence
    # or bound it trivially. Best to keep it unless running into numerical problems
    # with it. Added a nontrivial bound when removing the final term
    if KEEP_M:
        num_opt = len(T)
    else:
        num_opt = len(T) - 1
        ent = 2 * q * (1-q) * W[-1] / log(2)

    for k in range(num_opt):
        ck = W[k]/(T[k] * log(2))

        # Get the k-th objective function
        new_objective = objective(T[k], q, Z, P)
        if T[k]!=1:
            a_i = 3/2 * max(1/T[k] , 1/(1-T[k]))
            moment_inequality_constraints = [a_i-z*Dagger(z) for z in Z]+[a_i-Dagger(z)*z for z in Z]
        else:
            moment_inequality_constraints = []

        # For relaxation of previos constraints
        moment_inequality_constraints += momentequality_constraints
        SDP.set_objective(new_objective)
        if not commutative_constraints:
            remove_commutative_constraints(SDP)
        # SDP.process_constraints(momentequalities=momentequality_constraints,momentinequalities=moment_inequality_constraints)
        SDP.process_constraints(momentinequalities=moment_inequality_constraints)
        # Define MOSEK parameters for tighter (or looser) tolerance
        # mosek_params = {
        #     # Keep feasibility strict so the matrix is valid Quantum/NPA
        #     "mosek.dparam.intpnt_co_tol_pfeas": 1.0e-9,
        #     "mosek.dparam.intpnt_co_tol_dfeas": 1.0e-9,
            
        #     # Relax the optimality gap slightly for speed
        #     "mosek.dparam.intpnt_co_tol_rel_gap": 1.0e-6,
        # }
        # SDP.solve(solver='mosek', solverparameters=mosek_params)
        SDP.solve(solver='mosek')

        if SDP.status == 'optimal':
            # 1 contributes to the constant term
            ent += ck * (1 + SDP.dual)
        else:
            # If we didn't solve the SDP well enough then just bound the entropy
            # trivially
            ent = 0
            print('Bad solve: ', k, SDP.status)
            # ------------------ Minimize distance to ideal distributions ------------------
            # optimal_LP = add_noise_and_minimize(res["AB_test_probs"], born_lp)
            # optimal_SP = add_noise_and_minimize(res["AT_probs"], born_sp)
            # for s, data in optimal_LP.items():
            #     print(f"Long Setting {s}:")
            #     print(f"   Orig Dist: {data['original_distance']:.10f}")
            #     print(f"   Min Dist : {data['min_distance']:.10f}")
            #     print(f"   Lambda   : {data['optimal_noise']:.10f}")
            #     res["AB_test_probs"][s] = data['closest_distribution']
            # for s, data in optimal_SP.items():
            #     print(f"Short Setting {s}:")
            #     print(f"   Orig Dist: {data['original_distance']:.10f}")
            #     print(f"   Min Dist : {data['min_distance']:.10f}")
            #     print(f"   Lambda   : {data['optimal_noise']:.10f}")
            #     res["AT_probs"][s] = data['closest_distribution']
            # ------------------ End minimize distance ------------------
            break

    return ent

def get_subs(P: ncp.Probability, Z):
    """
    Returns any substitution rules to use with ncpol2sdpa. E.g. projections and
    commutation relations.
    """

    #Alice and Bob measurements are projectors
    subs = P.substitutions

    # Finally we note that Alice and distant BobL's (but not BobS's) operators should All commute with Eve's ops 
    Alice_ops = P.get_extra_monomials("A")
    BobL_ops = [P([j],[k],"B") for k in range(2,len(B_config)) for j in range(B_config[k]-1)]
    for a in Alice_ops+BobL_ops:
        for z in Z:
            subs.update({z*a : a*z, Dagger(z)*a : a*Dagger(z)}) # Commutative Constraints
    return subs

def extra_mono_func(P: ncp.Probability, Z):
    """
    Returns additional monomials to add to sdp relaxation.
    """

    monos = []

    # Add ABZ
    ZZ = Z + [Dagger(z) for z in Z]
    all_As = P.get_extra_monomials("A")
    all_Bs = P.get_extra_monomials("B")
    for a in all_As:
        for b in all_Bs:
            for z in ZZ:
                monos += [a*b*z]


    ###Add monos appearing in objective function according to paper
    for z in Z:
        monos += [P([0],[0],"A")*Dagger(z)*z]
    
    ###Add some more monos
    monos += [a*a_*z*b for a in all_As for a_ in all_As for b in all_Bs[:2]  for z in ZZ]
    monos += [a*a_*b*z for a in all_As for a_ in all_As for b in all_Bs for z in ZZ]
    monos += [z*a*b for a in all_As for b in all_Bs[:2] for z in ZZ]
    # monos += [a*b*b_*_ for a in all_As for b in all_Bs for b_ in all_Bs for z in ZZ]
    # monos += [a*z*b*bb for a in all_As for b in all_Bs for b_ in all_Bs for z in ZZ]
    # monos += [a*b*z*b_ for a in all_As for b in all_Bs for b_ in all_Bs for z in ZZ]
    return monos[:]
    
def objective(ti, q, Z, P):
    """
    Returns the objective fun for the faster computations.
        Only tey generation on X=0

        ti     --    i-th node of the quadraturedrature
        q      --    bit flip probability (not considered)
    """
    obj = 0.0    
    F = [P([0],[0],"A"), P([1],[0],"A")]   # POVM for Alices key gen measurement

    for a in range(A_config[0]):
        b = (a + 1) % 2                     # (a + 1 mod 2)
        M = (1-q) * F[a] + q * F[b]         # Noisy preprocessing povm element
        # From paper
        obj += M * (Z[a] + Dagger(Z[a]) + (1-ti)*Dagger(Z[a])*Z[a]) + ti*Z[a]*Dagger(Z[a])

    return obj

def score_constraints(P, res):
    """
    Returns the moment equality constraints for the distribution.
    """

    constraints = []
    for a in range(nA):
        for b in range(nB):
            for x in range(nX):
        # might need to change nB, because we have unbinned, and only 2 input variables
                for y in range(nY):
                    if not (x == 0 and y == 0):
                        # Because of config [nB,nB,nB,nB,nB], we need to shift y by 2 for BobL test rounds
                        constraints += [P([a,b],[x,y+2]) - res["AB_test_probs"][(x, y)][(a,b)]]
                    if y < 2:
                        constraints += [P([a,b],[x,y]) - res["AT_probs"][(x, y)][(a,b)]]

    return constraints[:]

def score_inequality_constraints(P, res):
    """
    Returns the moment equality constraints for the distribution.
    """

    constraints = []
    for a in range(nA):
        for b in range(nB):
            for x in range(nX):
        # might need to change nB, because we have unbinned, and only 2 input variables
                for y in range(nY):
                    if not (x == 0 and y == 0):
                        # Because of config [nB,nB,nB,nB,nB], we need to shift y by 2 for BobL test rounds
                        constraints += [P([a,b],[x,y+2]) - res["AB_test_probs"][(x, y)][(a,b)] + epsilon]
                        constraints += [-P([a,b],[x,y+2]) + res["AB_test_probs"][(x, y)][(a,b)] + epsilon]
                    if y < 2:
                        constraints += [P([a,b],[x,y]) - res["AT_probs"][(x, y)][(a,b)] + epsilon]
                        constraints += [-P([a,b],[x,y]) + res["AT_probs"][(x, y)][(a,b)] + epsilon]

    return constraints[:]

def SDP_init(res):
    P = ncp.Probability(A_config , B_config)
    Z = ncp.generate_operators('Z', A_config[0], hermitian=False)

    subs = get_subs(P, Z)             # substitutions used in ncpol2sdpa
    extra_monos = extra_mono_func(P, Z)
    
    ops = ncp.flatten([P.get_all_operators(),Z])
    obj = objective(1, 0, Z, P)    # Placeholder objective function

    placeholder_moment_inequality_constraints = [T[0]-z*Dagger(z) for z in Z]+[T[0]-Dagger(z)*z for z in Z] # Adding commutative constraints
    placeholder_moment_inequality_constraints += score_inequality_constraints(P, res)
    sdp = SdpRelaxation(ops, verbose = False, normalized=True)
    sdp.get_relaxation(level = 2,
                        # momentequalities = score_constraints(P, res),
                        momentinequalities = placeholder_moment_inequality_constraints,
                        objective = obj,
                        substitutions = subs,
                        extramonomials = extra_monos)
    start_time = datetime.datetime.now(); print(start_time)
    try:
        # Devetak-Winter formula
        # keyrate_ = compute_entropy(sdp, 0, score_constraints(P, res), Z, P) - res["H_A_given_B"]
        keyrate_ = compute_entropy(sdp,0,score_inequality_constraints(P, res), Z, P) - res["H_A_given_B"]
        if keyrate_ < 0:
            print("keyrate below 0")
    except Exception as error:
        print("Error in computing keyrate: ", error)
        pass
    return keyrate_, sdp.status 
# --------------------------------------------
def pretty_print_P(name, P, is_Key=False):
    """
    Pretty prints the probability table P with a name.
    """
    
    with np.printoptions(precision=4, suppress=True):
        print(f"{name}:")
        if not is_Key:
            for key in sorted(P.keys()):
                print(f"  Setting {key}: {P[key]}")
                sum_probs = sum(P[key].values())
                print(f"    (Sum of probabilities: {sum_probs})")
        else:
            print(f"  {P}")
            sum_probs = sum(P.values())
            print(f"    (Sum of probabilities: {sum_probs})")
            
def normalize_P(P):
    """
    Normalizes the probability table P.
    """
    for key in P.keys():
        total = sum(P[key].values())
        if total > 0:
            for ab in P[key].keys():
                P[key][ab] /= total
    
    return P  

def enforce_routed_ns(P_sp_raw, P_lp_raw):
    """
    Projects raw probability tables onto the No-Signaling (NS) polytope
    using Least Squares minimization.
    
    Enforces:
    1. Normalization & Positivity.
    2. Alice's NS (Global): P(a|x) must be consistent across ALL y (Long Path)
       and ALL z (Short Path). This fixes the switch loophole.
    3. Bob's/Test's NS: P(b|y) must be independent of x, and P(c|z) independent of x.
    """
    
    # --- 1. Helper to parse structure and unique keys ---
    def get_keys(d):
        inputs = list(d.keys())
        # Assume outcomes are the same for all inputs, grab from first
        outcomes = list(d[inputs[0]].keys()) 
        return inputs, outcomes

    sp_inputs, sp_outcomes = get_keys(P_sp_raw)
    lp_inputs, lp_outcomes = get_keys(P_lp_raw)
    
    # Identify unique x, y, z values
    # Structure assumption: inputs are tuples (x, z) or (x, y)
    all_x = sorted(list(set([k[0] for k in sp_inputs] + [k[0] for k in lp_inputs])))
    all_z = sorted(list(set([k[1] for k in sp_inputs])))
    all_y = sorted(list(set([k[1] for k in lp_inputs])))
    
    # Map outcomes to indices (e.g., (0,0) -> 0, (0,1) -> 1 ...)
    out_map = {k: i for i, k in enumerate(sp_outcomes)}
    num_outcomes = len(sp_outcomes) # Typically 4 for 2 qubits (00, 01, 10, 11)
    
    # --- 2. Define Variables ---
    # We create a variable vector for every input setting
    # vars_sp[(x,z)] is a vector of size 4 (outcomes)
    vars_sp = {k: cp.Variable(num_outcomes, nonneg=True) for k in sp_inputs}
    vars_lp = {k: cp.Variable(num_outcomes, nonneg=True) for k in lp_inputs}
    
    # --- 3. Define Constraints ---
    constraints = []
    
    # A. Normalization (Sum of probs = 1)
    for v in vars_sp.values(): constraints.append(cp.sum(v) == 1)
    for v in vars_lp.values(): constraints.append(cp.sum(v) == 1)
    
    # B. Alice's Global Non-Signaling (The "Switch" Constraint)
    # We define a variable for Alice's marginal P(a|x) for each x
    # Alice has 2 outcomes (0 or 1), derived from the joint outcome (a,b)
    # Assuming standard mapping: (0,0)->A=0, (0,1)->A=0, (1,0)->A=1, (1,1)->A=1
    # We need a matrix to sum over Bob's outcomes to get Alice's marginals
    # shape (2, 4): [[1, 1, 0, 0], [0, 0, 1, 1]]
    marg_matrix_A = np.array([[1, 1, 0, 0], [0, 0, 1, 1]]) 
    
    marg_A = {x: cp.Variable(2) for x in all_x}

    # Constrain SP rounds to match Alice's global marginal
    for (x, z), v in vars_sp.items():
        # Sum over 'c' to get P(a|x, z), must equal P(a|x)
        constraints.append(marg_matrix_A @ v == marg_A[x])
        
    # Constrain LP rounds to match Alice's global marginal
    for (x, y), v in vars_lp.items():
        # Sum over 'b' to get P(a|x, y), must equal P(a|x)
        constraints.append(marg_matrix_A @ v == marg_A[x])
        
    # C. Bob/Test Non-Signaling
    # Bob's marginal P(b|y) must be independent of x
    # Matrix to sum over Alice's outcomes: [[1, 0, 1, 0], [0, 1, 0, 1]]
    marg_matrix_B = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
    
    # Enforce P(c|z) independent of x (Test Device)
    for z in all_z:
        # Collect all variables with this z
        current_vars = [vars_sp[(x, z)] for x in all_x if (x, z) in vars_sp]
        if len(current_vars) > 1:
            base_marg = marg_matrix_B @ current_vars[0]
            for v in current_vars[1:]:
                constraints.append(marg_matrix_B @ v == base_marg)

    # Enforce P(b|y) independent of x (Bob Device)
    for y in all_y:
        current_vars = [vars_lp[(x, y)] for x in all_x if (x, y) in vars_lp]
        if len(current_vars) > 1:
            base_marg = marg_matrix_B @ current_vars[0]
            for v in current_vars[1:]:
                constraints.append(marg_matrix_B @ v == base_marg)

    # --- 4. Define Objective (Least Squares) ---
    obj_terms = []
    
    for k, v in vars_sp.items():
        raw_vec = np.array([P_sp_raw[k].get(out, 0) for out in sp_outcomes])
        obj_terms.append(cp.sum_squares(v - raw_vec))
        
    for k, v in vars_lp.items():
        raw_vec = np.array([P_lp_raw[k].get(out, 0) for out in sp_outcomes])
        obj_terms.append(cp.sum_squares(v - raw_vec))
        
    prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
    
    # --- 5. Solve and Reconstruct ---
    try:
        prob.solve()
    except cp.SolverError:
        prob.solve(solver=cp.SCS) # Fallback solver

    def reconstruct(vars_dict, outcomes):
        new_dict = {}
        for k, v in vars_dict.items():
            vals = v.value
            # Clip small negatives due to float precision and re-normalize
            vals = np.maximum(vals, 0)
            vals /= np.sum(vals)
            new_dict[k] = {out: vals[i] for i, out in enumerate(outcomes)}
        return new_dict

    return reconstruct(vars_sp, sp_outcomes), reconstruct(vars_lp, sp_outcomes)

def verify_nonsignaling(data_table, label, shots=10000, alpha=0.05):
    print(f"\n--- {label} Results ---")
    settings_x = sorted(list(set(s[0] for s in data_table.keys())))
    settings_y = sorted(list(set(s[1] for s in data_table.keys())))
    
    # Alice Test: Independence from Bob's setting y
    for x in settings_x:
        observations = []
        relevant_ys = [y for y in settings_y if (x, y) in data_table]
        if len(relevant_ys) < 2: continue
        
        for y in relevant_ys:
            p = data_table[(x, y)]
            # Alice's outcome 'a' is the second value in (b, a)
            a0 = (p.get((0, 0), 0) + p.get((1, 0), 0)) * shots
            a1 = (p.get((0, 1), 0) + p.get((1, 1), 0)) * shots
            observations.append([a0, a1])
        
        _, p_val, _, _ = chi2_contingency(observations)
        status = "FAIL" if p_val < alpha else "PASS"
        print(f"Alice @ x={x}: {status} (p = {p_val:.10f})")

    # Bob Test: Independence from Alice's setting x
    for y in settings_y:
        observations = []
        relevant_xs = [x for x in settings_x if (x, y) in data_table]
        if len(relevant_xs) < 2: continue
        
        for x in relevant_xs:
            p = data_table[(x, y)]
            # Bob's outcome 'b' is the first value in (b, a)
            b0 = (p.get((0, 0), 0) + p.get((0, 1), 0)) * shots
            b1 = (p.get((1, 0), 0) + p.get((1, 1), 0)) * shots
            observations.append([b0, b1])
        
        _, p_val, _, _ = chi2_contingency(observations)
        status = "FAIL" if p_val < alpha else "PASS"
        print(f"Bob   @ y={y}: {status} (p = {p_val:.10f})")

def add_noise_and_minimize(P_empirical, P_ideal, metric='TVD'):
    """
    For each measurement setting in P_empirical, finds the optimal amount of white noise
    to add such that the distance to P_ideal is minimized.
    Args:
        P_empirical: dict keyed by setting -> dict of (a,b) -> probabilities
        P_ideal: dict keyed by setting -> dict of (a,b) -> probabilities
        metric: Distance metric to use ('TVD' for Total Variation Distance)
    Returns:
    results: dict keyed by setting -> dict with keys:
        'original_distance': distance before noise
        'min_distance': distance after optimal noise
        'optimal_noise': optimal noise parameter
        'closest_distribution': the resulting closest distribution after noise
    """
    outcomes = [(0,0), (0,1), (1,0), (1,1)]
    P_white = np.array([0.25, 0.25, 0.25, 0.25])
    results = {}
    
    for setting in P_ideal:
        if setting not in P_empirical: continue
        
        vec_emp = np.array([P_empirical[setting].get(o, 0.0) for o in outcomes])
        vec_ideal = np.array([P_ideal[setting].get(o, 0.0) for o in outcomes])
        
        def calc_dist(p_vec, q_vec):
            return 0.5 * np.sum(np.abs(p_vec - q_vec))

        orig_dist = calc_dist(vec_emp, vec_ideal)
        
        def objective(lam):
            p_mixed = (1 - lam) * vec_emp + lam * P_white
            return calc_dist(p_mixed, vec_ideal)
            
        res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
        
        # Calculate the actual best distribution vector: (1-L)*Emp + L*Noise
        best_vec = (1 - res.x) * vec_emp + res.x * P_white
        best_dist = {outcomes[i]: best_vec[i] for i in range(len(outcomes))}
        
        results[setting] = {
            'original_distance': orig_dist,
            'min_distance': res.fun,
            'optimal_noise': res.x,
            'closest_distribution': best_dist
        }
        
    return results


def distance_avg(P_empirical, P_ideal, metric='TVD'):
    """
    For each measurement setting in P_empirical it finds the distance between its born ideal distribution
    and the empirical one.
    Args:
        P_empirical: dict keyed by setting -> dict of (a,b) -> probabilities
        P_ideal: dict keyed by setting -> dict of (a,b) -> probabilities
        metric: Distance metric to use ('TVD' for Total Variation Distance)
    Returns:
    results: 
        'original_distance': average distance before noise
    """
    outcomes = [(0,0), (0,1), (1,0), (1,1)]
    orig_dist = 0.0
    
    for setting in P_ideal:
        if setting not in P_empirical: continue
        
        vec_emp = np.array([P_empirical[setting].get(o, 0.0) for o in outcomes])
        vec_ideal = np.array([P_ideal[setting].get(o, 0.0) for o in outcomes])
        
        def calc_dist(p_vec, q_vec):
            return 0.5 * np.sum(np.abs(p_vec - q_vec))

        orig_dist += calc_dist(vec_emp, vec_ideal)
        
    return orig_dist/len(P_ideal)

def add_noise_to_P_empirical(P_empirical, noise):
    """
    We add white noise to the empirical distribution P_empirical.
    Since SDP should be optimal for White Noise.
    
    """
    outcomes = [(0,0), (0,1), (1,0), (1,1)]
    P_white = np.array([0.25, 0.25, 0.25, 0.25])
    results = {}
    
    for setting in P_empirical:
        if setting not in P_empirical: continue
        
        vec_emp = np.array([P_empirical[setting].get(o, 0.0) for o in outcomes])
        
        
        best_vec = (1 - noise) * vec_emp + noise * P_white
        best_dist = {outcomes[i]: best_vec[i] for i in range(len(outcomes))}
        
        results[setting] = best_dist
        
    return results

def load_born_ideal_distributions(v):
    """
    Loads the ideal Born distributions for both short and long paths from JSON files.
    """
    # ------------------- Use pre-simulated data from results folder -------------------
    # Read Born probabilities from json file in results and convert into dict format
    try:
        with open(ROOT_DIR / 'Born_Tables.json', 'r') as f:
            born_ideal_data = json.load(f)
            if v == 1.0:
                v = '1'
            else:
                v = str(v)
            born_ideal_data = born_ideal_data[v]
            born_sp = born_ideal_data['AT_probs']
            born_lp = born_ideal_data['AB_test_probs']
            born_key = born_ideal_data['AB_key_probs']
            def convert_keys_to_tuples(d):
                """
                Recursively converts dictionary keys from string representations of tuples 
                (e.g., "(0, 0)") back to actual tuple objects (e.g., (0, 0)).
                """
                new_dict = {}
                for k, v in d.items():
                    # 1. Recursively process the value if it's a dictionary
                    if isinstance(v, dict):
                        v = convert_keys_to_tuples(v)
                        
                    # 2. Process the key
                    new_key = k
                    # Check if key is a string and looks like a tuple
                    if isinstance(k, str) and k.strip().startswith('(') and k.strip().endswith(')'):
                        try:
                            # safely parse the string "(0,0)" into the tuple (0,0)
                            parsed_key = ast.literal_eval(k)
                            if isinstance(parsed_key, tuple):
                                new_key = parsed_key
                        except (ValueError, SyntaxError):
                            # If parsing fails, keep the original string key
                            pass
                    
                    new_dict[new_key] = v
                    
                return new_dict
            
            born_sp = convert_keys_to_tuples(born_sp)
            born_lp = convert_keys_to_tuples(born_lp)
            del born_lp[(0,0)]  # Remove key generation round from long path ideal
            born_key = convert_keys_to_tuples(born_key)
    except FileNotFoundError:
        print(f"Born ideal with visibility {v} not found in results folder.")
    
    return born_sp, born_lp, born_key

def calculate_hoeffding_error(n: int, confidence: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """
    Calculates the necessary epsilon for Hoeffding's inequality given sample size and confidence level.
    
    Formula: P(|X_bar - mu| >= epsilon) <= 2 * exp(-2 * n * epsilon^2 / (b - a)^2)
    
    Args:
        n (int): Sample size (number of observations).
        confidence (float): The desired confidence level (1 - probability bound).
        lower (float): The minimum possible value of the variable (a).
        upper (float): The maximum possible value of the variable (b).
        
    Returns:
        float: epsilon value for the inequality.
    """
    
    return np.sqrt(-np.log((1 - confidence) / 2) * (upper - lower)**2 / (2 * n))
# --------------------------------------------
    
if __name__ == "__main__":
    ROOT_DIR = Path(".")
    # ROOT_DIR = Path("./code_handin")
    nA = 2; nB = 2; nX = 2; nY = 3
    # Since the best results are obtained when both BobS and BobL bin no-clicks into +1 for test rounds, we only consider this case here.

    KEEP_M = False   
    consider_no_clicks = True                  # Whether to consider no-click events in the SDP -> The whole point
    
    q = QuantumRegister(2, "q")   # q[0]=Alice, q[1]=travelling qubit
    c = ClassicalRegister(2, "c")
    
    A_config = [nA,nA]
    B_config = [nB,nB,nB,nB,nB] ##The first two are for BobS and the second two are for BobL
    # We only consider the case where both BobS and BobL bin no-clicks into +1 for test rounds, only key generation round, which is not necessary for the SDP, remains unbinned.
    
    gaussian_order = 6                           # Number of nodes / 2 in gaussian quadrature
    T, W = generate_quadrature(gaussian_order)      # Nodes, weights of quadrature

    p_S=0.3
    p_X=(0.5,0.5)
    p_Z=(0.5,0.5)
    p_Y=(0.7,0.15,0.15)

    # --------------------------- 
    # --------------------------- Iterate through different experimental settings --------------------------- 
    # --------------------------- 
    start_time = datetime.datetime.now()    

    with open(ROOT_DIR / 'settings.json', 'r') as f:
        settings = json.load(f)["aer"]
        for setting in settings:
            N = setting["numeral_settings"]["N_rounds"]          # Number of rounds
            n_shots = setting["numeral_settings"]["n_shots"]      # Number of shots per round
            hoeffding_confidence = setting["numeral_settings"]["hoeffding_confidence"]  # Desired confidence level for Hoeffding's inequality
            epsilon = calculate_hoeffding_error(n_shots, hoeffding_confidence)  # Tolerance for inequality constraints in SDP
            # epsilon = 1e-3  # Tolerance for inequality constraints in SDP
            
            # ------------- Flags to control SDP constraints -------------
            increase_p_long = setting["flags"]["increase_p_long"]
            decrease_v = setting["flags"]["decrease_v"]
            # ----------------------------------------------------------
            # Tolerance for inequality constraints in SDP
            # ------------- Parameters to adjust experimenting with different settings -------------
            init_LP_noise = setting["numeral_settings"]["init_LP_noise"]      # Initial long-path noise parameter
            p_long = init_LP_noise
            
            init_v = setting["numeral_settings"]["init_v"]          # Initial visibility
            v = init_v
            
            decline_margin = setting["numeral_settings"]["decline"]  # Margin to decline visibility or increase p_long
            # ------------------------------------------------------------------------------
            
            x_list = [random.choices([0,1], p_X)[0] for _ in range(N)]
            s_list = [random.choices([0, 1], [p_S, 1-p_S])[0] for _ in range(N)]
            z_list = [random.choices([0, 1], p_Z)[0] for _ in range(N)]      # used if s=0
            y_list = [random.choices([0, 1, 2], p_Y)[0] for _ in range(N)]      # used if s=1

                

            sdp_status = "optimal"
            try:
                while v>=0.95 and p_long<=0.1 and str.lower(sdp_status) == "optimal":
# --------------------------- START: Process the empirical distributions --------------------------
                    noise_model = apply_noise(v=v)
                    backend = AerSimulator(noise_model=noise_model)
                    start_time_rounds = datetime.datetime.now()
                    print("Start of rounds: ", start_time_rounds)
                    circuits = []
                    for i in range(N):
                        circuits.append(
                            routed_round_circuit(
                                x=x_list[i],
                                s=s_list[i],
                                z=z_list[i] if s_list[i] == 0 else None,
                                y=y_list[i] if s_list[i] == 1 else None,
                                p_long=p_long
                            )
                        )
                    probs_list = run_circuits_get_probs(circuits, shots=n_shots, backend=backend) #, seed=random.randint(1,1e6)) 
                    P_sp, P_lp, N_sp, N_lp = accumulate_tables(probs_list, x_list, s_list, z_list, y_list)
                    
                    # ---------------------------------------------------------
                    P_key = P_lp[(0,0)]  # Key generation round is x=0,y=0
                    P_lp_param = {k:v for k,v in P_lp.items() if k != (0,0)} # All test rounds for BobL used for Parameter Estimation in SDP
                    print("End of rounds: ", datetime.datetime.now())
                    
                    # S_lp = chsh_from_table(P_lp, 1, 2)   # BobL uses y=1,2 -> y=0 is keygen round
# --------------------------- END: Process the empirical distributions --------------------------
                    born_sp, born_lp, born_key = load_born_ideal_distributions(v)
                    # for commutative in [True, False]: / obviously it is always the same
                    for commutative in [True]:
                        # Commutative constraints for the same qiskit simulated data
                        commutative_constraints = commutative
                        # ---------------------------
                        outputs_of_outputs = {}
                        etaAs = [0.99,0.96]
                        for etaA in etaAs:
                            etaBS = etaA; 
                            # etaBLs = np.arange(0.7,0.87,0.05).tolist() + np.linspace(0.92,0.99,8).tolist()[:-1]
                            etaBLs = np.arange(0.5,0.78,0.05).tolist() + np.linspace(0.8,1,9).tolist()
                            etaBLs.reverse()
                            outputs = []
                            for etaBL in etaBLs:
                                processed_P_sp = P_sp.copy()
                                processed_P_lp = P_lp.copy()
                                # START: Apply transmission losses to the empirical distributions ---------
                                if consider_no_clicks:
                                    # Apply losses with no-click events considered
                                    for key in processed_P_sp.keys():
                                        p_matrix = dict_to_matrix(processed_P_sp[key])
                                        p_matrix_eta = apply_losses_with_noclick(p_matrix, etaA, etaBS, keep_no_click=False)  # BobS always bins no-clicks into +1 for test rounds
                                        processed_P_sp[key] = {(a,b): p_matrix_eta[a,b] for a in range(p_matrix_eta.shape[0]) for b in range(p_matrix_eta.shape[1])}

                                    for key in processed_P_lp.keys():
                                        p_matrix = dict_to_matrix(processed_P_lp[key])
                                        p_matrix_eta = apply_losses_with_noclick(p_matrix, etaA, etaBL, keep_no_click=False)  # BobL always bins no-clicks into +1 for test rounds
                                        processed_P_lp[key] = {(a,b): p_matrix_eta[a,b] for a in range(p_matrix_eta.shape[0]) for b in range(p_matrix_eta.shape[1])}
                                processed_P_key = processed_P_lp[(0,0)]  # Key generation round is x=0,y=0
                                processed_P_lp_param = {k:v for k,v in processed_P_lp.items() if k != (0,0)} # All test rounds for BobL used for Parameter Estimation in SDP
                                # END --------------------------------------------------------------------
                                # ------ START: Calculating parameters for keyrate estimation --------
                                S_sp = chsh_from_table(processed_P_sp, 0, 1)   # BobS uses z=0,1
                                Hc = H_A_given_B(processed_P_key)
                                res = {"AT_probs": processed_P_sp, "AB_test_probs": processed_P_lp_param, "AB_key_probs": processed_P_key,  "S": float(S_sp), "H_A_given_B": float(Hc)}
                                # ------ END: Calculating parameters for keyrate estimation --------
                                SDP_start_time = datetime.datetime.now(); 
                                print("Start of SDP: ", SDP_start_time)

                                keyrate_, sdp_status = SDP_init(res)
                                
                                outputs.append({
                                    "Status": sdp_status,
                                    "etaBL": etaBL,
                                    "keyrate": keyrate_,
                                    "Average Distance To Ideal Distributions": {
                                        "SP": distance_avg(processed_P_sp, born_sp),
                                        "LP": distance_avg(processed_P_lp, born_lp),
                                        "Key": distance_avg({(0,0):processed_P_key}, born_key)
                                    },
                                    "Qiskit_data": res
                                })
                                SDP_end_time = datetime.datetime.now()
                                print("End of SDP for: ", SDP_end_time)
                                print(f"Key-Rate with visibility {v}, long-path noise {p_long} and etaBL {etaBL}: {keyrate_}")
                                if keyrate_ < 0:
                                    break
                            print(f"Finished etaA={etaA} for visibility {v} and p_long {p_long}.")
                            outputs_of_outputs[str(etaA)] = outputs
                        ### Save the outputs to a file --------------------------------
                        def stringify_keys(data):
                            if isinstance(data, dict):
                                # Convert keys to string and recurse through values
                                return {str(k): stringify_keys(v) for k, v in data.items()}
                            elif isinstance(data, list):
                                # Recurse through list items
                                return [stringify_keys(i) for i in data]
                            else:
                                # Return the value as is (base case)
                                return data
                        results_dir = ROOT_DIR / 'results'
                        results_dir.mkdir(parents=True, exist_ok=True)
                        with open(results_dir / (str(SDP_start_time).replace(" ", "_").replace(":", "_")+f'_chsh_bin_vis_{v:.2f}_pLong_{p_long:.2f}_comm_{commutative_constraints}.json'), 'w') as f:
                            json.dump({
                                "settings": {
                                    "Commutative Constraints": commutative_constraints,
                                    "N_rounds": N,
                                    "Shots per round": n_shots,
                                    "Relaxation margin for constraints, Hoeffding Confidence": [epsilon, hoeffding_confidence],
                                    "visibility": v,
                                    "p_long": p_long,
                                    "init_v": init_v,
                                    "init_LP_noise": init_LP_noise,
                                    "decline": decline_margin,
                                    
                                    },
                                "keyrate_results": stringify_keys(outputs_of_outputs)
                                }, f)
                        # --------------------------------------------------------------
                    if outputs_of_outputs["0.99"][0]["keyrate"] < 0:
                        print("First Keyrate already below 0, breaking noise loop on v:", v, "and p_long:", p_long)
                        break
                    elif increase_p_long or decrease_v:
                        if increase_p_long:
                            p_long += decline_margin
                        if decrease_v:
                            v -= decline_margin
                print("Exited while loop with v:", v, "and p_long:", p_long, "and sdp status:", sdp_status)
            except Exception as err:
                print("Error message: ", err)
                pass

            
            end_time = datetime.datetime.now()
            print(f"Finished at :", end_time)

    print("Total runtime:", datetime.datetime.now() - start_time)