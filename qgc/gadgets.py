"""Section 6.1: the teleportation and correction gadgets.

Lemma 6.2 factors "correct, then teleport" into three circuits

    Lambda3 . Lambda2(R, l, s, t) . Lambda1(l)  ==  TP_{l,s,t} . (R on u)

Enc applies Lambda1 and the randomizer A; Dec applies Corr = Lambda2 . A^dag
and then Lambda3. Lambda2 is built here too because Enc needs it to tabulate
the correction function (Section 6.2.1), even though in CRE mode Enc never
applies it.

C1, C2, C3 and Gamma are the pieces of Proposition 6.3, proved in Appendix C,
which pushes a PX-group correction R past the teleportation measurement.

Every circuit here is over the gadget register order  u, v, z, b_0..b_kappa, x.
The Lemma 6.2 identity holds only when b starts in |0>, and it returns b to |0>.
"""
import numpy as np
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import random_clifford

from qgc.helper import get_bit
from qgc.structure import Correction


def _registers(circuit, kappa, with_x=True):
    """Add z, b_0..b_kappa and optionally x to a circuit that already has u, v."""
    z = QuantumRegister(kappa, name="z")
    circuit.add_register(z)
    b = []
    for i in range(kappa + 1):
        b.append(AncillaRegister(kappa + 1, name="b{}".format(i)))
        circuit.add_register(b[i])
    x = None
    if with_x:
        x = QuantumRegister(kappa, name="x")
        circuit.add_register(x)
    return z, b, x


def _uzb_mapping(kappa, z, b):
    """Qubit list for a subcircuit acting on u, z, b only."""
    mapping = [0, *z]
    for i in range(kappa + 1):
        mapping.extend(b[i])
    return mapping


# ------------------------------------------------------- Enc side: C1, Lambda1
def construct_c1(l_z_xor: int, kappa: int) -> QuantumCircuit:
    """Hadamards on z, the parity gate +(r), then the descending fan-outs.

    Appendix C: conjugating the fan-out CX(r) by Hadamards turns it into the
    parity gate, which is what lets R be pushed through.
    """
    circuit = QuantumCircuit(1)
    z, b, _ = _registers(circuit, kappa, with_x=False)

    for i in range(0, kappa):
        circuit.h(z[i])
        if get_bit(l_z_xor, i):
            circuit.cx(z[i], 0)                 # parity +(r) into u

    circuit.cx(0, b[0])                         # descending fan-outs
    for i in range(kappa):
        circuit.cx(z[i], b[i + 1])

    return circuit


def construct_lambda1(l_z: tuple[int, ...], l_x: tuple[int, ...],
                      kappa: int) -> QuantumCircuit:
    """The part of the teleportation gadget Enc can apply without knowing R."""
    circuit = QuantumCircuit(2)
    z, b, x = _registers(circuit, kappa)

    circuit.cx(0, 1)

    for i in range(0, kappa):
        if get_bit(l_z[0], i):
            circuit.x(z[i])
        if get_bit(l_x[0], i):
            circuit.x(x[i])

    circuit.compose(construct_c1(l_z[0] ^ l_z[1], kappa),
                    qubits=_uzb_mapping(kappa, z, b), inplace=True)

    l_x_xor = l_x[0] ^ l_x[1]
    for i in range(0, kappa):
        if get_bit(l_x_xor, i):
            circuit.cx(1, x[i])                 # fan-out from v writes l_{x,e}

    return circuit


def construct_classical_teleport(l_z: tuple[int, ...], l_x: tuple[int, ...],
                                 s_x: bool, t_x: bool, kappa: int) -> QuantumCircuit:
    """Figure 5: the classical teleportation gadget, for a classical input bit.

    Against Figure 3 this drops the Hadamard on u and the fan-out into z, so
    there is no Z-basis measurement: d is pinned to 0 and z always holds
    l_{z,0}. What is left is CNOTs and bit flips, which a classical party can
    run -- the paper's "classical encoding for classical inputs".

    With the EPR pair in |rr>, u = y and v = r, the CNOT makes v = r XOR y = e,
    the fan-out writes l_{x,e} into x, and the far half of the pair is left
    holding r = e XOR y: the input one-time-padded with e, which the evaluator
    recovers from the label. b is never touched and stays |0>, so no Lambda3.
    """
    circuit = QuantumCircuit(2)
    z, _, x = _registers(circuit, kappa)

    circuit.cx(0, 1)                            # v = r XOR y = e

    for i in range(kappa):
        if get_bit(l_z[0], i):
            circuit.x(z[i])                     # z = l_{z,0}, always
        if get_bit(l_x[0], i):
            circuit.x(x[i])

    l_x_xor = l_x[0] ^ l_x[1]
    for i in range(kappa):
        if get_bit(l_x_xor, i):
            circuit.cx(1, x[i])                 # x = l_{x,e}

    if s_x:
        circuit.x(0)
    if t_x:
        circuit.x(1)
    return circuit


def sample_a(kappa: int, rng=None) -> QuantumCircuit:
    """A uniformly random element of the randomization group R_kappa.

    `rng` must be a numpy Generator. Passing an int here would seed every draw
    identically and make A a constant, which silently destroys privacy: the
    paper needs A uniform so that Corr = Lambda2(R).A^dag is uniform over all
    correction gadgets and hence hides R (Section 6.1.2). Correctness holds for
    any fixed A, so no correctness test can catch that.
    """
    rng = np.random.default_rng() if rng is None else rng
    circuit = QuantumCircuit(2)
    z, b, x = _registers(circuit, kappa)

    circuit.compose(random_clifford(1, rng).to_circuit(), qubits=0, inplace=True)
    circuit.compose(random_clifford(1, rng).to_circuit(), qubits=1, inplace=True)
    for i in range(0, kappa):
        circuit.compose(random_clifford(1, rng).to_circuit(), qubits=[x[i]], inplace=True)
        circuit.compose(random_clifford(1, rng).to_circuit(), qubits=[z[i]], inplace=True)

    # single-qubit Cliffords on the diagonal, two-qubit Cliffords on each
    # mirrored pair (b_ij, b_ji) -- exactly the slots of R_kappa
    for j in range(kappa + 1):
        circuit.compose(random_clifford(1, rng).to_circuit(),
                        qubits=[b[j][j]], inplace=True)
        for i in range(j):
            circuit.compose(random_clifford(2, rng).to_circuit(),
                            qubits=[b[i][j], b[j][i]], inplace=True)

    return circuit


# --------------------------------------------- Dec side: Gamma, C2, Lambda2, C3
def construct_gamma(p: int, r: int, kappa: int) -> QuantumCircuit:
    """Appendix C Case 3: the CZ layer that absorbs a P correction.

    CZ(b_0j, b_j0) for every j with r_j = 1, and CZ(b_ij, b_ji) for every
    i < j with r_i = r_j = 1. All disjoint, hence depth one.
    """
    circuit = QuantumCircuit()
    b = []
    for i in range(kappa + 1):
        b.append(AncillaRegister(kappa + 1, name='b{}'.format(i)))
        circuit.add_register(b[i])

    if p == 0:
        return circuit

    for j in range(1, kappa + 1):
        if get_bit(r, j - 1):
            circuit.cz(b[0][j], b[j][0])
        for i in range(1, j):
            if get_bit(r, j - 1) and get_bit(r, i - 1):
                circuit.cz(b[i][j], b[j][i])

    return circuit


def construct_c2(corr: Correction, r: int, s_x: bool, s_z: bool,
                 kappa: int) -> QuantumCircuit:
    """Appendix C Circuit 14: R = X^x Z^z P^p pushed past the fan-outs.

    P^p on u and P^{p r_i} on z_i, plus Gamma; then Z, then X, with the X also
    landing on all of b_0. Finally the s randomization, Z^{s_x} X^{s_z}.
    """
    circuit = QuantumCircuit(1)
    z, b, _ = _registers(circuit, kappa, with_x=False)

    if corr.p == 1:
        circuit.s(0)
        for i in range(kappa):
            if get_bit(r, i):
                circuit.s(z[i])
    mapping = []
    for i in range(0, kappa + 1):
        mapping.extend(b[i])
    circuit.compose(construct_gamma(corr.p, r, kappa), qubits=mapping, inplace=True)

    if corr.z == 1:
        circuit.z(0)
        for i in range(kappa):
            if get_bit(r, i):
                circuit.z(z[i])
    if corr.x ^ s_z:
        circuit.x(b[0])
    if corr.x:
        circuit.x(0)
    if s_x:
        circuit.z(0)
    if s_z:
        circuit.x(0)

    return circuit


def construct_lambda2(corr: Correction, l_z: tuple[int, ...], l_x: tuple[int, ...],
                      s_x: bool, s_z: bool, t_x: bool, t_z: bool,
                      kappa: int) -> QuantumCircuit:
    """C2 on (u, z, b), then R3 on x and R4 with the t randomization on v.

    Lies in R_kappa, which is what lets it be described as a gate word per slot.
    """
    circuit = QuantumCircuit(2)
    z, b, x = _registers(circuit, kappa)

    circuit.compose(construct_c2(corr, l_z[0] ^ l_z[1], s_x, s_z, kappa),
                    qubits=_uzb_mapping(kappa, z, b), inplace=True)

    # R2 = X^x propagated past the fan-out from v gives R3 on x and R4 on v
    l_x_xor = l_x[0] ^ l_x[1]
    if corr.x:
        circuit.x(1)
        for i in range(kappa):
            if get_bit(l_x_xor, i):
                circuit.x(x[i])

    if t_x:
        circuit.x(1)
    if t_z:
        circuit.z(1)

    return circuit


def construct_c3(kappa: int) -> QuantumCircuit:
    """The ascending fan-outs and the closing Hadamards. Independent of R, s."""
    circuit = QuantumCircuit(1)
    z, b, _ = _registers(circuit, kappa, with_x=False)

    circuit.cx(0, b[0])
    circuit.h(0)
    for i in range(kappa):
        circuit.cx(z[i], b[i + 1])
        circuit.h(z[i])

    return circuit


def construct_lambda3(kappa: int) -> QuantumCircuit:
    """C3 on (u, z, b). Applied by Dec, after Corr."""
    circuit = QuantumCircuit(2)
    z, b, _ = _registers(circuit, kappa)
    circuit.compose(construct_c3(kappa), qubits=_uzb_mapping(kappa, z, b),
                    inplace=True)
    return circuit
