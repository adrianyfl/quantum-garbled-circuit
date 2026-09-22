"""Equation (6.1): propagating Pauli corrections through a gate.

For a p-qubit gate U and single-qubit Paulis P_1..P_p there are PX-group
R_1..R_p with U (P_1 (x) ... (x) P_p) = (R_1^dag (x) ... (x) R_p^dag) U. The
gate set C_2 u {T} is chosen precisely so this holds; gateset.py enforces it.

apply_coherent_correction is the direct-mode path only, where the encoder
applies the correction itself in the clear. That is a reference oracle for
testing and has no privacy; the real scheme garbles the correction instead
(see correction_function and cdec).
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.quantum_info import Clifford, Pauli

from qgc.helper import POINTER_BIT, get_bit
from qgc.structure import Correction


def propagate_correction(operation: Instruction,
                         input_frames: list[tuple[bool, bool]]) -> list[Correction]:
    """(d, e) per input wire -> the R each output wire needs."""
    name = operation.name.lower()

    # T X^e Z^d = P^e X^e Z^d T, so the T gate is the one that leaves the
    # Pauli group and produces a P component
    if name == "t":
        (d, e) = input_frames[0]
        return [Correction(x=e, z=(d != e), p=e)]

    cliff = Clifford(operation)
    z_bits = np.array([d for d, e in input_frames], dtype=bool)
    x_bits = np.array([e for d, e in input_frames], dtype=bool)
    out_pauli = Pauli((z_bits, x_bits)).evolve(cliff, frame="s")
    return [
        Correction(x=bool(out_pauli.x[i]), z=bool(out_pauli.z[i]), p=False)
        for i in range(len(input_frames))
    ]


def get_witnesses(circuit: QuantumCircuit, segment, record, kappa):
    """The colour qubit holds v XOR lam; undo the known lam to leave v.

    Direct mode only, where the encoder legitimately knows the labels. Dec gets
    its bits from the d^w dictionary instead (see decode_label_bit in garble).
    """
    if get_bit(record.l_z[0], POINTER_BIT):
        circuit.x(segment.z[POINTER_BIT])
    if get_bit(record.l_x[0], POINTER_BIT):
        circuit.x(segment.x[POINTER_BIT])
    return segment.z[POINTER_BIT], segment.x[POINTER_BIT]


def linear_response(operation, num_inputs):
    """How each input frame bit on its own moves the correction.

    The Clifford part of the correction is linear in (d, e), so flipping one
    bit at a time recovers the whole map.
    """
    zero_frames = [(False, False)] * num_inputs
    baseline = propagate_correction(operation, zero_frames)
    if any(c.x or c.z or c.p for c in baseline):
        raise ValueError("propagate_correction has a nonzero baseline")

    responses = []
    for j in range(num_inputs):
        for which in ("d", "e"):
            frames = list(zero_frames)
            d, e = frames[j]
            frames[j] = (True, e) if which == "d" else (d, True)
            responses.append((j, which, propagate_correction(operation, frames)))
    return responses


def apply_coherent_correction(circuit, operation, input_witnesses, output_qubits):
    """Direct mode: apply the correction controlled on the witness qubits."""
    name = operation.name.lower()

    if name == "t":
        d_witness, e_witness = input_witnesses[0]
        out = output_qubits[0]
        circuit.cp(np.pi / 2, e_witness, out)
        circuit.cz(d_witness, out)
        circuit.cz(e_witness, out)
        circuit.cx(e_witness, out)
        return

    responses = linear_response(operation, len(input_witnesses))

    for j, which, result in responses:
        witness = input_witnesses[j][0 if which == "d" else 1]
        for k, corr in enumerate(result):
            if corr.x:
                circuit.cx(witness, output_qubits[k])
            if corr.z:
                circuit.cz(witness, output_qubits[k])

    p_contributors = [[] for _ in output_qubits]
    for j, which, result in responses:
        witness = input_witnesses[j][0 if which == "d" else 1]
        for k, corr in enumerate(result):
            if corr.p:
                p_contributors[k].append(witness)
    for k, contributors in enumerate(p_contributors):
        if len(contributors) > 1:
            raise NotImplementedError(
                "two P contributions on one output wire would not compose as a "
                "single controlled-phase")
        for witness in contributors:
            circuit.cp(np.pi / 2, witness, output_qubits[k])
