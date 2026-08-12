import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Clifford
from qiskit.circuit import Instruction

from helper import find_witness_bit, get_bit
from structure import Correction

# propagate correction through the given gate, supports Clifford + T
def propagate_correction(operation: Instruction, input_frames: list[tuple[bool, bool]]) -> list[Correction]:
    name = operation.name.lower()

    # T * X^e Z^d = P^e X^e Z^d T => R = X^e Z^(d^e) P^e
    if name == "t":
        (d, e) = input_frames[0]
        return [Correction(x=e, z=(d != e), p=e)]

    cliff = Clifford(operation)
    z_bits = np.array([d for d, e in input_frames], dtype=bool)
    x_bits = np.array([e for d, e in input_frames], dtype=bool)
    in_pauli = Pauli((z_bits, x_bits))
    out_pauli = in_pauli.evolve(cliff, frame="s")
    return [
        Correction(x=bool(out_pauli.x[i]), z=bool(out_pauli.z[i]), p=False)
        for i in range(len(input_frames))
    ]

# coherently extract d and e from a segment's z/x labels
def get_witnesses(circuit: QuantumCircuit, segment, record, kappa):
    i_z = find_witness_bit(record.l_z[0], record.l_z[1], kappa)
    i_x = find_witness_bit(record.l_x[0], record.l_x[1], kappa)
    if get_bit(record.l_z[0], i_z):
        circuit.x(segment.z[i_z])
    if get_bit(record.l_x[0], i_x):
        circuit.x(segment.x[i_x])
    return segment.z[i_z], segment.x[i_x]

# reconstruct correction through propagate_correction by input
def linear_response(operation, num_inputs):
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

# apply reconstructed correction from linear_response
def apply_coherent_correction(circuit, operation, input_witnesses, output_qubits):
    name = operation.name.lower()

    if name == "t":
        d_witness,e_witness = input_witnesses[0]
        out = output_qubits[0]
        circuit.cp(np.pi/2, e_witness, out)
        circuit.cz(d_witness, out)
        circuit.cz(e_witness, out)
        circuit.cx(e_witness, out)
        circuit.t(e_witness)
        circuit.cz(d_witness, e_witness)
        return

    num_inputs = len(input_witnesses)
    responses = linear_response(operation, num_inputs)

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
            raise NotImplementedError()
        for witness in contributors:
            circuit.cp(np.pi / 2, witness, output_qubits[k])