import numpy as np
from qiskit.quantum_info import Pauli, Clifford
from qiskit.circuit import Instruction
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