import itertools
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from randomization_group import describe_corr, desc_to_bits
from correction import propagate_correction
from evaluator import construct_lambda2
from structure import WireParams

def build_correction_function(operation: Instruction, params: list[WireParams]):
    p = operation.num_qubits
    table = {}
    for bits in itertools.product((0, 1), repeat=2*p):
        frames = [(bool(bits[2*j]), bool(bits[2*j+1])) for j in range(p)]
        Rs = propagate_correction(operation, frames)
        chunks = []
        for j in range(p):
            w = params[j]
            lambda2 = construct_lambda2(Rs[j], w.l_z, w.l_x, w.s_x, w.s_z, w.t_x, w.t_z, w.kappa)
            chunks.append(desc_to_bits(describe_corr(w.A, lambda2, w.kappa), w.kappa))
        table[bits] = ''.join(chunks)
    return table

# find dependencies on qubits
def dependency_profile(table, p):
    out_bits = len(next(iter(table.values())))
    deps = []
    for k in range(out_bits):
        dep = []
        for i in range(2*p):
            for bits in itertools.product((0, 1), repeat=2*p):
                if bits[i]: continue
                flipped = list(bits)
                flipped[i] = 1
                if table[bits][k] != table[tuple(flipped)][k]:
                    dep.append(i)
                    break
        deps.append(dep)
    return deps