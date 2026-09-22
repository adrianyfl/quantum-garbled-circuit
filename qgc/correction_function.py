"""Section 6.2.1: the correction function f_kappa, as a truth table.

For a p-qubit gate, f maps the 2p frame bits (z_1, x_1, ..., z_p, x_p) to the
canonical descriptions of the p correction gadgets Corr_j = Lambda2(R_j).A_j^dag.
Since p <= 2 the table has at most 16 rows, so it is tabulated directly rather
than garbled gate by gate -- a 2p-input lookup table is a perfectly good
decomposable randomized encoding, and its topology depends only on kappa and p,
as Section 6.2.1 requires.
"""
import itertools

from qiskit.circuit import Instruction

from qgc.correction import propagate_correction
from qgc.gadgets import construct_lambda2
from qgc.randomization_group import desc_to_bits, describe_corr
from qgc.structure import WireParams


def build_correction_function(operation: Instruction, params: list[WireParams]):
    """-> {frame bits -> description bitstring}, chunks concatenated per wire."""
    p = operation.num_qubits
    table = {}
    for bits in itertools.product((0, 1), repeat=2 * p):
        frames = [(bool(bits[2 * j]), bool(bits[2 * j + 1])) for j in range(p)]
        corrections = propagate_correction(operation, frames)
        chunks = []
        for j in range(p):
            w = params[j]
            lambda2 = construct_lambda2(corrections[j], w.l_z, w.l_x,
                                        w.s_x, w.s_z, w.t_x, w.t_z, w.kappa)
            desc, phase = describe_corr(w.A, lambda2, w.kappa)
            chunks.append(desc_to_bits(desc, w.kappa, phase))
        table[bits] = ''.join(chunks)
    return table
