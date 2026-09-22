"""Gate set, traced-out outputs, and the classical input gadget.

The Figure 5 gadget is classical reversible, so its stated semantics can be
checked exactly on bit vectors at any kappa -- far cheaper than a statevector,
and it covers widths the statevector cannot reach.
"""
import random

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qgc import gateset
from qgc.gadgets import construct_classical_teleport
from qgc.garble import garble_circuit
from qgc.helper import gadget_num_qubits, get_bit, sample_label


# ---------------------------------------------------------------- gate set
def test_clifford_t_passes_through_exactly():
    qc = QuantumCircuit(2)
    qc.h(0); qc.t(0); qc.cx(0, 1); qc.s(1); qc.x(0); qc.cz(0, 1)
    out, info = gateset.to_gate_set(qc)
    assert not info["approximate"] and info["synthesised"] == []
    assert gateset.offenders(out) == []
    assert gateset.synthesis_error(qc, out) < 1e-12


def test_tdg_is_rewritten():
    qc = QuantumCircuit(1)
    qc.tdg(0)
    out = gateset.rewrite_tdg(qc)
    assert {i.operation.name for i in out.data} == {"t", "sdg"}
    assert np.allclose(Operator(qc).data, Operator(out).data)
    assert gateset.offenders(out) == []


def test_arbitrary_rotation_is_flagged_as_approximate():
    """Qiskit synthesises rz without complaint; we must say that it did."""
    qc = QuantumCircuit(1)
    qc.rz(0.7, 0)
    out, info = gateset.to_gate_set(qc)
    assert info["approximate"] and info["synthesised"] == ["rz"]
    assert gateset.offenders(out) == []
    assert gateset.synthesis_error(qc, out) < 1e-10


def test_out_of_set_is_rejected_when_not_transpiling():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    assert gateset.offenders(qc) == ["ccx"]
    with pytest.raises(ValueError):
        gateset.assert_gate_set(qc)
    with pytest.raises(ValueError):
        garble_circuit(qc, 1, mode="direct", transpile_input=False)


def test_garble_transpiles_by_default():
    qc = QuantumCircuit(1)
    qc.rz(0.7, 0)
    random.seed(0)
    g = garble_circuit(qc, 1, mode="direct")
    assert g.gate_set["approximate"]
    assert gateset.offenders(g.original_circuit) == []


# ---------------------------------------------------------------- traced out
def test_traced_out_wires_get_no_dictionary_or_decoding():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    random.seed(0)
    full = garble_circuit(qc, 1, mode="direct")
    random.seed(0)
    one = garble_circuit(qc, 1, mode="direct", traced_out=(1,))
    random.seed(0)
    none_kept = garble_circuit(qc, 1, mode="direct", traced_out=(0, 1))

    assert len(full.output_registers) == 2
    assert len(one.output_registers) == 1
    assert len(none_kept.output_registers) == 0

    # Protocol 6 loops over O \ T, so decoding is linear in the kept outputs
    per_wire = len(full.decoder.data) // 2
    assert len(one.decoder.data) == per_wire
    assert len(none_kept.decoder.data) == 0
    assert none_kept.traced_out == frozenset({0, 1})

    # and the d^w registers really are gone from the encoding
    assert one.encoded_circuit.num_qubits == full.encoded_circuit.num_qubits - (5 * 1 + 2)


# ---------------------------------------------------------------- Figure 5
def gadget_layout(kappa):
    u, v = 0, 1
    z = list(range(2, 2 + kappa))
    b = list(range(2 + kappa, 2 + kappa + (kappa + 1) ** 2))
    x = list(range(2 + kappa + (kappa + 1) ** 2, gadget_num_qubits(kappa)))
    return u, v, z, b, x


def basis_eval(circuit, bits):
    bits = list(bits)
    index = {q: i for i, q in enumerate(circuit.qubits)}
    for inst in circuit.data:
        name = inst.operation.name
        q = [index[qb] for qb in inst.qubits]
        if name == "x":
            bits[q[0]] ^= 1
        elif name == "cx":
            bits[q[1]] ^= bits[q[0]]
        else:
            raise ValueError(f"{name} is not classically implementable")
    return bits


def test_classical_gadget_is_classically_implementable():
    """The paper's claim: CNOTs and bit flips only, nothing else."""
    for kappa in (1, 2, 4):
        random.seed(kappa)
        qc = construct_classical_teleport(sample_label(kappa), sample_label(kappa),
                                          True, True, kappa)
        assert {i.operation.name for i in qc.data} <= {"x", "cx"}


@pytest.mark.parametrize("kappa", [1, 2, 4, 8])
def test_classical_gadget_semantics(kappa):
    """Figure 5: |y,0,0,r> -> |y^sx, l_{z,0}, l_{x,e}, e^tx>, e = r^y."""
    random.seed(kappa)
    u, v, z, b, x = gadget_layout(kappa)
    for _ in range(4):
        l_z, l_x = sample_label(kappa), sample_label(kappa)
        for s_x in (False, True):
            for t_x in (False, True):
                qc = construct_classical_teleport(l_z, l_x, s_x, t_x, kappa)
                for y in (0, 1):
                    for r in (0, 1):
                        bits = [0] * gadget_num_qubits(kappa)
                        bits[u], bits[v] = y, r
                        end = basis_eval(qc, bits)
                        e = r ^ y

                        assert end[u] == y ^ int(s_x)
                        assert end[v] == e ^ int(t_x)
                        for i in range(kappa):
                            assert end[z[i]] == get_bit(l_z[0], i), "z must hold l_{z,0}"
                            assert end[x[i]] == get_bit(l_x[e], i), "x must hold l_{x,e}"
                        assert all(end[q] == 0 for q in b), "b must stay |0>"


def test_classical_input_changes_only_that_wire():
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    random.seed(0)
    plain = garble_circuit(qc, 1, mode="direct")
    random.seed(0)
    mixed = garble_circuit(qc, 1, mode="direct", classical_inputs=(0,))
    assert mixed.classical_inputs == frozenset({0})
    # same registers, fewer encoder ops (no C1/C3, no A A^dagger)
    assert mixed.encoded_circuit.num_qubits == plain.encoded_circuit.num_qubits
    assert len(mixed.encoded_circuit.data) < len(plain.encoded_circuit.data)
    assert len(mixed.decoder.data) == len(plain.decoder.data)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
