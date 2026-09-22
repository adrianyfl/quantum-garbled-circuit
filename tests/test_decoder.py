"""Dec must recover (d, e) from the d^w dictionary, never from the labels.

decode_label_bit is the piece that replaces the old get_witnesses shortcut,
which read the plaintext labels straight off the encoder's records. The
replacement sees only

  - the label register, holding l_{b,v} for whichever branch it is in
  - d^w, holding l_{b,0}

and must produce v. The case that matters is the coherent one: the label
register is in superposition over v, so the decoded bit has to end up
*entangled with the branch*, not measured out of it.
"""
import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qgc.garble import decode_label_bit
from qgc.helper import get_bit, sample_label


def layout(kappa):
    lab = list(range(kappa))
    dic = list(range(kappa, 2 * kappa))
    scr = list(range(2 * kappa, 3 * kappa))
    return lab, dic, scr, 3 * kappa


def basis_index(kappa, label_val, dict_val, bit=0):
    lab, dic, _, b = layout(kappa)
    idx = 0
    for i in range(kappa):
        if get_bit(label_val, i):
            idx |= 1 << lab[i]
        if get_bit(dict_val, i):
            idx |= 1 << dic[i]
    if bit:
        idx |= 1 << b
    return idx


def build(kappa, twice=False):
    lab, dic, scr, bit = layout(kappa)
    qc = QuantumCircuit(3 * kappa + 1)
    decode_label_bit(qc, lab, dic, scr, bit)
    if twice:
        decode_label_bit(qc, lab, dic, scr, bit)
    return qc


def gate_list(qc):
    return [(inst.operation.name, tuple(qc.find_bit(q).index for q in inst.qubits))
            for inst in qc.data]


def test_classical_branches():
    """Prepared in a definite branch v, the decoder must output exactly v."""
    for kappa in (1, 2, 3):
        random.seed(kappa)
        for _ in range(4):
            labels = sample_label(kappa)
            for v in (0, 1):
                start = basis_index(kappa, labels[v], labels[0])
                sv = Statevector.from_int(start, 2 ** (3 * kappa + 1)).evolve(build(kappa))
                nz = np.flatnonzero(np.abs(sv.data) > 1e-9)
                assert len(nz) == 1, "decoder left a superposition on a basis input"
                assert int(nz[0]) == start | (v << (3 * kappa)), (
                    f"kappa={kappa} labels={labels} v={v}: bit/scratch wrong")


def test_self_inverse():
    """Applying it twice clears the bit again -- that is how Dec uncomputes."""
    for kappa in (1, 2, 3):
        random.seed(10 + kappa)
        labels = sample_label(kappa)
        for v in (0, 1):
            start = basis_index(kappa, labels[v], labels[0])
            sv = Statevector.from_int(start, 2 ** (3 * kappa + 1)).evolve(build(kappa, twice=True))
            nz = np.flatnonzero(np.abs(sv.data) > 1e-9)
            assert len(nz) == 1 and int(nz[0]) == start, f"kappa={kappa} v={v}: not self-inverse"


def test_coherent_branch():
    """The real use: label register in superposition over v.

    (|l_0> + |l_1>)/sqrt2  must become  (|l_0>|0> + |l_1>|1>)/sqrt2, i.e. the
    decoded bit is entangled with the branch and nothing decoheres.
    """
    for kappa in (1, 2, 3):
        random.seed(20 + kappa)
        for _ in range(4):
            labels = sample_label(kappa)
            dim = 2 ** (3 * kappa + 1)

            start = np.zeros(dim, dtype=complex)
            for v in (0, 1):
                start[basis_index(kappa, labels[v], labels[0])] = 1 / np.sqrt(2)

            want = np.zeros(dim, dtype=complex)
            for v in (0, 1):
                want[basis_index(kappa, labels[v], labels[0], bit=v)] = 1 / np.sqrt(2)

            got = Statevector(start).evolve(build(kappa)).data
            assert np.allclose(got, want, atol=1e-9), (
                f"kappa={kappa} labels={labels}: coherent decode wrong")


def test_output_decoding_is_label_oblivious():
    """Dec's output stage must depend on the topology, not the randomness.

    Garble the same circuit under two seeds: the labels, the randomizers and
    s/t all change, so the old get_witnesses path -- which keyed its X gates off
    find_witness_bit(l_0, l_1) -- produced a different circuit each time. The
    dictionary-based decoder must produce the identical gate list.

    direct mode is used because there Dec is exactly the output-decoding stage.
    The same property now holds for CRE mode's add_cdec under point-and-permute;
    that half is covered by test_shape_is_independent_of_randomness in
    test_cdec.py.
    """
    from qiskit import QuantumCircuit as QC

    from qgc.garble import garble_circuit

    for kappa in (1, 2):
        for ops, n in ([("h", 0)], 1), ([("h", 0), ("cx", 0, 1)], 2):
            built = []
            for seed in (0, 12345):
                qc = QC(n)
                for op in ops:
                    getattr(qc, op[0])(*op[1:])
                random.seed(seed)
                built.append(garble_circuit(qc, kappa, mode="direct").decoder)
            assert gate_list(built[0]) == gate_list(built[1]), (
                f"kappa={kappa} n={n}: decoder depends on the encoding randomness")


if __name__ == "__main__":
    passed, failed = 0, 0
    for fn in (test_classical_branches, test_self_inverse,
               test_coherent_branch, test_output_decoding_is_label_oblivious):
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            failed += 1
    print(f"\n  {passed} passed, {failed} failed")
