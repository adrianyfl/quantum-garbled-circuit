"""The quantum decoder (add_cdec) must agree with the classical one (prg_cdec).

In the real pipeline the CRE-mode circuit is a few hundred qubits wide, so
add_cdec can never be simulated where it actually sits. Here it is lifted out
and run standalone on small synthetic garblings: the held labels and c^g are
prepared as basis states, add_cdec runs, and the description register is read
back.

add_cdec is built only from X / CX / CCX / MCX, so the final state must be a
single basis state. For every input assignment we check

  1. the description register equals prg_cdec's output exactly
  2. the decoder is clean -- ancilla back to |0>, labels and c^g untouched

(2) matters as much as (1): garble.py reuses one ancilla for every slot of
every gate, so a decoder that left it dirty would corrupt everything after it.

There is also a structural check that the garbling's shape, and hence the
decoder circuit, does not depend on the randomness. That is the point-and-
permute property Section 6.4 needs from CDec, and the reason the old
dependency-grouped garbling had to go.
"""
import itertools
import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qgc.cdec import add_cdec, cg_bits, cg_layout, cg_offset, prepare_cg
from qgc.helper import get_bit, sample_label
from qgc.prg_garble import prg_cdec, prg_garble

MAX_QUBITS = 20


def synthetic_garbling(n_in, kappa, out_bits):
    """A small garbling of a random n_in-input, out_bits-output table."""
    table = {}
    for bits in itertools.product((0, 1), repeat=n_in):
        table[bits] = ''.join(str(random.randint(0, 1)) for _ in range(out_bits))
    labels = [sample_label(kappa) for _ in range(n_in)]
    return table, labels, prg_garble(table, labels, kappa, prg="xor")


def run_add_cdec(g, held, kappa, mode):
    """Build and simulate the decoder. -> (description bits, clean, n_qubits)."""
    n_in, out_bits = g.n_in, g.out_bits
    n_cg = cg_layout(g)

    lab = [[i * kappa + b for b in range(kappa)] for i in range(n_in)]
    base = n_in * kappa
    out = [base + k for k in range(out_bits)]
    anc = base + out_bits
    cg = [anc + 1 + i for i in range(max(n_cg, 1))] if mode == "register" else []
    total = anc + 1 + len(cg)
    if total > MAX_QUBITS:
        return None, None, total

    qc = QuantumCircuit(total)
    for i in range(n_in):
        for b in range(kappa):
            if get_bit(held[i], b):
                qc.x(lab[i][b])
    if mode == "register":
        prepare_cg(qc, g, cg)

    # the basis state just prepared; everything but `out` must come back unchanged
    start = 0
    for i in range(n_in):
        for b in range(kappa):
            if get_bit(held[i], b):
                start |= 1 << lab[i][b]
    if mode == "register":
        for i, bit in enumerate(cg_bits(g)):
            if bit:
                start |= 1 << cg[i]

    add_cdec(qc, g, lab, out, anc, kappa,
             cg_qubits=cg if mode == "register" else None, mode=mode)

    sv = Statevector.from_instruction(qc)
    nz = np.flatnonzero(np.abs(sv.data) > 1e-9)
    assert len(nz) == 1, f"decoder left a superposition: {len(nz)} basis states"
    idx = int(nz[0])

    bits = ''.join(str((idx >> out[k]) & 1) for k in range(out_bits))
    out_mask = sum(1 << q for q in out)
    clean = (idx & ~out_mask) == (start & ~out_mask)
    return bits, clean, total


CASES = [
    # label,              n_in, kappa, out_bits
    ("1q gate, k=2",         2, 2, 3),
    ("1q gate, k=1",         2, 1, 3),
    ("1q gate, k=3",         2, 3, 2),
    ("1q gate, 1 out bit",   2, 2, 1),
    ("2q gate, k=1",         4, 1, 1),
    ("2q gate, k=2",         4, 2, 1),
]


def check_case(label, n_in, kappa, out_bits, mode, seed):
    random.seed(seed)
    table, labels, g = synthetic_garbling(n_in, kappa, out_bits)

    mismatched, dirty, width = [], [], None
    for bits in itertools.product((0, 1), repeat=n_in):
        held = [labels[i][bits[i]] for i in range(n_in)]
        want, _ = prg_cdec(g, held)
        assert want == table[bits], "classical decoder does not recover the table"
        got, clean, width = run_add_cdec(g, held, kappa, mode)
        if got is None:
            return None, width
        if got != want:
            mismatched.append((bits, want, got))
        if not clean:
            dirty.append(bits)
    return (mismatched, dirty), width


def test_add_cdec_matches_prg_cdec():
    for seed, (label, n_in, kappa, out_bits) in enumerate(CASES):
        for mode in ("register", "lookup"):
            res, _ = check_case(label, n_in, kappa, out_bits, mode, 100 + seed)
            if res is None:
                continue                      # too wide; reported by __main__
            mismatched, dirty = res
            assert not mismatched, f"{label} [{mode}]: {mismatched[:2]}"
            assert not dirty, f"{label} [{mode}]: ancilla/labels/c^g dirty on {dirty[:2]}"


def test_cg_addressing_agrees_with_layout():
    """cg_offset must index cg_bits exactly as the rows were stored.

    This is the invariant register mode rides on, and the only part of register
    mode a 2-qubit gate cannot exercise by simulation (16 rows needs 22 qubits).
    Checking it classically covers that gap.
    """
    for seed, (n_in, kappa, out_bits) in enumerate([(2, 2, 3), (4, 1, 2), (4, 2, 1)]):
        random.seed(200 + seed)
        _, _, g = synthetic_garbling(n_in, kappa, out_bits)
        bits = cg_bits(g)
        assert len(bits) == cg_layout(g) == g.num_rows() * g.out_bits
        for colour_bits, row in g.rows.items():
            for j, v in enumerate(row):
                assert bits[cg_offset(g, colour_bits, j)] == v, (
                    f"c^g addressing disagrees at {colour_bits}[{j}]")


def test_shape_is_independent_of_randomness():
    """The garbling's shape, and so the decoder circuit, is topology-only.

    Every wire's colour sits at POINTER_BIT and every table has all 2^n_in
    rows, so two garblings of the same topology under different randomness
    differ only in the stored bits. The old scheme keyed its controls off
    find_witness_bit and shaped c^g around the discovered dependency profile,
    so both varied with the labels.
    """
    for n_in, kappa, out_bits in [(2, 2, 3), (4, 2, 2)]:
        built = []
        for seed in (0, 999):
            random.seed(seed)
            _, _, g = synthetic_garbling(n_in, kappa, out_bits)
            lab = [[i * kappa + b for b in range(kappa)] for i in range(n_in)]
            base = n_in * kappa
            out = list(range(base, base + out_bits))
            anc = base + out_bits
            cg = list(range(anc + 1, anc + 1 + cg_layout(g)))
            qc = QuantumCircuit(anc + 1 + len(cg))
            add_cdec(qc, g, lab, out, anc, kappa, cg_qubits=cg, mode="register")
            built.append((sorted(g.rows), cg_layout(g),
                          [(i.operation.name, tuple(qc.find_bit(q).index for q in i.qubits))
                           for i in qc.data]))
        assert built[0][0] == built[1][0], "row addresses vary with randomness"
        assert built[0][1] == built[1][1], "c^g size varies with randomness"
        assert built[0][2] == built[1][2], "CDec circuit varies with randomness"


if __name__ == "__main__":
    passed = failed = skipped = 0
    print("add_cdec vs prg_cdec")
    for seed, (label, n_in, kappa, out_bits) in enumerate(CASES):
        for mode in ("register", "lookup"):
            res, width = check_case(label, n_in, kappa, out_bits, mode, 100 + seed)
            tag = f"{label} [{mode}]"
            if res is None:
                print(f"  [SKIP] {tag}: {width} qubits > {MAX_QUBITS}")
                skipped += 1
                continue
            mismatched, dirty = res
            ok = not mismatched and not dirty
            passed, failed = (passed + ok, failed + (not ok))
            print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: {2 ** n_in} rows x "
                  f"{out_bits} bits, {width}q")
            for bits, want, got in mismatched[:3]:
                print(f"           {bits}: want {want} got {got}")
            if dirty:
                print(f"           dirty on {dirty[:3]}")

    print("\nstructural")
    for fn in (test_cg_addressing_agrees_with_layout,
               test_shape_is_independent_of_randomness):
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            failed += 1
    print(f"\n  {passed} passed, {failed} failed, {skipped} skipped")
