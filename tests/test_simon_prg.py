"""SIMON as the real PRG: classical and reversible forms must agree.

A garbling at a kappa that keys SIMON is far past any statevector simulator --
kappa=64 alone puts two label registers at 128 qubits. But add_cdec is built
only from X / CX / CCX / MCX, so it is a permutation of basis states and can be
evaluated exactly by walking circuit.data over a vector of bits, linear in the
gate count. That is the same trick qiskit-simon uses, extended to MCX.

So the full SIMON-backed decoder is checked at kappa=64 with real (round
reduced) SIMON, at whatever width it happens to need.
"""
import itertools
import random

import pytest
from qiskit import QuantumCircuit

from qgc.cdec import add_cdec, cg_bits, cg_layout, prepare_cg
from qgc.helper import get_bit, sample_label
from qgc.prg_garble import prg_cdec, prg_garble
from qgc.simon_prg import (add_simon_mask, prg_simon, simon_block_qubits,
                           simon_variant)

ROUNDS = 4          # round reduced: keeps the emitted circuits small
KAPPA = 64


def basis_eval(circuit, bits):
    """Exact evaluation of a classical reversible circuit on a basis state.

    qiskit-simon's evaluate() covers X/CX/CCX/SWAP; add_cdec also emits MCX.
    """
    bits = list(bits)
    index = {q: i for i, q in enumerate(circuit.qubits)}
    for inst in circuit.data:
        name = inst.operation.name
        q = [index[qb] for qb in inst.qubits]
        if name == "x":
            bits[q[0]] ^= 1
        elif name == "cx":
            bits[q[1]] ^= bits[q[0]]
        elif name == "ccx":
            bits[q[2]] ^= bits[q[0]] & bits[q[1]]
        elif name == "mcx":
            acc = 1
            for c in q[:-1]:
                acc &= bits[c]
            bits[q[-1]] ^= acc
        elif name == "swap":
            bits[q[0]], bits[q[1]] = bits[q[1]], bits[q[0]]
        elif name in ("barrier", "id"):
            continue
        else:
            raise ValueError(f"{name} is not classical reversible")
    return bits


def test_variant_selection():
    for kappa, n in ((64, 16), (96, 24), (128, 32), (256, 64)):
        p = simon_variant(kappa)
        assert p.word_size == n and p.key_words == 4
        assert p.key_size == kappa and p.block_size == kappa // 2
    for bad in (1, 2, 32, 100, 192):
        with pytest.raises(ValueError):
            simon_variant(bad)


def test_simon_mask_matches_classical():
    """add_simon_mask must reproduce prg_simon bit for bit."""
    random.seed(0)
    for n_in, out_bits in ((1, 8), (2, 40), (2, 70), (4, 16)):
        width = simon_block_qubits(KAPPA, out_bits)
        labels = [random.getrandbits(KAPPA) for _ in range(n_in)]
        want = prg_simon(labels, KAPPA, 0, out_bits, rounds=ROUNDS)

        lab = [[i * KAPPA + b for b in range(KAPPA)] for i in range(n_in)]
        base = n_in * KAPPA
        out = list(range(base, base + out_bits))
        blk = list(range(base + out_bits, base + out_bits + width))
        qc = QuantumCircuit(base + out_bits + width)
        add_simon_mask(qc, lab, out, KAPPA, n_in, out_bits, blk, rounds=ROUNDS)

        start = [0] * qc.num_qubits
        for i in range(n_in):
            for b in range(KAPPA):
                start[lab[i][b]] = get_bit(labels[i], b)
        end = basis_eval(qc, start)

        assert [end[out[j]] for j in range(out_bits)] == want, \
            f"n_in={n_in} out_bits={out_bits}"

        # labels restored, block scratch back to |0>
        for i in range(n_in):
            for b in range(KAPPA):
                assert end[lab[i][b]] == get_bit(labels[i], b), "label not restored"
        assert all(end[q] == 0 for q in blk), "block scratch left dirty"


def test_shared_key_mask_matches_classical(monkeypatch):
    """Level 3 batches counter blocks under one key schedule; bits must not move.

    rounds=8 gives the schedule steps to share, and a batch of 2 over 3 blocks
    forces a short last batch.
    """
    from qgc import cost, optimization
    monkeypatch.setattr(cost, "SHARE_BATCH", 2)
    optimization.set_level(3)
    try:
        random.seed(3)
        for n_in, out_bits in ((1, 8), (2, 70), (4, 40)):
            width = simon_block_qubits(KAPPA, out_bits)
            assert width == 32 * min(2, -(-out_bits // 32))
            labels = [random.getrandbits(KAPPA) for _ in range(n_in)]
            want = prg_simon(labels, KAPPA, 0, out_bits, rounds=8)

            lab = [[i * KAPPA + b for b in range(KAPPA)] for i in range(n_in)]
            base = n_in * KAPPA
            out = list(range(base, base + out_bits))
            blk = list(range(base + out_bits, base + out_bits + width))
            qc = QuantumCircuit(base + out_bits + width)
            add_simon_mask(qc, lab, out, KAPPA, n_in, out_bits, blk, rounds=8)

            start = [0] * qc.num_qubits
            for i in range(n_in):
                for b in range(KAPPA):
                    start[lab[i][b]] = get_bit(labels[i], b)
            end = basis_eval(qc, start)

            assert [end[out[j]] for j in range(out_bits)] == want, f"n_in={n_in}"
            for i in range(n_in):
                for b in range(KAPPA):
                    assert end[lab[i][b]] == get_bit(labels[i], b), "label not restored"
            assert all(end[q] == 0 for q in blk), "block scratch left dirty"
    finally:
        optimization.set_level(optimization.DEFAULT)


@pytest.mark.parametrize("n_in,out_bits", [(2, 6), (2, 40), (4, 5)])
def test_add_cdec_with_simon(n_in, out_bits):
    """The whole decoder, with real SIMON, against the classical decoder."""
    random.seed(n_in * 100 + out_bits)
    table = {bits: ''.join(str(random.randint(0, 1)) for _ in range(out_bits))
             for bits in itertools.product((0, 1), repeat=n_in)}
    labels = [sample_label(KAPPA) for _ in range(n_in)]
    g = prg_garble(table, labels, KAPPA, prg="simon", rounds=ROUNDS)

    width = simon_block_qubits(KAPPA, out_bits)
    n_cg = cg_layout(g)
    lab = [[i * KAPPA + b for b in range(KAPPA)] for i in range(n_in)]
    base = n_in * KAPPA
    out = list(range(base, base + out_bits))
    anc = base + out_bits
    cg = list(range(anc + 1, anc + 1 + n_cg))
    blk = list(range(anc + 1 + n_cg, anc + 1 + n_cg + width))
    total = anc + 1 + n_cg + width

    for vals in itertools.product((0, 1), repeat=n_in):
        held = [labels[i][vals[i]] for i in range(n_in)]
        want, _ = prg_cdec(g, held)
        assert want == table[vals], "classical decoder does not recover the table"

        qc = QuantumCircuit(total)
        prepare_cg(qc, g, cg)                 # c^g is loaded by the circuit
        add_cdec(qc, g, lab, out, anc, KAPPA, cg_qubits=cg,
                 mode="register", block_qubits=blk)

        start = [0] * total
        for i in range(n_in):
            for b in range(KAPPA):
                start[lab[i][b]] = get_bit(held[i], b)

        end = basis_eval(qc, start)
        got = ''.join(str(end[out[j]]) for j in range(out_bits))
        assert got == want, f"vals={vals}: want {want} got {got}"

        assert end[anc] == 0, "ancilla dirty"
        assert all(end[q] == 0 for q in blk), "block scratch dirty"
        for i in range(n_in):
            for b in range(KAPPA):
                assert end[lab[i][b]] == get_bit(held[i], b), "label not restored"
        assert [end[q] for q in cg] == cg_bits(g), "c^g disturbed"


def test_tzap_simon_mask_matches_classical():
    """Level 5's tzap-optimised mask must still be prg_simon, bit for bit.

    The optimised circuit has H and T, so basis_eval cannot run it; Aer's MPS
    method does, and a permutation circuit must land on one outcome.
    """
    pytest.importorskip("tzap")
    pytest.importorskip("qiskit_aer")
    from tzap_optimize import simulate_basis

    from qgc import optimization
    optimization.set_level(5)
    try:
        random.seed(5)
        rounds = 8                   # so the shared key schedule has steps
        for n_in, out_bits in ((1, 8), (2, 40)):
            width = simon_block_qubits(KAPPA, out_bits)
            labels = [random.getrandbits(KAPPA) for _ in range(n_in)]
            want = prg_simon(labels, KAPPA, 0, out_bits, rounds=rounds)

            lab = [[i * KAPPA + b for b in range(KAPPA)] for i in range(n_in)]
            base = n_in * KAPPA
            out = list(range(base, base + out_bits))
            blk = list(range(base + out_bits, base + out_bits + width))
            qc = QuantumCircuit(base + out_bits + width)
            add_simon_mask(qc, lab, out, KAPPA, n_in, out_bits, blk, rounds=rounds)
            assert "ccx" not in qc.count_ops(), "level 5 should emit tzap's SIMON"

            start = [0] * qc.num_qubits
            for i in range(n_in):
                for b in range(KAPPA):
                    start[lab[i][b]] = get_bit(labels[i], b)
            end = simulate_basis(qc, start)

            assert [end[out[j]] for j in range(out_bits)] == want, f"n_in={n_in}"
            for i in range(n_in):
                for b in range(KAPPA):
                    assert end[lab[i][b]] == get_bit(labels[i], b), "label not restored"
            assert all(end[q] == 0 for q in blk), "block scratch left dirty"
    finally:
        optimization.set_level(optimization.DEFAULT)


def test_simon_and_xor_garble_the_same_table():
    """Only the mask differs; the plaintext table recovered must be identical."""
    random.seed(7)
    n_in, out_bits = 2, 12
    table = {bits: ''.join(str(random.randint(0, 1)) for _ in range(out_bits))
             for bits in itertools.product((0, 1), repeat=n_in)}
    labels = [sample_label(KAPPA) for _ in range(n_in)]
    gx = prg_garble(table, labels, KAPPA, prg="xor")
    gs = prg_garble(table, labels, KAPPA, prg="simon", rounds=ROUNDS)
    assert gx.rows.keys() == gs.rows.keys(), "point-and-permute addresses differ"
    assert any(gx.rows[k] != gs.rows[k] for k in gx.rows), "masks are identical?"
    for vals in itertools.product((0, 1), repeat=n_in):
        held = [labels[i][vals[i]] for i in range(n_in)]
        assert prg_cdec(gx, held)[0] == prg_cdec(gs, held)[0] == table[vals]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
