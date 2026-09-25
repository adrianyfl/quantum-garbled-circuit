"""The closed-form model must agree exactly with circuits that were built.

cost.py is what the resource study reports at kappa the machine cannot reach,
so it is only worth anything if it reproduces, to the gate, what garble_circuit
actually emits at kappa it can. Everything here builds a real circuit and
compares.

Enc's X-gate count is data dependent (prepare_cg and the d^w dictionary write
one X per set bit of random labels), so Enc is not modelled gate for gate. Dec
and the qubit counts are deterministic, and those are what the paper reports.
"""
import random

import pytest
from qiskit import QuantumCircuit

from qgc import cost
from qgc.garble import garble_circuit
from qgc.randomization_group import desc_bits_len, slot_structure

CIRCUITS = [
    ("H",             1, [("h", 0)]),
    ("T",             1, [("t", 0)]),
    ("H,T",           1, [("h", 0), ("t", 0)]),
    ("CX",            2, [("cx", 0, 1)]),
    ("H,CX",          2, [("h", 0), ("cx", 0, 1)]),
    ("H,CX,T",        2, [("h", 0), ("cx", 0, 1), ("t", 0)]),
    ("3-qubit chain", 3, [("h", 0), ("cx", 0, 1), ("cx", 1, 2)]),
]


def build(n, ops, kappa, mode, seed=0, **kw):
    qc = QuantumCircuit(n)
    for op in ops:
        getattr(qc, op[0])(*op[1:])
    random.seed(seed)
    return qc, garble_circuit(qc, kappa, mode=mode, **kw)


def arities(qc):
    return [inst.operation.num_qubits for inst in qc.data]


def test_desc_bits_matches_slot_enumeration():
    """The closed form must equal what slot_structure actually produces.

    The second assertion pins the polynomial for the current alphabet; it is
    written out from WORD_LEN and SYM_BITS so that changing either updates it
    rather than leaving a stale constant behind.
    """
    from qgc.gate_words import PHASE_BITS, SYM_BITS, WORD_LEN
    single = WORD_LEN[1] * SYM_BITS[1]
    pair = WORD_LEN[2] * SYM_BITS[2]
    for kappa in range(1, 9):
        assert cost.desc_bits(kappa) == desc_bits_len(kappa), f"kappa={kappa}"
        assert cost.desc_bits(kappa) == (
            (3 + 3 * kappa) * single + kappa * (kappa + 1) // 2 * pair + PHASE_BITS)


def test_words_are_shortest_and_exact():
    """Every word rebuilds its Clifford, and none exceeds WORD_LEN.

    WORD_LEN is the group's diameter, so a word longer than it would mean the
    table is wrong, and a WORD_LEN larger than the longest word would be pure
    padding -- each pad position still costs a full symbol sweep in
    add_controlled_word.
    """
    from qiskit.quantum_info import Clifford

    from qgc.gate_words import WORD_LEN, _word_table, word_to_circuit
    for n in (1, 2):
        table = _word_table(n)
        assert len(table) == (24 if n == 1 else 11520), f"n={n}: group incomplete"
        assert max(len(w) for w in table.values()) == WORD_LEN[n], (
            f"n={n}: WORD_LEN should be the diameter")
        for key, word in table.items():
            assert Clifford(word_to_circuit(word, n)).tableau.tobytes() == key


def test_slot_counts():
    for kappa in range(1, 7):
        singles, pairs = cost.slot_counts(kappa)
        actual = slot_structure(kappa)
        assert singles == sum(1 for s in actual if s[0] == 1), f"kappa={kappa}"
        assert pairs == sum(1 for s in actual if s[0] == 2), f"kappa={kappa}"
        assert singles + 2 * pairs == cost.gadget_qubits(kappa)


@pytest.mark.parametrize("mode", ["direct", "cre"])
@pytest.mark.parametrize("kappa", [1, 2])
def test_qubit_counts_exact(kappa, mode):
    for name, n, ops in CIRCUITS:
        if mode == "cre" and kappa > 1 and n > 2:
            continue                                  # keeps the suite quick
        qc, g = build(n, ops, kappa, mode)
        want = cost.qubit_counts(n, arities(qc), kappa, mode)
        got = g.encoded_circuit.num_qubits
        assert want["total"] == got, (
            f"{name} kappa={kappa} {mode}: model {want['total']}, built {got}")


@pytest.mark.parametrize("mode", ["direct", "cre"])
@pytest.mark.parametrize("kappa", [1, 2])
def test_dec_cost_exact(kappa, mode):
    """Dec is where the cost is, and it is fully deterministic."""
    for name, n, ops in CIRCUITS:
        if mode == "cre" and kappa > 1 and n > 2:
            continue
        qc, g = build(n, ops, kappa, mode)
        want = cost.dec_cost(n, arities(qc), kappa, mode)

        want_ops = sum(v for k, v in want.items() if k != "t")
        got_ops = len(g.decoder.data)
        assert want_ops == got_ops, (
            f"{name} kappa={kappa} {mode}: model {want_ops} ops, built {got_ops}\n"
            f"  model {want}\n  built {dict(cost.measured_histogram(g.decoder))}")

        assert want["t"] == cost.measured_t(g.decoder), (
            f"{name} kappa={kappa} {mode}: T model {want['t']}, "
            f"built {cost.measured_t(g.decoder)}")


def test_traced_out_is_modelled():
    """Discarded outputs cost no d^w and no decoding."""
    for traced in ((), (1,), (0, 1)):
        qc, g = build(2, [("cx", 0, 1)], 1, "direct", traced_out=traced)
        want = cost.qubit_counts(2, arities(qc), 1, "direct", traced_out=traced)
        assert want["total"] == g.encoded_circuit.num_qubits, f"traced={traced}"
        d = cost.dec_cost(2, arities(qc), 1, "direct", traced_out=traced)
        assert sum(v for k, v in d.items() if k != "t") == len(g.decoder.data)


def test_cg_matches_garbling():
    """c^g = 2^(2p) * p * desc_bits(kappa), with no dependence on randomness."""
    from qgc.cdec import cg_layout
    for kappa in (1, 2):
        for name, n, ops in CIRCUITS[:4]:
            qc, g = build(n, ops, kappa, "cre")
            for gate_id, gc in g.cre.items():
                p = g.gates[gate_id].operation.num_qubits
                assert cg_layout(gc.garbled) == 2 ** (2 * p) * p * cost.desc_bits(kappa)


def test_simon_mask_cost_exact():
    """The SIMON cost model against circuits actually emitted.

    kappa=64 with reduced rounds; the model is linear in `rounds`, so matching
    here pins the spec-round numbers too.
    """
    from qgc.simon_prg import add_simon_mask, simon_block_qubits

    kappa, rounds = 64, 4
    for n_in, out_bits in ((1, 8), (2, 40), (2, 70), (4, 16)):
        width = simon_block_qubits(kappa, out_bits)
        lab = [[i * kappa + b for b in range(kappa)] for i in range(n_in)]
        base = n_in * kappa
        out = list(range(base, base + out_bits))
        blk = list(range(base + out_bits, base + out_bits + width))
        qc = QuantumCircuit(base + out_bits + width)
        add_simon_mask(qc, lab, out, kappa, n_in, out_bits, blk, rounds=rounds)

        want = cost.simon_mask_cost(kappa, n_in, out_bits, rounds)
        assert want["toffoli"] == cost.measured_toffoli(qc), (
            f"n_in={n_in} out_bits={out_bits}: model {want['toffoli']}, "
            f"built {cost.measured_toffoli(qc)}")
        assert cost.measured_histogram(qc)["cx"] >= want["cx"], "copy-out CX missing"


@pytest.fixture
def at_level():
    from qgc import optimization
    yield optimization.set_level
    optimization.set_level(optimization.DEFAULT)


def test_simon_t_per_toffoli_by_level(at_level):
    """7 T unitary, 5 under tzap, 4 measured, 2 once uncomputation is free."""
    want = {0: 7, 1: 7, 2: 7, 3: 7, 4: 7, 5: 5, 6: 4, 7: 2}
    for level, per_toffoli in want.items():
        at_level(level)
        for kappa in (64, 128):
            r = cost.simon_cost(kappa, [1, 2])
            assert r["t"] == r["toffoli"] * per_toffoli, f"level={level} kappa={kappa}"


@pytest.mark.parametrize("level,per_toffoli,emits_ccx", [(5, 5, False), (6, 4, True)])
def test_simon_mask_t_exact_under_tzap(at_level, level, per_toffoli, emits_ccx):
    """Level 5 emits tzap's SIMON; level 6 goes back to CCX for the measured form."""
    pytest.importorskip("tzap")
    from qgc.simon_prg import add_simon_mask, simon_block_qubits

    at_level(level)
    kappa, rounds, n_in, out_bits = 64, 4, 2, 40
    width = simon_block_qubits(kappa, out_bits)
    lab = [[i * kappa + b for b in range(kappa)] for i in range(n_in)]
    base = n_in * kappa
    out = list(range(base, base + out_bits))
    blk = list(range(base + out_bits, base + out_bits + width))
    qc = QuantumCircuit(base + out_bits + width)
    add_simon_mask(qc, lab, out, kappa, n_in, out_bits, blk, rounds=rounds)

    toffoli = cost.simon_mask_cost(kappa, n_in, out_bits, rounds)["toffoli"]
    assert ("ccx" in qc.count_ops()) == emits_ccx
    assert cost.measured_t(qc) == toffoli * per_toffoli


@pytest.mark.parametrize("batch", [2, 64])
def test_shared_round_keys_saving_exact(at_level, monkeypatch, batch):
    """Level 3's CX and X saving, model against two emitted masks.

    rounds=8 so the key schedule has steps to share (it has none below m+1).
    batch=2 forces a short last batch; 64 puts every block in one.
    """
    from qgc.simon_prg import add_simon_mask, schedule_gates, simon_block_qubits

    monkeypatch.setattr(cost, "SHARE_BATCH", batch)
    kappa, rounds, n_in, out_bits = 64, 8, 2, 70           # 3 blocks per label

    def emit(level):
        at_level(level)
        width = simon_block_qubits(kappa, out_bits)
        lab = [[i * kappa + b for b in range(kappa)] for i in range(n_in)]
        base = n_in * kappa
        out = list(range(base, base + out_bits))
        blk = list(range(base + out_bits, base + out_bits + width))
        qc = QuantumCircuit(base + out_bits + width)
        add_simon_mask(qc, lab, out, kappa, n_in, out_bits, blk, rounds=rounds)
        return cost.measured_histogram(qc), cost.simon_mask_cost(kappa, n_in, out_bits, rounds)

    plain, _ = emit(2)
    shared, model = emit(3)
    sched = schedule_gates(kappa, rounds)
    assert sched["cx"] > 0
    saved_runs = 2 * (model["blocks"] - model["batches"])
    assert model["batches"] == n_in * -(-3 // batch)
    assert plain["ccx"] == shared["ccx"] == model["toffoli"]
    assert plain["cx"] - shared["cx"] == saved_runs * sched["cx"]
    assert plain["x"] - shared["x"] == saved_runs * sched["x"]


def test_simon_rejects_unusable_kappa():
    for kappa in (1, 2, 32, 100):
        with pytest.raises(ValueError):
            cost.simon_cost(kappa, [1])
    for kappa in (64, 96, 128, 256):
        r = cost.simon_cost(kappa, [2])
        assert r["word_size"] * r["key_words"] == kappa


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
