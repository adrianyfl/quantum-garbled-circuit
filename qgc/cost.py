"""Closed-form resource model. Nothing here builds a circuit.

At any kappa worth reporting the circuit cannot be constructed, let alone run:
a 2-qubit gate at kappa=128 needs a c^g register of about 20M qubits. But the
scheme's shape is fixed by the topology alone -- that is what point-and-permute
bought -- so every count is a closed form in (kappa, gate arities).

The per-slot costs are derived from ALPHABET and CONTROLLED rather than written
out, so they cannot drift from what gate_words actually emits. Everything here
is checked against built circuits at small kappa by tests/test_cost.py.

T-COST MODEL (parameters, not claims -- adjust to your convention):
    ccx                  T_TOFFOLI
    mcx with k controls  (k-1) * T_TOFFOLI
    cs, csdg, cp         T_CS
    ch                   T_CH
    p(pi/4)              1        p(pi/2) and p(pi) are Clifford
    everything else      0
"""
import math
from collections import Counter

from qgc import gate_words
from qgc.gate_words import PHASE_BITS

T_TOFFOLI = 7      # 7 T per Toffoli (4 with measurement and feed-forward)
T_CS = 3
T_CH = 4

# Set by qgc.optimization.set_level; see that module for what each means.
SHARE_ROUND_KEYS = False
TZAP_FACTOR = 1.0


def controlled_t(name):
    """T cost of CONTROLLED[name], read at call time so a level change lands."""
    return {"h": T_CH, "s": T_CS, "sdg": T_CS, "cx": T_TOFFOLI}.get(name, 0)


def mcx_t(controls):
    return max(controls - 1, 0) * T_TOFFOLI


# ---------------------------------------------------------------- shape
def gadget_qubits(kappa):
    """u, v, z, x and the (kappa+1)^2 register b -- one wire segment."""
    return 3 + 4 * kappa + kappa * kappa


def slot_counts(kappa):
    """(single-qubit slots, paired slots) of the randomization group."""
    return 3 + 3 * kappa, kappa * (kappa + 1) // 2


def desc_bits(kappa):
    """Description length of one correction gadget, phase bits included.

    Equals randomization_group.desc_bits_len, and expands to
    38 kappa^2 + 74 kappa + 39. Closed form so the model never has to
    enumerate O(kappa^2) slots.
    """
    singles, pairs = slot_counts(kappa)
    return (singles * (gate_words.WORD_LEN[1] * gate_words.SYM_BITS[1])
            + pairs * (gate_words.WORD_LEN[2] * gate_words.SYM_BITS[2]) + PHASE_BITS)


def wire_count(num_qubits, arities):
    """Segments in the encoding: one per input wire plus one per gate output."""
    return num_qubits + sum(arities)


# ---------------------------------------------------------------- qubits
def qubit_counts(num_qubits, arities, kappa, mode="cre", traced_out=()):
    wires = wire_count(num_qubits, arities)
    out_wires = num_qubits - len(frozenset(traced_out))
    counts = {
        "logical": num_qubits,
        "segments": wires * gadget_qubits(kappa),
        "output_regs": out_wires * (5 * kappa + 2),   # d^w 4k, compare k, 2 bits
        "cg": 0,
        "desc": 0,
        "anc": 0,
    }
    if mode == "cre":
        for p in arities:
            out_bits = p * desc_bits(kappa)
            counts["cg"] += max(2 ** (2 * p) * out_bits, 1)
            counts["desc"] += out_bits
            counts["anc"] += 1 + gate_words.max_onehot_width()
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    return counts


# ---------------------------------------------------------------- per-slot
def word_cost(n):
    """add_controlled_word for one n-qubit slot: (mcx, x, controlled gates, T).

    Per word position and per non-pad symbol: flip the symbol's zero bits, mcx
    onto the ancilla, apply the controlled gate, undo both.
    """
    w, length = gate_words.SYM_BITS[n], gate_words.WORD_LEN[n]
    symbols = range(1, len(gate_words.ALPHABET[n]))
    gate_t = sum(controlled_t(gate_words.ALPHABET[n][s][0]) for s in symbols)

    if n in gate_words.UNARY_ARITIES:
        # decode the address into one-hot and undo it: 2(2^w - 1) Toffolis,
        # then one singly-controlled gate per symbol
        toffoli = 2 * ((1 << w) - 1)
        return {
            "ccx": length * toffoli,
            "cx": length * toffoli,
            "x": length * 2,
            "cgate": length * len(symbols),
            "t": length * (gate_t + toffoli * T_TOFFOLI),
        }

    zeros = sum(sum(1 for b in range(w) if not ((s >> (w - 1 - b)) & 1))
                for s in symbols)
    mcx = 2 * length * len(symbols)
    return {
        "mcx": mcx,
        "x": 2 * zeros * length,
        "cgate": length * len(symbols),
        "t": length * gate_t + mcx * mcx_t(w),
    }


def lambda3_cost(kappa):
    """C3: ascending fan-outs then Hadamards. Clifford, so no T."""
    return {"cx": (kappa + 1) ** 2, "h": kappa + 1, "t": 0}


def decode_label_bit_cost(kappa):
    """One (d or e) bit recovered from the d^w dictionary."""
    return {"cx": 4 * kappa, "x": 2 * kappa + 1, "mcx": 1, "t": mcx_t(kappa)}


# ---------------------------------------------------------------- Dec
def cdec_cost(p, kappa):
    """add_cdec for one gate: open one of 2^(2p) rows, then unmask."""
    rows, controls = 2 ** (2 * p), 2 * p
    out_bits = p * desc_bits(kappa)
    return {
        # over all colour tuples the zero bits total controls * 2^(controls-1),
        # applied and undone
        "x": rows * controls,
        "mcx": 2 * rows,
        "ccx": rows * out_bits,
        "cx": out_bits * controls,                     # xor mask
        "t": 2 * rows * mcx_t(controls) + rows * out_bits * T_TOFFOLI,
    }


def gate_eval_cost(p, kappa):
    """GateEval(g): CDec, then the controlled words, phases and Lambda3."""
    singles, pairs = slot_counts(kappa)
    total = Counter(cdec_cost(p, kappa))
    per_wire = Counter()
    for n, count in ((1, singles), (2, pairs)):
        for key, value in word_cost(n).items():
            per_wire[key] += value * count
    for key, value in lambda3_cost(kappa).items():
        per_wire[key] += value
    per_wire["p"] += PHASE_BITS
    per_wire["t"] += 1                                  # only p(pi/4) is non-Clifford
    for key, value in per_wire.items():
        total[key] += value * p
    return dict(total)


def output_decode_cost(kappa):
    """Protocol 6 lines 5-7 for one output wire: 4 label decodes, then Z^d X^e."""
    total = Counter()
    for key, value in decode_label_bit_cost(kappa).items():
        total[key] += 4 * value
    total["cx"] += 1
    total["cz"] += 1
    return dict(total)


def dec_cost(num_qubits, arities, kappa, mode="cre", traced_out=()):
    """Dec's gate counts. `t` carries the TZAP factor if a level supplied one."""
    total = Counter()
    if mode == "cre":
        for p in arities:
            total.update(gate_eval_cost(p, kappa))
    for _ in range(num_qubits - len(frozenset(traced_out))):
        total.update(output_decode_cost(kappa))
    out = dict(total)
    out["t"] = round(out.get("t", 0) * TZAP_FACTOR)
    return out


# ---------------------------------------------------------------- SIMON
SIMON_ROUNDS = {64: 32, 96: 36, 128: 44, 256: 72}   # for n = kappa/4, m = 4


def simon_mask_cost(kappa, n_in, out_bits, rounds=None):
    """One add_simon_mask: n_in label streams, out_bits of mask.

    Two SIMON circuits per counter block, because the key register is the label
    register and has to come back unchanged.
    """
    n = kappa // 4
    rounds = SIMON_ROUNDS[kappa] if rounds is None else rounds
    blocks = n_in * math.ceil(out_bits / (2 * n))
    circuits = 2 * blocks
    return {
        "blocks": blocks,
        "circuits": circuits,
        "toffoli": circuits * rounds * n,
        "cx": n_in * out_bits,
    }


def simon_cost(kappa, arities, rounds=None):
    """Coherent PRG cost in Dec, with add_xor_mask replaced by real SIMON.

    Uses n = kappa/4, m = 4, so key size = kappa and block size = kappa/2.
    Only Dec pays this: Enc computes its masks classically.
    """
    if kappa not in SIMON_ROUNDS:
        raise ValueError(f"kappa must be one of {sorted(SIMON_ROUNDS)} to key "
                         f"SIMON at n=kappa/4, m=4; got {kappa}")
    n = kappa // 4
    rounds = SIMON_ROUNDS[kappa] if rounds is None else rounds
    blocks = copy_cx = 0
    for p in arities:
        one = simon_mask_cost(kappa, 2 * p, p * desc_bits(kappa), rounds)
        blocks += one["blocks"]
        copy_cx += one["cx"]
    circuits = 2 * blocks                     # compute + uncompute
    toffoli = circuits * rounds * n
    saved_cx = 0
    if SHARE_ROUND_KEYS:
        # Expanding the schedule once per key rather than per block. The
        # schedule is linear, so this is a CNOT saving only -- toffoli and t
        # are deliberately untouched.
        from qgc.simon_prg import schedule_cx
        keys = 2 * len(arities)
        saved_cx = max(circuits - keys, 0) * schedule_cx(kappa, rounds)
    return {
        "variant": f"Simon{2 * n}/{kappa}",
        "word_size": n, "key_words": 4, "rounds": rounds,
        "blocks": blocks,
        "circuits": circuits,
        "toffoli": toffoli,
        "cx": copy_cx,
        "t": round(toffoli * T_TOFFOLI * TZAP_FACTOR),
        "saved_cx": saved_cx,
        "block_scratch": 2 * n,
    }


# ---------------------------------------------------------------- measuring
def measured_histogram(circuit):
    h = Counter()
    for inst in circuit.data:
        name = inst.operation.name
        if name == "mcx":
            h["mcx"] += 1
            h[f"mcx{len(inst.qubits) - 1}"] += 1
        else:
            h[name] += 1
    return h


def measured_t(circuit):
    total = 0
    for inst in circuit.data:
        name = inst.operation.name
        if name == "ccx":
            total += T_TOFFOLI
        elif name == "mcx":
            total += mcx_t(len(inst.qubits) - 1)
        elif name in ("cs", "csdg", "cp"):
            total += T_CS
        elif name == "ch":
            total += T_CH
        elif name in ("t", "tdg"):
            total += 1
        elif name == "p":
            # the phase-bit gates: pi/4 is a T, pi/2 and pi are Clifford
            angle = float(inst.operation.params[0]) % (2 * math.pi)
            if abs(angle - math.pi / 4) < 1e-9:
                total += 1
    return total


def measured_toffoli(circuit):
    total = 0
    for inst in circuit.data:
        if inst.operation.name == "ccx":
            total += 1
        elif inst.operation.name == "mcx":
            total += max(len(inst.qubits) - 2, 0)
    return total
