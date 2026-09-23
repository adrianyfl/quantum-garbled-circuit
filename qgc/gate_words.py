"""Canonical representations of randomization-group elements, as gate words.

A correction gadget Corr = Lambda2 . A^dag lies in R_kappa, so it is a tensor
product of one- and two-qubit Cliffords, one per slot. Section 6.1.2 calls the
ordered list of those gates the canonical representation. Here each slot's
Clifford becomes a fixed-length word over a small alphabet, which the evaluator
can apply coherently with add_controlled_word.

THE PHASE BITS. Qiskit's Clifford carries Pauli signs but no global phase, so a
word reproduces its Clifford only up to a power of e^(i pi/4). That is harmless
for a standalone unitary and fatal here, because the evaluator applies the word
*controlled on the description register*, which is entangled with the (d, e)
teleportation branches -- a per-branch global phase is a relative phase across
them. word_phase recovers the lost exponent so it can be carried alongside the
word and reapplied; see PHASE_BITS and its use in randomization_group.describe.
"""
import cmath
import math
from functools import lru_cache

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Operator

# The cost of applying a word is driven by the alphabet twice over:
# add_controlled_word tests every non-pad symbol at every position, and the
# symbol register's width sets the MCX control count. So a MINIMAL generating
# set wins even though its words are longer -- see qgc/optimization.py.
#
# WORD_LEN is exactly the diameter of the Clifford group under the alphabet,
# measured by the breadth-first search in _word_table. Padding beyond it would
# be pure waste, since a pad position still costs a full symbol sweep.
ALPHABETS = {
    "minimal": {
        "alphabet": {
            1: [None, ('h', (0,)), ('s', (0,))],
            2: [None,
                ('cx', (0, 1)),
                ('h', (0,)), ('h', (1,)),
                ('s', (0,)), ('s', (1,))],
        },
        "word_len": {1: 6, 2: 13},
    },
    # The original convenience alphabet at its diameter. Shorter words than
    # "minimal" (3 and 11) but many more symbols to sweep.
    "redundant": {
        "alphabet": {
            1: [None, ('h', (0,)), ('s', (0,)), ('sdg', (0,)),
                ('x', (0,)), ('y', (0,)), ('z', (0,))],
            2: [None,
                ('cx', (0, 1)),
                ('h', (0,)), ('h', (1,)),
                ('s', (0,)), ('s', (1,)),
                ('sdg', (0,)), ('sdg', (1,)),
                ('x', (0,)), ('x', (1,)),
                ('y', (0,)), ('y', (1,)),
                ('z', (0,)), ('z', (1,))],
        },
        "word_len": {1: 3, 2: 11},
    },
}
# The unoptimised baseline: the redundant alphabet padded to the lengths Qiskit's
# synthesis needed, which are well past the group's diameter.
ALPHABETS["original"] = {"alphabet": ALPHABETS["redundant"]["alphabet"],
                         "word_len": {1: 4, 2: 19}}

ALPHABET_NAME = "minimal"
ALPHABET = {}
WORD_LEN = {}
SYM_BITS = {}
WORD_BITS = {}


def configure(name):
    """Select a gate-word alphabet. Call through qgc.optimization.set_level."""
    global ALPHABET_NAME, ALPHABET, WORD_LEN, SYM_BITS, WORD_BITS
    if name not in ALPHABETS:
        raise ValueError(f"unknown alphabet {name!r}, expected {sorted(ALPHABETS)}")
    ALPHABET_NAME = name
    ALPHABET = ALPHABETS[name]["alphabet"]
    WORD_LEN = ALPHABETS[name]["word_len"]
    SYM_BITS = {n: math.ceil(math.log2(len(ALPHABET[n]))) for n in (1, 2)}
    WORD_BITS = {n: WORD_LEN[n] * SYM_BITS[n] for n in (1, 2)}
    _word_table.cache_clear()


def _init():
    configure(ALPHABET_NAME)

# The slip is always an 8th root of unity: for a 1-qubit slot det(U) = c^2
# det(W) with Clifford determinants in {+-1, +-i}, and for a 2-qubit slot
# det(U) = c^4 det(W) with determinants in {+-1}. Either way c^8 = 1.
PHASE_BITS = 3
PHASE_DEN = 8


def word_phase(local, word, n) -> int:
    """k with Operator(local) == e^(i pi k/4) * Operator(word_to_circuit(word, n)).

    The asserts are load bearing. They hold because describe() only ever sees
    Cliffords -- A is drawn from R_kappa and Lambda2 is in R_kappa by Lemma 6.2
    -- so if a non-Clifford ever reaches here the phase would not be an 8th
    root and must fail loudly rather than be rounded.
    """
    a = np.asarray(Operator(word_to_circuit(word, n)).data).ravel()
    b = np.asarray(Operator(local).data).ravel()
    i = int(np.argmax(np.abs(a)))
    c = b[i] / a[i]
    assert np.allclose(a * c, b, atol=1e-7), "slot is not a phase multiple of its word"
    k = round(cmath.phase(c) / (math.pi / 4)) % PHASE_DEN
    assert abs(c - cmath.exp(1j * math.pi / 4 * k)) < 1e-7, f"{c} is not an 8th root"
    return k


@lru_cache(maxsize=None)
def _word_table(n):
    """Shortest word over ALPHABET[n] for every element of the Clifford group.

    Built once by breadth-first search: 24 elements for n=1, 11520 for n=2.
    This replaces Clifford.to_circuit(), which cannot be used with a minimal
    alphabet (it emits x, y, z and sdg) and which in any case produces words
    longer than the group's diameter. The lookup is also far cheaper than
    resynthesising, which matters because describe() calls it once per slot and
    there are O(kappa^2) slots.
    """
    ident = Clifford(QuantumCircuit(n))
    table = {ident.tableau.tobytes(): []}
    generators = [(sym, Clifford(word_to_circuit([sym], n)))
                  for sym in range(1, len(ALPHABET[n]))]
    frontier = [(ident, [])]
    while frontier:
        nxt = []
        for element, word in frontier:
            for sym, gen in generators:
                # Clifford.compose applies self then other, matching the order
                # word_to_circuit lays symbols out in
                candidate = element.compose(gen)
                key = candidate.tableau.tobytes()
                if key not in table:
                    table[key] = word + [sym]
                    nxt.append((candidate, word + [sym]))
        frontier = nxt
    return table


def clifford_to_word(cliff, n):
    word = _word_table(n)[cliff.tableau.tobytes()]
    assert len(word) <= WORD_LEN[n], f"word too long: {len(word)} > {WORD_LEN[n]}"
    return word + [0] * (WORD_LEN[n] - len(word))


def word_to_circuit(word, n):
    qc = QuantumCircuit(n)
    for sym in word:
        if sym == 0:                       # symbol 0 is the pad
            continue
        name, qs = ALPHABET[n][sym]
        getattr(qc, name)(*qs)
    return qc


def word_to_bits(word, n):
    w = SYM_BITS[n]
    return ''.join(format(s, f'0{w}b') for s in word)


def bits_to_word(bits, n):
    w = SYM_BITS[n]
    return [int(bits[i:i + w], 2) for i in range(0, len(bits), w)]


CONTROLLED = {
    'h':   lambda qc, a, t: qc.ch(a, t[0]),
    's':   lambda qc, a, t: qc.cs(a, t[0]),
    'sdg': lambda qc, a, t: qc.csdg(a, t[0]),
    'x':   lambda qc, a, t: qc.cx(a, t[0]),
    'y':   lambda qc, a, t: qc.cy(a, t[0]),
    'z':   lambda qc, a, t: qc.cz(a, t[0]),
    'cx':  lambda qc, a, t: qc.ccx(a, t[0], t[1]),
}


# Arities that use unary iteration, selected by qgc.optimization level 6. The
# naive sweep tests each symbol with its own pair of MCX; unary iteration
# decodes the symbol register into a one-hot register once per position and
# then every gate is singly controlled.
#
# It is NOT a win at every arity. Decoding costs 2(2^w - 1) Toffolis for the
# build and its inverse, against 2*symbols MCX for the sweep, so it pays only
# when there are enough symbols to amortise the decode. Measured: 1.35x at
# n=2 (5 symbols, w=3), and 0.71x -- a loss -- at n=1 (2 symbols, w=2).
UNARY_ARITIES = frozenset()


def onehot_width(n):
    """Scratch the unary decoder needs for an n-qubit slot, 0 if unused."""
    return (1 << SYM_BITS[n]) if n in UNARY_ARITIES else 0


def max_onehot_width():
    return max((onehot_width(n) for n in (1, 2)), default=0)


def _decode_onehot(qc, addr, onehot, uncompute=False):
    """One-hot over 2^w addresses, 2^w - 1 Toffolis.

    Start with address 0 active, then split every active branch on each address
    bit in turn. `addr` is little-endian, so onehot[sym] is the qubit that is
    set exactly when the register holds sym.
    """
    w = len(addr)
    steps = []
    for b in range(w):
        for j in range(1 << b):
            steps.append((j, b, j + (1 << b)))
    if not uncompute:
        qc.x(onehot[0])
        for j, b, hi in steps:
            qc.ccx(onehot[j], addr[b], onehot[hi])
            qc.cx(onehot[hi], onehot[j])
    else:
        for j, b, hi in reversed(steps):
            qc.cx(onehot[hi], onehot[j])
            qc.ccx(onehot[j], addr[b], onehot[hi])
        qc.x(onehot[0])


def add_controlled_word(qc, word_qubits, targets, n, ancilla, word_len=None,
                        onehot=None):
    """Apply the gate word held in `word_qubits` to `targets`.

    Naive form: for each position and each non-pad symbol, flip the symbol's
    zero bits so that "register equals this symbol" becomes "all ones", land
    that on the ancilla, apply the controlled gate, undo both.

    Unary form (USE_UNARY): decode the position's symbol register into a
    one-hot register once, then control each gate on its own one-hot qubit.
    Trades 2*symbols MCX for one 2^w-1 Toffoli decode and its inverse.

    The ancilla and the one-hot scratch both arrive and leave |0>.
    """
    w = SYM_BITS[n]
    length = word_len if word_len is not None else WORD_LEN[n]
    assert len(word_qubits) == length * w

    for pos in range(length):
        ctrl = word_qubits[pos * w:(pos + 1) * w]

        if n in UNARY_ARITIES:
            if onehot is None or len(onehot) < (1 << w):
                raise ValueError(f"unary form needs {1 << w} one-hot qubits")
            addr = list(reversed(ctrl))     # the register is MSB first
            _decode_onehot(qc, addr, onehot)
            for sym in range(1, len(ALPHABET[n])):
                name, qs = ALPHABET[n][sym]
                CONTROLLED[name](qc, onehot[sym], [targets[i] for i in qs])
            _decode_onehot(qc, addr, onehot, uncompute=True)
            continue

        for sym in range(1, len(ALPHABET[n])):
            name, qs = ALPHABET[n][sym]
            # bit b of the register holds bit (w-1-b) of sym (MSB first)
            flips = [ctrl[b] for b in range(w) if not ((sym >> (w - 1 - b)) & 1)]
            for q in flips:
                qc.x(q)
            qc.mcx(ctrl, ancilla)
            CONTROLLED[name](qc, ancilla, [targets[i] for i in qs])
            qc.mcx(ctrl, ancilla)
            for q in flips:
                qc.x(q)
    return qc


_init()
