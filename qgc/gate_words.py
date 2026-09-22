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

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ALPHABET = {
    1: [None, ('h', (0,)), ('s', (0,)), ('sdg', (0,)), ('x', (0,)), ('y', (0,)), ('z', (0,))],
    2: [None,
        ('cx', (0, 1)),
        ('h', (0,)), ('h', (1,)),
        ('s', (0,)), ('s', (1,)),
        ('sdg', (0,)), ('sdg', (1,)),
        ('x', (0,)), ('x', (1,)),
        ('y', (0,)), ('y', (1,)),
        ('z', (0,)), ('z', (1,))],
}
WORD_LEN = {1: 4, 2: 19}
SYM_BITS = {n: math.ceil(math.log2(len(ALPHABET[n]))) for n in (1, 2)}
WORD_BITS = {n: WORD_LEN[n] * SYM_BITS[n] for n in (1, 2)}

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


def clifford_to_word(cliff, n):
    lookup = {s: i for i, s in enumerate(ALPHABET[n]) if s is not None}
    word = []
    for inst in cliff.to_circuit().data:
        qs = tuple(inst.qubits[i]._index for i in range(len(inst.qubits)))
        word.append(lookup[(inst.operation.name, qs)])
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


def add_controlled_word(qc, word_qubits, targets, n, ancilla, word_len=None):
    """Apply the gate word held in `word_qubits` to `targets`.

    For each position and each non-pad symbol: flip the symbol's zero bits so
    that "register equals this symbol" becomes "all ones", land that on the
    ancilla, apply the controlled gate, then undo both. The ancilla arrives and
    leaves |0>.
    """
    w = SYM_BITS[n]
    length = word_len if word_len is not None else WORD_LEN[n]
    assert len(word_qubits) == length * w
    for pos in range(length):
        ctrl = word_qubits[pos * w:(pos + 1) * w]
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
