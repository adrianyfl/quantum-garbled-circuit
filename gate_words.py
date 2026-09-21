import cmath, math
from qiskit import QuantumCircuit
import numpy as np
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
PHASE_BITS = 3
PHASE_DEN = 8

"""
Add global phase to corresponding word
k with Operator(local) == e^(i*pi*k/4) * Operator(word_to_circuit(word,n))
"""
def word_phase(local, word, n) -> int:
    A = np.asarray(Operator(word_to_circuit(word, n)).data).ravel()
    B = np.asarray(Operator(local).data).ravel()
    i = int(np.argmax(np.abs(A)))
    c = B[i] / A[i]
    assert np.allclose(A * c, B, atol=1e-7), "slot is not a phase multiple of its word"
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
        if sym == 0:
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
    w = SYM_BITS[n]
    L = word_len if word_len is not None else WORD_LEN[n]
    assert len(word_qubits) == L * w
    for pos in range(L):
        ctrl = word_qubits[pos * w:(pos + 1) * w]
        for sym in range(1, len(ALPHABET[n])):          # symbol 0 is the pad -> no gate
            name, qs = ALPHABET[n][sym]
            # bit b of the register holds bit (w-1-b) of sym  (MSB-first)
            flips = [ctrl[b] for b in range(w) if not ((sym >> (w - 1 - b)) & 1)]
            for q in flips:
                qc.x(q)
            qc.mcx(ctrl, ancilla)
            tq = [targets[i] for i in qs]
            CONTROLLED[name](qc, ancilla, tq)
            qc.mcx(ctrl, ancilla)
            for q in flips:
                qc.x(q)
    return qc


def apply_word_cost(n, word_len=None):
    L = word_len if word_len is not None else WORD_LEN[n]
    syms = len(ALPHABET[n]) - 1
    return 2 * L * syms, L * syms