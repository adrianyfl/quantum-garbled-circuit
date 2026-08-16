from functools import lru_cache
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford
from helper import gadget_num_qubits
from gate_words import (clifford_to_word, word_to_circuit, word_to_bits,
                        bits_to_word, WORD_BITS, WORD_LEN)

def b_index(i, j, kappa):
    return 2 + kappa + i * (kappa + 1) + j


def key(c):
    return c.tableau.tobytes()


def generators(n):
    def mk(fn):
        qc = QuantumCircuit(n)
        fn(qc)
        return Clifford(qc)
    gens = [mk(lambda c: c.h(0)), mk(lambda c: c.s(0))]
    if n == 2:
        gens += [mk(lambda c: c.h(1)), mk(lambda c: c.s(1)), mk(lambda c: c.cx(0, 1))]
    return gens


@lru_cache(maxsize=None)
def build(n):
    """Enumerate the n-qubit Clifford group. TEST FIXTURE ONLY."""
    gens = generators(n)
    ident = Clifford(QuantumCircuit(n))
    index = {key(ident): 0}
    elems = [ident]
    frontier = [ident]
    while frontier:
        nxt = []
        for c in frontier:
            for g in gens:
                cand = c.compose(g)
                k = key(cand)
                if k not in index:
                    index[k] = len(elems)
                    elems.append(cand)
                    nxt.append(cand)
        frontier = nxt
    return index, elems


@lru_cache(maxsize=None)
def slot_structure(kappa):
    """[(arity, qubit tuple)] -- singles first, then pairs. Deterministic order."""
    pairs = []
    singles = list(range(gadget_num_qubits(kappa)))
    for j in range(kappa + 1):
        for i in range(j):
            pairs.append((b_index(i, j, kappa), b_index(j, i, kappa)))
            singles.remove(b_index(i, j, kappa))
            singles.remove(b_index(j, i, kappa))
    return [(1, (s,)) for s in singles] + [(2, p) for p in pairs]


@lru_cache(maxsize=None)
def desc_bits_len(kappa):
    return sum(WORD_BITS[s[0]] for s in slot_structure(kappa))


def describe(circuit: QuantumCircuit, kappa: int) -> dict:
    """Decompose a depth-one R_kappa circuit into one gate word per slot."""
    slots = slot_structure(kappa)
    slot_of = {}
    for s in slots:
        for q in s[1]:
            slot_of[q] = s

    buckets = {}
    for inst in circuit.data:
        if inst.operation.name in ("barrier", "delay"):
            continue
        idxs = [circuit.find_bit(q).index for q in inst.qubits]
        target_slots = {slot_of[i] for i in idxs}
        if len(target_slots) != 1:
            raise ValueError(f"{inst.operation.name} on {idxs} spans slots {target_slots}")
        buckets.setdefault(target_slots.pop(), []).append((inst.operation, idxs))

    desc = {}
    for s in slots:
        local = QuantumCircuit(s[0])
        for op, idxs in buckets.get(s, []):
            local.append(op, [0 if g == s[1][0] else 1 for g in idxs])
        desc[s] = clifford_to_word(Clifford(local), s[0])
    return desc


def describe_corr(a: QuantumCircuit, lambda2: QuantumCircuit, kappa: int) -> dict:
    """Corr = Lambda2 . A^dagger  ->  A^dagger applied first."""
    circuit = QuantumCircuit(gadget_num_qubits(kappa))
    circuit.compose(a.inverse(), inplace=True)
    circuit.compose(lambda2, inplace=True)
    return describe(circuit, kappa)


def rebuild(desc: dict, kappa: int) -> QuantumCircuit:
    circuit = QuantumCircuit(gadget_num_qubits(kappa))
    for s, word in desc.items():
        circuit.compose(word_to_circuit(word, s[0]), qubits=list(s[1]), inplace=True)
    return circuit


def desc_to_bits(desc: dict, kappa: int) -> str:
    return ''.join(word_to_bits(desc[s], s[0]) for s in slot_structure(kappa))


def bits_to_desc(bits: str, kappa: int) -> dict:
    desc, pos = {}, 0
    for s in slot_structure(kappa):
        w = WORD_BITS[s[0]]
        desc[s] = bits_to_word(bits[pos:pos + w], s[0])
        pos += w
    assert pos == len(bits), f"leftover bits: consumed {pos} of {len(bits)}"
    return desc


def slot_bit_offsets(kappa):
    """[(slot, start, length)] -- where each slot's word sits in the bitstring."""
    out, pos = [], 0
    for s in slot_structure(kappa):
        w = WORD_BITS[s[0]]
        out.append((s, pos, w))
        pos += w
    return out

print(len(build(2)[0]))