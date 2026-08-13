from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford
from functools import lru_cache
from helper import gadget_num_qubits

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

# build the table that maps n-qubit clifford to a number
@lru_cache(maxsize=None)
def build(n):
    gens = generators(n)
    ident = Clifford(QuantumCircuit(n))
    index = {key(ident): 0}
    elems = [ident]
    frontier = [ident]
    while frontier:
        next = []
        for c in frontier:
            for g in gens:
                cand = c.compose(g)
                k = key(cand)
                if k not in index:
                    index[k] = len(elems)
                    elems.append(cand)
                    next.append(cand)
        frontier = next
    return index, elems

# construct the pair + single structure based on kappa
def slot_structure(kappa):
    pairs = []
    singles = list(range(gadget_num_qubits(kappa)))
    for j in range(kappa + 1):
        for i in range(j):
            pairs.append((b_index(i, j, kappa), b_index(j, i, kappa)))
            singles.remove(b_index(i, j, kappa))
            singles.remove(b_index(j, i, kappa))
    slots = []
    for s in singles:
        slots.append((1, (s,)))
    for p in pairs:
        slots.append((2, p))
    return slots

# generate a description of a depth one Clifford circuit based on a clifford circuit
def describe(circuit: QuantumCircuit, kappa: int) -> dict[tuple[int, tuple[int]], int]:
    # qubit to slot table
    slots = slot_structure(kappa)
    slot_of = {}
    for s in slots:
        for q in s[1]:
            slot_of[q] = s

    # gather the gates on qubits
    buckets = {}
    for inst in circuit.data:
        if inst.operation.name in ("barrier", "delay"):
            continue
        idxs = [circuit.find_bit(q).index for q in inst.qubits]
        target_slots = {slot_of[i] for i in idxs}
        if len(target_slots) != 1:
            raise ValueError(f"{inst.operation.name} on {idxs} spans slots {target_slots}")
        buckets.setdefault(target_slots.pop(), []).append((inst.operation, idxs))

    # convert the gates to numerical values
    idx1, _ = build(1)
    idx2, _ = build(2)
    desc = {}
    for s in slots:
        local = QuantumCircuit(s[0])
        for op, idxs in buckets.get(s, []):
            local.append(op, [0 if g == s[1][0] else 1 for g in idxs])
        desc[s] = idx1[Clifford(local).tableau.tobytes()] if s[0] == 1 else idx2[Clifford(local).tableau.tobytes()]
    return desc

# rebuild the Clifford circuit based on a description
def rebuild(desc: dict[tuple[int, tuple[int]], int], kappa: int) -> QuantumCircuit:
    _, e1 = build(1)
    _, e2 = build(2)
    circuit = QuantumCircuit(gadget_num_qubits(kappa))
    for s, idx in desc.items():
        elems = e1 if s[0] == 1 else e2
        circuit.compose(elems[idx].to_circuit(), qubits=list(s[1]), inplace=True)
    return circuit

