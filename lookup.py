from qiskit import QuantumCircuit

def add_lookup(qc, table, controls, outs, ancilla):
    for row, bits in sorted(table.items()):
        flips = [controls[i] for i, r in enumerate(row) if r == 0]
        for q in flips:
            qc.x(q)
        qc.mcx(controls, ancilla)
        for j, b in enumerate(bits):
            if b:
                qc.cx(ancilla, outs[j])
        qc.mcx(controls, ancilla)
        for q in flips:
            qc.x(q)
    return qc

def lookup_cost(table):
    c = len(next(iter(table)))
    mcx = 2 * len(table)
    cnot = sum(sum(bits) for bits in table.values())
    return c, mcx, cnot