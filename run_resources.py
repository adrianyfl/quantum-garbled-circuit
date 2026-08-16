"""Resource counts for a set of circuits, in both evaluator modes.

Nothing is executed -- the counts are read off the constructed circuit.
Use optimization_level=0 if you ever transpile these, or A / A^dagger
(direct mode) will cancel and the count will be silently low.
"""
import random

from qiskit import QuantumCircuit

from qgc import garble_circuit
from resources import report, print_report

CASES = [
    ("H",            1, [("h", 0)]),
    ("T",            1, [("t", 0)]),
    ("H,T",          1, [("h", 0), ("t", 0)]),
    ("CX",           2, [("cx", 0, 1)]),
    ("H,CX",         2, [("h", 0), ("cx", 0, 1)]),
    ("H,CX,T",       2, [("h", 0), ("cx", 0, 1), ("t", 0)]),
    ("3-qubit chain",3, [("h", 0), ("cx", 0, 1), ("cx", 1, 2)]),
]


def build(name, n, ops, kappa, mode, prg="xor", seed=0):
    qc = QuantumCircuit(n)
    for op in ops:
        getattr(qc, op[0])(*op[1:])
    random.seed(seed)
    return garble_circuit(qc, kappa, mode=mode, prg=prg)


if __name__ == "__main__":
    for kappa in (1, 2):
        print(f"\n{'='*94}\nkappa = {kappa}\n{'='*94}")
        print(f"  {'circuit':15s} {'mode':7s} {'qubits':>8s} {'ops':>8s} {'depth':>8s} "
              f"{'toffoli':>9s} {'T':>10s} {'c^g':>7s} {'desc':>7s}")
        for name, n, ops in CASES:
            for mode in ("direct", "cre"):
                g = build(name, n, ops, kappa, mode)
                r = report(g, f"{name} [{mode}]")
                print(f"  {name:15s} {mode:7s} {r['encoded_qubits']:>8,d} {r['encoded_ops']:>8,d} "
                      f"{r['depth']:>8,d} {r['toffoli']:>9,d} {r['t_count']:>10,d} "
                      f"{r['cg_qubits']:>7,d} {r['desc_qubits']:>7,d}")