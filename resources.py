"""Resource accounting for the garbled circuit.

The circuit is never executed -- these counts are read off its structure.

T-COST MODEL (adjust to match whatever convention you report in):
  cx, cy, cz, h, s, sdg, x, y, z, swap : Clifford, 0 T
  ccx (Toffoli)                        : T_TOFFOLI
  mcx with k controls                  : (k-1) Toffolis  -> (k-1)*T_TOFFOLI
  cs, csdg                             : T_CS
  ch                                   : T_CH
  cp(theta)                            : T_CS   (it is a controlled phase)
These are the standard fault-tolerant decompositions; the constants are
parameters, not claims.
"""
from collections import Counter

T_TOFFOLI = 7      # 7 T per Toffoli (4 with measurement-and-feedforward)
T_CS = 3
T_CH = 4

CLIFFORD_1Q = {"h", "s", "sdg", "x", "y", "z", "id", "sx", "sxdg"}
CLIFFORD_2Q = {"cx", "cy", "cz", "swap"}


def gate_histogram(circuit):
    h = Counter()
    for inst in circuit.data:
        name = inst.operation.name
        if name == "mcx":
            h[f"mcx{len(inst.qubits) - 1}"] += 1
        else:
            h[name] += 1
    return h


def t_count(circuit):
    total = 0
    for inst in circuit.data:
        name = inst.operation.name
        nq = len(inst.qubits)
        if name == "ccx":
            total += T_TOFFOLI
        elif name == "mcx":
            k = nq - 1
            total += max(k - 1, 0) * T_TOFFOLI
        elif name in ("cs", "csdg", "cp"):
            total += T_CS
        elif name == "ch":
            total += T_CH
        elif name == "t" or name == "tdg":
            total += 1
    return total


def toffoli_count(circuit):
    total = 0
    for inst in circuit.data:
        name = inst.operation.name
        if name == "ccx":
            total += 1
        elif name == "mcx":
            total += max(len(inst.qubits) - 2, 0)
    return total


def report(qgc, label=""):
    c = qgc.encoded_circuit
    h = gate_histogram(c)
    return {
        "label": label,
        "logical_qubits": qgc.original_circuit.num_qubits,
        "logical_gates": len(qgc.original_circuit.data),
        "segments": len(qgc.segments),
        "encoded_qubits": c.num_qubits,
        "encoded_ops": len(c.data),
        "depth": c.depth(),
        "toffoli": toffoli_count(c),
        "t_count": t_count(c),
        "cg_qubits": sum(len(v["cg"]) for v in qgc.registers.values()),
        "desc_qubits": sum(len(v["desc"]) for v in qgc.registers.values()),
        "histogram": dict(sorted(h.items())),
    }


def print_report(r):
    print(f"  {r['label']}")
    for k in ("logical_qubits", "logical_gates", "segments", "encoded_qubits",
              "encoded_ops", "depth", "toffoli", "t_count", "cg_qubits", "desc_qubits"):
        print(f"     {k:16s} {r[k]:>12,d}")