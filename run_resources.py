"""Resource tables for the Quantum Garbled Circuits scheme.

Everything is closed form (see qgc/cost.py); no circuit is constructed, which
is the only way to report at a kappa that keys a real PRG. The small-kappa rows
are cross-checked against circuits that were actually built, so the large-kappa
rows rest on a model known to be exact where it can be tested.

    python run_resources.py
"""
import random

from qiskit import QuantumCircuit

from qgc import cost
from qgc.garble import garble_circuit

CIRCUITS = [
    ("H",             1, [("h", 0)]),
    ("T",             1, [("t", 0)]),
    ("CX",            2, [("cx", 0, 1)]),
    ("H,CX,T",        2, [("h", 0), ("cx", 0, 1), ("t", 0)]),
    ("3-qubit chain", 3, [("h", 0), ("cx", 0, 1), ("cx", 1, 2)]),
]

# kappa that key SIMON at n = kappa/4, m = 4. kappa=128 gives 128-bit labels,
# hence ~64-bit security against a Grover search; 256 is the conservative read.
CRYPTO_KAPPA = [64, 96, 128, 256]


def arities(n, ops):
    qc = QuantumCircuit(n)
    for op in ops:
        getattr(qc, op[0])(*op[1:])
    return qc, [inst.operation.num_qubits for inst in qc.data]


def human(x):
    for unit, size in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if x >= size:
            return f"{x / size:.2f}{unit}"
    return str(x)


def validate():
    print("=" * 78)
    print("VALIDATION -- closed form vs circuits actually built")
    print("=" * 78)
    print(f"  {'circuit':15s} {'kappa':>5s} {'mode':7s} "
          f"{'qubits (model/built)':>24s} {'dec ops':>18s} {'T':>14s}")
    ok = True
    for kappa in (1, 2):
        for mode in ("direct", "cre"):
            for name, n, ops in CIRCUITS:
                if mode == "cre" and (kappa > 1 or n > 2):
                    continue
                qc, ars = arities(n, ops)
                random.seed(0)
                g = garble_circuit(qc, kappa, mode=mode)
                mq = cost.qubit_counts(n, ars, kappa, mode)["total"]
                bq = g.encoded_circuit.num_qubits
                md = cost.dec_cost(n, ars, kappa, mode)
                mo = sum(v for k, v in md.items() if k != "t")
                bo = len(g.decoder.data)
                mt, bt = md["t"], cost.measured_t(g.decoder)
                good = (mq == bq and mo == bo and mt == bt)
                ok &= good
                print(f"  {name:15s} {kappa:>5d} {mode:7s} "
                      f"{mq:>11,d}/{bq:<11,d} {mo:>8,d}/{bo:<8,d} {mt:>6,d}/{bt:<6,d}"
                      f" {'' if good else '  MISMATCH'}")
    print(f"\n  {'all exact' if ok else 'MISMATCHES PRESENT'}\n")
    return ok


def qubit_table():
    print("=" * 78)
    print("ENCODING SIZE (qubits), CRE mode")
    print("=" * 78)
    print(f"  {'circuit':15s} {'kappa':>6s} {'segments':>14s} {'c^g':>14s} "
          f"{'desc':>14s} {'total':>14s}")
    for name, n, ops in CIRCUITS:
        _, ars = arities(n, ops)
        for kappa in [1, 2] + CRYPTO_KAPPA:
            c = cost.qubit_counts(n, ars, kappa, "cre")
            print(f"  {name:15s} {kappa:>6d} {human(c['segments']):>14s} "
                  f"{human(c['cg']):>14s} {human(c['desc']):>14s} "
                  f"{human(c['total']):>14s}")
        print()


def dec_table():
    print("=" * 78)
    print("DECODING COST, CRE mode (xor mask -- the PRG is priced separately)")
    print("=" * 78)
    print(f"  {'circuit':15s} {'kappa':>6s} {'dec ops':>12s} {'toffoli':>12s} "
          f"{'T':>12s}")
    for name, n, ops in CIRCUITS:
        _, ars = arities(n, ops)
        for kappa in [1, 2] + CRYPTO_KAPPA:
            d = cost.dec_cost(n, ars, kappa, "cre")
            ops_total = sum(v for k, v in d.items() if k != "t")
            print(f"  {name:15s} {kappa:>6d} {human(ops_total):>12s} "
                  f"{human(d.get('ccx', 0)):>12s} {human(d['t']):>12s}")
        print()


def simon_table():
    print("=" * 78)
    print("SIMON AS THE COHERENT PRG IN Dec  (n = kappa/4, m = 4)")
    print("=" * 78)
    print(f"  {'circuit':15s} {'kappa':>6s} {'variant':>14s} {'blocks':>10s} "
          f"{'toffoli':>12s} {'T':>12s}")
    for name, n, ops in CIRCUITS:
        _, ars = arities(n, ops)
        for kappa in CRYPTO_KAPPA:
            s = cost.simon_cost(kappa, ars)
            print(f"  {name:15s} {kappa:>6d} {s['variant']:>14s} "
                  f"{human(s['blocks']):>10s} {human(s['toffoli']):>12s} "
                  f"{human(s['t']):>12s}")
        print()


def totals():
    print("=" * 78)
    print("TOTAL T-COUNT IN Dec  (garbled-table lookup + SIMON)")
    print("=" * 78)
    print(f"  {'circuit':15s} {'kappa':>6s} {'lookup':>12s} {'simon':>12s} "
          f"{'total':>12s}")
    for name, n, ops in CIRCUITS:
        _, ars = arities(n, ops)
        for kappa in CRYPTO_KAPPA:
            d = cost.dec_cost(n, ars, kappa, "cre")
            s = cost.simon_cost(kappa, ars)
            print(f"  {name:15s} {kappa:>6d} {human(d['t']):>12s} "
                  f"{human(s['t']):>12s} {human(d['t'] + s['t']):>12s}")
        print()


if __name__ == "__main__":
    validate()
    qubit_table()
    dec_table()
    simon_table()
    totals()
    from qgc.gate_words import PHASE_BITS, SYM_BITS, WORD_LEN
    single = WORD_LEN[1] * SYM_BITS[1]
    pair = WORD_LEN[2] * SYM_BITS[2]
    print(f"Scaling, for the current alphabet ({single} bits per single slot, "
          f"{pair} per pair):")
    print(f"  desc_bits(k) = (3+3k)*{single} + k(k+1)/2*{pair} + {PHASE_BITS}"
          f"   per output wire")
    print("  c^g          = 2^(2p) * p * desc_bits(k)          per gate of arity p")
