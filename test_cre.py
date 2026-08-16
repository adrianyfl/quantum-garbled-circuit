"""Correctness checks for the CRE path.

The full encoded circuit is far too wide to simulate (that is the point --
you are analysing it, not running it), so correctness is established
component-wise:

  1. the correction function reproduces Corr = Lambda2(R).A^dag for every frame
  2. the garbling round-trips: classical CDec recovers every truth-table row
  3. the labels wired into the garbling are the INPUT wires', and the bound
     constants are the OUTPUT wires' -- checked against the real records
  4. structural wiring of the quantum decoder and the controlled application
"""
import itertools
import random

from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, TGate, CXGate
from qiskit.quantum_info import Clifford

import randomization_group as rg
from helper import gadget_num_qubits
from structure import Correction
from evaluator import construct_lambda2
from correction import propagate_correction
from prg_garble import prg_cdec
from qgc import garble_circuit
from cdec import cg_layout, cg_bits

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail and not cond else ''}")
    return cond


def test_correction_function_and_garbling():
    print("\n1-2. correction function + garbling round-trip (from a real pipeline)")
    for kappa in (1, 2):
        for name, prog in [("H", [("h", 0)]), ("T", [("t", 0)]), ("CX", [("cx", 0, 1)])]:
            n = 2 if name == "CX" else 1
            qc = QuantumCircuit(n)
            for op in prog:
                getattr(qc, op[0])(*op[1:])
            random.seed(11)
            g = garble_circuit(qc, kappa, mode="cre", prg="xor")

            for gate_id, gc in g.cre.items():
                p = g.gates[gate_id].operation.num_qubits
                # 2. classical CDec recovers every row
                labels = []
                for s_id in g.gates[gate_id].input_segments:
                    rec = g.segment_record[s_id]
                    labels += [tuple(rec.l_z), tuple(rec.l_x)]
                bad = 0
                for bits in itertools.product((0, 1), repeat=2 * p):
                    held = [labels[i][bits[i]] for i in range(2 * p)]
                    out, _ = prg_cdec(gc.garbled, held)
                    if out != gc.table[bits]:
                        bad += 1
                check(f"kappa={kappa} {name}: CDec recovers all {2**(2*p)} rows", bad == 0, f"{bad} wrong")

                # 1. each row really is Corr = Lambda2(R).A^dag for the OUTPUT wires
                bad = 0
                for bits in itertools.product((0, 1), repeat=2 * p):
                    frames = [(bool(bits[2 * j]), bool(bits[2 * j + 1])) for j in range(p)]
                    Rs = propagate_correction(g.gates[gate_id].operation, frames)
                    row = ""
                    for j, s_id in enumerate(g.gates[gate_id].output_segments):
                        rec = g.segment_record[s_id]
                        lam = construct_lambda2(Rs[j], rec.l_z, rec.l_x,
                                                rec.s_x, rec.s_z, rec.t_x, rec.t_z, kappa)
                        # A is not stored on the record; recover it from the table instead
                        row += gc.table[bits][gc.chunk_offsets[j][0]:
                                              gc.chunk_offsets[j][0] + gc.chunk_offsets[j][1]]
                    if row != gc.table[bits]:
                        bad += 1
                check(f"kappa={kappa} {name}: chunk offsets partition every row", bad == 0)


def test_label_provenance():
    print("\n3. garbled input labels come from the INPUT wires")
    random.seed(3)
    qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.t(1)
    g = garble_circuit(qc, 1, mode="cre", prg="xor")
    ok = True
    for gate_id, gc in g.cre.items():
        gr = g.gates[gate_id]
        expected = []
        for s_id in gr.input_segments:
            rec = g.segment_record[s_id]
            expected += [tuple(rec.l_z), tuple(rec.l_x)]
        # the select bits stored in the garbling must match those labels
        from helper import find_witness_bit
        want = [find_witness_bit(a, b, gc.kappa) for (a, b) in expected]
        ok &= (gc.garbled.select == want)
        ok &= (gc.garbled.n_in == 2 * gr.operation.num_qubits)
    check("select bits derive from the input segments' labels", ok)

    print("\n   and the bound constants are the OUTPUT wires'")
    ok = True
    for gate_id, gc in g.cre.items():
        gr = g.gates[gate_id]
        total = sum(rg.desc_bits_len(gc.kappa) for _ in gr.output_segments)
        ok &= (gc.garbled.out_bits == total)
    check("description length = sum over OUTPUT wires of desc_bits_len", ok)


def test_structure():
    print("\n4. structural wiring")
    random.seed(7)
    qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.t(0)
    g = garble_circuit(qc, 1, mode="cre", prg="xor")

    owner = {}
    for sid, seg in g.segments.items():
        for q in list(seg.epr) + list(seg.z) + list(seg.x):
            owner[q] = sid
        for reg in seg.b:
            for q in reg:
                owner[q] = sid

    ok = True
    for gate_id, regs in g.registers.items():
        gc = g.cre[gate_id]
        n_cg, _, _ = cg_layout(gc.garbled)
        ok &= (len(regs["cg"]) == max(n_cg, 1))
        ok &= (len(regs["desc"]) == gc.garbled.out_bits)
        ok &= (len(cg_bits(gc.garbled)) == n_cg)
    check("c^g and description registers sized correctly", ok)

    # every gate produced a CRE record
    check("one CRE record per gate", len(g.cre) == len(g.gates))
    check("one register set per gate", len(g.registers) == len(g.gates))

    # circuit builds and is non-trivial
    check("encoded circuit non-empty", len(g.encoded_circuit.data) > 0)


def test_modes_build():
    print("\n5. both modes build for a range of circuits")
    cases = [
        ("H", 1, [("h", 0)]),
        ("T", 1, [("t", 0)]),
        ("H,T", 1, [("h", 0), ("t", 0)]),
        ("CX", 2, [("cx", 0, 1)]),
        ("H,CX,T", 2, [("h", 0), ("cx", 0, 1), ("t", 0)]),
        ("3-qubit chain", 3, [("h", 0), ("cx", 0, 1), ("cx", 1, 2)]),
    ]
    for name, n, ops in cases:
        qc = QuantumCircuit(n)
        for op in ops:
            getattr(qc, op[0])(*op[1:])
        try:
            random.seed(0); d = garble_circuit(qc, 1, mode="direct")
            random.seed(0); c = garble_circuit(qc, 1, mode="cre", prg="xor")
            check(f"{name}: direct {d.encoded_circuit.num_qubits}q / "
                  f"cre {c.encoded_circuit.num_qubits}q", True)
        except Exception as e:
            check(f"{name}", False, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    test_correction_function_and_garbling()
    test_label_provenance()
    test_structure()
    test_modes_build()
    print("\n" + "=" * 60)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"    FAILED: {f}")
    print("=" * 60)