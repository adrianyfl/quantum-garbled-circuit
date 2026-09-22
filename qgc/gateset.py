"""The paper's universal gate set, and transpilation into it.

Section 6 assumes the circuit being encoded is over C_2 u {T}. The only
property used is Equation (6.1): for a p-qubit gate U and single-qubit Paulis
P_1..P_p there are PX-group R_1..R_p with

    U (P_1 (x) ... (x) P_p) = (R_1 (x) ... (x) R_p) U

which is what propagate_correction computes. Anything outside the set breaks
that, so garble_circuit routes its input through here first.

T-dagger is rewritten as T then S-dagger. Both are in the PX group so the
identity still holds, but propagate_correction special-cases the name "t", and
carrying a second non-Clifford name through it buys nothing.
"""
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Clifford, Operator

# swap, cz, y, z and sdg are redundant given h, s and cx, but they keep the
# transpiler's output short and readable.
CLIFFORD_T_BASIS = ["h", "s", "sdg", "x", "y", "z", "cx", "cz", "swap", "t", "tdg"]


def in_gate_set(operation) -> bool:
    """Is this operation in C_2 u {T}?"""
    if operation.name == "t":
        return True
    if operation.num_qubits > 2:
        return False
    try:
        Clifford(operation)
        return True
    except Exception:
        return False


def offenders(circuit) -> list:
    """Names of operations that are not in C_2 u {T}."""
    return sorted({inst.operation.name for inst in circuit.data
                   if not in_gate_set(inst.operation)})


def rewrite_tdg(circuit: QuantumCircuit) -> QuantumCircuit:
    """T-dagger -> T then S-dagger. Both diagonal, so the order is immaterial."""
    out = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for inst in circuit.data:
        if inst.operation.name == "tdg":
            out.t(inst.qubits[0])
            out.sdg(inst.qubits[0])
        else:
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def to_gate_set(circuit: QuantumCircuit, optimization_level: int = 1):
    """Rewrite a circuit over C_2 u {T}. -> (circuit, info).

    Gates already in the set pass through. Anything else is synthesised by the
    transpiler, which for arbitrary rotations is an APPROXIMATION -- Qiskit does
    not raise, it silently emits a long Clifford+T sequence. info["synthesised"]
    lists what was not exactly representable, so callers can say so.
    """
    # tdg is in C_2 u {T} via the rewrite below, so it is not an offender; rz
    # and friends are.
    synthesised = [n for n in offenders(circuit) if n != "tdg"]

    out = transpile(circuit, basis_gates=CLIFFORD_T_BASIS,
                    optimization_level=optimization_level)
    out = rewrite_tdg(out)

    remaining = offenders(out)
    if remaining:
        raise ValueError(
            f"could not rewrite into C_2 u {{T}}: {remaining} remain. "
            f"The transpiler emits only {CLIFFORD_T_BASIS}.")

    return out, {
        "synthesised": synthesised,
        "approximate": bool(synthesised),
        "ops_before": len(circuit.data),
        "ops_after": len(out.data),
    }


def assert_gate_set(circuit: QuantumCircuit):
    bad = offenders(circuit)
    if bad:
        raise ValueError(
            f"{bad} are outside C_2 u {{T}}; pass transpile_input=True or run "
            f"to_gate_set first. Section 6.1 needs Equation (6.1) to hold for "
            f"every gate.")
    return circuit


def synthesis_error(original: QuantumCircuit, rewritten: QuantumCircuit) -> float:
    """max |A c - B| over the two unitaries, up to global phase.

    Only for circuits small enough to build an Operator -- a spot check on the
    transpiler, not part of the pipeline.
    """
    import numpy as np
    a = Operator(original).data.ravel()
    b = Operator(rewritten).data.ravel()
    i = int(np.argmax(np.abs(a)))
    return float(np.max(np.abs(a * (b[i] / a[i]) - b)))
