from qiskit import QuantumCircuit

from structure import QGC


def decode_output(qgc: QGC, kappa: int | None = None) -> QuantumCircuit:
    """Dec applied to the encoding: Enc, then Dec, over the same registers.

    Enc and Dec are built as separate circuits by garble_circuit; this is only
    the composition, for simulating the round trip. Dec never reads the
    plaintext labels -- it recovers (d, e) from the d^w dictionary registers
    that Enc wrote (see decode_label_bit in qgc.py).

    `kappa` is accepted and ignored; it is no longer needed now that the
    decoder is prebuilt.
    """
    return qgc.encoded_circuit.compose(qgc.decoder)


def output_qubits(qgc: QGC) -> list:
    """The qubit of each output wire that holds F(y) after decoding."""
    return [qgc.segments[qgc.current_segment[q]].epr[1]
            for q in range(qgc.original_circuit.num_qubits)]
