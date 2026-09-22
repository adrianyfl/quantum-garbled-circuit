"""Dec applied to the encoding."""
from qiskit import QuantumCircuit

from qgc.structure import QGC


def decode_output(qgc: QGC, kappa: int | None = None) -> QuantumCircuit:
    """Enc, then Dec, over the same registers.

    garble_circuit builds the two separately; this is only the composition, for
    simulating the round trip. Dec never reads the plaintext labels -- it
    recovers (d, e) from the d^w dictionary registers that Enc wrote (see
    decode_label_bit in garble.py).

    `kappa` is accepted and ignored, since the decoder is prebuilt.
    """
    return qgc.encoded_circuit.compose(qgc.decoder)
