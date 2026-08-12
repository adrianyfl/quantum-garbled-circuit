from qiskit import QuantumCircuit

from correction import get_witnesses
from structure import QGC

def decode_output(qgc: QGC, kappa: int) -> QuantumCircuit:
    circuit = qgc.encoded_circuit.copy()
    for q, seg_id in qgc.current_segment.items():
        segment = qgc.segments[seg_id]
        record = qgc.segment_record[seg_id]
        d_qubit, e_qubit = get_witnesses(circuit, segment, record, kappa)
        circuit.cx(e_qubit, segment.epr[1])
        circuit.cz(d_qubit, segment.epr[1])
    return circuit