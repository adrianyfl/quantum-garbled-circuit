"""Section 6.2: the classical randomized encoding attached to one gate."""
from dataclasses import dataclass

from qgc.correction_function import build_correction_function
from qgc.prg_garble import prg_garble
from qgc.randomization_group import desc_bits_len
from qgc.structure import WireParams


@dataclass
class GateCRE:
    table: dict           # {frame bits -> description bitstring}, plaintext
    garbled: object       # PRGGarbled -- the offline part f_off, goes in c^g
    chunk_offsets: list   # [(start, length)] per output wire, into a table row
    kappa: int


def wire_params(record, a_circuit, kappa) -> WireParams:
    return WireParams(A=a_circuit,
                      l_z=tuple(record.l_z), l_x=tuple(record.l_x),
                      s_x=record.s_x, s_z=record.s_z,
                      t_x=record.t_x, t_z=record.t_z,
                      kappa=kappa)


def input_labels(input_records):
    """The 2p CRE input labels: one pair per (wire, basis)."""
    labels = []
    for rec in input_records:
        labels.append(tuple(rec.l_z))     # labels of the bit d_j
        labels.append(tuple(rec.l_x))     # labels of the bit e_j
    return labels


def build_gate_cre(operation, out_params, input_records, kappa, prg="xor",
                   rounds=None) -> GateCRE:
    table = build_correction_function(operation, out_params)
    garbled = prg_garble(table, input_labels(input_records), kappa, prg=prg,
                         rounds=rounds)
    offsets, pos = [], 0
    for w in out_params:
        n = desc_bits_len(w.kappa)
        offsets.append((pos, n))
        pos += n
    return GateCRE(table=table, garbled=garbled, chunk_offsets=offsets, kappa=kappa)
