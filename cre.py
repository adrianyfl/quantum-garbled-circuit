from dataclasses import dataclass

from correction_function import build_correction_function, dependency_profile
from prg_garble import prg_garble
from randomization_group import desc_bits_len
from structure import WireParams

@dataclass
class GateCRE:
    table: dict           # {input bits -> description bitstring}
    deps: list            # per output bit, which input bits it depends on
    garbled: object       # PRGGarbled -- the offline part f_off
    chunk_offsets: list   # [(start, length)] per output wire
    kappa: int


def wire_params(record, a_circuit, kappa) -> WireParams:
    return WireParams(A=a_circuit,
                      l_z=tuple(record.l_z), l_x=tuple(record.l_x),
                      s_x=record.s_x, s_z=record.s_z,
                      t_x=record.t_x, t_z=record.t_z,
                      kappa=kappa)


def input_labels(input_records):
    labels = []
    for rec in input_records:
        labels.append(tuple(rec.l_z))     # labels of bit d_j
        labels.append(tuple(rec.l_x))     # labels of bit e_j
    return labels


def build_gate_cre(operation, out_params, input_records, kappa, prg="xor") -> GateCRE:
    table = build_correction_function(operation, out_params)
    deps = dependency_profile(table, operation.num_qubits)
    garbled = prg_garble(table, deps, input_labels(input_records), kappa, prg=prg)
    offsets, pos = [], 0
    for w in out_params:
        n = desc_bits_len(w.kappa)
        offsets.append((pos, n))
        pos += n
    return GateCRE(table=table, deps=deps, garbled=garbled,
                   chunk_offsets=offsets, kappa=kappa)