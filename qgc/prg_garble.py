"""Point-and-permute garbling of the correction function (CEnc and CDec).

One table of 2^n_in rows, addressed by the colour bits of the input labels.
Row (v_1..v_n) is masked with the PRG keyed on the labels the evaluator would
actually hold for those values, so holding one label per wire opens exactly one
row, and the colours reveal nothing because each is a value XORed with an
unknown uniform bit.

The shape depends only on the topology -- n_in = 2p and out_bits = sum over the
output wires of desc_bits_len(kappa) -- never on the randomness. Section 6.4
requires that of CDec: "CDec only depends on the topology T_g of the correction
function". An earlier version grouped output bits by a discovered dependency
profile, which made both the table shape and the decoder circuit vary with the
labels, and had to go.
"""
import itertools
from dataclasses import dataclass, field

from qgc.helper import colour, get_bit
from qgc.simon_prg import prg_simon


def prg_xor(label_vals, kappa, tag, nbits, rounds=None):
    """A LINEAR stand-in with no security whatever -- a test fixture only.

    It exists so the garbling and its reversible decoder can be checked against
    each other at kappa far too small to key a cipher. The real PRG is SIMON.
    """
    out = []
    for j in range(nbits):
        m = 0
        for v in label_vals:
            m ^= get_bit(v, (tag + j) % max(kappa, 1))
        out.append(m)
    return out


PRGS = {"xor": prg_xor, "simon": prg_simon}


@dataclass
class PRGGarbled:
    """The offline part f_off for one gate: the garbled table."""
    kappa_in: int
    n_in: int
    out_bits: int
    rows: dict = field(default_factory=dict)      # colour tuple -> [masked bits]
    prg: str = "xor"
    rounds: int | None = None                     # SIMON round reduction, None = spec

    def num_rows(self):
        return 2 ** self.n_in


def prg_garble(table, labels, kappa_in, prg="xor", rounds=None):
    """Garble a 2p-input lookup table under point-and-permute."""
    f = PRGS[prg]
    n_in = len(labels)
    out_bits = len(next(iter(table.values())))
    g = PRGGarbled(kappa_in=kappa_in, n_in=n_in, out_bits=out_bits,
                   prg=prg, rounds=rounds)

    for vals in itertools.product((0, 1), repeat=n_in):
        plain = [int(b) for b in table[vals]]
        held = [labels[i][vals[i]] for i in range(n_in)]
        masks = f(held, kappa_in, 0, out_bits, rounds=rounds)
        g.rows[tuple(colour(h) for h in held)] = [p ^ m for p, m in zip(plain, masks)]

    assert len(g.rows) == 2 ** n_in, "colour collision -- sample_label is broken"
    return g


def prg_cdec(g: PRGGarbled, held):
    """Open the one row the held labels address, and unmask it."""
    f = PRGS[g.prg]
    row = g.rows[tuple(colour(h) for h in held)]
    masks = f(held, g.kappa_in, 0, g.out_bits, rounds=g.rounds)
    return ''.join(str(r ^ m) for r, m in zip(row, masks)), 1
