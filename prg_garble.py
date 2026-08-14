import hashlib, itertools
from dataclasses import dataclass, field
from helper import get_bit, find_witness_bit

def prg_sha256(label_vals, kappa, tag, nbits):
    out, ctr = [], 0
    while len(out) < nbits:
        h = hashlib.sha256()
        for v in label_vals:
            h.update(v.to_bytes((kappa + 7) // 8, 'little'))
        h.update(tag.to_bytes(4, 'little'))
        h.update(ctr.to_bytes(4, 'little'))
        for byte in h.digest():
            for b in range(8):
                out.append((byte >> b) & 1)
                if len(out) == nbits:
                    return out
        ctr += 1
    return out[:nbits]


def prg_xor(label_vals, kappa, tag, nbits):
    out = []
    for j in range(nbits):
        m = 0
        for v in label_vals:
            m ^= get_bit(v, (tag + j) % max(kappa, 1))
        out.append(m)
    return out


PRGS = {"sha256": prg_sha256, "xor": prg_xor}


@dataclass
class PRGGarbled:
    kappa_in: int
    n_in: int
    out_bits: int
    select: list = field(default_factory=list)
    constants: dict = field(default_factory=dict)
    groups: list = field(default_factory=list)   # (dep tuple, [output bit indices])
    tables: dict = field(default_factory=dict)   # group idx -> {row: [masked bits]}
    prg: str = "sha256"
    calls_garble: int = 0

    def size_bits(self):
        return sum(len(bits) for t in self.tables.values() for bits in t.values()) \
               + len(self.constants)


def prg_garble(table, deps, labels, kappa_in, prg="sha256"):
    f = PRGS[prg]
    n_in = len(labels)
    out_bits = len(next(iter(table.values())))
    g = PRGGarbled(kappa_in=kappa_in, n_in=n_in, out_bits=out_bits, prg=prg)
    g.select = [find_witness_bit(labels[i][0], labels[i][1], kappa_in) for i in range(n_in)]

    # group output bits by dependency set
    by_dep = {}
    for k, dep in enumerate(deps):
        if not dep:
            g.constants[k] = int(table[next(iter(table))][k])
            continue
        by_dep.setdefault(tuple(dep), []).append(k)
    g.groups = sorted(by_dep.items())

    for gi, (dep, ks) in enumerate(g.groups):
        tbl = {}
        for vals in itertools.product((0, 1), repeat=len(dep)):
            bits = [0] * n_in
            for pos, i in enumerate(dep):
                bits[i] = vals[pos]
            plain = [int(table[tuple(bits)][k]) for k in ks]
            lab_vals = [labels[i][vals[pos]] for pos, i in enumerate(dep)]
            masks = f(lab_vals, kappa_in, gi, len(ks))       # ONE call for the whole group
            g.calls_garble += 1
            row = tuple(get_bit(labels[i][vals[pos]], g.select[i]) for pos, i in enumerate(dep))
            tbl[row] = [p ^ m for p, m in zip(plain, masks)]
        assert len(tbl) == 2 ** len(dep), "select bits collided"
        g.tables[gi] = tbl
    return g


def prg_cdec(g: PRGGarbled, held):
    f = PRGS[g.prg]
    out = [None] * g.out_bits
    for k, v in g.constants.items():
        out[k] = str(v)
    calls = 0
    for gi, (dep, ks) in enumerate(g.groups):
        row = tuple(get_bit(held[i], g.select[i]) for i in dep)
        masks = f([held[i] for i in dep], g.kappa_in, gi, len(ks))   # ONE call per group
        calls += 1
        for pos, k in enumerate(ks):
            out[k] = str(g.tables[gi][row][pos] ^ masks[pos])
    return ''.join(out), calls