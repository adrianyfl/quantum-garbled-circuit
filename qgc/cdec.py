"""Protocol 7 step 1: run CDec coherently to recover the correction gadgets.

Controlled on the colour qubits of the held labels, open the one addressed row
of c^g into the description register, then unmask it with the PRG. Both the
control set and the row loop are fixed by (n_in, out_bits) alone, so this
circuit is the same for every choice of randomness -- which is what Section 6.4
asks of CDec.
"""
from qgc.helper import POINTER_BIT


def row_index(colour_bits):
    """Colour tuple -> row number, first element most significant."""
    idx = 0
    for c in colour_bits:
        idx = (idx << 1) | c
    return idx


def cg_layout(g):
    """Number of qubits c^g needs.

    The layout is rows in colour order, out_bits apiece, so a bit's address is
    arithmetic (see cg_offset). Materialising it as a dict would cost one entry
    per stored bit -- about 20M entries for a 2-qubit gate at kappa=128.
    """
    return g.num_rows() * g.out_bits


def cg_offset(g, colour_bits, j):
    return row_index(colour_bits) * g.out_bits + j


def cg_bits(g):
    bits = []
    for colour_bits in sorted(g.rows):
        bits.extend(g.rows[colour_bits])
    return bits


def prepare_cg(qc, g, cg_qubits):
    """Enc writes the garbled table into c^g, which is part of the encoding."""
    for i, b in enumerate(cg_bits(g)):
        if b:
            qc.x(cg_qubits[i])
    return sum(cg_bits(g))


def add_lookup(qc, table, controls, outs, ancilla):
    """Row lookup with the stored bits hardcoded as gates rather than held in
    c^g. Narrower than register mode, so it is what the tests use at widths a
    simulator can reach; garble.py always uses register mode, where c^g is part
    of the encoding rather than baked into Dec."""
    for row, bits in sorted(table.items()):
        flips = [controls[i] for i, r in enumerate(row) if r == 0]
        for q in flips:
            qc.x(q)
        qc.mcx(controls, ancilla)
        for j, b in enumerate(bits):
            if b:
                qc.cx(ancilla, outs[j])
        qc.mcx(controls, ancilla)
        for q in flips:
            qc.x(q)
    return qc


def add_xor_mask(qc, label_qubits, out_qubits, kappa, n_in, out_bits):
    """The reversible form of prg_xor: out[j] ^= XOR_i bit_(j mod kappa)(label_i)."""
    for j in range(out_bits):
        pos = j % max(kappa, 1)
        for i in range(n_in):
            qc.cx(label_qubits[i][pos], out_qubits[j])


def add_cdec(qc, g, label_qubits, out_qubits, ancilla, kappa,
             cg_qubits=None, mode="register", block_qubits=None):
    """Coherently open the addressed row of c^g and unmask it.

    `block_qubits` is the SIMON scratch block, required when g.prg == "simon".
    """
    if g.prg not in ("xor", "simon"):
        raise ValueError(f"no reversible mask circuit for prg={g.prg!r}")
    if g.prg == "simon" and block_qubits is None:
        raise ValueError("prg='simon' needs a block scratch register")

    controls = [label_qubits[i][POINTER_BIT] for i in range(g.n_in)]

    if mode == "register":
        assert cg_qubits is not None, "register mode needs c^g"
        for colour_bits in sorted(g.rows):
            flips = [controls[i] for i, c in enumerate(colour_bits) if c == 0]
            for q in flips:
                qc.x(q)
            qc.mcx(controls, ancilla)
            for j in range(g.out_bits):
                qc.ccx(ancilla, cg_qubits[cg_offset(g, colour_bits, j)], out_qubits[j])
            qc.mcx(controls, ancilla)
            for q in flips:
                qc.x(q)
    else:
        add_lookup(qc, g.rows, controls, out_qubits, ancilla)

    if g.prg == "xor":
        add_xor_mask(qc, label_qubits, out_qubits, kappa, g.n_in, g.out_bits)
    else:
        from qgc.simon_prg import add_simon_mask
        add_simon_mask(qc, label_qubits, out_qubits, kappa, g.n_in, g.out_bits,
                       block_qubits, rounds=g.rounds)
    return qc
