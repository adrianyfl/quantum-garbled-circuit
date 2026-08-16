from lookup import add_lookup


def cg_layout(g):
    table_pos, const_pos, pos = {}, {}, 0
    for gi, (dep, ks) in enumerate(g.groups):
        for row in sorted(g.tables[gi]):
            for j in range(len(ks)):
                table_pos[(gi, row, j)] = pos
                pos += 1
    for k in sorted(g.constants):
        const_pos[k] = pos
        pos += 1
    return pos, table_pos, const_pos


def cg_bits(g):
    bits = []
    for gi, (dep, ks) in enumerate(g.groups):
        for row in sorted(g.tables[gi]):
            bits.extend(g.tables[gi][row])
    for k in sorted(g.constants):
        bits.append(g.constants[k])
    return bits


def prepare_cg(qc, g, cg_qubits):
    for i, b in enumerate(cg_bits(g)):
        if b:
            qc.x(cg_qubits[i])
    return sum(cg_bits(g))


def add_xor_mask(qc, gi, dep, ks, label_qubits, out_qubits, kappa):
    for j, k in enumerate(ks):
        pos = (gi + j) % max(kappa, 1)
        for i in dep:
            qc.cx(label_qubits[i][pos], out_qubits[k])


# ---------------------------------------------------------------- decoder
def add_cdec(qc, g, label_qubits, out_qubits, ancilla, kappa,
             cg_qubits=None, mode="register"):
    assert g.prg == "xor", (
        "the reversible mask circuit implements the LINEAR prg only. "
        "prg='sha256' garbles correctly but would need a reversible hash to decode; "
        "that circuit is not built, so its cost is not in these counts.")
    if mode == "register":
        assert cg_qubits is not None, "register mode needs c^g"
        _, table_pos, const_pos = cg_layout(g)
        for k in sorted(g.constants):
            qc.cx(cg_qubits[const_pos[k]], out_qubits[k])
        for gi, (dep, ks) in enumerate(g.groups):
            controls = [label_qubits[i][g.select[i]] for i in dep]
            for row in sorted(g.tables[gi]):
                flips = [controls[i] for i, r in enumerate(row) if r == 0]
                for q in flips:
                    qc.x(q)
                qc.mcx(controls, ancilla)
                for j, k in enumerate(ks):
                    qc.ccx(ancilla, cg_qubits[table_pos[(gi, row, j)]], out_qubits[k])
                qc.mcx(controls, ancilla)
                for q in flips:
                    qc.x(q)
            add_xor_mask(qc, gi, dep, ks, label_qubits, out_qubits, kappa)
    else:
        for k, v in g.constants.items():
            if v:
                qc.x(out_qubits[k])
        for gi, (dep, ks) in enumerate(g.groups):
            controls = [label_qubits[i][g.select[i]] for i in dep]
            add_lookup(qc, g.tables[gi], controls, [out_qubits[k] for k in ks], ancilla)
            add_xor_mask(qc, gi, dep, ks, label_qubits, out_qubits, kappa)
    return qc


def cdec_cost(g, mode="register"):
    """(mcx, toffoli, cnot) for one gate's decoder."""
    mcx = toff = cnot = 0
    for gi, (dep, ks) in enumerate(g.groups):
        tbl = g.tables[gi]
        mcx += 2 * len(tbl)
        if mode == "register":
            toff += len(tbl) * len(ks)                  # one ccx per stored bit
        else:
            cnot += sum(sum(bits) for bits in tbl.values())
        cnot += len(ks) * len(dep)                      # xor mask
    if mode == "register":
        cnot += len(g.constants)
    return mcx, toff, cnot