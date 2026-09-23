# Quantum Garbled Circuits

An implementation and resource study of the Quantum Garbled Circuits scheme of
Brakerski and Yuen, *Quantum Garbled Circuits* ([arXiv:2006.01085v2][paper]),
over the **PRG route** — the computationally private instantiation, with SIMON
as the pseudorandom generator.

Section and protocol numbers throughout the code refer to that paper.

[paper]: https://arxiv.org/abs/2006.01085

## What the scheme does

A quantum randomized encoding turns a circuit `F` and a quantum input `y` into
a state `F̂(y)` from which `F(y)` can be decoded and nothing else. The
construction routes every wire through a teleportation gadget, which leaves a
Pauli correction the evaluator has to undo; the correction depends on
measurement outcomes the evaluator must not learn, so it is delivered as a
*classical* garbled circuit that the evaluator runs coherently.

Three moving parts:

- **Enc** (Protocols 3–5) applies the gate, then `Λ1` and a random `A ∈ R_κ` to
  each output wire, and writes the garbled correction table into `c^g`.
- **Dec** (Protocols 6–7) coherently decodes the correction from the labels it
  holds, applies it, applies `Λ3`, and finally reads the output wires against
  the `d^w` label dictionaries.
- **Sim** (Protocol 8) is **not implemented** — see *Status* below.

## Install and run

```bash
git submodule update --init          # qiskit-simon, the SIMON cipher circuits
pip install -r requirements.txt

pytest tests -q                      # 39 tests
python run_resources.py              # the resource tables
```

Python ≥ 3.10 (the code uses `X | None` annotations).

```python
from qgc import garble_circuit, decode_output

g = garble_circuit(circuit, kappa=1, mode="cre")
full = decode_output(g)              # Enc then Dec, over one register set
```

## Layout

| module | what it is |
|---|---|
| `qgc/helper.py` | labels, point-and-permute colours, gadget sizing |
| `qgc/structure.py` | records, wire segments, `QGC` result object |
| `qgc/gateset.py` | transpilation into C₂ ∪ {T} (Section 6) |
| `qgc/gadgets.py` | Section 6.1: `C1`–`C3`, `Γ`, `Λ1`–`Λ3`, `A`, the Figure 5 gadget |
| `qgc/correction.py` | Equation (6.1) Pauli propagation; direct-mode correction |
| `qgc/randomization_group.py` | `R_κ` slots, canonical descriptions, phase bits |
| `qgc/gate_words.py` | the gate alphabet, words, coherent application |
| `qgc/correction_function.py` | Section 6.2.1: the correction function as a table |
| `qgc/prg_garble.py` | point-and-permute garbling (classical CEnc/CDec) |
| `qgc/simon_prg.py` | SIMON as the PRG, classical and reversible |
| `qgc/cdec.py` | Protocol 7 step 1: CDec as a quantum circuit |
| `qgc/cre.py` | per-gate CRE assembly |
| `qgc/garble.py` | **Enc**, and the matching **Dec** circuit |
| `qgc/decoder.py` | composes Enc and Dec for simulation |
| `qgc/cost.py` | closed-form resource model |

`qiskit-simon/` is a submodule: the SIMON family as reversible Qiskit circuits.
Its modules are flat and top level, so `simon_prg` puts it on `sys.path` rather
than merging it into this package.

## Two modes

**`mode="cre"`** is the paper's scheme. The correction is garbled and the
evaluator decodes it coherently. It is also several hundred qubits wide for a
one-gate circuit at κ=1, so it cannot be simulated.

**`mode="direct"`** applies the correction at encode time, in the clear, with
explicit controlled gates. It has **no privacy** and is not the scheme — it is
a reference oracle, and the only configuration narrow enough to check end to
end against `F(y)`.

## Choosing κ

`κ` is the teleportation label length. Section 6.3 ties the teleportation
labels to the CRE input labels, so the label *is* the PRG seed, and κ is
therefore also the SIMON key size. That fixes the variant:

```
n = κ/4,  m = 4   i.e.  Simon(κ/2)/(κ)     for κ ∈ {64, 96, 128, 256}
```

Among the variants with `key_size = κ` this is the cheapest: a call costs `T·n`
Toffolis and yields `2n` mask bits, so Toffoli per mask bit is `T/2` — the word
size cancels and only the round count matters, which always picks the largest
`m`. κ=128 gives 128-bit labels, hence ~64-bit security against Grover; 256 is
the conservative reading.

κ also drives the gadget, which is `(κ+1)²` qubits per wire. Nothing at a
cryptographically meaningful κ can be constructed, let alone run — a 2-qubit
gate at κ=128 needs a `c^g` register of ~20M qubits. That is why `cost.py`
exists.

## Scaling

Both verified against built circuits:

```
desc_bits(κ) = (3+3κ)·12 + κ(κ+1)/2·39 + 3   per output wire
c^g          = 2^(2p) · p · desc_bits(κ)     per gate of arity p
```

The two constants are `WORD_LEN[n] · SYM_BITS[n]`, the bits a gate word costs
for a single and a paired slot. `run_resources.py` prints the formula for
whatever alphabet is configured.

Total T-count in Dec (table lookup + coherent SIMON):

| circuit | κ=64 | κ=128 | κ=256 |
|---|---|---|---|
| H | 44.3M | 227.7M | 1.41G |
| CX | 177.1M | 912.8M | 5.64G |
| H,CX,T | 265.3M | 1.37G | 8.45G |

SIMON dominates the table lookup by ~7.5× at κ=128. The coherent PRG is the
cost of this scheme.

### The gate word encoding

`WORD_LEN` is exactly the diameter of the Clifford group under `ALPHABET`,
measured by breadth-first search in `gate_words._word_table`, and words come
from that table rather than from Qiskit's `Clifford.to_circuit()`. Both choices
matter, because `add_controlled_word` sweeps *every* non-pad symbol at *every*
position:

- a word longer than the diameter is pure padding, and a pad position still
  costs a full sweep;
- the alphabet size sets both the sweep length and, through `SYM_BITS`, the
  MCX control count.

So the minimal generating set `{cx, h₀, h₁, s₀, s₁}` beats a convenience
alphabet with `x`, `y`, `z`, `sdg` even though its words are longer (diameter
13 against 11). Against the earlier 13-symbol, length-19 encoding this is worth
**~2.1× total T and ~1.9× qubits**, and it is exactly equivalent — the same
canonical, fixed-length, topology-only description.

## How it is verified

The encoding is far too wide to simulate, so correctness is established by
other means. In practice width has rarely been the binding constraint:

- **Basis-state evaluation.** `add_cdec` is built only from X/CX/CCX/MCX, so it
  permutes basis states and can be evaluated exactly over a bit vector, linear
  in gate count and independent of width. This checks the full SIMON-backed
  decoder at **κ=64**, and the Figure 5 gadget at **κ=8**.
- **A closed-form cost model**, reproducing built circuits *to the gate* at
  κ=1,2 for qubits, Dec op counts and T-counts, then extrapolated.
- **Structural invariants**, e.g. that the garbled table's shape and the
  decoder's gate list do not depend on the randomness.
- **Component unitary comparison** for Lemma 6.2 and the correction gadgets.
- **End to end** in direct mode: `Dec(Enc(F,y)) = F(y)` at fidelity
  1.000000000000, including classical inputs.

## Status

Implemented and verified:

- teleportation and correction gadgets (Section 6.1), checked against Lemma 6.2
- point-and-permute garbling with a topology-only CDec (Section 6.4)
- SIMON as a real, reversible, coherently-evaluable PRG
- Enc/Dec split, with `d^w` label dictionaries so Dec never sees the labels
- traced-out outputs (`O \ T`), classical inputs (Figure 5), transpilation
- a validated closed-form resource model

Not implemented:

- **`Sim` (Section 6.5) and `CSim`.** Privacy is therefore *assumed, not
  demonstrated*. This matters: correctness holds for any fixed `A`, so a bug
  that made the randomizer constant passed every test in this repo until it was
  found by inspection. A privacy-side check is structurally the only thing that
  catches that class of defect.
- **Decomposability.** Definition 4.1 wants `F̂ = (F̂_off, F̂_1, …, F̂_n)` on
  disjoint qubits; Enc is currently one circuit.
- **QNC⁰_f certification.** Nothing measures encoder depth or emits fan-out
  gates.
- **Per-wire κ_w.** Section 6.3 sets `κ_w = 1` on output wires; κ is uniform here.

Known limits of the verification:

- CRE mode has no end-to-end test (780 qubits for H at κ=1, and ~19k T gates, so
  no stabilizer shortcut). Its chain is verified link by link instead — table
  row → description bits → gate words → unitary — but never run as one piece.
- Traced-out outputs are checked structurally only; the smallest circuit that
  exercises them is 41 qubits.
- Nothing tests that `garble.py`'s `desc[start + off + b]` indexing agrees with
  `desc_to_bits`'s layout. That is the one unverified link in the CRE chain.
- The SIMON cost charges a full key schedule per counter block. Expanding the
  schedule once into a `T·n` scratch register and reusing it across the ~20k
  blocks per key at κ=128 would cut the SIMON column materially; it is not
  modelled.

## Notes on the implementation

Two things in the code are easy to get wrong and worth knowing about.

**Global phase is not optional here.** Qiskit's `Clifford` carries Pauli signs
but no global phase, so a gate word reproduces its Clifford only up to a power
of `e^{iπ/4}`. That is harmless for a standalone unitary and fatal here,
because the evaluator applies the word *controlled on the description
register*, which is entangled with the `(d,e)` teleportation branches — a
per-branch global phase is a relative phase across them. Three phase bits per
output wire carry the lost exponent; the exponent is always an 8th root of
unity by a determinant argument, and it factorises over slots, so it costs
O(1) per slot rather than an operator over the whole gadget.

**The SIMON key register is the label register.** `build_simon_encrypt` leaves
it holding the last `m` round keys, and Dec reads those labels again afterwards.
So every counter block is encrypted, copied out, and un-encrypted: two SIMON
circuits per block, not one.

## Acknowledgments

The scheme is due to Zvika Brakerski and Henry Yuen, *Quantum Garbled Circuits*,
arXiv:2006.01085v2. SIMON is from Beaulieu, Shors, Smith, Treatman-Clark, Weeks
and Wingers, *The SIMON and SPECK Families of Lightweight Block Ciphers*, NSA,
2013.

Parts of this repository were written with [Claude Code](https://claude.com/claude-code).
