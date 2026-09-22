"""SIMON as the garbling PRG, classically and as a reversible circuit.

The label IS the seed -- Section 6.3 ties the teleportation labels to the CRE
input labels -- so the label is the SIMON key and key_size must equal kappa.
Among the variants with key_size = kappa we take the largest m, because a call
costs T*n Toffolis and yields 2n mask bits, so Toffoli per mask bit is T/2: the
word size cancels and only the round count matters. That is always m = 4,

    n = kappa/4,  m = 4,  i.e. Simon(kappa/2)/(kappa)

which exists for kappa in {64, 96, 128, 256}.

Masking uses the usual multi-key form

    mask = XOR_i SIMON_{label_i}(tag || counter)

over the 2p input labels, in counter mode. A row stays hidden as long as any
one of its labels is unknown.

COMPUTE-COPY-UNCOMPUTE. The quantum key register is the label register, which
Dec reads again afterwards (the colour controls, and the output decoding), and
build_simon_encrypt leaves it holding the last m round keys. So every block is
encrypted, copied into the description register, and un-encrypted. That is two
SIMON circuits per block, and it is not optional.
"""
import os
import sys

# The submodule's modules are flat and top level (params, classical_simon, ...),
# so it goes on the path rather than into this package.
_SUBMODULE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "qiskit-simon")
if _SUBMODULE not in sys.path:
    sys.path.insert(0, _SUBMODULE)

from classical_simon import simon_encrypt          # noqa: E402
from params import simon_params                    # noqa: E402
from quantum_simon import build_simon_encrypt      # noqa: E402

# key size -> word size n, at m = 4
USABLE_KAPPA = {64: 16, 96: 24, 128: 32, 256: 64}


def simon_variant(kappa, rounds=None):
    """SimonParams for this kappa at n = kappa/4, m = 4."""
    if kappa not in USABLE_KAPPA:
        raise ValueError(
            f"kappa must be one of {sorted(USABLE_KAPPA)} to key SIMON at "
            f"n=kappa/4, m=4; got {kappa}. Round reduction lowers T, not the "
            f"key size, so there is no small-kappa SIMON.")
    return simon_params(2 * USABLE_KAPPA[kappa], kappa, rounds)


def simon_block_qubits(kappa):
    """Width of the scratch block register add_simon_mask needs."""
    return 2 * USABLE_KAPPA[kappa]


def counter_block(tag, ctr, width):
    """A distinct block per (tag, counter)."""
    return ((tag << 32) | ctr) & ((1 << width) - 1)


def block_bit_qubit(block_qubits, n, i):
    """Qubit holding bit i of the 2n-bit block integer.

    quantum_simon lays the block out as register x then y, with x holding the
    left word, and the block is (L << n) | R. So the low n bits live in y,
    which is the second half of the register.
    """
    return block_qubits[n + i] if i < n else block_qubits[i - n]


def prg_simon(label_vals, kappa, tag, nbits, rounds=None):
    """mask = XOR_i SIMON_{label_i}(tag || counter), truncated to nbits."""
    params = simon_variant(kappa, rounds)
    width = params.block_size
    out = [0] * nbits
    for key in label_vals:
        produced, ctr = 0, 0
        while produced < nbits:
            cipher = simon_encrypt(params.block_size, params.key_size,
                                   counter_block(tag, ctr, width), key,
                                   rounds=params.rounds)
            for t in range(width):
                if produced >= nbits:
                    break
                out[produced] ^= (cipher >> t) & 1
                produced += 1
            ctr += 1
    return out


def add_simon_mask(qc, label_qubits, out_qubits, kappa, n_in, out_bits,
                   block_qubits, tag=0, rounds=None):
    """out ^= XOR_i SIMON_{label_i}(tag || counter), reversibly.

    `block_qubits` is 2n scratch, which must arrive |0> and is left |0>.
    `label_qubits[i]` is the kappa-qubit label register, used as the key and
    restored exactly.
    """
    params = simon_variant(kappa, rounds)
    n, width = params.word_size, params.block_size
    blk = list(block_qubits)
    if len(blk) != width:
        raise ValueError(f"block scratch must be {width} qubits, got {len(blk)}")

    enc = build_simon_encrypt(params.block_size, params.key_size,
                              key=None, rounds=params.rounds)
    dec = enc.inverse()

    for i in range(n_in):
        key_q = list(label_qubits[i])
        if len(key_q) != kappa:
            raise ValueError(f"label register must be {kappa} qubits")
        produced, ctr = 0, 0
        while produced < out_bits:
            loaded = counter_block(tag, ctr, width)
            flips = [block_bit_qubit(blk, n, t) for t in range(width)
                     if (loaded >> t) & 1]
            for q in flips:
                qc.x(q)
            qc.compose(enc, qubits=blk + key_q, inplace=True)
            for t in range(width):
                if produced >= out_bits:
                    break
                qc.cx(block_bit_qubit(blk, n, t), out_qubits[produced])
                produced += 1
            qc.compose(dec, qubits=blk + key_q, inplace=True)
            for q in flips:
                qc.x(q)
            ctr += 1
    return qc
