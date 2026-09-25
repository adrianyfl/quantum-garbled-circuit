"""Selectable optimization levels.

Each level adds exactly one thing to the one below it, so the resource study
can attribute a saving to a specific change rather than to "optimization".
Level 0 is the unoptimised baseline and stays available for that reason.

    from qgc import optimization
    optimization.set_level(0)          # baseline
    optimization.set_level(2)          # default

Levels 0-5 are unitary circuits; 6 and 7 assume mid-circuit measurement and
feed-forward, so they come last. tzap (level 5) and the measured Toffoli
(level 6) both cheapen SIMON's Toffolis and do not stack: tzap phase-folds the
7-T decomposition down to 5 T, the measured form is 4 T. SIMON takes whichever
is cheaper (cost.simon_tzap), so from level 6 on it is emitted as plain
X/CX/CCX again, for the measured Toffoli to replace.

Levels 1-5 change the circuit that is emitted. Level 6 changes the T-cost
convention. Level 7 is a projected saving that is NOT in the emitted circuit --
see `and_gadget`.

None of these touch security: every level emits a fixed-length, canonical,
topology-only description, and the PRG computes the same function.
"""
from dataclasses import dataclass

from qgc import cost, gate_words, randomization_group


@dataclass(frozen=True)
class Level:
    name: str
    alphabet: str
    toffoli_t: int
    share_round_keys: bool
    tzap: bool
    note: str
    unary: frozenset = frozenset()
    and_gadget: bool = False


LEVELS = {
    0: Level("none", "original", 7, False, False,
             "unoptimised baseline: 13-symbol alphabet padded to 19 positions"),
    1: Level("tight-words", "redundant", 7, False, False,
             "word length cut to the group's diameter; a pad position still "
             "costs a full symbol sweep, so the padding was pure waste"),
    2: Level("minimal-alphabet", "minimal", 7, False, False,
             "minimal generating set: fewer symbols to sweep and a narrower "
             "symbol register, at the price of longer words"),
    3: Level("shared-round-keys", "minimal", 7, True, False,
             "SIMON encrypts counter blocks in batches of cost.SHARE_BATCH under "
             "one key register, so the key schedule runs once per batch instead "
             "of once per block. Linear, so it saves CX and X only -- no T -- and "
             "costs 2n scratch qubits per block in a batch"),
    4: Level("unary-iteration", "minimal", 7, True, False,
             "one-hot decode per word position instead of a two-MCX test per "
             "symbol. Applied to 2-qubit slots only: measured 1.35x there, but "
             "0.71x -- a LOSS -- at arity 1, where 2 symbols cannot amortise "
             "the decode", unary=frozenset({2})),
    5: Level("tzap", "minimal", 7, True, True,
             "SIMON emitted as tzap-optimised Clifford+T: phase folding merges "
             "the two T phases on each x qubit, 7 -> 5 T per Toffoli, proven "
             "equivalent in qiskit-simon. Dec is not passed through tzap",
             unary=frozenset({2})),
    6: Level("measured-toffoli", "minimal", 4, True, True,
             "Toffoli costed at 4 T, assuming measurement and feed-forward; "
             "SIMON takes this over tzap's 5 T",
             unary=frozenset({2})),
    7: Level("and-gadgets", "minimal", 4, True, True,
             "SIMON's Toffolis as AND gadgets: 4 T to compute, 0 T to uncompute. "
             "PROJECTED -- needs mid-circuit measurement and feed-forward, so "
             "the emitted circuit is unchanged and only the model moves",
             unary=frozenset({2}), and_gadget=True),
}

DEFAULT = 2
_current = None


def set_level(level: int):
    """Apply an optimization level globally."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level}, expected {sorted(LEVELS)}")
    spec = LEVELS[level]

    gate_words.configure(spec.alphabet)
    gate_words.UNARY_ARITIES = spec.unary
    randomization_group.desc_bits_len.cache_clear()

    cost.T_TOFFOLI = spec.toffoli_t
    cost.SHARE_ROUND_KEYS = spec.share_round_keys
    cost.AND_GADGET = spec.and_gadget
    cost.TZAP = spec.tzap

    global _current
    _current = level
    return spec


def current():
    return _current, LEVELS[_current]


set_level(DEFAULT)
