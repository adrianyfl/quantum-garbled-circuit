"""Selectable optimization levels.

Each level adds exactly one thing to the one below it, so the resource study
can attribute a saving to a specific change rather than to "optimization".
Level 0 is the unoptimised baseline and stays available for that reason.

    from qgc import optimization
    optimization.set_level(0)          # baseline
    optimization.set_level(2)          # default

Levels 0-2 change the circuit that is emitted. Level 3 changes only the T-cost
convention. Level 4 is a projected saving that is NOT in the emitted circuit --
see `share_round_keys`. Level 5 needs an external tool and a measured factor.

None of these touch security: every level emits a fixed-length, canonical,
topology-only description, and the PRG is unchanged.
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
    3: Level("measured-toffoli", "minimal", 4, False, False,
             "Toffoli costed at 4 T, assuming measurement and feed-forward"),
    4: Level("shared-round-keys", "minimal", 4, True, False,
             "SIMON's key schedule expanded once per key instead of per counter "
             "block. PROJECTED: saves CNOTs only, and is not yet emitted"),
    5: Level("tzap", "minimal", 4, True, True,
             "external TZAP pass over the repeated units; needs a measured "
             "factor, which this machine cannot produce (no cargo)"),
    6: Level("unary-iteration", "minimal", 4, True, False,
             "one-hot decode per word position instead of a two-MCX test per "
             "symbol. Applied to 2-qubit slots only: measured 1.35x there, but "
             "0.71x -- a LOSS -- at arity 1, where 2 symbols cannot amortise "
             "the decode", unary=frozenset({2})),
    7: Level("and-gadgets", "minimal", 4, True, False,
             "SIMON's Toffolis as AND gadgets: 4 T to compute, 0 T to uncompute. "
             "PROJECTED -- needs mid-circuit measurement and feed-forward, so "
             "the emitted circuit is unchanged and only the model moves",
             unary=frozenset({2}), and_gadget=True),
}

DEFAULT = 2
_current = None


def set_level(level: int, tzap_factor: float | None = None):
    """Apply an optimization level globally.

    `tzap_factor` is the measured T-count ratio from an actual TZAP run. It has
    no default on purpose: level 5 must not report a guessed number.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown level {level}, expected {sorted(LEVELS)}")
    spec = LEVELS[level]

    gate_words.configure(spec.alphabet)
    gate_words.UNARY_ARITIES = spec.unary
    randomization_group.desc_bits_len.cache_clear()

    cost.T_TOFFOLI = spec.toffoli_t
    cost.SHARE_ROUND_KEYS = spec.share_round_keys
    cost.AND_GADGET = spec.and_gadget

    if spec.tzap:
        if tzap_factor is None:
            raise ValueError(
                "level 5 needs tzap_factor=<measured T ratio> from a real TZAP "
                "run. Export the repeated units with tools/export_units.py, run "
                "tzap on them, and pass the measured ratio here.")
        cost.TZAP_FACTOR = tzap_factor
    else:
        cost.TZAP_FACTOR = 1.0

    global _current
    _current = level
    return spec


def current():
    return _current, LEVELS[_current]


set_level(DEFAULT)
