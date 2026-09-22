"""Quantum Garbled Circuits (Brakerski and Yuen, arXiv:2006.01085v2).

A resource study of the QGC scheme over the PRG (computationally private)
route, with SIMON as the pseudorandom generator.

    from qgc import garble_circuit, decode_output

    g = garble_circuit(circuit, kappa=1, mode="cre")
    full = decode_output(g)          # Enc then Dec, for simulation

See README.md for the module map and what is and is not verified.
"""
from qgc.decoder import decode_output
from qgc.garble import garble_circuit

__all__ = ["garble_circuit", "decode_output"]
