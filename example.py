from pathlib import Path
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qgc import garble_circuit
import numpy as np
import random

random.seed(0)
np.random.seed(0)

# --------------------------------
# Input state: |+0>
# --------------------------------
input_prep = QuantumCircuit(2, name="input")
input_prep.h(0)
# Qubit 1 remains in |0>


# --------------------------------
# Program: CNOT
# --------------------------------
program = QuantumCircuit(2, name="program")
program.cx(0, 1)


# --------------------------------
# Convert the program to use EPR pairs
# --------------------------------
garbled_circuit = garble_circuit(
    program,
    input_prep=input_prep,
    kappa= 2
)


# --------------------------------
# Ordinary circuit for comparison
# --------------------------------
reference_circuit = input_prep.compose(program)


# --------------------------------
# Draw and save circuits
# --------------------------------
output_directory = Path("circuit_diagrams")
output_directory.mkdir(exist_ok=True)

circuits = {
    "Input Preparation": input_prep,
    "Original Program": program,
    "Original Circuit": reference_circuit,
    "EPR Converted Circuit": garbled_circuit.encoded_circuit,
}

for title, circuit in circuits.items():
    figure = circuit.draw(
        output="mpl",
        fold=-1,          # Keep the circuit on one continuous row
        idle_wires=True,
    )

    figure.suptitle(title)

    filename = title.lower().replace(" ", "_").replace("+", "plus")
    figure.savefig(
        output_directory / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(figure)