from qiskit import QuantumCircuit
from wire_epr_conversion import convert_wire_to_epr

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

# Pass the input separately from the program
converted_circuit = convert_wire_to_epr(
    program,
    input_prep=input_prep,
)

# Ordinary circuit for comparison
reference_circuit = input_prep.compose(program)

print("Input Preparation:")
print(input_prep)

print("Original Program:")
print(program)

print("Ordinary Input + Program:")
print(reference_circuit)

print("EPR Converted Circuit:")
print(converted_circuit)