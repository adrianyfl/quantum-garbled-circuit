from structure import *
from helper import get_bit
from qiskit import AncillaRegister
from qiskit.quantum_info import random_clifford
import random

# construct C1
def construct_c1(l_z_xor: int, kappa: int) -> QuantumCircuit:
    circuit = QuantumCircuit(1)
    z = QuantumRegister(kappa, name='z')
    circuit.add_register(z)

    # register that uses (kappa + 1)^2 qubits
    b = []
    for i in range(kappa + 1):
        b.append(AncillaRegister(kappa + 1, name="b{}".format(i)))
        circuit.add_register(b[i])

    # apply h to all z qubits + cx for labels
    for i in range(0, kappa):
        circuit.h(z[i])
        if get_bit(l_z_xor, i):
            circuit.cx(z[i], 0)

    # descending fanout
    circuit.cx(0, b[0])
    for i in range(kappa):
        circuit.cx(z[i], b[i + 1])

    return circuit

# construct Lambda1
def construct_lambda1(l_z: tuple[int,...], l_x: tuple[int,...], kappa: int) -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    z = QuantumRegister(kappa, name="z")
    circuit.add_register(z)
    b = []
    for i in range(kappa + 1):
        b.append(AncillaRegister(kappa + 1, name="b{}".format(i)))
        circuit.add_register(b[i])
    x = QuantumRegister(kappa, name="x")
    circuit.add_register(x)

    circuit.cx(0, 1)

    # apply l_z0 and l_x0 to z and x
    for i in range(0, kappa):
        if get_bit(l_z[0], i):
            circuit.x(z[i])
        if get_bit(l_x[0], i):
            circuit.x(x[i])

    # add C1 to the circuit
    c1 = construct_c1(l_z[0] ^ l_z[1], kappa)
    mapping = [0, *z]
    for i in range(0, kappa+1):
        mapping.extend(b[i])
    circuit.compose(c1, qubits=mapping, inplace=True)

    # apply l_x1 ^ l_x0
    l_x_xor = l_x[0] ^ l_x[1]
    for i in range(0, kappa):
        if get_bit(l_x_xor, i):
            circuit.cx(1, x[i])

    return circuit

# sample A from the clifford group
def sample_a(kappa: int) -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    z = QuantumRegister(kappa, name='z')
    circuit.add_register(z)
    b = []
    for i in range(kappa + 1):
        b.append(AncillaRegister(kappa + 1, name="b{}".format(i)))
        circuit.add_register(b[i])
    x = QuantumRegister(kappa, name='x')
    circuit.add_register(x)

    # single qubit
    cliff_circ0 = random_clifford(1).to_circuit()
    circuit.compose(cliff_circ0, qubits=0 ,inplace=True)
    cliff_circ3 = random_clifford(1).to_circuit()
    circuit.compose(cliff_circ3, qubits=1, inplace=True)
    for i in range(0, kappa):
        cliff_circ1 = random_clifford(1).to_circuit()
        circuit.compose(cliff_circ1, qubits=[x[i]], inplace=True)
        cliff_circ2 = random_clifford(1).to_circuit()
        circuit.compose(cliff_circ2, qubits=[z[i]], inplace=True)

    # b registers
    for j in range(kappa + 1):
        cliff_circ_diag = random_clifford(1).to_circuit()
        circuit.compose(cliff_circ_diag, qubits=[b[j][j]], inplace=True)
        for i in range(j):
            cliff_circ4 = random_clifford(2).to_circuit()
            circuit.compose(cliff_circ4, qubits=[b[i][j], b[j][i]], inplace=True)

    return circuit