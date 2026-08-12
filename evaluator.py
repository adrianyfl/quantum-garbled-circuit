from structure import *
from helper import get_bit
from qiskit import AncillaRegister

def construct_c3(kappa: int) -> QuantumCircuit:
    circuit = QuantumCircuit(1)
    z = QuantumRegister(kappa, name="z")
    circuit.add_register(z)

    # b register that uses (kappa + 1) ** 2 qubits
    b = []
    for i in range(kappa + 1):
        b.append(AncillaRegister(kappa + 1, name='b{}'.format(i)))
        circuit.add_register(b[i])

    #ascending fanout + h gate
    circuit.cx(0, b[0])
    circuit.h(0)
    for i in range(kappa):
        circuit.cx(z[i], b[i + 1])
        circuit.h(z[i])

    return circuit

def construct_lambda3(kappa: int) -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    z = QuantumRegister(kappa, name="z")
    circuit.add_register(z)
    b = []
    for i in range(kappa + 1):
        b.append(AncillaRegister(kappa + 1, name='b{}'.format(i)))
        circuit.add_register(b[i])
    x = QuantumRegister(kappa, name="x")
    circuit.add_register(x)

    # apply c3
    mapping = [0, *z]
    for i in range(kappa + 1):
        mapping.extend(b[i])
    c3_circ = construct_c3(kappa)
    circuit.compose(c3_circ, qubits=mapping, inplace=True)
    return circuit

def construct_gamma(p: int, r: int, kappa:int) -> QuantumCircuit:
    circuit = QuantumCircuit()

    # b registers
    b = []
    for i in range(kappa + 1):
        b.append(AncillaRegister(kappa + 1, name='b{}'.format(i)))
        circuit.add_register(b[i])

    # check p
    if p == 0:
        return circuit

    # apply all czs
    for j in range(1,kappa + 1):
        if get_bit(r, j-1):
            circuit.cz(b[0][j], b[j][0])
        for i in range(1, j):
            if get_bit(r, j-1) and get_bit(r, i-1):
                circuit.cz(b[i][j], b[j][i])

    return circuit

def construct_c2(corr: Correction, r: int, s_x: bool, s_z: bool, kappa: int) -> QuantumCircuit:
    circuit = QuantumCircuit(1)
    z = QuantumRegister(kappa, name="z")
    circuit.add_register(z)
    b = []
    for i in range(kappa + 1):
        b.append(AncillaRegister(kappa + 1, name="b{}".format(i)))
        circuit.add_register(b[i])

    # P correction
    if corr.p == 1:
        circuit.s(0)
        for i in range(kappa):
            if get_bit(r, i):
                circuit.s(z[i])
    gamma_circuit = construct_gamma(corr.p, r, kappa)
    mapping = []
    for i in range(0, kappa+1):
        mapping.extend(b[i])
    circuit.compose(gamma_circuit, qubits=mapping, inplace=True)

    # X + Z correction + qotp
    if corr.z == 1:
        circuit.z(0)
        for i in range(kappa):
            if get_bit(r, i):
                circuit.z(z[i])
    if corr.x ^ s_z:
        circuit.x(b[0])
    if corr.x:
        circuit.x(0)
    if s_x:
        circuit.z(0)
    if s_z:
        circuit.x(0)

    return circuit

def construct_lambda2(
        corr: Correction,
        l_z: tuple[int, ...],
        l_x: tuple[int, ...],
        s_x: bool,
        s_z: bool,
        t_x: bool,
        t_z: bool,
        kappa: int
) -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    z = QuantumRegister(kappa, name="z")
    circuit.add_register(z)
    b = []
    for i in range(kappa + 1):
        b.append(AncillaRegister(kappa + 1, name='b{}'.format(i)))
        circuit.add_register(b[i])
    x = QuantumRegister(kappa, name="x")
    circuit.add_register(x)

    # construct C2
    c2_circuit = construct_c2(corr, l_z[0] ^ l_z[1], s_x, s_z, kappa)
    mapping = [0, *z]
    for i in range(0, kappa+1):
        mapping.extend(b[i])
    circuit.compose(c2_circuit, qubits=mapping, inplace=True)

    # apply R3 and R4
    l_x_xor = l_x[0] ^ l_x[1]
    if corr.x:
        circuit.x(1)
        for i in range(kappa):
            if get_bit(l_x_xor, i):
                circuit.x(x[i])

    # apply qotp
    if t_x:
        circuit.x(1)
    if t_z:
        circuit.z(1)

    return circuit