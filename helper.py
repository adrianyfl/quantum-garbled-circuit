import random

# get bit from int
def get_bit(number, index):
    return (number >> index) & 1

# sample label
def sample_label(kappa: int) -> tuple[int, ...]:
    l = [random.getrandbits(kappa), random.getrandbits(kappa)]
    while l[0] == l[1]:
        l[1] = random.getrandbits(kappa)
    return tuple(l)

# find where two labels differ
def find_witness_bit(l0: int, l1: int, kappa: int) -> int:
    for i in range(kappa):
        if get_bit(l0, i) != get_bit(l1, i):
            return i
    raise ValueError("labels are identical -- sample_label should have prevented this")

# calculate size of the gadget based on kappa
def gadget_num_qubits(kappa: int) -> int:
    return 3 + 4 * kappa + kappa * kappa