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