"""Labels, point-and-permute colours, and gadget sizing."""
import random

# Point-and-permute colour position. Every wire's two labels are sampled to
# differ here, so the evaluator always reads the colour at the SAME index. That
# is what makes CDec depend on the topology alone (Section 6.4: "CDec only
# depends on the topology T_g of the correction function"), rather than on
# where a particular pair of labels happened to differ.
POINTER_BIT = 0


def get_bit(number, index):
    return (number >> index) & 1


def set_bit(number, index, value):
    return (number & ~(1 << index)) | ((value & 1) << index)


def sample_label(kappa: int) -> tuple[int, ...]:
    """Two labels whose colour bits differ.

    The colour of the label for value v is v XOR lam, where lam is a uniformly
    random bit. Since lam is unknown to the evaluator, reading a colour reveals
    nothing about v -- but it still selects a unique table row, which is the
    whole point of point-and-permute.

    At kappa = 1 the label IS its colour, so there is no hiding at all. That is
    fine for the test fixtures and useless for anything else.
    """
    lam = random.getrandbits(1)
    l0 = set_bit(random.getrandbits(kappa), POINTER_BIT, lam)
    l1 = set_bit(random.getrandbits(kappa), POINTER_BIT, lam ^ 1)
    return (l0, l1)


def colour(label: int) -> int:
    """The point-and-permute colour of a label."""
    return get_bit(label, POINTER_BIT)


def gadget_num_qubits(kappa: int) -> int:
    """u, v, z, x and the (kappa+1)^2 register b -- one wire segment."""
    return 3 + 4 * kappa + kappa * kappa
