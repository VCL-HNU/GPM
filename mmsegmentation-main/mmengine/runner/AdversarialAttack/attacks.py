
from .fgsm import FGSM
from .l1_pgd import L1PGD
from .l2_pgd import L2PGD
from .linf_pgd import LinfPGD

methods = dict(
    FGSM=FGSM,
    L1PGD=L1PGD,
    L2PGD=L2PGD,
    LinfPGD=LinfPGD,
)

def get_method(type, *args, **kwargs):
    return methods[type](*args, **kwargs)
