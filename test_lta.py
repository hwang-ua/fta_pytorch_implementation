import torch

from core.lta import *
from core.network import *



def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    test_ary = np.array([[0.1, 0.5, -0.1, -0.5]])

    # use LTA
    random_seed(0)
    lta = LTA(tiles=10, bound_low=-1, bound_high=1, eta=0.2, input_dim=2)
    net = FCNetwork("cpu", 4, [16], 2, head_activation=lta)
    print("Use LTA:", net(test_ary))

    # without LTA
    random_seed(0)
    net = FCNetwork("cpu", 4, [16], 2)
    print("Without LTA:", net(test_ary))