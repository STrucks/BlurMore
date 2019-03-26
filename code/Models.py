from chainer import ChainList
from chainer import Chain
import chainer.links as L
import chainer.functions as F

import numpy as np


class MLP(Chain):
    """Multilayer perceptron"""

    def __init__(self, n_output=2, n_hidden=5):
        #super(MLP, self).__init__(L.Linear(None, n_hidden),L.Linear(n_hidden, n_hidden),L.Linear(n_hidden, n_hidden),L.Linear(n_hidden, n_output))
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_hidden)
            self.l2 = L.Linear(n_hidden, n_hidden)
            self.l3 = L.Linear(n_hidden, n_output)

    def __call__(self, x):
        """
        for layer in self.children():
            x = F.relu(layer(x))
        activation = F.softmax(x).data
        return activation.astype(np.float32)
        """
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return F.softmax(self.l3(h))

