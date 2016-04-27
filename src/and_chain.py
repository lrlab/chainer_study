from chainer import Chain
import chainer.functions as F
import chainer.links as L

class AndChain(Chain):
    def __init__(self):
        super(AndChain, self).__init__(
            l1 = L.Linear(2, 1)
        )

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        return h
