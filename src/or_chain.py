from chainer import Chain
import chainer.functions as F
import chainer.links as L

class OrChain(Chain):
    def __init__(self):
        super(OrChain, self).__init__(
            l1 = L.Linear(2, 1)
        )

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        return h
