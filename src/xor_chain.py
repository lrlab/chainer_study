from chainer import Chain
import chainer.functions as F
import chainer.links as L

class XorChain(Chain):
    def __init__(self):
        n_unit = 10
        super(XorChain, self).__init__(
            l1 = L.Linear(2, n_unit),
            l2 = L.Linear(n_unit, 1),
        )

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        o = self.l2(h)
        return o
