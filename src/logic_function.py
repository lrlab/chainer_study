#coding: utf-8
"""Logic Function

Usage: logic_function.py --type <typ> --epoch <n_epoch>
       logic_function.py -h | --help
Options:
    --type      logic function type (and, or, xor)
    -h, --help  show this help message and exit

"""

from docopt import docopt
import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
import numpy as np
from and_chain import AndChain
from or_chain import OrChain
from xor_chain import XorChain

def main():
    args = docopt(__doc__)
    typ = args["<typ>"]
    n_epoch = int(args["<n_epoch>"])


    if typ == 'and':
        model = AndChain()
        source = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
        target = [[0.0], [0.0], [0.0], [1.0]]

    elif typ == 'or':
        model = OrChain()
        source = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
        target = [[0.0], [1.0], [1.0], [1.0]]

    elif typ == 'xor':
        model = XorChain()
        source = [[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
        target = [[0.0], [1.0], [1.0], [0.0]]
    else:
        raise Exception("Argument Error")

    optimizer = optimizers.Adam()
    optimizer.setup(model)
    N = len(source)

    dataset = {}
    dataset['source'] = np.array(source).astype(np.float32)
    dataset['target'] = np.array(target).astype(np.float32)


    for epoch in xrange(1, n_epoch+1):
        perm = np.random.permutation(N)

        x = chainer.Variable(dataset['source'][perm])
        t = chainer.Variable(dataset['target'][perm])

        model.zerograds()
        y = model(x)

        loss = F.mean_squared_error(y, t)
        loss.backward()
        optimizer.update()


        if epoch % 1000 == 0:
            #誤差と正解率を計算
            loss_val = loss.data

            print 'epoch:', epoch
            print 'x:\n', x.data
            print 't:\n', t.data
            print 'y:\n', y.data
            print 'model.l1.W:\n', model.l1.W.data
            print 'model.l1.b:\n', model.l1.b.data

            print('train mean loss={}'.format(loss_val)) # 訓練誤差, 正解率
            print ' - - - - - - - - - '

            if loss_val < 1e-7:
                break

if __name__ == '__main__':
    main()
