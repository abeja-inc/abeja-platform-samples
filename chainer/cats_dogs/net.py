import chainer
import chainer.functions as F
import chainer.links as L

class SimpleNet(chainer.Chain):
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(3, 32, 3, pad=1)
            self.b1 = L.BatchNormalization(32)
            self.c2 = L.Convolution2D(32, 32, 3, pad=1)
            self.b2 = L.BatchNormalization(32)
            self.c3 = L.Convolution2D(32, 64, 3, pad=1)
            self.b3 = L.BatchNormalization(64)
            self.c4 = L.Convolution2D(64, 128, 3, pad=1)
            self.b4 = L.BatchNormalization(128)
            self.c5 = L.Convolution2D(128, 256, 3, pad=1)
            self.b5 = L.BatchNormalization(256)
            self.l6 = L.Linear(16 * 256, 128)
            self.l7 = L.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.max_pooling_2d(x, ksize=2, stride=2)

        x = F.relu(self.b2(self.c2(x)))
        x = F.max_pooling_2d(x, ksize=2, stride=2)

        x = F.relu(self.b3(self.c3(x)))
        x = F.max_pooling_2d(x, ksize=2, stride=2)

        x = F.relu(self.b4(self.c4(x)))
        x = F.max_pooling_2d(x, ksize=2, stride=2)

        x = F.relu(self.b5(self.c5(x)))
        x = F.max_pooling_2d(x, ksize=2, stride=2)

        x = F.dropout(F.relu(self.l6(x)))

        return self.l7(x)
