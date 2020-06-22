import mxnet
from mxnet import initializer, init
from mxnet.gluon import nn, rnn
from mxnet.gluon.model_zoo import vision


def get_model(args):
    '''Setup network'''
    ctx = [mxnet.gpu(gpu_id) for gpu_id in args.gpu]

    net = MVRNN(cnn_arch='vgg11_bn', cnn_feature_length=4096, num_views=args.num_views, num_class=args.num_classes,
                pretrained=True, pretrained_cnn=args.pretrained_cnn, ctx=ctx)
    if args.checkpoint:
        net.load_parameters(args.checkpoint, ctx=ctx)
    else:
        net.initialize(init=init.MSRAPrelu(), ctx=ctx)
    net.hybridize()
    net.collect_params().setattr('grad_req', 'add')
    net._cnn2.collect_params().setattr('lr_mult', args.output_lr_mult)

    return net


class MVRNN(nn.HybridBlock):
    def __init__(self, cnn_arch, cnn_feature_length, num_views, num_class, rnn_type='lstm', pretrained=True,
                 pretrained_cnn=None,
                 ctx=None, **kwargs):
        super(MVRNN, self).__init__(**kwargs)
        self.arch_name = cnn_arch
        self.pretrained = pretrained
        self.pretrained_cnn = pretrained_cnn
        self.num_views = num_views
        self.use_rnn = rnn_type
        self.cnn_feature_length = cnn_feature_length
        self.rnn_hidden_size = cnn_feature_length // 2
        self.rnn_layers = 1
        self.rnn_dropout = 0
        self.rnn_bidirectional = True

        with self.name_scope():
            if pretrained_cnn:
                print('using pretrained cnn1')
                cnn = vision.get_model(cnn_arch, classes=num_class, ctx=ctx)
                cnn.load_parameters(pretrained_cnn, ctx=ctx)
                self._cnn2 = cnn.output
            else:
                cnn = vision.get_model(cnn_arch, pretrained=pretrained, ctx=ctx)
                self._cnn2 = vision.get_model(cnn_arch, classes=num_class).output
                # self._cnn2 = nn.Dense(num_class)
            self._cnn1 = cnn.features

            assert rnn_type in ['lstm', 'gru']
            if rnn_type == 'lstm':
                self._rnn = rnn.LSTM(hidden_size=self.rnn_hidden_size, num_layers=self.rnn_layers,
                                     dropout=self.rnn_dropout, bidirectional=self.rnn_bidirectional, layout="NTC")
            if rnn_type == 'gru':
                self._rnn = rnn.GRU(hidden_size=self.rnn_hidden_size, num_layers=self.rnn_layers,
                                    dropout=self.rnn_dropout, bidirectional=self.rnn_bidirectional, layout="NTC")

    def get_features(self, x):
        x = x.reshape((-1, 3, 224, 224))  # [batch_size * num_views, 3, 224, 224]
        x = self._cnn1(x)  # [batch_size * num_views, 4096]
        x = x.reshape((-1, self.num_views, self.cnn_feature_length))  # [batch_size, num_views, 4096]
        x = self._rnn(x)  # [batch_size, num_views, 4096]
        return x

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = x.reshape((-1, 3, 224, 224))  # [batch_size * num_views, 3, 224, 224]
        x = self._cnn1(x)  # [batch_size * num_views, 4096]
        x = x.reshape((-1, self.num_views, self.cnn_feature_length))  # [batch_size, num_views, 4096]
        x = self._rnn(x)  # [batch_size, num_views, 4096]
        return self._cnn2(x.reshape((-1, self.rnn_hidden_size * 2)))  # [batch_size * num_views, num_class]

    def initialize(self, init=initializer.Uniform(), ctx=None, verbose=False, force_reinit=False):
        if self.pretrained:
            if not self.pretrained_cnn:
                self._cnn2.initialize(init, ctx)
            self._rnn.initialize(init, ctx)
        else:
            super().initialize(init, ctx, verbose, force_reinit)

