import argparse
import datetime
import os
import sys
from pathlib import Path

import mxnet
from mxnet import gluon
from mxnet import init
from mxnet.gluon.data.vision.transforms import Compose, ToTensor, Normalize

import utils
from datahelper import MultiViewImageDataset
from model import MVRNN


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('ViewSequenceNet')

    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--batch_update_period', type=int, default=32,
                        help='do back propagation after every 64 batches')
    parser.add_argument('--epoch', default=100, type=int, help='the number of epochs')
    parser.add_argument('--lr', default=1e-5, type=float, help='the initial learning rate')
    parser.add_argument('--gpu', type=int, nargs='+', default=(0,), help='gpu indices, support multi gpus')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer for training')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_period', type=int, default=80, help='learning rate decay period')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--output_lr_mult', type=float, default=1.0, help='lr multiplier for output layer')
    parser.add_argument('--dataset_path', type=str, default='/media/zenn/files/dataset/modelnet10-multiview',
                        help='path to the dataset')
    parser.add_argument('--checkpoint', type=str, default=None, help='location of the checkpoint')
    parser.add_argument('--pretrained_cnn', type=str, default=None, help='location of the 2d pretrained_cnn checkpoint')

    parser.add_argument('--num_views', type=int, default=12, help='number of views')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--label_smoothing', default=False, action='store_true',
                        help='whether to use the label smoothing')
    parser.add_argument('--fix_cnn', default=False, action='store_true', help='whether to fix the cnn')
    parser.add_argument('--from_epoch', default=0, type=int, help='start from epoch(for training from a checkpoint)')
    parser.add_argument('--use_sample_weights', default=False, action='store_true',
                        help='whether to use sample_weights for cross entropy loss')
    return parser.parse_args()


def main(args):
    '''create dir'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = Path('./experiment/checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = Path('./experiment/logs/')
    log_dir.mkdir(exist_ok=True)

    ctx = [mxnet.gpu(gpu_id) for gpu_id in args.gpu]

    '''initialize the network'''
    net = MVRNN(cnn_arch='vgg11_bn', cnn_feature_length=4096, num_views=args.num_views, num_class=args.num_classes,
                pretrained=True, pretrained_cnn=args.pretrained_cnn, ctx=ctx)
    if args.checkpoint:
        net.load_parameters(args.checkpoint, ctx=ctx)
    else:
        net.initialize(init=init.MSRAPrelu(), ctx=ctx)
    net.hybridize()
    '''set grad_req to 'add' to manually aggregate gradients'''
    net.collect_params().setattr('grad_req', 'add')
    net._cnn2.collect_params().setattr('lr_mult', args.output_lr_mult)

    '''Setup loss function'''
    loss_fun = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=not args.label_smoothing)

    '''Loading dataset'''
    train_ds = MultiViewImageDataset(os.path.join(args.dataset_path, 'train'), args.num_views,
                                     transform=Compose([
                                         ToTensor(),
                                         Normalize(mean=(0.485, 0.456, 0.406),
                                                   std=(0.229, 0.224, 0.225))]))
    test_ds = MultiViewImageDataset(os.path.join(args.dataset_path, 'test'), args.num_views,
                                    transform=Compose([
                                        ToTensor(),
                                        Normalize(mean=(0.485, 0.456, 0.406),
                                                  std=(0.229, 0.224, 0.225))]))
    loader = gluon.data.DataLoader
    train_data = loader(train_ds, args.batch_size, shuffle=True, last_batch='keep', num_workers=4)
    test_data = loader(test_ds, args.batch_size, shuffle=False, last_batch='keep', num_workers=4)

    current_time = datetime.datetime.now()
    time_str = '%d-%d-%d--%d-%d-%d' % (
        current_time.year, current_time.month, current_time.day, current_time.hour, current_time.minute,
        current_time.second)
    log_filename = time_str + '.txt'
    checkpoint_name = 'checkpoint_' + time_str
    checkpoint_dir = Path(os.path.join(checkpoints_dir, checkpoint_name))
    checkpoint_dir.mkdir(exist_ok=True)

    with open(os.path.join(log_dir, log_filename, ), 'w') as log_out:
        try:
            kv = mxnet.kv.create('device')
            utils.log_string(log_out, sys.argv[0])
            utils.train(net, train_data, test_data, loss_fun, kv, log_out, str(checkpoint_dir), args)
        except Exception as e:
            raise e


if __name__ == "__main__":
    args = parse_args()
    print(str(args))
    main(args)
