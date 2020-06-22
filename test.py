import os

import argparse

import importlib

import mxnet
from mxnet import gluon
from mxnet.gluon.data.vision.transforms import Compose, ToTensor, Normalize

import utils
from datahelper import MultiViewImageDataset


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('ViewSequenceNet')

    parser.add_argument('--model', type=str, default='model', help='name of the model file')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--batch_update_period', type=int, default=64,
                        help='do back propagation after every 64 batches')
    parser.add_argument('--gpu', type=int, nargs='+', default=(0,), help='')
    parser.add_argument('--dataset_path', type=str, default='/media/zenn/files/dataset/modelnet10-multiview',
                        help='path to the dataset')
    parser.add_argument('--checkpoint', type=str, default=None, help='location of the checkpoint')
    parser.add_argument('--num_views', type=int, default=12, help='number of views')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = importlib.import_module(args.model)
    net = model.get_model(args)
    metric = mxnet.metric.Accuracy()
    test_ds = MultiViewImageDataset(os.path.join(args.dataset_path, 'test'), args.num_views,
                                    transform=Compose([
                                        ToTensor(),
                                        Normalize(mean=(0.485, 0.456, 0.406),
                                                  std=(0.229, 0.224, 0.225))]))
    loader = gluon.data.DataLoader
    test_data = loader(test_ds, args.batch_size, shuffle=False, last_batch='keep')
    ctx = [mxnet.gpu(gpu_id) for gpu_id in args.gpu]
    print(
        'test on dataset %s, acc %s ' % (
            args.dataset_path, utils.test(metric, ctx, net, test_data, num_views=args.num_views,
                                          num_class=args.num_classes)))
