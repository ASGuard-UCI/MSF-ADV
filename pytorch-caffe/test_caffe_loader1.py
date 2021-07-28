# 2017.12.16 by xiaohang
import sys
from caffenet import *
import numpy as np
import argparse
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time

class CaffeDataLoader:
    def __init__(self, protofile):
        caffe.set_mode_cpu()
        self.net = caffe.Net(protofile, 'aaa', caffe.TRAIN)

    def next(self):
        output = self.net.forward()
        data = self.net.blobs['data'].data
        label = self.net.blobs['label'].data
        return data, label

def create_network(protofile, weightfile):
    net = CaffeNet(protofile)
    if args.cuda:
        net.cuda()
    print(net)
    net.load_weights(weightfile)
    net.train()
    return net

def forward_network(net, data, label):
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    if args.cuda:
        data = Variable(data.cuda())
        label = Variable(label.cuda())
    else:
        data = Variable(data)
        label = Variable(label)
    blobs = net(data, label)
    return blobs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert caffe to pytorch')
    parser.add_argument('--data_protofile', default='', type=str)
    parser.add_argument('--net_protofile', default='', type=str)
    parser.add_argument('--weightfile', default='', type=str)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    args = parser.parse_args()
    print(args)
    
    data_protofile = args.data_protofile
    net_protofile = args.net_protofile
    weightfile = args.weightfile

    data_loader = CaffeDataLoader(data_protofile)
    net = create_network(net_protofile, weightfile)
    net.set_verbose(False)

    for i in range(10):
        data, label = data_loader.next()
        print('data shape', data.shape)
        blobs = forward_network(net, data, label)
        blob_names = blobs.keys()
        for blob_name in blob_names:
            if args.cuda:
                blob_data = blobs[blob_name].data.cpu().numpy()
            else:
                blob_data = blobs[blob_name].data.numpy()
            print('[%d] %-30s pytorch_shape: %-20s mean: %f' % (i, blob_name, blob_data.shape, blob_data.mean()))
