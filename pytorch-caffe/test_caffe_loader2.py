# 2017.12.16 by xiaohang
import sys
from caffenet import *
import numpy as np
import argparse
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time

def create_network(protofile, weightfile):
    net = CaffeNet(protofile)
    if args.cuda:
        net.cuda()
    print(net)
    net.load_weights(weightfile)
    net.train()
    return net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert caffe to pytorch')
    parser.add_argument('--protofile', default='', type=str)
    parser.add_argument('--weightfile', default='', type=str)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    args = parser.parse_args()
    print(args)
    
    protofile = args.protofile
    weightfile = args.weightfile

    net = create_network(protofile, weightfile)
    net.set_verbose(False)

    for i in range(10):
        blobs = net()
        blob_names = blobs.keys()
        for blob_name in blob_names:
            if args.cuda:
                blob_data = blobs[blob_name].data.cpu().numpy()
            else:
                blob_data = blobs[blob_name].data.numpy()
            print('[%d] %-30s pytorch_shape: %-20s mean: %f' % (i, blob_name, blob_data.shape, blob_data.mean()))
