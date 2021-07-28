# 2017.12.16 by xiaohang
import sys
from caffenet import *
import numpy as np
import argparse
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time

def load_image(imgfile):
    image = caffe.io.load_image(imgfile)
    transformer = caffe.io.Transformer({'data': (1, 3, args.height, args.width)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([args.meanB, args.meanG, args.meanR]))
    transformer.set_raw_scale('data', args.scale)
    transformer.set_channel_swap('data', (2, 1, 0))

    image = transformer.preprocess('data', image)
    image = image.reshape(1, 3, args.height, args.width)
    return image

def load_synset_words(synset_file):
    lines = open(synset_file).readlines()
    synset_dict = dict()
    for i, line in enumerate(lines):
        synset_dict[i] = line.strip()
    return synset_dict

def forward_pytorch(protofile, weightfile, image, num_times):
    #torch.backends.cudnn.enabled = True
    net = CaffeNet(protofile, width=args.width, height=args.height)
    if args.cuda:
        net.cuda()
    print(net)
    net.load_weights(weightfile)
    net.set_verbose(False)
    net.eval()
    image = torch.from_numpy(image)
    if args.cuda:
        image = Variable(image.cuda())
    else:
        image = Variable(image)
    for i in range(num_times):
        t0 = time.time()
        blobs = net(image)
        t1 = time.time()
        print('pytorch forward%d: %f' % (i, t1-t0))
    return t1-t0, blobs, net.models

# Reference from:
def forward_caffe(protofile, weightfile, image, num_times):
    if args.cuda:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    net.blobs['data'].reshape(1, 3, args.height, args.width)
    net.blobs['data'].data[...] = image
    for i in range(num_times):
        t0 = time.time()
        output = net.forward()
        t1 = time.time()
        print('caffe forward%d: %f' % (i, t1-t0))
    return t1 - t0, net.blobs, net.params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert caffe to pytorch')
    parser.add_argument('--protofile', default='', type=str)
    parser.add_argument('--weightfile', default='', type=str)
    parser.add_argument('--imgfile', default='', type=str)
    parser.add_argument('--height', default=224, type=int)
    parser.add_argument('--width', default=224, type=int)
    parser.add_argument('--meanB', default=104, type=float)
    parser.add_argument('--meanG', default=117, type=float)
    parser.add_argument('--meanR', default=123, type=float)
    parser.add_argument('--scale', default=255, type=float)
    parser.add_argument('--synset_words', default='', type=str)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    args = parser.parse_args()
    print(args)
    
    
    protofile = args.protofile
    weightfile = args.weightfile
    imgfile = args.imgfile

    image = load_image(imgfile)
    pytorch_time, pytorch_blobs, pytorch_models = forward_pytorch(protofile, weightfile, image, 10)
    caffe_time, caffe_blobs, caffe_params = forward_caffe(protofile, weightfile, image, 10)
