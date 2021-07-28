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

def forward_pytorch(protofile, weightfile, image):
    net = CaffeNet(protofile, width=args.width, height=args.height, omit_data_layer=True, phase='TEST')
    if args.cuda:
        net.cuda()
    print(net)
    net.load_weights(weightfile)
    net.eval()
    image = torch.from_numpy(image)
    if args.cuda:
        image = Variable(image.cuda())
    else:
        image = Variable(image)
    t0 = time.time()
    blobs = net(image)
    t1 = time.time()
    return t1-t0, blobs, net.models

# Reference from:
def forward_caffe(protofile, weightfile, image):
    if args.cuda:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    net.blobs['data'].reshape(1, 3, args.height, args.width)
    net.blobs['data'].data[...] = image
    t0 = time.time()
    output = net.forward()
    t1 = time.time()
    return t1-t0, net.blobs, net.params

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
    time_pytorch, pytorch_blobs, pytorch_models = forward_pytorch(protofile, weightfile, image)
    time_caffe, caffe_blobs, caffe_params = forward_caffe(protofile, weightfile, image)

    print('pytorch forward time %d', time_pytorch)
    print('caffe forward time %d', time_caffe)

    layer_names = pytorch_models.keys()
    blob_names = pytorch_blobs.keys()
    print('------------ Parameter Difference ------------')
    for layer_name in layer_names:
        if type(pytorch_models[layer_name]) in [nn.Conv2d, nn.Linear, Scale, Normalize]:
            pytorch_weight = pytorch_models[layer_name].weight.data
            if args.cuda:
                pytorch_weight = pytorch_weight.cpu().numpy()
            else:
                pytorch_weight = pytorch_weight.numpy()
            caffe_weight = caffe_params[layer_name][0].data
            weight_diff = abs(pytorch_weight - caffe_weight).sum()
            if type(pytorch_models[layer_name].bias) == Parameter:
                pytorch_bias = pytorch_models[layer_name].bias.data
                if args.cuda:
                    pytorch_bias = pytorch_bias.cpu().numpy()
                else:
                    pytorch_bias = pytorch_bias.numpy()
                caffe_bias = caffe_params[layer_name][1].data
                bias_diff = abs(pytorch_bias - caffe_bias).sum()
                print('%-30s       weight_diff: %f        bias_diff: %f' % (layer_name, weight_diff, bias_diff))
            else:
                print('%-30s       weight_diff: %f' % (layer_name, weight_diff))
        elif type(pytorch_models[layer_name]) == nn.BatchNorm2d:
            if args.cuda:
                pytorch_running_mean = pytorch_models[layer_name].running_mean.cpu().numpy()
                pytorch_running_var = pytorch_models[layer_name].running_var.cpu().numpy()
            else:
                pytorch_running_mean = pytorch_models[layer_name].running_mean.numpy()
                pytorch_running_var = pytorch_models[layer_name].running_var.numpy()
            caffe_running_mean = caffe_params[layer_name][0].data/caffe_params[layer_name][2].data[0]
            caffe_running_var = caffe_params[layer_name][1].data/caffe_params[layer_name][2].data[0]
            running_mean_diff = abs(pytorch_running_mean - caffe_running_mean).sum()
            running_var_diff = abs(pytorch_running_var - caffe_running_var).sum()
            print('%-30s running_mean_diff: %f running_var_diff: %f' % (layer_name, running_mean_diff, running_var_diff))
    
    print('------------ Output Difference ------------')
    for blob_name in blob_names:
        if args.cuda:
            pytorch_data = pytorch_blobs[blob_name].data.cpu().numpy()
        else:
            pytorch_data = pytorch_blobs[blob_name].data.numpy()
        caffe_data = caffe_blobs[blob_name].data
        diff = abs(pytorch_data - caffe_data).sum()
        print('%-30s pytorch_shape: %-20s caffe_shape: %-20s output_diff: %f' % (blob_name, pytorch_data.shape, caffe_data.shape, diff/pytorch_data.size))

    if args.synset_words != '':
        print('------------ Classification ------------')
        synset_dict = load_synset_words(args.synset_words)
        if 'prob' in blob_names:
            if args.cuda:
                pytorch_prob = pytorch_blobs['prob'].data.cpu().view(-1).numpy()
            else:
                pytorch_prob = pytorch_blobs['prob'].data.view(-1).numpy()
            caffe_prob = caffe_blobs['prob'].data[0]
            print('pytorch classification top1: %f %s' % (pytorch_prob.max(), synset_dict[pytorch_prob.argmax()]))
            print('caffe   classification top1: %f %s' % (caffe_prob.max(), synset_dict[caffe_prob.argmax()]))
