# 2017.12.16 by xiaohang
import sys
from caffenet import *
import numpy as np
import argparse
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time

def load_synset_words(synset_file):
    lines = open(synset_file).readlines()
    synset_dict = dict()
    for i, line in enumerate(lines):
        synset_dict[i] = line.strip()
    return synset_dict

def forward_caffe(protofile, weightfile):
    if args.cuda:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    if args.phase == 'TRAIN':
        net = caffe.Net(protofile, weightfile, caffe.TRAIN)
    else:
        net = caffe.Net(protofile, weightfile, caffe.TEST)
    t0 = time.time()
    output = net.forward()
    t1 = time.time()
    return t1-t0, net.blobs, net.params

def forward_pytorch(protofile, weightfile, data, label):
    channels = data.shape[1]
    height = data.shape[2]
    width = data.shape[3]
    net = CaffeNet(protofile, width=width, height=height, channels=channels, omit_data_layer = True, phase=args.phase)
    if args.cuda:
        net.cuda()
    print(net)
    net.load_weights(weightfile)
    net.train()
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    if args.cuda:
        data = Variable(data.cuda())
        label = Variable(label.cuda())
    else:
        data = Variable(data)
        label = Variable(label)
    t0 = time.time()
    blobs = net(data, label)
    t1 = time.time()
    return t1-t0, blobs, net.models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert caffe to pytorch')
    parser.add_argument('--protofile', default='', type=str)
    parser.add_argument('--weightfile', default='', type=str)
    parser.add_argument('--phase', default='TRAIN', type=str)
    parser.add_argument('--synset_words', default='', type=str)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    #parser.add_argument('--omit_data_layer', action='store_true', help='whether to omit_data_layer')

    args = parser.parse_args()
    print(args)
    
    
    protofile = args.protofile
    weightfile = args.weightfile

    time_caffe, caffe_blobs, caffe_params = forward_caffe(protofile, weightfile)
    data = caffe_blobs['data'].data
    label = caffe_blobs['label'].data
    time_pytorch, pytorch_blobs, pytorch_models = forward_pytorch(protofile, weightfile, data, label)

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
        #if blob_name == 'mbox_loss':
        #    print('pytorch mbox_loss', pytorch_data)
        #    print('caffe mbox_loss', caffe_data)

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
