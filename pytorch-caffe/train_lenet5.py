#python train.py --solver mnist/lenet_solver.prototxt --gpu 0,1
from __future__ import print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from caffenet import CaffeNet, ParallelCaffeNet
from prototxt import parse_solver

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Train Caffe Example')
parser.add_argument('--gpu', help='gpu ids e.g "0,1,2,3"')
parser.add_argument('--solver', help='the solver prototxt')
parser.add_argument('--model', help='the network definition prototxt')
parser.add_argument('--snapshot', help='the snapshot solver state to resume training')
parser.add_argument('--weights', help='the pretrained weight')
args = parser.parse_args()

solver        = parse_solver(args.solver)
protofile     = solver['net']
base_lr       = float(solver['base_lr'])
momentum      = float(solver['momentum'])
weight_decay  = float(solver['weight_decay'])
test_iter     = int(solver['test_iter'])
max_iter      = int(solver['max_iter'])
test_interval = int(solver['test_interval'])
snapshot      = int(solver['snapshot'])
snapshot_prefix = solver['snapshot_prefix']

torch.manual_seed(int(time.time()))
if args.gpu:
    torch.cuda.manual_seed(int(time.time()))

net = CaffeNet(protofile)
net.set_verbose(False)
net.set_train_outputs('loss')
net.set_eval_outputs('loss', 'accuracy')
print(net)

if args.gpu:
    device_ids = args.gpu.split(',')
    device_ids = [int(i) for i in device_ids]
    print('device_ids', device_ids)
    if len(device_ids) > 1:
        print('---- Multi GPUs ----')
        net = ParallelCaffeNet(net.cuda(), device_ids=device_ids)
        #net = nn.DataParallel(net.cuda(), device_ids=device_ids)
    else:
        print('---- Single GPU ----')
        net.cuda()

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)

if args.weights:
    state = torch.load(args.weights)
    start_epoch = state['batch']+1
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('loaded state %s' % (args.weights))

net.train()
buf = Variable(torch.zeros(len(device_ids)).cuda())
for batch in range(max_iter):
    if (batch+1) % test_interval == 0:
        net.eval()
        average_accuracy = 0.0
        average_loss = 0.0
        for i in range(test_iter):
            loss, accuracy = net(buf)
            average_accuracy += accuracy.data.mean()
            average_loss += loss.data.mean()
        average_accuracy /= test_iter
        average_loss /= test_iter
        print('[%d]  test loss: %f\ttest accuracy: %f' % (batch+1, average_loss, average_accuracy))
        net.train()
    else:
        optimizer.zero_grad()
        loss = net(buf).mean()
        loss.backward()
        optimizer.step()
        if (batch+1) % 100 == 0:
            print('[%d] train loss: %f' % (batch+1, loss.data[0]))
        

    if (batch+1) % snapshot == 0:
        savename = '%s_batch%08d.pth' % (snapshot_prefix, batch+1)
        print('save state %s' % (savename))
        state = {'batch': batch+1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()}
        torch.save(state, savename)
