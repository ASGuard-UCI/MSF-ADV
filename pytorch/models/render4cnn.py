import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from functools                              import reduce
from torch.legacy.nn.Module                 import Module as LegacyModule
from torch.legacy.nn.utils                  import clear
from torch.nn._functions.thnn.normalization import CrossMapLRN2d

import numpy as np
from pdb import set_trace as st

class Render4CNN(nn.Module):
    def __init__(self, finetune=False, weights = None, weights_path = None, num_classes = 12):
        super().__init__()

        # Normalization layers
        # norm1 = Lambda(lambda x,lrn=SpatialCrossMapLRN_temp(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data)))
        # norm2 = Lambda(lambda x,lrn=SpatialCrossMapLRN_temp(*(5, 0.0001, 0.75, 1)): Variable(lrn.forward(x.data)))
        norm1 = nn.LocalResponseNorm(5, 0.0001, 0.75, 1)
        norm2 = nn.LocalResponseNorm(5, 0.0001, 0.75, 1)

        # conv layers
        conv1 = nn.Conv2d(3, 96, (11, 11), (4,4))
        relu1 = nn.ReLU()
        pool1 = nn.MaxPool2d( (3,3), (2,2), (0,0), ceil_mode=True)

        conv2 = nn.Conv2d(96, 256, (5, 5), (1,1), (2,2), 1,2)
        relu2 = nn.ReLU()
        pool2 = nn.MaxPool2d( (3,3), (2,2), (0,0), ceil_mode=True)

        conv3 = nn.Conv2d(256,384,(3, 3),(1, 1),(1, 1))
        relu3 = nn.ReLU()

        conv4 = nn.Conv2d(384,384,(3, 3),(1, 1),(1, 1),1,2)
        relu4 = nn.ReLU()

        conv5 = nn.Conv2d(384,256,(3, 3),(1, 1),(1, 1),1,2)
        relu5 = nn.ReLU()
        pool5 = nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)


        # inference layers
        fc6     = nn.Linear(9216,4096)
        relu6   = nn.ReLU()

        fc7     = nn.Linear(4096,4096)
        relu7   = nn.ReLU()

        drop6 = nn.Dropout(0.5)
        drop7 = nn.Dropout(0.5)

        azim     = nn.Linear(4096,num_classes * 360)
        elev     = nn.Linear(4096,num_classes * 360)
        tilt     = nn.Linear(4096,num_classes * 360)


        if weights == 'lua':

            state_dict = torch.load(weights_path)

            conv1.weight.data.copy_(state_dict['0.weight'])
            conv1.bias.data.copy_(state_dict['0.bias'])
            conv2.weight.data.copy_(state_dict['4.weight'])
            conv2.bias.data.copy_(state_dict['4.bias'])
            conv3.weight.data.copy_(state_dict['8.weight'])
            conv3.bias.data.copy_(state_dict['8.bias'])
            conv4.weight.data.copy_(state_dict['10.weight'])
            conv4.bias.data.copy_(state_dict['10.bias'])

            conv5.weight.data.copy_(state_dict['12.weight'])
            conv5.bias.data.copy_(state_dict['12.bias'])
            fc6.weight.data.copy_(state_dict['16.1.weight'])
            fc6.bias.data.copy_(state_dict['16.1.bias'])
            fc7.weight.data.copy_(state_dict['19.1.weight'])
            fc7.bias.data.copy_(state_dict['19.1.bias'])

            if num_classes == 3 and (state_dict['22.1.weight'].size()[0] > 360*3):
                azim.weight.data.copy_( torch.cat([  state_dict['22.1.weight'][360*4:360*5, :],  state_dict['22.1.weight'][360*5:360*6, :], state_dict['22.1.weight'][360*8:360*9, :] ], dim = 0) )
                elev.weight.data.copy_( torch.cat([  state_dict['24.1.weight'][360*4:360*5, :],  state_dict['24.1.weight'][360*5:360*6, :], state_dict['24.1.weight'][360*8:360*9, :] ], dim = 0) )
                tilt.weight.data.copy_( torch.cat([  state_dict['26.1.weight'][360*4:360*5, :],  state_dict['26.1.weight'][360*5:360*6, :], state_dict['26.1.weight'][360*8:360*9, :] ], dim = 0) )

                azim.bias.data.copy_(   torch.cat([  state_dict['22.1.bias'][360*4:360*5], state_dict['22.1.bias'][360*5:360*6], state_dict['22.1.bias'][360*8:360*9] ], dim = 0) )
                elev.bias.data.copy_(   torch.cat([  state_dict['24.1.bias'][360*4:360*5], state_dict['24.1.bias'][360*5:360*6], state_dict['24.1.bias'][360*8:360*9] ], dim = 0) )
                tilt.bias.data.copy_(   torch.cat([  state_dict['26.1.bias'][360*4:360*5], state_dict['26.1.bias'][360*5:360*6], state_dict['26.1.bias'][360*8:360*9] ], dim = 0) )
            else:
                azim.weight.data.copy_(state_dict['22.1.weight'])
                elev.weight.data.copy_(state_dict['24.1.weight'])
                tilt.weight.data.copy_(state_dict['26.1.weight'])
                azim.bias.data.copy_(state_dict['22.1.bias'])
                elev.bias.data.copy_(state_dict['24.1.bias'])
                tilt.bias.data.copy_(state_dict['26.1.bias'])

        elif weights == 'npy':
            state_dict = np.load(weights_path).item()

            # Convert parameters to torch tensors
            for key in state_dict.keys():
                state_dict[key]['weight'] = torch.from_numpy(state_dict[key]['weight'])
                state_dict[key]['bias']   = torch.from_numpy(state_dict[key]['bias'])


            conv1.weight.data.copy_(state_dict['conv1']['weight'])
            conv1.bias.data.copy_(state_dict['conv1']['bias'])
            conv2.weight.data.copy_(state_dict['conv2']['weight'])
            conv2.bias.data.copy_(state_dict['conv2']['bias'])
            conv3.weight.data.copy_(state_dict['conv3']['weight'])
            conv3.bias.data.copy_(state_dict['conv3']['bias'])
            conv4.weight.data.copy_(state_dict['conv4']['weight'])
            conv4.bias.data.copy_(state_dict['conv4']['bias'])

            conv5.weight.data.copy_(state_dict['conv5']['weight'])
            conv5.bias.data.copy_(state_dict['conv5']['bias'])
            fc6.weight.data.copy_(state_dict['fc6']['weight'])
            fc6.bias.data.copy_(state_dict['fc6']['bias'])
            fc7.weight.data.copy_(state_dict['fc7']['weight'])
            fc7.bias.data.copy_(state_dict['fc7']['bias'])

            if num_classes == 3:
                azim.weight.data.copy_( torch.cat([  state_dict['pred_azimuth'][  'weight'][360*4:360*5, :],  state_dict['pred_azimuth'][  'weight'][360*5:360*6, :], state_dict['pred_azimuth'][  'weight'][360*8:360*9, :] ], dim = 0) )
                elev.weight.data.copy_( torch.cat([  state_dict['pred_elevation']['weight'][360*4:360*5, :],  state_dict['pred_elevation']['weight'][360*5:360*6, :], state_dict['pred_elevation']['weight'][360*8:360*9, :] ], dim = 0) )
                tilt.weight.data.copy_( torch.cat([  state_dict['pred_tilt'][     'weight'][360*4:360*5, :],  state_dict['pred_tilt'][     'weight'][360*5:360*6, :], state_dict['pred_tilt'][     'weight'][360*8:360*9, :] ], dim = 0) )

                azim.bias.data.copy_(   torch.cat([  state_dict['pred_azimuth']['bias'][360*4:360*5], state_dict['pred_azimuth']['bias'][360*5:360*6], state_dict['pred_azimuth']['bias'][360*8:360*9] ], dim = 0) )
                elev.bias.data.copy_(   torch.cat([  state_dict['pred_elevation']['bias'][360*4:360*5], state_dict['pred_elevation']['bias'][360*5:360*6], state_dict['pred_elevation']['bias'][360*8:360*9] ], dim = 0) )
                tilt.bias.data.copy_(   torch.cat([  state_dict['pred_tilt']['bias'][360*4:360*5], state_dict['pred_tilt']['bias'][360*5:360*6], state_dict['pred_tilt']['bias'][360*8:360*9] ], dim = 0) )
            else:
                azim.weight.data.copy_(state_dict['fc-azimuth']['weight'])
                elev.weight.data.copy_(state_dict['fc-elevation']['weight'])
                tilt.weight.data.copy_(state_dict['fc-tilt']['weight'])
                azim.bias.data.copy_(state_dict['fc-azimuth']['bias'])
                elev.bias.data.copy_(state_dict['fc-elevation']['bias'])
                tilt.bias.data.copy_(state_dict['fc-tilt']['bias'])

        # Define Network
        self.conv4 = nn.Sequential( conv1, relu1, pool1, norm1,
                                    conv2, relu2, pool2, norm2,
                                    conv3, relu3,
                                    conv4, relu4)

        # self.conv4 = nn.Sequential( conv1, relu1, pool1,
                                    # conv2, relu2, pool2,
                                    # conv3, relu3,
                                    # conv4, relu4)

        self.conv5 = nn.Sequential( conv5,  relu5,  pool5)

        self.infer = nn.Sequential( fc6,    relu6,  drop6,
                                    fc7,    relu7,  drop7)

        if finetune:
            self.conv4.requires_grad = False
            self.conv5.requires_grad = False


        self.azim = nn.Sequential( azim )
        self.elev = nn.Sequential( elev )
        self.tilt = nn.Sequential( tilt )

    def forward(self, images):
        # tmp = torch.zeros(images.size(), requires_grad=True)
        # tmp = images
        # features = self.conv4(tmp)
        features = self.conv4(images)
        features = self.conv5(features)
        features = features.view(features.size(0), 9216)
        features = self.infer(features)
        # loss = torch.sum(features)
        # loss.backward()
        # print(torch.sum(tmp.grad))
        # st()
        azim = self.azim(features)
        elev = self.elev(features)
        tilt = self.tilt(features)

        return azim, elev, tilt


# class SpatialCrossMapLRN_temp(LegacyModule):

#     def __init__(self, size, alpha=1e-4, beta=0.75, k=1, gpuDevice=0):
#         super(SpatialCrossMapLRN_temp, self).__init__()

#         self.size = size
#         self.alpha = alpha
#         self.beta = beta
#         self.k = k
#         self.scale = None
#         self.paddedRatio = None
#         self.accumRatio = None
#         self.gpuDevice = gpuDevice

#     def updateOutput(self, input):
#         assert input.dim() == 4

#         if self.scale is None:
#             self.scale = input.new()

#         if self.output is None:
#             self.output = input.new()

#         batchSize = input.size(0)
#         channels = input.size(1)
#         inputHeight = input.size(2)
#         inputWidth = input.size(3)

#         if input.is_cuda:
#             self.output = self.output.cuda(self.gpuDevice)
#             self.scale = self.scale.cuda(self.gpuDevice)

#         self.output.resize_as_(input)
#         self.scale.resize_as_(input)

#         # use output storage as temporary buffer
#         inputSquare = self.output
#         torch.pow(input, 2, out=inputSquare)

#         prePad = int((self.size - 1) / 2 + 1)
#         prePadCrop = channels if prePad > channels else prePad

#         scaleFirst = self.scale.select(1, 0)
#         scaleFirst.zero_()
#         # compute first feature map normalization
#         for c in range(prePadCrop):
#             scaleFirst.add_(inputSquare.select(1, c))

#         # reuse computations for next feature maps normalization
#         # by adding the next feature map and removing the previous
#         for c in range(1, channels):
#             scalePrevious = self.scale.select(1, c - 1)
#             scaleCurrent = self.scale.select(1, c)
#             scaleCurrent.copy_(scalePrevious)
#             if c < channels - prePad + 1:
#                 squareNext = inputSquare.select(1, c + prePad - 1)
#                 scaleCurrent.add_(1, squareNext)

#             if c > prePad:
#                 squarePrevious = inputSquare.select(1, c - prePad)
#                 scaleCurrent.add_(-1, squarePrevious)

#         self.scale.mul_(self.alpha / self.size).add_(self.k)

#         torch.pow(self.scale, -self.beta, out=self.output)
#         self.output.mul_(input)

#         return self.output

#     def updateGradInput(self, input, gradOutput):
#         assert input.dim() == 4

#         batchSize = input.size(0)
#         channels = input.size(1)
#         inputHeight = input.size(2)
#         inputWidth = input.size(3)

#         if self.paddedRatio is None:
#             self.paddedRatio = input.new()
#         if self.accumRatio is None:
#             self.accumRatio = input.new()
#         self.paddedRatio.resize_(channels + self.size - 1, inputHeight, inputWidth)
#         self.accumRatio.resize_(inputHeight, inputWidth)

#         cacheRatioValue = 2 * self.alpha * self.beta / self.size
#         inversePrePad = int(self.size - (self.size - 1) / 2)

#         self.gradInput.resize_as_(input)
#         torch.pow(self.scale, -self.beta, out=self.gradInput).mul_(gradOutput)

#         self.paddedRatio.zero_()
#         paddedRatioCenter = self.paddedRatio.narrow(0, inversePrePad, channels)
#         for n in range(batchSize):
#             torch.mul(gradOutput[n], self.output[n], out=paddedRatioCenter)
#             paddedRatioCenter.div_(self.scale[n])
#             torch.sum(self.paddedRatio.narrow(0, 0, self.size - 1), 0, out=self.accumRatio)
#             for c in range(channels):
#                 self.accumRatio.add_(self.paddedRatio[c + self.size - 1])
#                 self.gradInput[n][c].addcmul_(-cacheRatioValue, input[n][c], self.accumRatio)
#                 self.accumRatio.add_(-1, self.paddedRatio[c])

#         return self.gradInput

#     def clearState(self):
#         clear(self, 'scale', 'paddedRatio', 'accumRatio')
#         return super(SpatialCrossMapLRN_temp, self).clearState()

# class SpatialCrossMapLRNFunc(Function):
#      def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
#          self.size = size
#          self.alpha = alpha
#          self.beta = beta
#          self.k = k

#      def forward(self, input):
#          self.save_for_backward(input)
#          self.lrn = SpatialCrossMapLRNOld(self.size, self.alpha, self.beta, self.k)
#          self.lrn.type(input.type())
#          return self.lrn.forward(input)

#      def backward(self, grad_output):
#          input, = self.saved_tensors
#          return self.lrn.backward(input, grad_output)

# class SpatialCrossMapLRN(nn.Module):
#      def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
#          super(SpatialCrossMapLRN, self).__init__()
#          self.size = size
#          self.alpha = alpha
#          self.beta = beta
#          self.k = k

#      def forward(self, input):
#          return CrossMapLRN2d(self.size, alpha = self.alpha, beta = self.beta, k = self.k)(input)
#          # return SpatialCrossMapLRNFunc(self.size, alpha = self.alpha, beta = self.beta, k = self.k)(input)

# class LambdaBase(nn.Sequential):
#     def __init__(self, fn, *args):
#         super(LambdaBase, self).__init__(*args)
#         self.lambda_func = fn

#     def forward_prepare(self, input):
#         output = []
#         for module in self._modules.values():
#             output.append(module(input))
#         return output if output else input

# class Lambda(LambdaBase):
#     def forward(self, input):
#         return self.lambda_func(self.forward_prepare(input))

# class LambdaMap(LambdaBase):
#     def forward(self, input):
#         return list(map(self.lambda_func,self.forward_prepare(input)))

# class LambdaReduce(LambdaBase):
#     def forward(self, input):
#         return reduce(self.lambda_func,self.forward_prepare(input))