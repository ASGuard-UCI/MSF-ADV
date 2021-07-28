import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
# from IPython import embed

class clickhere_cnn(nn.Module):
    def __init__(self, renderCNN, weights_path = None, num_classes = 12):
        super(clickhere_cnn, self).__init__()

        # Image Stream
        self.conv4 = renderCNN.conv4
        self.conv5 = renderCNN.conv5

        fc6     = nn.Linear(9216,4096)
        relu6   = nn.ReLU()
        fc7     = nn.Linear(4096,4096)
        relu7   = nn.ReLU()
        drop6 = nn.Dropout(0.5)
        drop7 = nn.Dropout(0.5)


        #Keypoint Stream
        kp_map      = nn.Linear(2116,2116)
        kp_class    = nn.Linear(34,34)
        kp_fuse     = nn.Linear(2150,169)

        # Fused layer
        fc8     = nn.Linear(4096 + 384, 4096)
        relu8   = nn.ReLU()
        drop8 = nn.Dropout(0.5)

        # Prediction layers
        azim        = nn.Linear(4096, num_classes * 360)
        elev        = nn.Linear(4096, num_classes * 360)
        tilt        = nn.Linear(4096, num_classes * 360)

        if weights_path:
            npy_dict = np.load(weights_path).item()

            state_dict = npy_dict
            # Convert parameters to torch tensors
            for key in npy_dict.keys():
                state_dict[key]['weight'] = torch.from_numpy(npy_dict[key]['weight'])
                state_dict[key]['bias']   = torch.from_numpy(npy_dict[key]['bias'])

            self.conv4[0].weight.data.copy_(state_dict['conv1']['weight'])
            self.conv4[0].bias.data.copy_(state_dict['conv1']['bias'])
            self.conv4[4].weight.data.copy_(state_dict['conv2']['weight'])
            self.conv4[4].bias.data.copy_(state_dict['conv2']['bias'])
            self.conv4[8].weight.data.copy_(state_dict['conv3']['weight'])
            self.conv4[8].bias.data.copy_(state_dict['conv3']['bias'])
            self.conv4[10].weight.data.copy_(state_dict['conv4']['weight'])
            self.conv4[10].bias.data.copy_(state_dict['conv4']['bias'])
            self.conv5[0].weight.data.copy_(state_dict['conv5']['weight'])
            self.conv5[0].bias.data.copy_(state_dict['conv5']['bias'])

            fc6.weight.data.copy_(state_dict['fc6']['weight'])
            fc6.bias.data.copy_(state_dict['fc6']['bias'])
            fc7.weight.data.copy_(state_dict['fc7']['weight'])
            fc7.bias.data.copy_(state_dict['fc7']['bias'])
            fc8.weight.data.copy_(state_dict['fc8']['weight'])
            fc8.bias.data.copy_(state_dict['fc8']['bias'])

            kp_map.weight.data.copy_(state_dict['fc-keypoint-map']['weight'])
            kp_map.bias.data.copy_(state_dict['fc-keypoint-map']['bias'])
            kp_class.weight.data.copy_(state_dict['fc-keypoint-class']['weight'])
            kp_class.bias.data.copy_(state_dict['fc-keypoint-class']['bias'])
            kp_fuse.weight.data.copy_(state_dict['fc-keypoint-concat']['weight'])
            kp_fuse.bias.data.copy_(state_dict['fc-keypoint-concat']['bias'])

            if num_classes == 3 and (state_dict['pred_azimuth']['weight'].size()[0] > 360*3):
                azim.weight.data.copy_( torch.cat([  state_dict['pred_azimuth'][  'weight'][360*4:360*5, :],  state_dict['pred_azimuth'][  'weight'][360*5:360*6, :], state_dict['pred_azimuth'][  'weight'][360*8:360*9, :] ], dim = 0) )
                elev.weight.data.copy_( torch.cat([  state_dict['pred_elevation']['weight'][360*4:360*5, :],  state_dict['pred_elevation']['weight'][360*5:360*6, :], state_dict['pred_elevation']['weight'][360*8:360*9, :] ], dim = 0) )
                tilt.weight.data.copy_( torch.cat([  state_dict['pred_tilt'][     'weight'][360*4:360*5, :],  state_dict['pred_tilt'][     'weight'][360*5:360*6, :], state_dict['pred_tilt'][     'weight'][360*8:360*9, :] ], dim = 0) )

                azim.bias.data.copy_(   torch.cat([  state_dict['pred_azimuth']['bias'][360*4:360*5], state_dict['pred_azimuth']['bias'][360*5:360*6], state_dict['pred_azimuth']['bias'][360*8:360*9] ], dim = 0) )
                elev.bias.data.copy_(   torch.cat([  state_dict['pred_elevation']['bias'][360*4:360*5], state_dict['pred_elevation']['bias'][360*5:360*6], state_dict['pred_elevation']['bias'][360*8:360*9] ], dim = 0) )
                tilt.bias.data.copy_(   torch.cat([  state_dict['pred_tilt']['bias'][360*4:360*5], state_dict['pred_tilt']['bias'][360*5:360*6], state_dict['pred_tilt']['bias'][360*8:360*9] ], dim = 0) )
            else:
                azim.weight.data.copy_( state_dict['pred_azimuth'  ]['weight'] )
                elev.weight.data.copy_( state_dict['pred_elevation']['weight'] )
                tilt.weight.data.copy_( state_dict['pred_tilt'     ]['weight'] )

                azim.bias.data.copy_( state_dict['pred_azimuth'  ]['bias'] )
                elev.bias.data.copy_( state_dict['pred_elevation']['bias'] )
                tilt.bias.data.copy_( state_dict['pred_tilt'     ]['bias'] )

        self.pool_map    = nn.Sequential(nn.MaxPool2d( (5,5), (5,5), (1,1), ceil_mode=True))
        self.map_linear  = nn.Sequential( kp_map )
        self.cls_linear  = nn.Sequential( kp_class )
        self.kp_softmax  = nn.Sequential( kp_fuse, nn.Softmax() )

        self.infer = nn.Sequential(fc6, relu6, drop6, fc7, relu7, drop7)
        self.fusion = nn.Sequential(fc8, relu8, drop8)


        self.azim = nn.Sequential(azim)
        self.elev = nn.Sequential(elev)
        self.tilt = nn.Sequential(tilt)

        if weights_path == None:
            self.init_weights()


    def init_weights(self):

        self.infer[0].weight.data.normal_(0.0, 0.01)
        self.infer[0].bias.data.fill_(0)
        self.infer[3].weight.data.normal_(0.0, 0.01)
        self.infer[3].bias.data.fill_(0)

        # Intialize weights for KP stream
        self.map_linear[0].weight.data.normal_(0.0, 0.01)
        self.map_linear[0].bias.data.fill_(0)
        self.cls_linear[0].weight.data.normal_(0.0, 0.01)
        self.cls_linear[0].bias.data.fill_(0)
        self.kp_softmax[0].weight.data.normal_(0.0, 0.01)
        self.kp_softmax[0].bias.data.fill_(0)

        # Initialize weights for fusion and inference
        self.fusion[0].weight.data.normal_(0.0, 0.01)
        self.fusion[0].bias.data.fill_(0)

        self.azim[0].weight.data.normal_(0.0, 0.01)
        self.azim[0].bias.data.fill_(0)
        self.elev[0].weight.data.normal_(0.0, 0.01)
        self.elev[0].bias.data.fill_(0)
        self.tilt[0].weight.data.normal_(0.0, 0.01)
        self.tilt[0].bias.data.fill_(0)


    def forward(self, images, kp_map, kp_class):
        # Image Stream
        features_conv4 = self.conv4(images)
        features_conv5 = self.conv5(features_conv4)
        features_conv5 = features_conv5.view(features_conv5.size(0), -1)
        features_fc7   = self.infer(features_conv5)

        # Keypoint Stream
        # KP map scaling performed in dataset class
        # kp_map       = self.pool_map(kp_map)
        kp_map_flat  = kp_map.view(kp_map.size(0), -1)
        features_map = self.map_linear(kp_map_flat)
        features_cls = self.cls_linear(kp_class)

        # Concatenate the two keypoint feature vectors
        # In deploy file, map over class
        features_kp = torch.cat([features_map, features_cls], dim = 1)

        # Softmax followed by reshaping into a 13x13
        # Conv4 as shape batch * 384 * 13 * 13
        features_kp = self.kp_softmax(features_kp)
        features_kp = features_kp.view(kp_map.size(0),1, 13, 13)

        # Attention -> Elt. wise product, then summation over x and y dims
        attention_mul   = features_kp * features_conv4
        attention_kp    = attention_mul.sum(3).sum(2)

        # Concatenate fc7 and attended features
        features_fused = torch.cat([features_fc7, attention_kp], dim = 1)
        features_fused = self.fusion(features_fused)

        # Final inference
        azim = self.azim(features_fused)
        elev = self.elev(features_fused)
        tilt = self.tilt(features_fused)

        return azim, tilt, elev
