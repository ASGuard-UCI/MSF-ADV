import torch
import torch.nn as nn
import numpy    as np

from torch.autograd     import Function, Variable
from torchvision.models import vgg16



class vgg_tm(nn.Module):
    def __init__(self, weights_path):
        super(vgg_tm, self).__init__()

        VGG = vgg16()

        # Copy over weights
        state_dict = np.load(weights_path).item()

        for key in state_dict.keys():
            state_dict[key]['weight'] = torch.from_numpy(state_dict[key]['weight'])
            state_dict[key]['bias']   = torch.from_numpy(state_dict[key]['bias'])

        VGG.features[00].weight.data.copy_(state_dict['conv1_1']['weight'])
        VGG.features[00].bias.data.copy_(state_dict['conv1_1']['bias'])
        VGG.features[02].weight.data.copy_(state_dict['conv1_2']['weight'])
        VGG.features[02].bias.data.copy_(state_dict['conv1_2']['bias'])

        VGG.features[05].weight.data.copy_(state_dict['conv2_1']['weight'])
        VGG.features[05].bias.data.copy_(state_dict['conv2_1']['bias'])
        VGG.features[07].weight.data.copy_(state_dict['conv2_2']['weight'])
        VGG.features[07].bias.data.copy_(state_dict['conv2_2']['bias'])

        VGG.features[10].weight.data.copy_(state_dict['conv3_1']['weight'])
        VGG.features[10].bias.data.copy_(state_dict['conv3_1']['bias'])
        VGG.features[12].weight.data.copy_(state_dict['conv3_2']['weight'])
        VGG.features[12].bias.data.copy_(state_dict['conv3_2']['bias'])
        VGG.features[14].weight.data.copy_(state_dict['conv3_3']['weight'])
        VGG.features[14].bias.data.copy_(state_dict['conv3_3']['bias'])

        VGG.features[17].weight.data.copy_(state_dict['conv4_1']['weight'])
        VGG.features[17].bias.data.copy_(state_dict['conv4_1']['bias'])
        VGG.features[19].weight.data.copy_(state_dict['conv4_2']['weight'])
        VGG.features[19].bias.data.copy_(state_dict['conv4_2']['bias'])
        VGG.features[21].weight.data.copy_(state_dict['conv4_3']['weight'])
        VGG.features[21].bias.data.copy_(state_dict['conv4_3']['bias'])

        VGG.features[24].weight.data.copy_(state_dict['conv5_1']['weight'])
        VGG.features[24].bias.data.copy_(state_dict['conv5_1']['bias'])
        VGG.features[26].weight.data.copy_(state_dict['conv5_2']['weight'])
        VGG.features[26].bias.data.copy_(state_dict['conv5_2']['bias'])
        VGG.features[28].weight.data.copy_(state_dict['conv5_3']['weight'])
        VGG.features[28].bias.data.copy_(state_dict['conv5_3']['bias'])


        VGG.classifier[0].weight.data.copy_(state_dict['fc6']['weight'])
        VGG.classifier[0].bias.data.copy_(state_dict['fc6']['bias'])
        VGG.classifier[3].weight.data.copy_(state_dict['fc7']['weight'])
        VGG.classifier[3].bias.data.copy_(state_dict['fc7']['bias'])

        classifier_modules = list(VGG.classifier.children())[0:6]

        VGG.classifier = nn.Sequential(*classifier_modules)

        self.features = VGG.features
        self.classifier = VGG.classifier

        # Viewpoint estimation layers

        azim     = nn.Linear(4096,4320)
        elev     = nn.Linear(4096,4320)
        tilt     = nn.Linear(4096,4320)

        self.azim = nn.Sequential( azim )
        self.elev = nn.Sequential( elev )
        self.tilt = nn.Sequential( tilt )

        self.azim[0].weight.data.normal_(0.0, 0.02)
        self.azim[0].bias.data.fill_(0)
        self.elev[0].weight.data.normal_(0.0, 0.02)
        self.elev[0].bias.data.fill_(0)
        self.tilt[0].weight.data.normal_(0.0, 0.02)
        self.tilt[0].bias.data.fill_(0)


    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return self.azim(x), self.elev(x), self.tilt(x)
