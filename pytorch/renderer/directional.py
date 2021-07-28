import torch

from torch import nn
import torch.nn.functional as nnfunc


class DirectionalRenderer:
    def __init__(self, clip=None, output_shading=False):
        self.clip = clip
        self.output_shading = output_shading

    def __call__(self, normal, reflectance, direction, intensity):
        '''
        Computed the rendered image assuming diffuse shading:
            max(cos(angle<d, n>), 0) * intensity.

        :param normal: a tensor of b x 3 x h x w, no need to be normalized.

        :param reflectance: a tensor of b x c x h x w.

        :param direction: a tensor of b x 3 x l representing the vectors of lighting
            directions of _l_ directional light sources in each batch.
            No need to be normalized.

        :param intensity: a tensor of b x c x l representing the _l_ lighting intensity
            in different channels.

        :return: a tensor of b x c x h x w, representing the rendered image.
        '''

        normal = nnfunc.normalize(normal) # b x 3 x h x w
        direction = nnfunc.normalize(direction, p=2, dim=1) # b x 3 x l

        # For each lighting, each pixel in each image there is a
        # max(cos(theta), 0)
        d_dot_n = torch.einsum('bmn,bmrc->bnrc', (direction, normal)).clamp(min=0)

        # Compute the shading at each pixel
        shading = torch.einsum('bnrc,bkn->bkrc', (d_dot_n, intensity))

        color = reflectance * shading
        if self.clip:
            color = torch.clamp(color, min=self.clip[0], max=self.clip[1])

        if self.output_shading:
            return color, shading
        else:
            return color
