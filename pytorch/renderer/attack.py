import torch
from torch import nn
import numpy as np

import torch.nn.functional as nnfunc


class AddSphere(nn.Module):
    '''
    Axis:
       y
       |
       |
       |
       /--------- x
      /
     /    u_theta
    z
    '''

    def __init__(self, height, width, sensor_scale):
        super().__init__()

        self.height = height
        self.width = width
        self.scaler = width / 2 / sensor_scale
        r, c = np.mgrid[:height, :width]
        y = -(r - (height-1) / 2) / self.scaler
        x = (c - (width-1) / 2) / self.scaler

        x = x[np.newaxis, :, :, np.newaxis] # 1 x h x w x 1
        y = y[np.newaxis, :, :, np.newaxis] # 1 x h x w x 1

        self.register_buffer('x', torch.from_numpy(x.astype(np.float32)))
        self.register_buffer('y', torch.from_numpy(y.astype(np.float32)))


    def __call__(self, depth, normal, refl, mask, geometry, color):
        '''
        Compute the new depth, normal, reflectance, and mask, after inserting _n_
        colored spheres into the scene.
        :param depth: b x 1 x h x w tensor
        :param normal: b x 3 x h x w tensor
        :param refl: b x c x h x w tensor
        :param mask: b x 1 x h x w tensor
        :param sersor_scale: the screen width is 2 * sensor_scale
        :param geometry: b x 4 x n tensor, in (x, y, z, radius) format for each sphere
        :param color: b x c x n tensor, representing the color of the n spheres
        :return: (new_depth, new_normal, new_refl, new_mask)
        '''

        height, width = depth.size()[2:]
        assert self.height == height
        assert self.width == width

        center_x, center_y, center_z, radius = geometry.unsqueeze(2).split(1, dim=1) # b x 1 x 1 x n
        diff_x = self.x - center_x # b x h x w x n
        diff_y = self.y - center_y # b x h x w x n
        tmp = radius * radius - diff_x * diff_x - diff_y * diff_y # b x h x w x n
        mask_sphere = (tmp >= 0) # b x h x w x n
        tmp = tmp.clamp(min=0)

        diff_z = torch.sqrt(tmp)
        depth_sphere = (diff_z + center_z) * self.scaler # b x h x w x n
        depth_sphere[~mask_sphere] = float('-inf') # b x h x w x n
        normal_sphere = torch.stack([diff_x, diff_y, diff_z], dim=1) # b x 1 x h x w x n
        normal_sphere = nnfunc.normalize(normal_sphere)
        refl_sphere = color.unsqueeze(2).unsqueeze(3).expand(-1, -1, height, width, -1) # b x c x h x w x n

        all_depth = torch.cat([depth.unsqueeze(-1), depth_sphere.unsqueeze(1)], dim=4) # b x 1 x h x w x (n+1)
        all_normal = torch.cat([normal.unsqueeze(-1), normal_sphere], dim=4) # b x 3 x h x w x (n+1)
        all_refl = torch.cat([refl.unsqueeze(-1), refl_sphere], dim=4) # b x c x h x w x (n+1)
        all_mask = torch.cat([mask.unsqueeze(-1), mask_sphere.unsqueeze(1)], dim=4) # b x 1 x h x w x (n+1)

        new_mask = all_mask.any(dim=4, keepdim=False) # b x 1 x h x w
        new_depth, indices = all_depth.max(dim=4, keepdim=False) # b x 1 x h x w
        indices_3 = indices.expand([-1, 3, -1, -1]).unsqueeze(-1) # b x 3 x h x w x 1

        # out(b, c, i, j, k) = in(b, c, i, j, index(b, c, i, j, k)
        new_normal = torch.gather(all_normal, dim=4, index=indices_3).squeeze(-1)
        new_refl = torch.gather(all_refl, dim=4, index=indices_3).squeeze(-1)

        return new_depth, new_normal, new_refl, new_mask
