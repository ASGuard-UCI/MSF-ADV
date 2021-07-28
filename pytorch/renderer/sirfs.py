import torch

from torch import nn
import torch.nn.functional as nnfunc


def directional_world_to_camera(direction, angle):
    '''
    Compute the lighting directions in camera space.

    :param direction: the directions in world space, a tensor of b x 3.
    :param angle: the camera angles [theta, phi, psi], a tensor of b x 3.
    :return: the directions in camera space.
    '''
    # TODO: @Dawei implementation
    raise NotImplementedError



def directional_sph_coeff(direction, intensity):
    '''
    Compute the spherical harmonics (SPH) coefficients, given a set of directional
    lights and their parameters. This function is diffferentiable. For more details,
    see the paper "An Efficient Representation for Irradiance Environment Maps".

    :param direction: a tensor of b x 3 representing the vectors of lighting
    directions of _b_ directional light sources. No need to be normalized.
    :param intensity: a tensor of b x c representing the lighting intensity in different
    channels.

    :return: a tensor of b x c x 9, representing the computed SPH coefficients in
    different channels.
    '''
    # Constants
    const_c = [0.282095, 0.488603, 1.092548, 0.315392, 0.546274]

    lighting_vector = nnfunc.normalize(direction, p=2, dim=1) # unit vector of b x 3
    x, y, z = torch.split(lighting_vector, 1, dim=1) # each b x 1

    # Compute the SPH coefficients
    coeff_00 = const_c[0] * intensity           # b x c
    coeff_1_1 = const_c[1] * y * intensity
    coeff_10 = const_c[1] * z * intensity
    coeff_11 = const_c[1] * x * intensity
    coeff_2_2 = const_c[2] * x * y * intensity
    coeff_2_1 = const_c[2] * y * z * intensity
    coeff_20 = const_c[3] * (3 * z * z - 1) * intensity
    coeff_21 = const_c[2] * x * z * intensity
    coeff_22 = const_c[4] * (x * x - y * y) * intensity

    return torch.stack([
        coeff_00,
        coeff_1_1, coeff_10, coeff_11,
        coeff_2_2, coeff_2_1, coeff_20, coeff_21, coeff_22,
    ], dim=2) # b x c x 9



class SIRFSRenderer(nn.Module):
    '''
    A differentiable renderer from the paper "Shape, Illumination Reflectance from Shading".
    Axis:

       y
       |
       |
       |
       /--------- x
      /
     /
    z

    Assume the camera is orthographic.
    '''
    def __init__(self, clip=None, output_normal_shading=False, exp=False):
        '''
        :param clip: None if not clipping, otherwise clip = (low, high)
        :param output_normal_shading: whether output the computed normal map and the shading in forward call.
        :param exp: whether apply exp to the final shading.
        '''
        super().__init__()
        # In torch implementation, the conv2d is actually cross-correlation!
        self.register_buffer('h3y', torch.Tensor([[
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]
        ]]) / 8)
        self.register_buffer('h3x', -torch.Tensor(self.h3y.transpose(2, 3)))
        self.register_buffer('coeff', torch.Tensor([0.429043, 0.511664, 0.743125, 0.886227, 0.247708]))
        self.output_normal_shading = output_normal_shading

        self.renderer = SIRFSRendererNormal(clip=clip, output_shading=output_normal_shading, exp=exp)


    def forward(self, depth, reflectance, light_harmonics, mask=None):
        ''' Render an image based on the depth map, reflectance map, and the lighting
            spherical harmonics efficients.
            See the supplementary material of the paper for details.

            :param depth: torch tensor of size b x 1 x h x w.
            :param light_harmonics: torch tensor of size b x c x 9.
        '''

        # Compute the normals from the depth image
        n_x_num = nn.functional.conv2d(depth, self.h3x, padding=1)
        n_y_num = nn.functional.conv2d(depth, self.h3y, padding=1)
        mat_b = torch.sqrt(1 + n_x_num * n_x_num + n_y_num * n_y_num)

        # The three components of the normal map, all b x 1 x h x w.
        n_x = n_x_num / mat_b
        n_y = n_y_num / mat_b
        n_z = 1 / mat_b
        n_1 = torch.ones_like(n_z, dtype=n_z.dtype)
        normals = torch.cat([n_x, n_y, n_z], dim=1)

        if self.output_normal_shading:
            color, shading = self.renderer(normals, reflectance, light_harmonics, mask=mask)
            return color, normals, shading
        else:
            return self.renderer(normals, reflectance, light_harmonics, mask=mask)


class SIRFSRendererNormal(nn.Module):
    def __init__(self, clip=None, output_shading=False, exp=False, bg_type=None):
        '''
        :param clip: None if not clipping, otherwise clip = (low, high)
        :param output_shading: whether output the computed normal map and the shading in forward call.
        :param exp: whether apply exponential
        :param bg_type: the background type. None: do not mask foreground/background.
            black/white: pure background. envmap: pure color background computed from lighting.
            envmap_ps: smooth background computed from lighting, assuming the camera
            is perspective. plate: assume there is a infinitly large white plate at infinitly far,
            perpenticular to the camera. plate_y: same, but with the plate normal heading +y.
        '''
        super().__init__()
        # In torch implementation, the conv2d is actually cross-correlation!
        self.register_buffer('coeff', torch.Tensor([0.429043, 0.511664, 0.743125, 0.886227, 0.247708]))

        assert clip is None or len(clip) == 2
        self.clip = clip
        self.output_shading = output_shading
        self.exp = exp

        assert bg_type in [None, 'black', 'white', 'envmap', 'envmap_ps', 'plate']
        self.bg_type = bg_type


    def forward(self, normal, reflectance, light_harmonics, mask=None):
        ''' Render an image based on the depth map, reflectance map, and the lighting
            spherical harmonics efficients.
            See the supplementary material of the paper for details.

            :param normal: torch tensor of size b x 3 x h x w.
            :param light_harmonics: torch tensor of size b x c x 9.
        '''
        normal = nnfunc.normalize(normal)
        n_x, n_y, n_z = torch.split(normal, 1, dim=1)

        n_1 = torch.ones_like(n_z, dtype=n_z.dtype)

        # Compute the shading according to the normal and lighting
        l = light_harmonics.permute(2, 0, 1) # 9 x b x c
        c = self.coeff # 5

        mat_m_1 = torch.stack([c[0] * l[8], c[0] * l[4], c[0] * l[7], c[1] * l[3]])
        mat_m_2 = torch.stack([c[0] * l[4], -c[0] * l[8], c[0] * l[5], c[1] * l[1]])
        mat_m_3 = torch.stack([c[0] * l[7], c[0] * l[5], c[2] * l[6], c[1] * l[2]])
        mat_m_4 = torch.stack([c[1] * l[3], c[1] * l[1], c[1] * l[2], c[3] * l[0] - c[4] * l[6]])
        mat_m = torch.stack([mat_m_1, mat_m_2, mat_m_3, mat_m_4]) # 4 x 4 x b x c
        mat_m = mat_m.permute(2, 3, 0, 1) # b x c x 4 x 4
        n_vec = torch.cat([n_x, n_y, n_z, n_1], dim=1) # b x 4 x h x w

        # Indices: b: batch; c: channel; i, j: pixel location; k, l: 4 x 4.
        log_shading = torch.einsum('bikl,bcij,bjkl->bckl', (n_vec, mat_m, n_vec)) # b x c x h x w

        if self.exp:
            shading = torch.exp(log_shading)
        else:
            shading = log_shading

        color = reflectance * shading
        if self.clip:
            color = torch.clamp(color, min=self.clip[0], max=self.clip[1])

        # Compute background
        if self.bg_type is not None:
            assert mask is not None, 'mask cannot be None when computing background!'

            mask_3 = torch.cat((mask,) * 3, 1)
            umask_3 = ~mask_3

            if self.bg_type == 'white':
                color[umask_3] = 1
            elif self.bg_type == 'black':
                color[umask_3] = 0
            elif self.bg_type == 'plate':
                # Render with plate normal (0, 0, 1)
                color = color * mask_3.to(color) + (c[2] * l[6] + 2 * c[1] * l[2] + c[3] * l[0] - c[4] * l[6]).unsqueeze(2).unsqueeze(3) * umask_3.to(color)
            elif self.bg_type == 'envmap':
                color = color * mask_3.to(color) + (c[0] * l[0] + c[1] * l[2] + 2 * c[3] * l[6]).unsqueeze(2).unsqueeze(3) * umask_3.to(color)
            elif self.bg_type == 'envmap_ps':
                raise NotImplementedError


        if self.output_shading:
            return color, shading
        else:
            return color
