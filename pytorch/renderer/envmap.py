'''
1. Compute SPH coefficients from an envmap.
2. Reconstruct an envmap from SPH coefficients.
'''

import numpy as np


C = [0.282095, 0.488603, 1.092548, 0.315392, 0.546274]


def envmap_compute(array):
    '''
    Compute the spherical harmonics coefficients from the hdr image, by
    computing the integral of the basis and the array.
       y
       |
       |
       |
       /--------- x
      /
     /    u_theta
    z
    U wrapping: from x to z then -x then -z.
    V wrapping: from -y to +y.

    :param array: h x w x c image.
    :return: c x 9 numpy array.
    '''

    # From u, v coordinates to x, y, z
    height, width, chann = array.shape
    v, u = np.mgrid[:height, :width].astype(np.float32) + 0.5 # Symmetrical
    u = np.repeat(u[:, :, np.newaxis], chann, axis=2) # h x w x c
    v = np.repeat(v[:, :, np.newaxis], chann, axis=2) # h x w x c

    u_theta = u / width * 2 * np.pi
    v_phi = (0.5 - v / height) * np.pi

    x = np.cos(u_theta) * np.cos(v_phi)
    y = np.sin(v_phi)
    z = np.sin(u_theta) * np.cos(v_phi)

    # All pre-computed basis values have size h x w x c
    basis_00 = C[0]

    basis_1_1 = C[1] * y
    basis_10 = C[1] * z
    basis_11 = C[1] * x

    basis_2_2 = C[2] * x * y
    basis_2_1 = C[2] * y * z
    basis_20 = C[3] * (3 * z * z - 1)
    basis_21 = C[2] * x * z
    basis_22 = C[4] * (x * x - y * y)

    d_area = np.cos(v_phi) * (2 * np.pi / width) * (np.pi / height) # dS = cos(phi) d_phi d_theta

    # Each coefficient: c
    coeff_00 = (basis_00 * array * d_area).sum(0).sum(0)
    coeff_1_1 = (basis_1_1 * array * d_area).sum(0).sum(0)
    coeff_10 = (basis_10 * array * d_area).sum(0).sum(0)
    coeff_11 = (basis_11 * array * d_area).sum(0).sum(0)
    coeff_2_2 = (basis_2_2 * array * d_area).sum(0).sum(0)
    coeff_2_1 = (basis_2_1 * array * d_area).sum(0).sum(0)
    coeff_20 = (basis_20 * array * d_area).sum(0).sum(0)
    coeff_21 = (basis_21 * array * d_area).sum(0).sum(0)
    coeff_22 = (basis_22 * array * d_area).sum(0).sum(0)

    return np.stack([
            coeff_00, coeff_1_1, coeff_10, coeff_11, coeff_2_2, coeff_2_1, coeff_20, coeff_21, coeff_22
        ], axis=1) # c x 9


def envmap_reconstruct(coeff, height, width):
    '''
    :param coeff: c x 9 numpy array.
    :param height: int
    :param width: int
    :return: height x width x c numpy array.
    '''
    chann = coeff.shape[0]
    v, u = np.mgrid[:height, :width].astype(np.float32) + 0.5 # Symmetrical
    u = np.repeat(u[:, :, np.newaxis], chann, axis=2) # h x w x c
    v = np.repeat(v[:, :, np.newaxis], chann, axis=2) # h x w x c

    u_theta = u / width * 2 * np.pi
    v_phi = (0.5 - v / height) * np.pi

    x = np.cos(u_theta) * np.cos(v_phi)
    y = np.sin(v_phi)
    z = np.sin(u_theta) * np.cos(v_phi)


    # Reconstruct the envmap using stored coefficients
    coeff = coeff.transpose() # 9 x c

    value_00 = C[0] * coeff[0]

    value_1_1 = C[1] * y * coeff[1]
    value_10 = C[1] * z * coeff[2]
    value_11 = C[1] * x * coeff[3]

    value_2_2 = C[2] * x * y * coeff[4]
    value_2_1 = C[2] * y * z * coeff[5]
    value_20 = C[3] * (3 * z * z - 1) * coeff[6]
    value_21 = C[2] * x * z * coeff[7]
    value_22 = C[4] * (x * x - y * y) * coeff[8]

    value = value_00 + value_1_1 + value_10 + value_11 + value_2_2 + value_2_1 \
        + value_20 + value_21 + value_22

    return value


def envmap_to_directional(array):
    '''
    Compute the directional light directions and intensities from the hdr image,
    by computing the integral in each infinitesimal spherical area.
       y
       |
       |
       |
       /--------- x
      /
     /    u_theta
    z
    U wrapping: from x to z then -x then -z.
    V wrapping: from -y to +y.

    :param array: h x w x c image.
    :return: (direction, intensity), where direction is a numpy array of
        3 x (h x w), and intensity is a numpy array of c x (h x w).
    '''

    # From u, v coordinates to x, y, z
    height, width, chann = array.shape
    v, u = np.mgrid[:height, :width].astype(np.float32) + 0.5 # Symmetrical

    u_theta = u / width * 2 * np.pi
    v_phi = (0.5 - v / height) * np.pi

    x = np.cos(u_theta) * np.cos(v_phi) # h x w
    y = np.sin(v_phi)                   # h x w
    z = np.sin(u_theta) * np.cos(v_phi) # h x w

    direction = np.stack((x, y, z), axis=0).reshape(3, height * width)

    d_area = np.cos(v_phi) * (2 * np.pi / width) * (np.pi / height) # dS = cos(phi) d_phi d_theta

    # Inside each area the light intensity is a considered constant
    intensity = (d_area[:, :, np.newaxis] * array).reshape(height * width, chann).transpose()

    return direction, intensity
