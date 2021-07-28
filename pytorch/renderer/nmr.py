'''Neural Mesh Renderer helper functions.
Axes:

    y
    |
    |
    |  z
    | /
    |/
    O-------------- x
Angles:
    Elevation: -y -> +y, -90 -> 90
    Azimuth: from +x to +z to -x to -z to +x, 0 -> 360
'''
import torch
import math
import numpy as np


def combine_multiple_renderings(depths, masks, images):
    '''Combine renderings of multiple shapes using the depth
    and the rendered depths/sillhouettes/images
    :param depths: a list of tensors of shape b x h x w
    :param masks: a list of tensors of shape b x h x w
    :param images: a list of tensors of shape b x c x h x w
    '''


    all_depth = torch.cat([t.unsqueeze(-1) for t in depths], dim=-1) # b x h x w x n
    all_mask = torch.cat([t.unsqueeze(-1) for t in masks], dim=-1) # b x h x w x n
    all_image = torch.cat([t.unsqueeze(-1) for t in images], dim=-1) # b x 3 x h x w x n

    combined_mask, _ = all_mask.max(dim=-1) # b x h x w
    combined_depth, indices = all_depth.min(dim=-1) # b x h x w, b x h x w
    indices_3 = indices.unsqueeze(1).unsqueeze(-1).expand([-1, 3, -1, -1, -1]) # b x 3 x h x w x 1

    # out(b, c, i, j, k) = in(b, c, i, j, index(b, c, i, j, k))
    combined_images = torch.gather(all_image, dim=-1, index=indices_3).squeeze(-1)

    return combined_depth, combined_mask, combined_images


def scale(points, sc):
    return points * sc


def translate(points, vec):
    '''
    :param points: n x 3
    :param vec: 3
    '''
    if not isinstance(vec, torch.Tensor):
        vec = points.new_tensor(vec)

    return points + vec.view(1, -1)


def rotate(points, axis, angle, norm_axis=True):
    '''
    :param points: n x 3
    :param axis: 3
    '''
    if not isinstance(axis, torch.Tensor):
        axis = points.new_tensor(axis)
    if not isinstance(angle, torch.Tensor):
        angle = points.new_tensor(angle)

    th = angle / 180 * math.pi
    if norm_axis:
        axis = axis / (axis * axis).sum().sqrt()
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]
    cth = torch.cos(th)
    sth = torch.sin(th)
    r11 = cth + ux * ux * (1 - cth)
    r12 = ux * uy * (1 - cth) - uz * sth
    r13 = ux * uz * (1 - cth) + uy * sth
    r21 = uy * ux * (1 - cth) + uz * sth
    r22 = cth + uy * uy * (1 - cth)
    r23 = uy * uz * (1 - cth) - ux * sth
    r31 = uz * ux * (1 - cth) - uy * sth
    r32 = uz * uy * (1 - cth) + ux * sth
    r33 = cth + uz * uz * (1 - cth)

    rot_t = torch.stack([
            torch.stack([r11, r21, r31]),
            torch.stack([r12, r22, r32]),
            torch.stack([r13, r23, r33])
        ], dim=0)


    return np.dot(points, rot_t)
        # points @ rot_t


def lighting_from_envmap(array):
    '''
    Compute the sun directional light direction and intensity from the hdr image,
    and the ambience light color.
       y
       |
       |
       |  z
       | /
       |/
       O--------- x
    U wrapping: from -x to -z then x then z.
    V wrapping: from -y to +y.

    :param array: h x w x c image.
    :return: (direction, direction_color, ambient_color),
    '''

    # From u, v coordinates to x, y, z
    height, width, chann = array.shape

    v, u, _ = np.unravel_index(array.argmax(), array.shape)
    sun_color = array[v, u]
    sky_color = np.mean(array[:, (u+width//2) % width, :], axis=0)
    sun_color /= sun_color.max()

    u_theta = (u / width - 0.5) * 2 * np.pi
    v_phi = (0.5 - v / height) * np.pi

    x = np.cos(u_theta) * np.cos(v_phi) # h x w
    y = np.sin(v_phi)                   # h x w
    z = np.sin(u_theta) * np.cos(v_phi) # h x w

    direction = [x, y, z]

    return direction, sun_color, sky_color



def sample_image_from_envmap(array, image_size, azimuth, fov=60):
    '''
       y
       |
       |
       |  z
       | /
       |/
       O--------- x
    U wrapping: from x to -z then -x then z, and rotated by azimuth degrees.
    V wrapping: from -y to +y.
    azimuth or theta: from +x to +z then -x then -z
    phi: from -y to +y
    :return: pespective_image, light_array
    '''
    height, width, _ = array.shape

    eye_depth = 1 / math.tan(math.radians(fov/2))
    y, x = np.mgrid[:image_size, :image_size].astype(np.float32) + 0.5 # Symmetrical
    y = (0.5 - y / image_size) * 2
    x = (x / image_size - 0.5) * 2
    z = eye_depth * np.ones_like(x)

    u_theta = np.arctan2(-z, x) + np.radians(azimuth)
    v_phi = np.arctan2(y, np.sqrt(x*x + z*z))
    u = (u_theta / np.pi / 2 * width).astype(np.int32) % width
    v = np.clip(((0.5 - v_phi / np.pi) * height).astype(np.int32), 0, height-1)

    ret_0 = array[v, u, 0]
    ret_1 = array[v, u, 1]
    ret_2 = array[v, u, 2]

    image = np.stack((ret_0, ret_1, ret_2), 2)

    return image


def sample_max_light_from_envmap(array, light_size, azimuth, fov=60, debug=False):
    '''
       y
       |
       |
       |  z
       | /
       |/
       O--------- x
    U wrapping: from x to -z then -x then z.
    V wrapping: from -y to +y.
    azimuth or theta: from +x to +z then -x then -z
    theta: from +x to -z then -x then +z
    phi: from -y to +y
    :return: pespective_image, light_array
    '''
    height, width, _ = array.shape

    v, u = np.mgrid[:height, :width].astype(np.float32) + 0.5 # Symmetrical
    v_phi = (0.5 - v / height) * np.pi
    u_theta = -u / width * 2 * np.pi

    d_area = np.cos(v_phi) / (height * width)
    # weight = np.sqrt(array.sum(2))
    weight = array.sum(2) ** 2
    weight[v_phi < 0] = 0

    prob_each_grid = weight * d_area
    prob_flatten = prob_each_grid.flatten() / prob_each_grid.sum()

    # DEBUG: uncomment
    # indices = list(np.ndindex((height, width)))
    # sampled_indices = np.random.choice(list(range(height * width)), size=light_size, p=prob_flatten)
    # sampled = [indices[s] for s in sampled_indices]

    # DEBUG: remove
    s0, s1, _ = np.unravel_index(array.argmax(), array.shape)
    sampled = [(s0, s1)] * light_size


    light_color = []
    light_direction = []

    sum_ratio = 0
    for s in sampled:
        light_color.append(array[s[0], s[1], :] / weight[s[0], s[1]])
        sum_ratio += 1 / weight[s[0], s[1]]
        theta = -s[1] / width * 2 * np.pi - np.radians(azimuth)
        phi = (0.5 - s[0] / height) * np.pi

        x = np.cos(theta) * np.cos(phi)
        y = np.sin(phi)
        z = -np.sin(theta) * np.cos(phi)
        light_direction.append([x, y, z])

    if debug:
        from skimage.draw import circle
        annotated = array.copy()
        rr, cc = circle(s0, s1, 20, array.shape[:2])
        annotated[rr, cc, :] = np.array([[[0, 0, 1]]])

        return np.array(light_direction), np.array(light_color) / sum_ratio, annotated

    return np.array(light_direction), np.array(light_color) / sum_ratio



def sample_light_from_envmap(array, light_size, azimuth, fov=60):
    '''
       y
       |
       |
       |  z
       | /
       |/
       O--------- x
    U wrapping: from x to -z then -x then z.
    V wrapping: from -y to +y.
    azimuth or theta: from +x to +z then -x then -z
    theta: from +x to -z then -x then +z
    phi: from -y to +y
    :return: pespective_image, light_array
    '''
    height, width, _ = array.shape

    v, u = np.mgrid[:height, :width].astype(np.float32) + 0.5 # Symmetrical
    v_phi = (0.5 - v / height) * np.pi
    u_theta = -u / width * 2 * np.pi

    d_area = np.cos(v_phi) / (height * width)
    # weight = np.sqrt(array.sum(2))
    weight = array.sum(2) ** 2
    weight[v_phi < 0] = 0

    prob_each_grid = weight * d_area
    prob_flatten = prob_each_grid.flatten() / prob_each_grid.sum()

    # DEBUG: uncomment
    indices = list(np.ndindex((height, width)))
    sampled_indices = np.random.choice(list(range(height * width)), size=light_size, p=prob_flatten)
    sampled = [indices[s] for s in sampled_indices]

    # DEBUG: remove
    # s0, s1, _ = np.unravel_index(array.argmax(), array.shape)
    # sampled = [(s0, s1)] * light_size


    light_color = []
    light_direction = []

    sum_ratio = 0
    for s in sampled:
        light_color.append(array[s[0], s[1], :] / weight[s[0], s[1]])
        sum_ratio += 1 / weight[s[0], s[1]]
        theta = -s[1] / width * 2 * np.pi - np.radians(azimuth)
        phi = (0.5 - s[0] / height) * np.pi

        x = np.cos(theta) * np.cos(phi)
        y = np.sin(phi)
        z = -np.sin(theta) * np.cos(phi)
        light_direction.append([x, y, z])

    return np.array(light_direction), np.array(light_color) / sum_ratio


def ambient_from_envmap(array):
    '''
       y
       |
       |
       |  z
       | /
       |/
       O--------- x
    U wrapping: from x to -z then -x then z.
    V wrapping: from -y to +y.
    azimuth or theta: from +x to +z then -x then -z
    theta: from +x to -z then -x then +z
    phi: from -y to +y
    :return: pespective_image, light_array
    '''
    height, width, _ = array.shape

    v, u = np.mgrid[:height, :width].astype(np.float32) + 0.5 # Symmetrical
    v_phi = (0.5 - v / height) * np.pi
    u_theta = -u / width * 2 * np.pi

    d_area = np.cos(v_phi) / (height * width)
    all_area = d_area.sum()
    summed = array * np.stack((d_area,)*3, 2)
    return summed.sum(0).sum(0) / all_area


def save_obj(filename, vertices, faces):
    if isinstance(vertices, torch.Tensor):
        assert vertices.ndimension() == 2
        assert vertices.size(1) == 3

        assert isinstance(faces, torch.Tensor)
        assert faces.ndimension() == 2
        assert faces.size(1) == 3

        vertices = vertices.tolist()
        faces = faces.tolist()


    with open(filename, 'w') as f:
        f.write('# OBJ file\n')
        for v in vertices:
            f.write("v {} {} {}\n".format(*v))

        for p in faces:
            f.write('f ')
            for i in p:
                f.write('{:d} '.format(i + 1))
            f.write('\n')
