from __future__ import division
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import sys


def get_ray(x_final, y_final, z_final):
    length = torch.sqrt(torch.pow(x_final, 2) + torch.pow(y_final, 2) + torch.pow(z_final, 2))
    ray_direction = torch.stack([torch.div(x_final, length), torch.div(y_final, length), torch.div(z_final, length)],
                                dim=1)

    return ray_direction, length


def render(ray_direction, length, vertex, face, i_final):
    max_dist = 70
    vertex = vertex.cuda()
    face = face.cuda()
    i_final = i_final.cuda()
    mask = filtermask(ray_direction, vertex)
    mask_mesh = (mask == 1).flatten()
    mask_bg = (mask == 0).flatten()
    mesh_ray_direction = ray_direction[mask_mesh]
    mesh_i = i_final[mask_mesh]
    mesh_ray_length = length[mask_mesh]
    # print(mesh_ray_length.shape)
    mesh_ray_distance, mesh_ray_intersection = render_sub(mesh_ray_direction, mesh_ray_length, vertex, face)
    mesh_ray_distance = mesh_ray_distance.cuda()
    mesh_ray_intersection = mesh_ray_intersection.cuda()
    bg_ray_distance = length[mask_bg]
    bg_ray_intersection = (ray_direction[mask_bg].t() * length[mask_bg]).t()
    bg_i = i_final[mask_bg]
    ray_intersection = torch.cat([mesh_ray_intersection, bg_ray_intersection], dim=0)

    condition = torch.eq(mesh_ray_distance, mesh_ray_length)
    i = torch.where(condition, mesh_i, torch.ones_like(mesh_i) * 144. / 255.)
    ii = torch.cat([i, bg_i], dim=0)

    point_cloud = torch.cat([ray_intersection, ii.reshape(-1, 1)], dim=1)

    return point_cloud

def render_div(ray_direction, length, vertex, face, i_final):
    max_dist = 70
    # print(length.shape)
    vertex = vertex.cuda()
    face = face.cuda()
    i_final = i_final.cuda()
    mask = filtermask(ray_direction, vertex)
    mask_mesh = (mask == 1).flatten()
    mask_bg = (mask == 0).flatten()
    mesh_ray_direction = ray_direction[mask_mesh]
    mesh_i = i_final[mask_mesh]
    mesh_ray_length = length[mask_mesh]
    # print(mesh_ray_length.shape)
    mesh_ray_distance, mesh_ray_intersection = render_sub(mesh_ray_direction, mesh_ray_length, vertex, face)
    mesh_ray_distance = mesh_ray_distance.cuda()
    mesh_ray_intersection = mesh_ray_intersection.cuda()

    bg_ray_distance = length[mask_bg]
    bg_ray_intersection = (ray_direction[mask_bg].t() * length[mask_bg]).t()
    bg_i = i_final[mask_bg]
    whole_ray_intersection = (ray_direction.t() * length).t()
    whole_i = i_final

    ray_intersection = torch.cat([mesh_ray_intersection, bg_ray_intersection], dim=0)

    condition = torch.eq(mesh_ray_distance, mesh_ray_length)
    i = torch.where(condition, mesh_i, torch.ones_like(mesh_i) * 144. / 255.)
    ii = torch.cat([i, bg_i], dim=0)


    point_cloud_mesh = torch.cat([mesh_ray_intersection, i.reshape(-1, 1)], dim=1)
    point_cloud = torch.cat([whole_ray_intersection, whole_i.reshape(-1, 1)], dim=1)
    point_cloud[mask_mesh] = point_cloud_mesh
    return point_cloud.data.cpu(), mask_bg.data.cpu(), mask_mesh.data.cpu(), len(ray_intersection)

def render_sub(ray_direction, length, vertex, face):
    ray_direction = ray_direction.cuda()
    length = length.cuda()
    vertex = vertex.cuda()
    face = face.cuda()
    face = face.long()
    epsilon = sys.float_info.epsilon
    max_dist = 70
    lidar_origin = torch.zeros(3).cuda()
    n_rays = ray_direction.shape[0]

    meshes = torch.nn.functional.embedding(face, vertex)
    n_meshes = meshes.shape[0]
    edge_1 = meshes[:, 1] - meshes[:, 0]
    edge_2 = meshes[:, 2] - meshes[:, 0]
    origin_tile = torch.unsqueeze(lidar_origin, 0).repeat([n_meshes, 1])
    origin_tile = origin_tile.cuda()
    meshes = meshes.cuda()
    T = origin_tile - meshes[:, 0]
    rays = torch.reshape(ray_direction, [-1, 1, 3]).repeat([1, n_meshes, 1])
    edge_1 = torch.reshape(edge_1, [1, -1, 3]).repeat([n_rays, 1, 1])
    edge_2 = torch.reshape(edge_2, [1, -1, 3]).repeat([n_rays, 1, 1])

    T = torch.reshape(T, [1, -1, 3]).repeat([n_rays, 1, 1])
    P = torch.cross(rays, edge_2)

    det = torch.sum(torch.mul(edge_1, P), 2)
    det_sign = torch.sign(det)
    det_sign_tile = torch.reshape(det_sign, [n_rays, n_meshes, 1]).repeat([1, 1, 3])
    det_sign_tile = det_sign_tile.cuda()
    T = T.cuda()
    T = torch.mul(T, det_sign_tile)
    det = torch.abs(det)
    P = P.cuda()
    u = torch.sum(torch.mul(T, P), 2)
    Q = torch.cross(T, edge_1)
    v = torch.sum(torch.mul(rays, Q), 2)
    t = torch.sum(torch.mul(edge_2, Q), 2)
    t /= (det + epsilon)
    t = torch.where(det < epsilon, torch.ones_like(t) * max_dist, t)
    t = torch.where(u < 0, torch.ones_like(t) * max_dist, t)
    t = torch.where(u > det, torch.ones_like(t) * max_dist, t)
    t = torch.where(v < 0, torch.ones_like(t) * max_dist, t)
    t = torch.where(u + v > det, torch.ones_like(t) * max_dist, t)
    t = torch.where(t < 0, torch.ones_like(t) * max_dist, t)
    min_t, _ = torch.min(t, dim=1)  # here!
    bg_dist = length
    min_t, _ = torch.min(torch.stack([min_t, bg_dist], 0), 0)
    ray_distance = min_t
    min_t_tile = torch.unsqueeze(min_t, 1).repeat([1, 3])
    origin_tile = torch.unsqueeze(lidar_origin, 0).repeat([n_rays, 1])
    ray_intesction = origin_tile + min_t_tile * ray_direction
    return ray_distance, ray_intesction


def loadmesh(path, x_of=9, y_of=0, r=0.5):
    r = 0.3
    plydata = PlyData.read(path)
    z_of = -1.73 + r / 2.
    from scipy.spatial.transform import Rotation as R
    x = torch.FloatTensor(plydata['vertex']['x']).cuda() * r
    y = torch.FloatTensor(plydata['vertex']['y']).cuda() * r
    z = torch.FloatTensor(plydata['vertex']['z']).cuda() * r
    vertex = torch.stack([x, y, z], dim=1)
    r = R.from_euler('zxy', [0, 90, 0], degrees=True)
    rotation = torch.tensor(r.as_matrix(), dtype=torch.float)
    rotation = rotation.cuda()
    vertex = vertex.cuda()
    # print(c_v_c)
    vertex = vertex.permute(1, 0)
    # print(self.rotation.shape)
    vertex = torch.matmul(rotation, vertex)
    vertex = vertex.permute(1, 0)
    vertex[:,0] = vertex[:,0] + x_of
    vertex[:, 1] = vertex[:, 1] + y_of
    vertex[:, 2] = vertex[:, 2] + z_of

    face = plydata['face'].data['vertex_indices']
    face = torch.Tensor(np.vstack(face)).cuda().long()
    return vertex, face
def savemesh(path_r, path_w, vet, face, x_of=5, y_of=0, r=0.5):
    plydata = PlyData.read(path_r)
    z_of = -1.73 + r / 2.
    plydata['vertex']['x'] = (vet[:, 0] - x_of) / r
    plydata['vertex']['y'] = (vet[:, 1] - y_of) / r
    plydata['vertex']['z'] = (vet[:, 2] - z_of) / r
    plydata.write(path_w)
    return


def local_translate(vertices, vector):
    return vertices + torch.reshape(vector, (1, 3))


def filtermask(xyz, vertex):
    ptsnew = torch.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 0] = torch.sqrt(xy + xyz[:, 2] ** 2)
    ptsnew[:, 1] = torch.atan2(xyz[:, 2], torch.sqrt(xy))  # for elevation angle defined from XY-plane up
    ptsnew[:, 2] = torch.atan2(xyz[:, 1], xyz[:, 0])

    with torch.no_grad():
        vertexnew = torch.zeros(vertex.shape)
        xy = vertex[:, 0] ** 2 + vertex[:, 1] ** 2
        vertexnew[:, 0] = torch.sqrt(xy + vertex[:, 2] ** 2)
        vertexnew[:, 1] = torch.atan2(vertex[:, 2], torch.sqrt(xy))  # for elevation angle defined from XY-plane up
        vertexnew[:, 2] = torch.atan2(vertex[:, 1], vertex[:, 0])

        min_v = torch.min(vertexnew[:, 1]) - np.pi / 18 / 3
        max_v = torch.max(vertexnew[:, 1]) + np.pi / 18 / 3
        min_h = torch.min(vertexnew[:, 2]) - np.pi / 18 / 3
        max_h = torch.max(vertexnew[:, 2]) + np.pi / 18 / 3

        mask = torch.gt(ptsnew[:, 2], min_h)
        mask = torch.lt(ptsnew[:, 2], max_h) * mask
        mask = torch.gt(ptsnew[:, 1], min_v) * mask
        mask = torch.lt(ptsnew[:, 1], max_v) * mask

    return mask