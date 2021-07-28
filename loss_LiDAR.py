import torch
import sys
import torch.nn.functional as F

def lossRenderAttack(outputPytorch, vertex, vertex_og, face, mu):
    face = face.long()
    loss_center = (vertex.mean(0) - vertex_og.mean(0)) ** 2
    loss_center = loss_center.sum()

    inv_res_x = 0.5 * float(512) / 60

    x_var = vertex[:, 0]
    y_var = vertex[:, 1]


    fx = torch.floor(x_var * 512.0 / 120 + 512.0 / 2).long()
    fy = torch.floor(y_var * 512.0 / 120 + 512.0 / 2).long()

    mask = torch.zeros((512, 512)).cuda().index_put((fx, fy), torch.ones(fx.shape).cuda())
    mask1 = torch.where(torch.mul(mask, outputPytorch[1]) >= 0.5, torch.ones_like(mask), torch.zeros_like(mask))

    loss_object = torch.sum(torch.mul(mask1, outputPytorch[2])) / (torch.sum(mask1 + 0.000000001))

    class_probs = (torch.mul(mask1, outputPytorch[5]).sum(2).sum(2) / torch.sum(mask1 + 0.000000001))[0]
    loss_class = class_probs[0] - class_probs[1]
    # print class_probs

    loss_distance_1 = torch.sum(torch.sqrt(torch.pow(vertex[:, 0] - vertex_og[:, 0] + sys.float_info.epsilon, 2) +
                                           torch.pow(vertex[:, 1] - vertex_og[:, 1] + sys.float_info.epsilon, 2) +
                                           torch.pow(vertex[:, 2] - vertex_og[:, 2] + sys.float_info.epsilon,
                                                     2)))  # + sys.float_info.epsilon to prevent zero gradient
    # print(vertex_og.shape)
    zmin = vertex_og[:, 2].min()
    loss_z = (vertex[:, 2].min() - zmin) ** 2

    def calc_dis(vertex):
        meshes = torch.nn.functional.embedding(face, vertex)
        edge_1 = meshes[:, 1] - meshes[:, 0]
        edge_2 = meshes[:, 2] - meshes[:, 0]
        edge_3 = meshes[:, 1] - meshes[:, 2]

        dis = torch.stack([torch.sqrt(torch.pow(edge_1[:, 0], 2) +
                                      torch.pow(edge_1[:, 1], 2) +
                                      torch.pow(edge_1[:, 2], 2)), torch.sqrt(torch.pow(edge_2[:, 0], 2) +
                                                                              torch.pow(edge_2[:, 1], 2) +
                                                                              torch.pow(edge_2[:, 2], 2)),
                           torch.sqrt(torch.pow(edge_3[:, 0], 2) +
                                      torch.pow(edge_3[:, 1], 2) +
                                      torch.pow(edge_3[:, 2], 2))], dim=1)

        return dis

    dis = calc_dis(vertex)
    dis_og = calc_dis(vertex_og)

    loss_distance_2 = torch.sum((dis - dis_og) ** 2)

    beta = 0.5
    labda = 100.0

    # loss_distance = loss_distance_1 + beta * loss_distance_2
    loss_distance = loss_distance_2
    # loss = mu * loss_distance + loss_object
    loss_distance_ = F.relu(loss_distance - 1.0)
    # loss_distance = 0.0
    loss = 5.0 * loss_object + beta * loss_center + loss_z + loss_distance_2 * 0.2

    # loss_class += mu*loss_distance
    # loss_distance = loss_distance

    # return loss, loss_object, loss_distance,
    return loss, loss_object, loss_distance, loss_center, loss_z