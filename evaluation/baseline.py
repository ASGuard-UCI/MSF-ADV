
import neural_renderer as nr
from pytorch.renderer import nmr
import torch
import torch.autograd as autograd
import argparse
import cv2
from c2p_segmentation import *
import loss_LiDAR
import numpy as np
import cluster
import os
from xyz2grid import *
import render
from plyfile import *
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from pytorch.yolo_models.utils_yolo import *
from pytorch.yolo_models.darknet import Darknet
from scipy.optimize import differential_evolution

def read_cali(path):
    file1 = open(path, 'r')
    Lines = file1.readlines()
    for line in Lines:
        if 'R:' in line:
            rotation = line.split('R:')[-1]
        if 'T:' in line:
            translation = line.split('T:')[-1]
    tmp_r = rotation.split(' ')
    tmp_r.pop(0)
    tmp_r[-1] = tmp_r[-1].split('\n')[0]
    rota_matrix = []

    for i in range(3):
        tt = []
        for j in range(3):
            tt.append(float(tmp_r[i * 3 + j]))
        rota_matrix.append(tt)
    rota_matrix = np.array(rota_matrix)
    tmp_t = translation.split(' ')
    tmp_t.pop(0)
    tmp_t[-1] = tmp_t[-1].split('\n')[0]
    trans_matrix = [float(tmp_t[i]) for i in range(3)]
    trans_matrix = np.array(trans_matrix)
    return rota_matrix, trans_matrix

def predict_convert(image_var, model, class_names, reverse=False):
    # pred = model.get_spec_layer( (image_var - mean_var ) / std_dv_var, 0).max(1)[1]
    pred, _ = model(image_var)
    # print(np.array(pred).shape)
    boxes = []
    img_vis = []
    pred_vis = []
    vis = []
    i = 0
    boxes.append(nms(pred[0][i] + pred[1][i] + pred[2][i], 0.4))
    img_vis.append((image_var[i].cpu().data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

    pred_vis.append(plot_boxes(Image.fromarray(img_vis[i]), boxes[i], class_names=class_names))
    vis = np.array(pred_vis[i][0])
    return np.array(vis), np.array(boxes)

class attack_msf():
    def __init__(self, args):
        self.args = args
        self.num_pos = 1
        self.threshold = 0.4
        self.root_path = './data/'
        self.pclpath = 'pcd/'
        self.rotation = torch.tensor(np.array([[1., 0., 0.],
                                               [0., 0., -1.],
                                               [0., 1., 0.]]), dtype=torch.float)
        self.protofile = self.root_path + 'deploy.prototxt'
        self.weightfile = self.root_path + 'deploy.caffemodel'
        self.outputs = ['instance_pt', 'category_score', 'confidence_score',
                   'height_pt', 'heading_pt', 'class_score']
        self.esp = args.epsilon
        self.direction_val, self.dist_val = self.load_const_features('./data/features_1.out')

    def init_render(self, image_size = 416):
        self.image_size = image_size
        self.renderer = nr.Renderer(image_size=image_size, camera_mode='look_at',
                                    anti_aliasing=False, light_direction=(0, 0, 0))
        exr = cv2.imread('./data/dog.exr', cv2.IMREAD_UNCHANGED)
        self.renderer.light_direction = [1, 3, 1]

        ld, lc, ac = nmr.lighting_from_envmap(exr)
        self.renderer.light_direction = ld
        self.renderer.light_color = lc
        self.renderer.ambient_color = ac

    def load_const_features(self, fname):

        print("Loading dircetion, dist")
        features_filename = fname

        features = np.loadtxt(features_filename)
        features = np.swapaxes(features, 0, 1)
        features = np.reshape(features, (1, 512, 512, 8))

        direction = np.reshape(features[:, :, :, 3], (1, 512, 512, 1))
        dist = np.reshape(features[:, :, :, 6], (1, 512, 512, 1))
        return torch.tensor(direction).cuda().float(), torch.tensor(dist).cuda().float()

    def model_val_lidar(self, protofile, weightfile):
        net = CaffeNet(protofile, phase='TEST')
        # torch.cuda.set_device(0)
        net.cuda()
        net.load_weights(weightfile)
        net.set_train_outputs(outputs)
        net.set_eval_outputs(outputs)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        return net

    def load_LiDAR_model(self, ):
        self.LiDAR_model = generatePytorch(self.protofile, self.weightfile).cuda()
        self.LiDAR_model_val = self.model_val_lidar(self.protofile, self.weightfile)

    def load_model_(self):

        namesfile = './pytorch/yolo_models/data_yolo/coco.names'
        class_names = load_class_names(namesfile)
        single_model = Darknet('./pytorch/yolo_models/cfg/yolov3.cfg')
        single_model.load_weights('./data/yolov3.weights')
        model = single_model
        self.model = model.cuda()
        self.model.eval()

    def load_pc_mesh(self, path):
        PCL_path = path

        # loading ray_direction and distance for the background pcd
        self.PCL = loadPCL(PCL_path, True)
        x_final = torch.FloatTensor(self.PCL[:, 0]).cuda()
        y_final = torch.FloatTensor(self.PCL[:, 1]).cuda()
        z_final = torch.FloatTensor(self.PCL[:, 2]).cuda()
        self.i_final = torch.FloatTensor(self.PCL[:, 3]).cuda()
        self.ray_direction, self.length = render.get_ray(x_final, y_final, z_final)

    def load_mesh(self, path, r, x_of=7, y_of=0):
        z_of = -1.73 + r / 2.
        plydata = PlyData.read(path)
        x = torch.FloatTensor(plydata['vertex']['x']) * r
        y = torch.FloatTensor(plydata['vertex']['y']) * r
        z = torch.FloatTensor(plydata['vertex']['z']) * r
        self.object_v = torch.stack([x, y, z], dim=1).cuda()

        self.object_f = plydata['face'].data['vertex_indices']
        self.object_f = torch.tensor(np.vstack(self.object_f)).cuda()

        rotation = lidar_rotation.cuda()
        self.object_v = self.object_v.cuda()
        self.object_v = self.object_v.permute(1, 0)
        self.object_v = torch.matmul(rotation, self.object_v)
        self.object_v = self.object_v.permute(1, 0)
        self.object_v[:, 0] += x_of
        self.object_v[:, 1] += y_of
        self.object_v[:, 2] += z_of

        self.object_ori = self.object_v.clone()

        camera_v = self.object_v.clone()
        camera_v = camera_v.permute(1, 0)
        r, t = torch.tensor(self.rota_matrix).cuda().float(), torch.tensor(self.trans_matrix).cuda().float()
        r_c = R.from_euler('zxy', [0, 180, 180], degrees=True)
        camera_rotation = torch.tensor(r_c.as_matrix(), dtype=torch.float).cuda()
        camera_v = torch.matmul(camera_rotation, camera_v)
        camera_v = torch.matmul(r, camera_v)
        camera_v = camera_v.permute(1, 0)

        camera_v += t

        c_v_c = camera_v.cuda()

        self.vn, idxs = self.set_neighbor_graph(self.object_f, c_v_c)
        self.vn_tensor = torch.Tensor(self.vn).view(-1).cuda().long()
        self.idxs_tensor = torch.Tensor(idxs.copy()).cuda().long()

        self.object_t = torch.tensor(self.object_v.new_ones(self.object_f.shape[0], 1, 1, 1, 3)).cuda()
        # color red
        self.object_t[:, :, :, :, 1] = 0.3
        self.object_t[:, :, :, :, 2] = 0.3
        self.object_t[:, :, :, :, 0] = 0.3
        self.mean_gt = self.object_ori.mean(0).data.cpu().clone().numpy()
        return self.object_v


    def set_learning_rate(self, optimizer, learning_rate):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    def tv_loss_(self, image, ori_image):
        noise = image - ori_image
        loss = torch.mean(torch.abs(noise[:, :, :, :-1] - noise[:, :, :, 1:])) + torch.mean(
            torch.abs(noise[:, :, :-1, :] - noise[:, :, 1:, :]))
        return loss

    def load_bg(self, path, h=416, w=416):
        background = cv2.imread(path)
        background = cv2.resize(background, (h, w))
        background = background[:, :, ::-1] / 255.0
        self.background = background.astype(np.float32)

    def compute_total_variation_loss(self, img1, img2):
        diff = img1 - img2
        tv_h = ((diff[:,:,1:,:] - diff[:,:,:-1,:]).pow(2)).sum()
        tv_w = ((diff[:,1:,:,:] - diff[:,:-1,:,:]).pow(2)).sum()
        return tv_h + tv_w

    def l2_loss(self, desk_t, desk_v, ori_desk_t, ori_desk_v):
        t_loss = torch.nn.functional.mse_loss(desk_t, ori_desk_t)
        v_loss = torch.nn.functional.mse_loss(desk_v, ori_desk_v)
        return v_loss, t_loss

    def rendering_img(self, xxx):
        self.load_mesh(args.object, r=0.15)
        xxx = torch.tensor(xxx.astype(np.float32)).cuda()
        self.object_v += xxx.view(-1, 3)

        u_offset = 0
        v_offset = 0

        num_class = 80
        threshold = 0.5
        batch_size = 1
        l_c_c_ori = self.object_ori
        with torch.no_grad():
            ray_direction = self.ray_direction.data.cuda()
            length = self.length.data.cuda()
            i_final, vertex, vertex_orig, face, pytorchModel = self.i_final.data.cuda, self.object_v.data.cuda(), l_c_c_ori.data.cuda, self.cone_f.data.cuda(), self.LiDAR_model.cuda()
            point_cloud = render.render(ray_direction, length, vertex, face,
                                                       i_final)
            grid_hard = xyzi2gridhard(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], point_cloud[:, 3])

            featureM = gridi2feature_v2(grid_hard)

            outputPytorch = pytorchModel(featureM)
            lossValue, loss_object, loss_distance, loss_center, loss_z = loss_LiDAR.lossRenderAttack(outputPytorch, vertex, vertex_orig, face,
                                                               0.05).data.cpu().numpy()

            camera_v = self.object_v.clone()
            camera_v = camera_v.permute(1, 0)

            r, t = torch.tensor(self.rota_matrix).cuda().float(), torch.tensor(self.trans_matrix).cuda().float()
            r_c = R.from_euler('zxy', [0, 180, 180], degrees=True)
            camera_v = torch.matmul(r, camera_v)
            camera_v = camera_v.permute(1, 0)
            camera_v = camera_v.permute(1, 0)
            camera_rotation = torch.tensor(r_c.as_matrix(), dtype=torch.float).cuda()
            camera_v = torch.matmul(camera_rotation, camera_v)
            camera_v = camera_v.permute(1, 0)
            camera_v += t

            c_v_c = camera_v.cuda()
            image_tensor = self.renderer.render(c_v_c.unsqueeze(0), self.object_f.unsqueeze(0), self.object_t.unsqueeze(0))[0].cuda()
            mask_tensor = self.renderer.render_silhouettes(c_v_c.unsqueeze(0), self.object_f.unsqueeze(0)).cuda()
            background_tensor = torch.from_numpy(self.background.transpose(2, 0, 1)).cuda()
            fg_mask_tensor = torch.zeros(background_tensor.size())
            new_mask_tensor = mask_tensor.repeat(3, 1, 1)
            fg_mask_tensor[:, u_offset: u_offset + self.image_size,
            v_offset: v_offset + self.image_size] = new_mask_tensor
            fg_mask_tensor = fg_mask_tensor.byte().cuda()
            new_mask_tensor = new_mask_tensor.byte().cuda()

            background_tensor.masked_scatter_(fg_mask_tensor, image_tensor.masked_select(new_mask_tensor))

            final_image = torch.clamp(background_tensor.float(), 0, 1)[None]

            final, outputs = self.model(final_image)

            num_pred = 0.0
            removed = 0.0
            for index, out in enumerate(outputs):
                num_anchor = out.shape[1] // (num_class + 5)
                out = out.view(batch_size * num_anchor, num_class + 5, out.shape[2], out.shape[3])
                cfs = torch.nn.functional.sigmoid(out[:, 4]).cuda()
                mask = (cfs >= threshold).type(torch.FloatTensor).cuda()
                num_pred += torch.numel(cfs)
                removed += torch.sum((cfs < threshold).type(torch.FloatTensor)).data.cpu().numpy()

                loss = torch.sum(mask * ((cfs - 0) ** 2 - (1 - cfs) ** 2))

                if adv_total_loss is None:
                    adv_total_loss = loss
                else:
                    adv_total_loss += loss
            total_loss = 12 * (F.relu(torch.clamp(adv_total_loss, min=0) - 0.01) / 5.0)
            total_loss += lossValue
        return total_loss.data.cpu().numpy()


    def savemesh(self, path_r, path_w, vet, r):
        plydata = PlyData.read(path_r)
        plydata['vertex']['x'] = vet[:, 0] / r
        plydata['vertex']['y'] = vet[:, 1] / r
        plydata['vertex']['z'] = vet[:, 2] / r

        plydata.write(path_w)
        return

    def set_neighbor_graph(self, f, vn, degree=1):
        max_len = 0
        face = f.cpu().data.numpy()
        vn = vn.data.cpu().tolist()
        for i in range(len(face)):
            v1, v2, v3 = face[i]
            for v in [v1, v2, v3]:
                vn[v].append(v2)
                vn[v].append(v3)
                vn[v].append(v1)

        # two degree
        for i in range(len(vn)):
            vn[i] = list(set(vn[i]))
        for de in range(degree - 1):
            vn2 = [[] for _ in range(len(vn))]
            for i in range(len(vn)):
                for item in vn[i]:
                    vn2[i].extend(vn[item])

            for i in range(len(vn2)):
                vn2[i] = list(set(vn2[i]))
            vn = vn2
        max_len = 0
        len_matrix = []
        for i in range(len(vn)):
            vn[i] = list(set(vn[i]))
            len_matrix.append(len(vn[i]))

        idxs = np.argsort(len_matrix)[::-1][:len(len_matrix) // 1]
        max_len = len_matrix[idxs[0]]
        print("max_len: ", max_len)

        vns = np.zeros((len(idxs), max_len))
        # for i in range( len(vn)):
        for i0, i in enumerate(idxs):
            for j in range(max_len):
                if j < len(vn[i]):
                    vns[i0, j] = vn[i][j]
                else:
                    vns[i0, j] = i
        return vns, idxs

    def read_cali(self, path):
        file1 = open(path, 'r')
        Lines = file1.readlines()
        for line in Lines:
            if 'R:' in line:
                rotation = line.split('R:')[-1]
            if 'T:' in line:
                translation = line.split('T:')[-1]
        tmp_r = rotation.split(' ')
        tmp_r.pop(0)
        tmp_r[-1] = tmp_r[-1].split('\n')[0]
        # print(tmp_r)
        rota_matrix = []

        for i in range(3):
            tt = []
            for j in range(3):
                tt.append(float(tmp_r[i * 3 + j]))
            rota_matrix.append(tt)
        self.rota_matrix = np.array(rota_matrix)
        tmp_t = translation.split(' ')
        tmp_t.pop(0)
        tmp_t[-1] = tmp_t[-1].split('\n')[0]
        # print(tmp_t)
        trans_matrix = [float(tmp_t[i]) for i in range(3)]
        self.trans_matrix = np.array(trans_matrix)

    def random_obj(self, object_v):
        self.x_r = random.uniform(5.0, 35.0)
        self.y_r = random.uniform(-0.3, 0.3)
        self.j_r = random.uniform(-5.0, 5.0)
        object_v[:, 0] += self.x_r
        object_v[:, 1] += self.y_r
        object_v = object_v.permute(1, 0)
        r_c = R.from_euler('zxy', [self.j_r, 0, 0], degrees=True)
        rota = torch.tensor(r_c.as_matrix(), dtype=torch.float).cuda()
        object_v = torch.matmul(rota, object_v)
        object_v = torch.matmul(r, object_v)
        object_v = object_v.permute(1, 0)
        return object_v

    def de_random_obj(self, object_v):
        object_v[:, 0] -= self.x_r
        object_v[:, 1] -= self.y_r
        object_v = object_v.permute(1, 0)
        r_c = R.from_euler('zxy', [-self.j_r, 0, 0], degrees=True)
        rota = torch.tensor(r_c.as_matrix(), dtype=torch.float).cuda()
        object_v = torch.matmul(rota, object_v)
        object_v = torch.matmul(r, object_v)
        object_v = object_v.permute(1, 0)
        return object_v


r = R.from_euler('zxy', [10,80,4], degrees=True)
lidar_rotation = torch.tensor(r.as_matrix(), dtype=torch.float).cuda()

if __name__ == '__main__':
    import timeit
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj' '--obj', dest='object', default="./object/object.ply")
    parser.add_argument('-obj_save' '--obj_save', dest='object_save', default="./object/obj_save")
    parser.add_argument('-lidar' '--lidar', dest='lidar')
    parser.add_argument('-cam' '--cam', dest='cam')
    parser.add_argument('-cali' '--cali', dest='cali')
    parser.add_argument('-o' '--opt', dest='opt', default="pgd")
    parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.2)
    parser.add_argument('-it', '--iteration', dest='iteration', type=int, default=1000)
    args = parser.parse_args()

    obj = attack_msf(args)
    obj.load_model_()
    obj.load_LiDAR_model()
    obj.read_cali(args.cali)
    obj_v = obj.load_mesh(args.object, 0.15)
    obj.load_bg(args.cam)
    obj.init_render()
    obj.load_pc_mesh(args.lidar)
    # obj.rendering_img(args.lidar)
    bound = []
    final_res = []
    final_time = []
    final_score = []
    for rr_bound in range(obj_v.shape[0] * 3):
        bound.append((-0.2, 0.2))
    start = timeit.default_timer()

    result = differential_evolution(obj.rendering_img, bound, maxiter=1000, popsize=3, workers=1, disp=True,
                                    updating='deferred')
    stop = timeit.default_timer()
    final_res.append(result.x)
    final_time.append(stop - start)
    final_score.append(result.fun)
    np.savez('./baseline_evo.npz', result_x=np.array(final_res),
             result_fun=np.array(final_score), result_time=np.array(final_time))
