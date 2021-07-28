from utils_yolo import *
from cfg_yolo import parse_cfg
from region_loss import RegionLoss
from darknet import Darknet


import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import Function
import math
from torchvision import transforms
import logging
from segment import parse_args, DRNSeg, SegList, SegListSeg
import os.path as osp
import json
import numpy as np
import drn
from PIL import Image
from PIL import ImageFilter
from random import randint
from pdb import set_trace as st
from models import FlowNet2CSS
import pose.models as models

from skimage import io, morphology, transform
from skimage.transform import rescale, resize, downscale_local_mean

from networks.resample2d_package.modules.resample2d import Resample2d
from skimage import io
import os
import sys
import pyflow

np.random.seed(0)
CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)

def convert_to_coordinate(array):
    ## Assume that array has size Batch x num_class x side_len x side_len
    coords = np.zeros(list(array.shape[:2])+[2])
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            coords[i, j] = np.unravel_index(array[i, j].argmax(), array.shape[2:])
    return coords


## First version of test pose
def vis_pose(pred_array):
    # Assume pred_array is of size 16*64*64
    size = pred_array.shape[1:]
    canvas = np.zeros(list(size)+[3])
    coords = convert_to_coordinate(pred_array[np.newaxis])[0]
    for i in range(pred_array.shape[0]):
        canvas[coords[i].astype(np.int)[0], coords[i].astype(np.int)[1]] = CITYSCAPE_PALETTE[i]
    canvas = transform.rescale(canvas, 4)
    canvas = (canvas*255).astype(np.uint8)
    return canvas

def convert_img(img, noise=False):
    img2 = img.data.cpu().numpy()
    if img2.ndim == 4:
        img2 = img2[0]
    if noise:
        img2 = (img2 + 1 ) /2.0
    img2 = np.transpose(img2, [1, 2, 0]) * 255.0
    return img2


def cpu2var(cpu_data, requires_grad=False, volatile=False):
    Tensor = torch.cuda.FloatTensor
    input_tensor = Tensor(cpu_data.size())
    input_tensor.copy_(cpu_data)
    input_var = Variable(input_tensor, requires_grad=requires_grad, volatile=volatile)
    return input_var

def predict_convert(image_var, model, class_names, reverse=False):
    # pred = model.get_spec_layer( (image_var - mean_var ) / std_dv_var, 0).max(1)[1]
    pred,_ = model( image_var)
    
    boxes = nms(pred[0][0]+pred[1][0]+pred[2][0], 0.4)
    img_vis = (image_var.cpu().data.numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        
    pred_vis = plot_boxes(Image.fromarray(img_vis), boxes, 
                                     class_names=class_names)
    # pred_vis = vis_seg(pred)
    if reverse:
        vis = np.concatenate( [pred_vis, img_vis], axis=1)
    else:
        vis = np.concatenate( [img_vis, pred_vis], axis=1)
    return vis, boxes

def compute_metric(pred1, pred2):
    ious = []
    for i, box1 in enumerate( pred1):
        best_iou = 0
        for j, box2 in enumerate(pred2):
            iou = bbox_iou(box1, box2, x1y1x2y2=False)
            # if iou > best_iou and pred2[j][6] == box1[6]:
            if iou > best_iou :

                best_iou = iou
                best_j = j
        ious.append(best_iou)
    if len(ious) == 0:
        ious.append(0)
    return np.mean(ious)

# compute flow by pyflow 

def flow_warp_pyflow(args, adv_img2_var, adv_img_var1):
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    adv_img1 = adv_img_var1.data.cpu().numpy()[0]
    adv_img2 = adv_img2_var.data.cpu().numpy()[0]
    adv_img1 = adv_img1.transpose(1,2,0).astype(float)
    adv_img2 = adv_img2.transpose(1,2,0).astype(float)
    adv_img1 = np.ascontiguousarray(adv_img1, dtype=float)
    adv_img2 = np.ascontiguousarray(adv_img2, dtype=float)

    u, v, im2W = pyflow.coarse2fine_flow(
                adv_img1, adv_img2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                nSORIterations, colType)
    h, w, c = adv_img2.shape

    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    flow[:, :, 0] = flow[:, :, 0] / float(h)
    flow[:, :, 1] = flow[:, :, 1] / float(w)
    flow = -flow
    flow = Variable(torch.from_numpy(flow.astype(np.float32)).cuda())
    rand = Variable(torch.randn(flow.size()) * args.flownoise_stdv ).cuda()
    warped_adv_im2 = forward_stn(flow + rand, grid, adv_img_var1).clamp(0.0, 1.0)
    return warped_adv_im2

def flow_warp(args, adv_img2_var, adv_img_var1, flow_model):
    global normalization_factor
    # st()# 416 448 32 
    # print(adv_img2_var.size(), adv_img_var1.size())
    adv_img2_var_new = F.pad(adv_img2_var , (16, 16, 16, 16))
    adv_img_var1_new = F.pad(adv_img_var1, (16, 16, 16, 16))
    # st()
    catenated_data = torch.cat([adv_img2_var_new.unsqueeze(2), adv_img_var1_new.unsqueeze(2)], dim = 2)
    flow = flow_model(catenated_data)
    flow_transpose = torch.transpose(torch.transpose(flow, 1, 2), 2, 3)
    # st()
    flow_transpose = flow_transpose/normalization_factor
    if args.rand:
        rand = Variable( (torch.randn(flow_transpose.size()) * args.flownoise_stdv).cuda(), requires_grad=False, volatile=True)
        # rand = Variable( (torch.randn(flow_transpose.size()) * 0.002).cuda(), requires_grad=False, volatile=True)

    else:
        rand = Variable( (torch.randn(flow_transpose.size()) * 0.000 ).cuda(), requires_grad=False, volatile=True)
    # print(flow_transpose.shape)
    # Forward warping of image1
    warped_adv_im2 = forward_stn(flow_transpose + rand, grid, adv_img_var1_new).clamp(0.0, 1.0)
    warped_adv_im2 = warped_adv_im2[:, :, 16:-16, 16:-16]
    return warped_adv_im2

def save_image(save_path, img, seg=False):
    if not seg:
        img = img * 255.0
    img = np.array(img).astype(np.uint8)
    img_pil = Image.fromarray(img)
    w, h, _ = img.shape
    # img_pil = img_pil.resize((w // 16, h // 16))
    img_pil.save(save_path)

# def create_grid(opt, img_size=[416, 416]):
def create_grid(opt, img_size=[448, 448]):

    M = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0] * opt.batch_size)
    M_cpu = torch.from_numpy(M).view(1, 2, 3)
    M_gpu = cpu2var(M_cpu, requires_grad=False)
    # pdb.set_trace()
    grid = F.affine_grid(M_gpu, torch.Size([opt.batch_size, 1] + img_size))
    return grid
def forward_stn(flow,grid, real):
    #transfer to BNWH
    grid2 = grid + flow
    grid3 = grid2.clamp(-1, 1)
    ########
    # pdb.set_trace()
    fake = F.grid_sample(real, grid3)
    return fake

def regular_defense():
    namesfile = 'data_yolo/coco.names'
    class_names = load_class_names(namesfile)
    
    args = parse_args()
    interval = int(args.attack_interval)
    starting = int(args.starting)
    ending = int(args.ending)
    args.target = args.save_dir
    # args.save_dir = args.target
    if args.ibm:
        prefix = os.environ['DATA_DIR']
    else:
        prefix = './'
  

    args.use_cuda = torch.cuda.is_available()
    single_model = Darknet('cfg/yolov3.cfg')
    single_model.load_weights( osp.join( prefix, 'yolov3.weights') )
    model = nn.DataParallel(single_model)

    if args.use_cuda:
        model = model.cuda()
    model.eval()


    flow_model = FlowNet2CSS(args)
    # flow_model = torch.nn.DataParallel(flow_model)
    flow_model_name  = osp.join( prefix,  'FlowNet2-CSS-ft-sd_checkpoint.pth.tar')
    flow_model_ckpt = torch.load(flow_model_name)
    flow_model.load_state_dict(flow_model_ckpt['state_dict'])
    if args.use_cuda:
        flow_model = flow_model.cuda()
        flow_model = torch.nn.DataParallel(flow_model)
    flow_model.eval()
 
    # grid = None
    # zeros = None
    # normalization_factor = None
    phase = args.phase

    data_dir = args.data_dir

    # save_dir = '_'.join(['video_HPE_{}_attack_eot_multiple'.format(args.attack_phase), args.attack_method, args.save_dir, args.video_name, 'firt_frame_adv', str(args.noise_init), 'interval', str(args.attack_interval)])
    noise_init_bool = False


    if args.ibm:
        if args.attack_phase == 'regular':
            save_dir = '_'.join(['video_detection_{}_attack_eot_multiple'.format('regular'), args.attack_method, args.save_dir, args.video_name, 'firt_frame_adv', 'False', 'interval', '0'])
            save_dir = osp.join(save_dir, '_'.join(['darknet', str(args.threshold), args.bound_type, str(args.clip), 'interval', str(0), 'sample_size', str(1), 'flownoise_stdv', str(0.002)]))
        else:
            save_dir = '_'.join(['video_detection_{}_attack_eot_multiple'.format(args.attack_phase), args.attack_method, args.save_dir, args.video_name, 'firt_frame_adv', str(args.noise_init), 'interval', str(args.attack_interval)])
            save_dir = osp.join(save_dir, '_'.join(['darknet', str(args.threshold), args.bound_type, str(args.clip), 'interval', str(args.attack_interval), 'sample_size', str(args.sample_size), 'flownoise_stdv', str(args.flownoise_stdv)]))

        save_dir = osp.join(os.environ['DATA_DIR'], save_dir)

    else:
        if args.attack_phase == 'regular':
            save_dir = '_'.join(['video_detection_{}_attack_eot_multiple'.format('regular'), args.attack_method, args.save_dir, args.video_name, 'firt_frame_adv', 'False', 'interval', '0'])
            save_dir = osp.join(save_dir, '_'.join(['darknet', str(args.threshold), args.bound_type, str(args.clip), 'interval', str(0), 'sample_size', str(1), 'flownoise_stdv', str(0.002)]))
        else:
            save_dir = '_'.join(['video_detection_{}_attack_eot_multiple'.format(args.attack_phase), args.attack_method, args.save_dir, args.video_name, 'firt_frame_adv', str(args.noise_init), 'interval', str(args.attack_interval)])
            save_dir = osp.join('yolo_data', save_dir, '_'.join(['darknet', str(args.threshold), args.bound_type, str(args.clip), 'interval', str(args.attack_interval), 'sample_size', str(args.sample_size), 'flownoise_stdv', str(args.flownoise_stdv)]))


    noise_dir = osp.join(save_dir, 'noise')
    noise_format = noise_dir
    print(save_dir)
    assert(osp.exists(save_dir))
    assert(osp.exists(noise_dir))
    # assert(osp.exists(vis_dir))

    starting = 0
    ending = None
    if args.starting is not None:
        starting = args.starting
    if args.ending is not None:
        ending = args.ending

    if args.ibm:
        if args.sample_size > 1:
            vis_dir = osp.join( os.environ["RESULT_DIR"] , 'vis_sample_yolo')
            result_dir = osp.join( os.environ["RESULT_DIR"], 'result_sample_yolo')            
        else:
            vis_dir = osp.join( os.environ["RESULT_DIR"] , 'vis_yolo')
            result_dir = osp.join( os.environ["RESULT_DIR"], 'result_yolo')
    else:
        if not args.transfer:
            if args.sample_size > 1:
                vis_dir = 'vis_sample_yolo'
                result_dir = 'result_sample_yolo'
            else:
                vis_dir = 'vis_yolo'
                result_dir = 'result_yolo'
        else:
            if args.sample_size > 1:
                vis_dir = 'vis_sample_yolo_transfer'
                result_dir = 'result_sample_yolo_transfer'
            else:
                vis_dir = 'vis_yolo_transfer'
                result_dir = 'result_yolo_transfer'

    vis_dir = osp.join(vis_dir, args.video_name)
    try:
        if not osp.exists(vis_dir):
            os.makedirs(vis_dir)

        if not osp.exists(result_dir):
            os.makedirs(result_dir)
    except: 
        pass    

    starting = args.starting 
    ending = args.ending
    interval = args.attack_interval
    # fname_format =  osp.join( args.noise_dir, '{}.jpg')
    # fname_format = osp.join(prefix, 'video_hpe', args.video_name, '{}.jpg')
    if args.ibm:
        if args.video_name.startswith("stuttgart_00"):
            path = osp.join(prefix, "leftImg8bit/demoVideo/", args.video_name)
        elif args.video_name == "choreography" or args.video_name == "dog-control":
            path = osp.join(prefix, "DAVIS", "JPEGImages", "480p", args.video_name)
        else:
             raise NotImplementedError
    else:
        if args.video_name.startswith("stuttgart_00"):
            path = osp.join("/home/xiaocw/data/cityscapes/", "leftImg8bit/demoVideo/", args.video_name)
        elif args.video_name == "choreography" or args.video_name == "dog-control":
            path = osp.join("/home/xiaocw/", "DAVIS", "JPEGImages", "480p", args.video_name)
    if args.video_name.startswith("stuttgart_00"):
        offset = 1
        img_format = "stuttgart_00_000000_0{}_leftImg8bit.png"
    else:
        offset = 0
        img_format = "{}.jpg"
    
    fname_format = osp.join(path, img_format)
    # if args.video_name == 'video':
    #     offset = 600
    # else:
    #     offset = 0

    iteration_matrix = np.zeros((9, 9))
    # Savet
    global grid
    if args.transfer:
        grid = create_grid(args, [416, 416])
    else:
        grid = create_grid(args)
    zeros = Variable(torch.zeros([1, 3, single_model.width, single_model.height]), requires_grad=False, volatile=True)
    if args.use_cuda:
        grid = grid.cuda()
        zeros = zeros.cuda()
    global normalization_factor
    # normalization_factor = torch.from_numpy(np.array([single_model.width , single_model.height])).type(torch.FloatTensor)
    normalization_factor = torch.from_numpy(np.array([448 , 448])).type(torch.FloatTensor)

    normalization_factor = Variable( normalization_factor.cuda(), requires_grad=False, volatile=True)
    
    acc_resultss = []
    map_resultss = []
    results = []
    missing_items = []
    if args.video_name == 'stuttgart_00':
        offset = 1
    else:
        offset = 0
    for i in range(starting + interval, ending):

        img_name = fname_format.format( str(i + offset).zfill(5) )
        img1 = np.array(Image.open(img_name).resize(( single_model.width, single_model.height), Image.BILINEAR) )
        # data = data.astype(np.float32) / 255. 
        
        img1 = img1.astype(np.float32) / 255.0
        # print(data_np.shape)
        # data = torch.from_numpy(np.transpose(data_np, (2, 0, 1))[np.newaxis]/255.0)
        # data = data.type(torch.FloatTensor)
        
        noise_name = osp.join( noise_format, "{}.npy".format(i))
        if not osp.exists(noise_name):
            print("file not found ", noise_name)
            continue
        noise = np.load(noise_name)
        if args.noise_init:
            print("add noise")
            img1 = np.clip( img1 + noise, 0, 1)   
        img1 = img1[np.newaxis].transpose(0, 3, 1, 2)
        img_var1 = Variable(torch.from_numpy(img1).cuda(), requires_grad=False, volatile=True)
        # vis_seg(img_var1)
        img_vis1, img_pred1 = predict_convert(img_var1, model, class_names)
        missing = False
        for j in range(i - interval, i):
            noise_name = osp.join( noise_format, "{}.npy".format(j ))
            if not osp.exists(noise_name):
                print("file not found ", noise_name)
                missing = True
                break
        if missing == True:
            missing_items.append(i)
            continue
        for j in range(i - interval, i):

            img_name = fname_format.format( str(j + offset).zfill(5) )
            
            noise_name = osp.join( noise_format, "{}.npy".format(j ))

            print("noise j ", j , noise_name)
            print("img j ", j, img_name)
            # img_name = fname_format.format( str(j + offset).zfill(6) )
            img2 = np.array(Image.open(img_name).resize(( single_model.width, single_model.height), Image.BILINEAR) )
            # data = data.astype(np.float32) / 255. 
            
            img2 = img2.astype(np.float32) / 255.0
        

            # img2 = io.imread(img_name)[:, :, :3]
            # img2 = img2.astype(np.float32) / 255.
            noise = np.load(noise_name)

            adv_img2 = np.clip(img2 + noise, 0, 1)
            adv_img2 = adv_img2[np.newaxis].transpose(0, 3, 1, 2)
            img2 = img2[np.newaxis].transpose(0, 3, 1, 2)
            adv_img_var2 = Variable(torch.from_numpy(adv_img2).cuda(), requires_grad=False, volatile=True)
            img_var2 = Variable(torch.from_numpy(img2).cuda(), requires_grad=False, volatile=True)

            adv_img_vis2, adv_img_pred2 = predict_convert(adv_img_var2, model, class_names)
            img_vis2, img_pred2 = predict_convert(img_var2, model, class_names)

            # get flow 
            if args.transfer:
                warp_img_var2 = flow_warp_pyflow(args, img_var2, img_var1)
                warp_adv_img2 = flow_warp_pyflow(args, adv_img_var2, img_var1)
            
            else:
                warp_img_var2 = flow_warp(args, img_var2, img_var1, flow_model)
                warp_adv_img2 = flow_warp(args, adv_img_var2, img_var1, flow_model)
            
            warp_img_vis2, warp_img_pred2 = predict_convert(warp_img_var2, model, class_names, True)

            warp_adv_img_vis2, warp_adv_img_pred2 = predict_convert(warp_adv_img2, model, class_names, True)


            adv_acc= compute_metric(warp_adv_img_pred2, adv_img_pred2)
            benign_acc = compute_metric(warp_img_pred2, img_pred2)
            print(i, j, "adv: ", adv_acc)
            print("bengin: ", benign_acc)
            results.append([benign_acc, adv_acc])
            vis_res = []
            vis_res.append(img_vis1)
            vis_res.append(img_vis2)
            vis_res.append(warp_img_vis2)
            vis_res.append(adv_img_vis2)
            vis_res.append(warp_adv_img_vis2)
            vis_res = np.concatenate(vis_res)
            # if i == 8 and j == 7:
            #     save_image(osp.join("tmp", "{}_{}_{}_{}_{}_{}_{}_rand.jpg".format(i, j, args.attack_method, args.noise_init, args.attack_interval, 'rand', args.target, args.sample_size) ), vis_res, True)
            #     st()
            # if np.isnan(benign_acc) or np.isnan(adv_acc) or benign_acc == 0:

            #     save_image(osp.join("tmp", "{}_{}_{}_{}_{}_{}_{}_rand.jpg".format(i, j, args.attack_method, args.noise_init, args.attack_interval, 'rand', args.target, args.sample_size) ), vis_res, True)
            #     st()

            # save_image(osp.join(vis_dir, "{}_{}_{}_{}_{}_{}_{}_rand.jpg".format(i, j, args.attack_method, args.noise_init, args.attack_interval, 'rand', args.target, args.sample_size) ), vis_res, True)

        if args.attack_phase == 'adaptive':
            if args.transfer:
                np.save('{}/result_{}_{}_{}_{}_{}_rand_{}_adaptive_{}_transfer.npy'.format(result_dir, args.video_name, args.attack_interval,  args.target, args.attack_method, args.noise_init, args.flownoise_stdv, args.sample_size), {"data": results, "miss":missing_items})

            else:
                np.save('{}/result_{}_{}_{}_{}_{}_rand_{}_adaptive_{}.npy'.format(result_dir, args.video_name, args.attack_interval,  args.target, args.attack_method, args.noise_init, args.flownoise_stdv, args.sample_size), {"data": results, "miss":missing_items})
        else:
            if args.rand:
                np.save('{}/result_{}_{}_{}_{}_{}_rand_{}_{}.npy'.format(result_dir, args.video_name, args.attack_interval,  args.target, args.attack_method, args.noise_init, args.flownoise_stdv, args.sample_size), {"data": results, "miss":missing_items})
            else:
                np.save('{}/result_{}_{}_{}_{}_{}_normal_{}.npy'.format(result_dir, args.video_name, args.attack_interval,  args.target, args.attack_method, args.noise_init, args.sample_size), {"data": results, "miss":missing_items})

        sys.stdout.flush()

if __name__ == '__main__':
    regular_defense()
