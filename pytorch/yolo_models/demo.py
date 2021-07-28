from utils_yolo import *
from cfg_yolo import parse_cfg
from region_loss import RegionLoss
from darknet import Darknet
import os.path as osp
import os

import torch 
from torch import nn 
from torch.autograd import Variable 
import torch.optim as optim
from torchvision import transforms

def save_image(save_path, img, seg=False):
    path = '/'.join( save_path.split('/')[0:-1])
    if not osp.exists(path):
        os.makedirs(path)
    if not seg:
        img = img * 255.0
    img = np.array(img).astype(np.uint8)
    img_pil = Image.fromarray(img)
    w, h, _ = img.shape
    # img_pil = img_pil.resize((w // 16, h // 16))
    img_pil.save(save_path)

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

def run():
    prefix = './'
    namesfile = 'data_yolo/coco.names'
    class_names = load_class_names(namesfile)
    # use_cuda = torch.cuda.is_available()
    single_model = Darknet('cfg/yolov3.cfg')
    single_model.load_weights( osp.join( prefix, 'yolov3.weights') )
    model = nn.DataParallel(single_model)

    model = model.cuda()
    model.eval()
    img_name = '/home/xiaocw/3D/3D_attack/debug/tmp/addsphere/color.png'
    img1 = np.array(Image.open(img_name).resize(( single_model.width, single_model.height), Image.BILINEAR) )
    # data = data.astype(np.float32) / 255. 

    img1 = img1.astype(np.float32) / 255.0
    img1 = img1[np.newaxis].transpose(0,3,1,2)
    img_var1 = Variable(torch.from_numpy(img1).cuda(), requires_grad=False, volatile=True)
    vis, pred = predict_convert(img_var1, model, class_names)
    save_image('figures/tmp.jpg', vis)
run()