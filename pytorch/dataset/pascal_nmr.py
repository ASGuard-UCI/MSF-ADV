import cv2
import os


from torch.utils import data

import config
import torch
import numpy as np
import os.path as osp
import subprocess

from ipdb import set_trace as st
from collections import defaultdict
from utils import list_files, makedir_if_not_exist

import neural_renderer as nr
from renderer import nmr


from PIL import Image 


def random_sample_direction():
    xyz = np.random.randn(3)
    xyz /= np.sqrt(np.square(xyz).sum())

    return xyz


class PascalNmrDataset(data.Dataset):
    def __init__(self, pascal3droot=config.PASCAL3D_ROOT, image_size=224, n_total=10000):
        super().__init__()
        self.n_total = n_total
        self.classes = [
            'aeroplane',
            'bicycle',
            'boat',
            'bottle',
            'bus',
            'car',
            'chair',
            'diningtable',
            'motorbike',
            'sofa',
            'train',
            'tvmonitor'
        ]
        self.obj_files = defaultdict(list)

        for name in self.classes:
            off_folder = osp.join(pascal3droot, 'CAD', name)
            off_files = list_files(off_folder, '.off')
            obj_folder = makedir_if_not_exist(osp.join(pascal3droot, 'obj'))

            for off_name in off_files:
                bname = osp.basename(off_name)[:-len('.off')]
                obj_name = osp.join(obj_folder, name + '_' + bname + '.obj')

                if not osp.exists(obj_name):
                    subprocess.check_call(['../scripts/meshconv', off_name, '-c', 'obj', '-tri', '-o', obj_name[:-len('.obj')]])
                self.obj_files[name].append(obj_name)

        self.renderer = nr.Renderer(image_size=224, camera_mode='look_at')


    def __getitem__(self, index):
        # First, random select a class
        class_idx = np.random.randint(len(self.classes))
        class_name = self.classes[class_idx]

        # Then randomly choose a model
        obj_name = np.random.choice(self.obj_files[class_name])
        # Random renderer setting
        angle = np.random.rand() * 360
        elevation = np.random.rand() * 45
        distance = 2
        self.renderer.eye = nr.get_points_from_angles(distance, elevation, angle)

        # Load obj
        vertices, faces = nr.load_obj(obj_name)
        vertices = nmr.rotate(vertices, [1, 0, 0], -90)
        textures = vertices.new_ones(faces.shape[0], 1, 1, 1, 3)

        vertices.unsqueeze_(0)
        faces.unsqueeze_(0)
        textures.unsqueeze_(0)

        # Generate random lighting
        while True:
            lighting = random_sample_direction()
            v1 = np.array(self.renderer.eye)
            v2 = np.array(lighting)

            if np.dot(v1, v2) / np.sqrt(np.square(v1).sum() * np.square(v2).sum()) < 0.5:
                continue
            if lighting[1] < 0:
                lighting[1] = -lighting[1]

            break


        self.renderer.light_direction = lighting
        self.renderer.light_intensity_ambient = np.random.rand() * 0.1 + 0.4
        self.renderer.light_intensity_directional = 1 - self.renderer.light_intensity_ambient

        image_tensor = self.renderer.render(vertices, faces, textures)
        image_tensor = torch.clamp(image_tensor, 0, 1)

        return image_tensor.squeeze(0), class_idx


    def __len__(self):
        return self.n_total


class PascalNmrTestDataset(data.Dataset):
    def __init__(self, folder='../data/pascal3d_test', pascal3droot=config.PASCAL3D_ROOT, n_sample=40):
        super().__init__()
        self.classes = [
            'aeroplane',
            'bicycle',
            'boat',
            'bottle',
            'bus',
            'car',
            'chair',
            'diningtable',
            'motorbike',
            'sofa',
            'train',
            'tvmonitor'
        ]
        self.obj_files = defaultdict(list)

        self.image_files = []
        self.labels = []

        for idx, name in enumerate(self.classes):
            off_folder = osp.join(pascal3droot, 'CAD', name)
            off_files = list_files(off_folder, '.off')
            obj_folder = makedir_if_not_exist(osp.join(pascal3droot, 'obj'))

            for off_name in sorted(off_files):
                bname = osp.basename(off_name)[:-len('.off')]

                for i in range(n_sample):
                    image_filename = osp.join(folder, '{}_{}_{}.jpg'.format(name, bname, i))

                    assert osp.isfile(image_filename)
                    self.image_files.append(image_filename)
                    self.labels.append(idx)


    def __getitem__(self, index):
        return (cv2.imread(self.image_files[index])[:, :, ::-1].transpose(2, 0, 1) / 255.0).astype(np.float32), self.labels[index]


    def __len__(self):
        return len(self.image_files)

class Pascal3DRenderDataset(data.Dataset):
    def __init__(self, folder='../data/pascal3d_renderings_sample', transforms=None):
        files = list_files(folder, '.jpg')
        self.filenames = []
        for file in files:
            filename = osp.join(folder, file)
            self.filenames.append(filename)
        self.transforms = transforms

    def __getitem__(self, index):
        fname = self.filenames[index]
        img = Image.open(fname)
        img = self.transforms(img)
        b_fname = fname.split('/')[-1]
        return img, b_fname.split('_')[0], b_fname
    def __len__(self):
        return len(self.filenames)





