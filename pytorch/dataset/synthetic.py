import cv2
import os

from utils import registry

from torch.utils import data

import config
import numpy as np

from PIL import Image
import torchvision.transforms   as transforms
import pandas
import time
from pdb import set_trace as st
from utils import label_to_probs

@registry.register('Dataset', 'Synthetic')
class SyntheticDataset(data.Dataset):
    def __init__(self):
        # TODO0
        pass

    def __len__(self):
        # TODO: return the size of the dataset
        raise NotImplementedError

    def __getitem__(self, index):
        '''
        Return the numpy arrays
        '''
        # TODO
        raise NotImplementedError


@registry.register('Dataset', 'SyntheticOffline')
class SyntheticOfflineDataset(data.Dataset):
    def __init__(self, split, dataset_root=config.SYNTHETIC_OFFLINE_DIR, im_size=227, num_classes=12):
        assert split in ['train', 'valid', 'train_easy', 'valid_easy']
        self.split = split
        csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir, 'data', 'synset_' + split + '.csv')
        start_time = time.time()
        # Load instance data from csv-file
        im_paths, info, obj_cls, vp_labels = self.csv_to_instances(csv_path)
        print( "csv file length: ", len(im_paths) )

        print(config.SYNTHETIC_OFFLINE_DIR)    
        self.root           = dataset_root
        self.loader         = self.pil_loader
        self.im_paths   = im_paths
        self.info       = info
        self.obj_cls    = obj_cls
        self.vp_labels  = vp_labels
        self.flip       = [False] * len(im_paths)

        self.im_size        = im_size
        self.num_classes    = num_classes
        self.num_instances  = len(self.im_paths)
        # assert(transform   != None)

        transform   = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(   mean=(0., 0., 0.),
                                     std=(1./255., 1./255., 1./255.)
                                 ),
            transforms.Normalize(   mean=(104, 116.668, 122.678),
                                    std=(1., 1., 1.)
                                )
        ])

        self.transform      = transform

        # Set weights for loss
        class_hist          = np.histogram(obj_cls, range(0, self.num_classes+1))[0]
        mean_class_size     = np.mean(class_hist)
        self.loss_weights   = mean_class_size / class_hist

        # Print out dataset stats
        print( "Dataset loaded in ", time.time() - start_time, " secs." )
        print( "Dataset size: ", self.num_instances )

    def __getitem__(self, index):
        """
            Args:
            index (int): Index
            Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        # Load and transform image
        if self.root == None:
            im_path = self.im_paths[index]
        else:
            im_path = os.path.join(self.root, "Image", self.split, self.im_paths[index])

        info    = self.info[index]
        obj_cls = self.obj_cls[index]
        view    = self.vp_labels[index]
        flip    = self.flip[index]

        # Transform labels
        azim, elev, tilt = label_to_probs(  view,
                                            obj_cls,
                                            flip,
                                            num_classes = self.num_classes)

        # Load and transform image
        img = self.loader(im_path, flip = flip)
        if self.transform is not None:
            img = self.transform(img)


        # construct unique key for statistics -- only need to generate imid and year
        _bb     = str(0) + '-' + str(0) + '-' + str(0) + '-' + str(0)
        key_uid = self.im_paths[index] + '_'  + _bb + '_objc' + str(obj_cls) + '_kpc' + str(0)

        return img, azim, elev, tilt, obj_cls, key_uid

    def __len__(self):
        return self.num_instances

    """
        Loads images and applies the following transformations
            1. convert all images to RGB
            2. crop images using bbox (if provided)
            3. resize using LANCZOS to rescale_size
            4. convert from RGB to BGR
            5. (? not done now) convert from HWC to CHW
            6. (optional) flip image

        TODO: once this works, convert to a relative path, which will matter for
              synthetic data dataset class size.
    """
    def pil_loader(self, path, flip = False):
        # open path as file to avoid ResourceWarning
        # link: (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')

                # Convert to BGR from RGB
                r, g, b = img.split()
                img = Image.merge("RGB", (b, g, r))

                # img = img.crop(box=bbox)

                # verify that imresize uses LANCZOS
                img = img.resize( (self.im_size, self.im_size), Image.LANCZOS)

                # flip image
                if flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                return img

    
    def csv_to_instances(self, csv_path):
        df   = pandas.read_csv(csv_path, sep=' ')
        data = df.values
        data_split = np.split(data, [0, 1, 4, 7, 8], axis=1)
        del(data_split[0])
        image_paths = np.squeeze(data_split[0]).tolist()
        info      = data_split[1].tolist()
        viewpoints  = data_split[2].tolist()
        obj_class   = np.squeeze(data_split[3]).tolist()

        return image_paths, info, obj_class, viewpoints

