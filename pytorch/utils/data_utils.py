import numpy as np
import os
import sys
from scipy import linalg as linAlg

from IPython import embed

from pdb import set_trace as st
from PIL import Image

# from ipdb import set_trace as st

def label_to_probs(view_angles, object_class, flip, num_classes = 12):
    '''
    Returns three arrays for the viewpoint labels, one for each rotation axis.
    A label is given by 360 * object_class_id + angle
    :return:
    '''
    # Calculate object multiplier
    obj_mult = object_class

    # extract angles
    azim = view_angles[0] % 360
    elev = view_angles[1] % 360
    tilt = view_angles[2] % 360

    if flip:
        # print ("Previous angle (%d, %d, %d) " % (azim, elev, tilt)),
        azim = (360-azim) % 360
        tilt = (-1 *tilt) % 360
        # print (". Flipped angle (%d, %d, %d) " % (azim, elev, tilt))

    # Loss parameters taken directly from Render4CNN paper
    azim_band_width = 7     # 15 in paper
    elev_band_width = 2     # 5 in paper
    tilt_band_width = 2     # 5 in paper

    azim_sigma = 5
    elev_sigma = 3
    tilt_sigma = 3

    azim_label = np.zeros((num_classes*360), dtype=np.float)
    elev_label = np.zeros((num_classes*360), dtype=np.float)
    tilt_label = np.zeros((num_classes*360), dtype=np.float)

    # calculate probabilities
    azim_band, azim_prob = calc_viewloss_vec(azim_band_width, azim_sigma)
    elev_band, elev_prob = calc_viewloss_vec(elev_band_width, elev_sigma)
    tilt_band, tilt_prob = calc_viewloss_vec(tilt_band_width, tilt_sigma)
    # print(azim_band)
    # print(obj_mult)
    # print(azim_prob)
    for i in azim_band:
        # print(azim, obj_mult, np.mod(azim + i + 360, 360) + 360 * obj_mult, azim_prob[i + azim_band_width], azim_prob)
        # embed()
        ind = np.mod(azim + i + 360, 360) + 360 * obj_mult
        azim_label[ind] = azim_prob[i + azim_band_width]

    for j in elev_band:
        ind = np.mod(elev + j + 360, 360) + 360 * obj_mult
        elev_label[ind] = elev_prob[j + elev_band_width]

    for k in tilt_band:
        ind = np.mod(tilt + k + 360, 360) + 360 * obj_mult
        tilt_label[ind] = tilt_prob[k + tilt_band_width]

    return azim_label, elev_label, tilt_label

def calc_viewloss_vec(size, sigma):
    band    = np.linspace(-1*size, size, 1 + 2*size, dtype=np.int16)
    vec     = np.linspace(-1*size, size, 1 + 2*size, dtype=np.float)
    prob    = np.exp(-1 * abs(vec) / sigma)
    # prob    = prob / np.sum(prob)

    return band, prob

def csv_to_instances(csv_path):
    import pandas
    df   = pandas.read_csv(csv_path, sep=',')
    data = df.values

    data_split = np.split(data, [0, 1, 5, 6, 9], axis=1)
    del(data_split[0])

    image_paths = np.squeeze(data_split[0]).tolist()
    bboxes      = data_split[1].tolist()
    obj_class   = np.squeeze(data_split[2]).tolist()
    viewpoints  = data_split[3].tolist()

    return image_paths, bboxes, obj_class, viewpoints

def pil_loader(path, bbox = None ,flip = False, im_size = 227):
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
            # img = img.resize( (im_size, im_size), Image.LANCZOS)

            # flip image
            # if flip:
                # img = img.transpose(Image.FLIP_LEFT_RIGHT)

            return img
