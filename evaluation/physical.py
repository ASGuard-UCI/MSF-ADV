
import numpy as np
from PIL import Image
import os


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


total_res = dict()
root = '***'
for j in range(1,100):
    for i in range(500):
        print("#################: ", i)
        padding = i * 50 + 900

        path_ben = root + 'benign_' + str(j) + '.jpeg'
        path_adv = root + 'adv_' + str(j) + '.jpeg'

        left, top, right, bottom = 300, 500, 300 + 1920, 500 + 1080
        image = Image.open(path_ben)
        image = add_margin(image, padding, padding, padding, padding, (255, 255, 255))
        image = image.resize((2388, 1900))
        image = image.crop((left, top, right, bottom))
        name_ben = root + 'benign_' + str(j) + '.jpg'
        image.save(name_ben)
        cmd = 'cp ' + name_ben + ' /apollo/modules/perception/testdata/camera/lib/obstacle/detector/yolo/img/test.jpg'
        os.system(cmd)
        os.system('./bazel-bin/modules/perception/camera/tools/offline/offline_obstacle_pipeline')
        res_path = '/apollo/result/test.txt'
        f = open(res_path, "r")
        text = f.read()
        print(text)
        if 'TRAFFIC' in text:
            save_path = '/apollo/save_paper/ben_' + str(i) + '/'
            cmd = 'mkdir ' + save_path
            os.system(cmd)
            cmd = 'cp /apollo/result/test* ' + save_path
            os.system(cmd)
        else:
            continue



