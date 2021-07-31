# MSF-ADV
MSF-ADV is a novel physical-world adversarial attack method, which can fool the Multi Sensor Fusion (MSF) based autonomous driving (AD) perception
in the victim autonomous vehicle (AV) to fail in detecting a front obstacle and thus
crash into it.

## Paper

[IEEE S&P 2021](https://www.ieee-security.org/TC/SP2021/index.html) Invisible for both Camera and LiDAR: Security of Multi-Sensor Fusion based Perception in Autonomous Driving Under Physical-World Attacks

Yulong Cao*, Ningfei Wang*, Chaowei Xiao*, Dawei Yang* (co-first authors), Jin Fang, Ruigang Yang, Qi Alfred Chen, Mingyan Liu, and Bo Li

To appear in the 42nd IEEE Symposium on Security and Privacy (IEEE S&P), May 2021 (Acceptance rate 12.0% = 117/972)


Invisible for both Camera and LiDAR: Security of Multi-Sensor Fusion based Perception in Autonomous Driving Under Physical-World Attacks

Author: Yulong Cao*, Ningfei Wang*, Chaowei Xiao*, Dawei Yang*, Jin Fang, Ruigang Yang, Qi Alfred Chen, Mingyan Liu, Bo Li (*Co-first authors)

Website: https://sites.google.com/view/cav-sec/msf-adv

![title](imgs/framework.png)

This is the code for the paper [Invisible for both Camera and LiDAR: Security of Multi-Sensor Fusion based Perception in Autonomous Driving Under Physical-World Attacks](https://www.computer.org/csdl/proceedings-article/sp/2021/893400b302/1t0x9btzenu) accepted by IEEE S&P 2021.

The arxiv link to the paper: https://arxiv.org/abs/2106.09249

## Installation
Install the required environments with the requirements.txt file using [ANACONDA](https://www.anaconda.com/products/individual)
```
$ conda create --name <env> --file requirements.txt
```

## Command line
### Download the target model
You can find the model through the [official Baidu Apollo GitHub](https://github.com/ApolloAuto/apollo) and [YOLO website](https://pjreddie.com/darknet/yolo/).

### Generating the adversarial object

 ```
 python attack.py [-obj] [-obj_save] [-lidar] [-cam] [-cali] [-e] [-o] [-it] 
 ```

| Argument | Description |
| -------- | ----------- |
| `-e` | Constrained max changing for the object vetex|
|`-o` | Optimization method: pgd and adam |
|`-it` | Max iteration number |
|`-obj` | Initial benign 3D object path |
|`-obj_save` | Adversarial 3D object saving dir |
|`-lidar` | LiDAR point cloud data path |
|`-cam` | Camera image data path |
|`-cali` | Calibration file path |

### Example for generating the adversarial object

 ```
 python attack.py -obj ./object/object.ply -obj_save ./object/obj_save -lidar ./data/lidar.bin -cam ./data/cam.png -cali ./data/cali.txt -e 0.2 -o pgd -it 1000 
 ```

 ### Evaluation
 The source code for evaluating the generated adversarial 3D object are in evaluation folder.

# Citation
 If you use the code or find this project helpful, please consider citing our paper.
```
@inproceedings{sp:2021:ningfei:msf-adv,
  title={{Invisible for both Camera and LiDAR: Security of Multi-Sensor Fusion based Perception in Autonomous Driving Under Physical World Attacks}},
  author={Yulong Cao and Ningfei Wang and Chaowei Xiao and Dawei Yang and Jin Fang and Ruigang Yang and Qi Alfred Chen and Mingyan Liu and Bo Li},
  booktitle={Proceedings of the 42nd IEEE Symposium on Security and Privacy (IEEE S\&P 2021)},
  year={2021},
  month = {May}
}
```
