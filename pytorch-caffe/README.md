## caffe2pytorch
This tool aims to load caffe prototxt and weights directly in pytorch without explicitly converting model from caffe to pytorch.

### Usage
```
from caffenet import *

def load_image(imgfile):
    import caffe
    image = caffe.io.load_image(imgfile)
    transformer = caffe.io.Transformer({'data': (1, 3, args.height, args.width)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([args.meanB, args.meanG, args.meanR]))
    transformer.set_raw_scale('data', args.scale)
    transformer.set_channel_swap('data', (2, 1, 0))

    image = transformer.preprocess('data', image)
    image = image.reshape(1, 3, args.height, args.width)
    return image

def forward_pytorch(protofile, weightfile, image):
    net = CaffeNet(protofile)
    print(net)
    net.load_weights(weightfile)
    net.eval()
    image = torch.from_numpy(image)
    image = Variable(image)
    blobs = net(image)
    return blobs, net.models

imgfile = 'data/cat.jpg'
protofile = 'resnet50/deploy.prototxt'
weightfile = 'resnet50/resnet50.caffemodel'
image = load_image(imgfile)
pytorch_blobs, pytorch_models = forward_pytorch(protofile, weightfile, image)

```

### Todos
- [x] support forward classification networks: AlexNet, VGGNet, GoogleNet, [ResNet](http://pan.baidu.com/s/1kVm4ly3), [ResNeXt](https://pan.baidu.com/s/1pLhk0Zp#list/path=%2F), DenseNet
- [x] support forward detection networks: [SSD300](https://drive.google.com/open?id=0BzKzrI_SkD1_WVVTSmQxU0dVRzA), [S3FD](https://github.com/sfzhang15/SFD), FPN

### Supported Layers
Each layer in caffe will have a corresponding layer in pytorch. 
- [x] Convolution
- [x] InnerProduct
- [x] BatchNorm
- [x] Scale
- [x] ReLU
- [x] Pooling
- [x] Reshape
- [x] Softmax
- [x] Accuracy
- [x] SoftmaxWithLoss
- [x] Dropout
- [x] Eltwise
- [x] Normalize
- [x] Permute
- [x] Flatten
- [x] Slice
- [x] Concat
- [x] PriorBox
- [x] LRN : gpu version is ok, cpu version produce big difference
- [x] DetectionOutput: support batchsize=1, num_classes=1 forward
- [x] Crop
- [x] Deconvolution
- [x] MultiBoxLoss

### Verify between caffe and pytorch
The script verify.py can verify the parameter and output difference between caffe and pytorch.
```
python verify_time.py --protofile resnet50/deploy.prototxt --weightfile resnet50/resnet50.caffemodel --imgfile data/cat.jpg --meanB 104.01 --meanG 116.67 --meanR 122.68 --scale 255 --height 224 --width 224 --synset_words data/synset_words.txt --cuda
python verify_deploy.py --protofile resnet50/deploy.prototxt --weightfile resnet50/resnet50.caffemodel --imgfile data/cat.jpg --meanB 104.01 --meanG 116.67 --meanR 122.68 --scale 255 --height 224 --width 224 --synset_words data/synset_words.txt --cuda
```
Note: 
1. synset_words.txt contains class information, each line represents the description of a class.
2. resnet50 is downloaded from [BaiduYun](http://pan.baidu.com/s/1kVm4ly3)

Outputs:
```
pytorch forward0: 0.887461
pytorch forward1: 0.077649
pytorch forward2: 0.022872
pytorch forward3: 0.021319
pytorch forward4: 0.018347
pytorch forward5: 0.018263
pytorch forward6: 0.018249
pytorch forward7: 0.018289
pytorch forward8: 0.018553
pytorch forward9: 0.018290
caffe forward0: 0.106477
caffe forward1: 0.025897
caffe forward2: 0.025949
caffe forward3: 0.026127
caffe forward4: 0.025875
caffe forward5: 0.025874
caffe forward6: 0.025646
caffe forward7: 0.025932
caffe forward8: 0.025827
caffe forward9: 0.025849
```
```
------------ Parameter Difference ------------
conv1                                weight_diff: 0.000000        bias_diff: 0.000000
bn_conv1                       running_mean_diff: 0.000000 running_var_diff: 0.000000
scale_conv1                          weight_diff: 0.000000        bias_diff: 0.000000
res2a_branch1                        weight_diff: 0.000000
bn2a_branch1                   running_mean_diff: 0.000000 running_var_diff: 0.000000
scale2a_branch1                      weight_diff: 0.000000        bias_diff: 0.000000
res2a_branch2a                       weight_diff: 0.000000
bn2a_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale2a_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res2a_branch2b                       weight_diff: 0.000000
bn2a_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale2a_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res2a_branch2c                       weight_diff: 0.000000
bn2a_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale2a_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res2b_branch2a                       weight_diff: 0.000000
bn2b_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale2b_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res2b_branch2b                       weight_diff: 0.000000
bn2b_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale2b_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res2b_branch2c                       weight_diff: 0.000000
bn2b_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale2b_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res2c_branch2a                       weight_diff: 0.000000
bn2c_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale2c_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res2c_branch2b                       weight_diff: 0.000000
bn2c_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale2c_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res2c_branch2c                       weight_diff: 0.000000
bn2c_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale2c_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res3a_branch1                        weight_diff: 0.000000
bn3a_branch1                   running_mean_diff: 0.000000 running_var_diff: 0.000000
scale3a_branch1                      weight_diff: 0.000000        bias_diff: 0.000000
res3a_branch2a                       weight_diff: 0.000000
bn3a_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale3a_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res3a_branch2b                       weight_diff: 0.000000
bn3a_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale3a_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res3a_branch2c                       weight_diff: 0.000000
bn3a_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale3a_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res3b_branch2a                       weight_diff: 0.000000
bn3b_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale3b_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res3b_branch2b                       weight_diff: 0.000000
bn3b_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale3b_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res3b_branch2c                       weight_diff: 0.000000
bn3b_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale3b_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res3c_branch2a                       weight_diff: 0.000000
bn3c_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale3c_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res3c_branch2b                       weight_diff: 0.000000
bn3c_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale3c_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res3c_branch2c                       weight_diff: 0.000000
bn3c_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale3c_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res3d_branch2a                       weight_diff: 0.000000
bn3d_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale3d_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res3d_branch2b                       weight_diff: 0.000000
bn3d_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale3d_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res3d_branch2c                       weight_diff: 0.000000
bn3d_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale3d_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res4a_branch1                        weight_diff: 0.000000
bn4a_branch1                   running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4a_branch1                      weight_diff: 0.000000        bias_diff: 0.000000
res4a_branch2a                       weight_diff: 0.000000
bn4a_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4a_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res4a_branch2b                       weight_diff: 0.000000
bn4a_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4a_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res4a_branch2c                       weight_diff: 0.000000
bn4a_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4a_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res4b_branch2a                       weight_diff: 0.000000
bn4b_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4b_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res4b_branch2b                       weight_diff: 0.000000
bn4b_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4b_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res4b_branch2c                       weight_diff: 0.000000
bn4b_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4b_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res4c_branch2a                       weight_diff: 0.000000
bn4c_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4c_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res4c_branch2b                       weight_diff: 0.000000
bn4c_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4c_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res4c_branch2c                       weight_diff: 0.000000
bn4c_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4c_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res4d_branch2a                       weight_diff: 0.000000
bn4d_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4d_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res4d_branch2b                       weight_diff: 0.000000
bn4d_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4d_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res4d_branch2c                       weight_diff: 0.000000
bn4d_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4d_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res4e_branch2a                       weight_diff: 0.000000
bn4e_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4e_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res4e_branch2b                       weight_diff: 0.000000
bn4e_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4e_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res4e_branch2c                       weight_diff: 0.000000
bn4e_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4e_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res4f_branch2a                       weight_diff: 0.000000
bn4f_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4f_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res4f_branch2b                       weight_diff: 0.000000
bn4f_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4f_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res4f_branch2c                       weight_diff: 0.000000
bn4f_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale4f_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res5a_branch1                        weight_diff: 0.000000
bn5a_branch1                   running_mean_diff: 0.000000 running_var_diff: 0.000000
scale5a_branch1                      weight_diff: 0.000000        bias_diff: 0.000000
res5a_branch2a                       weight_diff: 0.000000
bn5a_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale5a_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res5a_branch2b                       weight_diff: 0.000000
bn5a_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale5a_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res5a_branch2c                       weight_diff: 0.000000
bn5a_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale5a_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res5b_branch2a                       weight_diff: 0.000000
bn5b_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale5b_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res5b_branch2b                       weight_diff: 0.000000
bn5b_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale5b_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res5b_branch2c                       weight_diff: 0.000000
bn5b_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale5b_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
res5c_branch2a                       weight_diff: 0.000000
bn5c_branch2a                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale5c_branch2a                     weight_diff: 0.000000        bias_diff: 0.000000
res5c_branch2b                       weight_diff: 0.000000
bn5c_branch2b                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale5c_branch2b                     weight_diff: 0.000000        bias_diff: 0.000000
res5c_branch2c                       weight_diff: 0.000000
bn5c_branch2c                  running_mean_diff: 0.000000 running_var_diff: 0.000000
scale5c_branch2c                     weight_diff: 0.000000        bias_diff: 0.000000
------------ Output Difference ------------
data                           output_diff: 0.000000
conv1                          output_diff: 0.000000
pool1                          output_diff: 0.000000
res2a_branch1                  output_diff: 0.000000
res2a_branch2a                 output_diff: 0.000000
res2a_branch2b                 output_diff: 0.000000
res2a_branch2c                 output_diff: 0.000000
res2a                          output_diff: 0.000000
res2b_branch2a                 output_diff: 0.000000
res2b_branch2b                 output_diff: 0.000000
res2b_branch2c                 output_diff: 0.000001
res2b                          output_diff: 0.000000
res2c_branch2a                 output_diff: 0.000000
res2c_branch2b                 output_diff: 0.000001
res2c_branch2c                 output_diff: 0.000001
res2c                          output_diff: 0.000001
res3a_branch1                  output_diff: 0.000001
res3a_branch2a                 output_diff: 0.000000
res3a_branch2b                 output_diff: 0.000000
res3a_branch2c                 output_diff: 0.000001
res3a                          output_diff: 0.000000
res3b_branch2a                 output_diff: 0.000000
res3b_branch2b                 output_diff: 0.000000
res3b_branch2c                 output_diff: 0.000000
res3b                          output_diff: 0.000000
res3c_branch2a                 output_diff: 0.000000
res3c_branch2b                 output_diff: 0.000000
res3c_branch2c                 output_diff: 0.000000
res3c                          output_diff: 0.000000
res3d_branch2a                 output_diff: 0.000000
res3d_branch2b                 output_diff: 0.000000
res3d_branch2c                 output_diff: 0.000001
res3d                          output_diff: 0.000001
res4a_branch1                  output_diff: 0.000001
res4a_branch2a                 output_diff: 0.000000
res4a_branch2b                 output_diff: 0.000000
res4a_branch2c                 output_diff: 0.000001
res4a                          output_diff: 0.000001
res4b_branch2a                 output_diff: 0.000000
res4b_branch2b                 output_diff: 0.000000
res4b_branch2c                 output_diff: 0.000001
res4b                          output_diff: 0.000000
res4c_branch2a                 output_diff: 0.000000
res4c_branch2b                 output_diff: 0.000000
res4c_branch2c                 output_diff: 0.000001
res4c                          output_diff: 0.000000
res4d_branch2a                 output_diff: 0.000000
res4d_branch2b                 output_diff: 0.000000
res4d_branch2c                 output_diff: 0.000001
res4d                          output_diff: 0.000000
res4e_branch2a                 output_diff: 0.000000
res4e_branch2b                 output_diff: 0.000000
res4e_branch2c                 output_diff: 0.000001
res4e                          output_diff: 0.000000
res4f_branch2a                 output_diff: 0.000000
res4f_branch2b                 output_diff: 0.000000
res4f_branch2c                 output_diff: 0.000001
res4f                          output_diff: 0.000000
res5a_branch1                  output_diff: 0.000002
res5a_branch2a                 output_diff: 0.000000
res5a_branch2b                 output_diff: 0.000000
res5a_branch2c                 output_diff: 0.000001
res5a                          output_diff: 0.000001
res5b_branch2a                 output_diff: 0.000000
res5b_branch2b                 output_diff: 0.000000
res5b_branch2c                 output_diff: 0.000001
res5b                          output_diff: 0.000001
res5c_branch2a                 output_diff: 0.000000
res5c_branch2b                 output_diff: 0.000000
res5c_branch2c                 output_diff: 0.000002
res5c                          output_diff: 0.000001
pool5                          output_diff: 0.000000
fc1000                         output_diff: 0.000001
prob                           output_diff: 0.000000
------------ Classification ------------
pytorch classification top1: 0.193016 n02113023 Pembroke, Pembroke Welsh corgi
caffe   classification top1: 0.193018 n02113023 Pembroke, Pembroke Welsh corgi
```
