tensorboard_logdir = '/home/xiaocw/tensorboard_logs'

import os

pascal3d_root           = '/home/xiaocw/3D/pascal3d/'

root_dir                = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
render4cnn_weights      =  os.path.join(root_dir, 'model_weights/render4cnn.pth')
ft_render4cnn_weights   =  os.path.join(root_dir, 'model_weights/ryan_render.npy')
clickhere_weights       =  os.path.join(root_dir, 'model_weights/ch_cnn.npy')
