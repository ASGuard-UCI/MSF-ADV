from __future__ import division, print_function
import numpy as np
from scipy.stats import norm
import pypcd
import caffe
import sys
sys.path.append('./pytorch-caffe')
from caffenet import *
root_path = './cnnseg/velodyne64/'
from ground_detector_simple import *
from xyz2grid import *
import inject_cube

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

height_ = 512
width_ = 512
range_ = 60
min_height_ = -5.0
max_height_ = 5.0

outputs = ['instance_pt', 'category_score', 'confidence_score',
			'height_pt', 'heading_pt', 'class_score']

def rc2grid(r,c):
    return r*512+c


class Obstacle:

	def __init__(self, idx):
		self.id = idx
		self.PCL = []

	def add(self,pt):
		self.PCL.append(pt)

	def numPT(self):
		return len(self.PCL)

	def getPCL(self):
		return self.PCL

	def get_mean(self):
		return self.PCL.mean(9)



def mapPointToGrid(PCL):

	point2grid = []

	pos_x = -1
	pos_y = -1
	inv_res_x = 0.5 * float(width_) / range_
	size = PCL.shape[0]
	for idx in range(size):
		if (PCL[idx,2] <= min_height_ or PCL[idx,2] >= max_height_):
			point2grid.append(-1)
			continue
		pos_x, pos_y = groupPc2Pixel(PCL[idx,:], inv_res_x, range_);
		if (pos_y < 0 or pos_y >= height_ or pos_x < 0 or pos_x >= width_):
			point2grid.append(-1)
			continue
		point2grid.append(pos_y * width_ + pos_x)

	return point2grid

def groupPc2Pixel(PT, inv_res_x, range_):

	fx = range_ - PT[0] * 512.0 / 120 + 512.0 / 2
	fy = range_ - PT[1] * 512.0 / 120 + 512.0 / 2
	x = -1 if fx < 0 else int(fx)
	y = -1 if fy < 0 else int(fy)
	return x,y


def loadPCL(PCL,flag = True):

	if flag:
		PCL = np.fromfile(PCL, dtype=np.float32)
		PCL = PCL.reshape((-1,4))

	else:
		PCL = pypcd.PointCloud.from_path(PCL)
		PCL = np.array(tuple(PCL.pc_data.tolist()))
		PCL = np.delete(PCL, -1, axis = 1)

	return PCL

def generateFM(PCL, PCLConverted):

	max_height_data_ = []
	mean_height_data_ = []
	count_data_ = []
	top_intensity_data_ = []
	mean_intensity_data_ = []
	nonempty_data_ = []

	mapSize = height_ * width_
	size = PCL.shape[0]
	for i in range(mapSize):
		max_height_data_.append(-5.0)
		mean_height_data_.append(0.0)
		count_data_.append(0.0)
		top_intensity_data_.append(0.0)
		mean_intensity_data_.append(0.0)
		nonempty_data_.append(0.0)

	for i in range(size):
		idx = PCLConverted[i]
		if idx == -1:
			continue
		pz = PCL[i,2]
		pi = PCL[i,3] #/ 255.0
		if max_height_data_[idx] < pz:
			max_height_data_[idx] = pz
			top_intensity_data_[idx] = pi;
		mean_height_data_[idx] += float(pz)
		mean_intensity_data_[idx] += float(pi);
		count_data_[idx] += 1.0;

	for i in range(mapSize):
		if count_data_[i] <= sys.float_info.epsilon:
			max_height_data_[i] = 0.0
			count_data_[i] = math.log(1)
		else:
			mean_height_data_[i] /= count_data_[i]
			mean_intensity_data_[i] /= count_data_[i]
			nonempty_data_[i] = 1.0
			count_data_[i] = math.log(int(count_data_[i])+1)

	max_height_data_ = np.array(max_height_data_).reshape(-1,512)
	mean_height_data_ = np.array(mean_height_data_).reshape(-1,512)
	count_data_ = np.array(count_data_).reshape(-1,512)
	top_intensity_data_ = np.array(top_intensity_data_).reshape(-1,512)
	mean_intensity_data_ = np.array(mean_intensity_data_).reshape(-1,512)
	nonempty_data_ = np.array(nonempty_data_).reshape(-1,512)

	FM = [max_height_data_, mean_height_data_, count_data_, top_intensity_data_,
		mean_intensity_data_, nonempty_data_]

	return FM


def generatePytorch(protofile, weightfile):
	net = CaffeNet(protofile, phase='TEST')
	# torch.cuda.set_device(0)
	net.cuda()
	net.load_weights(weightfile)
	net.set_train_outputs(outputs)
	net.set_eval_outputs(outputs)
	return net

def generateCaffe(protofile, weightfile):
	caffe.set_device(0)
	caffe.set_mode_gpu()
	net = caffe.Net(protofile, weightfile, caffe.TEST)
	return net


def generateNormalMask(target_center):

	mask = np.zeros((512,512))
	x = np.linspace(norm.ppf(0.01),norm.ppf(0.5), 512)

	for r in range(512):
		for c in range(512):
			mask[r,c] = norm.pdf(x[511 - abs(r - target_center[0])]) * norm.pdf(x[511 - abs(c - target_center[1])])
			#mask[r,c] = 1 # for test
	return mask

def preProcess(PCLmain,PCLtarget):

	PCL = loadPCL(PCLmain,True)
	PCL = inject_cube.injectCylinder(PCL,0.5,0.25,7,0,-1.73)
	PCL = PCL[:,:4].astype('float32')

	target_obs = loadPCL(PCLtarget,True)

	aset = set([tuple(x) for x in PCL])
	bset = set([tuple(x) for x in target_obs])

	PCL_except_obs = np.array([x for x in aset - bset])

	return PCL,PCL_except_obs,target_obs

def twod2threed(obj, label_map, PCL, PCLConverted):

	obstacle = []
	cluster_id_list = []
	for obs in obj:
		cluster_id_list.append(obs[-1][1])
		obstacle.append(Obstacle(obs[-1][1]))

	size = PCL.shape[0]
	for i in range(size):
		idx = PCLConverted[i]
		if idx < 0:
			continue
		pt = PCL[i,:]
		label = label_map[idx]
		if label < 1: ## means if ==0
			continue
		if label in cluster_id_list and pt[2] <= obj[cluster_id_list.index(label)][-1][-1] + 0.5:
			obstacle[cluster_id_list.index(label)].add(pt)

	obstacle = [obs for obs in obstacle if obs.numPT() >= 3]

	ground = []
	_,_,ground_model,_ = my_ransac(np.delete(PCL, -1, axis = 1))
	for i in range(PCL.shape[0]):
		z = (-ground_model[0] * PCL[i,0] - -ground_model[1] * PCL[i,1] - ground_model[3]) / ground_model[2]
		if PCL[i,2] < (z + 0.25):
			# can directly remove ground here
			ground.append(PCL[i,:])
	ground = np.array(ground)

	index = []
	for idx,obs in enumerate(obstacle):
		num_pt = obs.numPT()
		pc = np.array(obs.getPCL())
		for i in range(pc.shape[0]):
			z = (-ground_model[0] * pc[i,0] - -ground_model[1] * pc[i,1] - ground_model[3]) / ground_model[2]
			if pc[i,2] < (z + 0.25):
				num_pt -= 1
		if num_pt < 3:
			index.append(idx)

	obstacle = [obs for obs in obstacle if obstacle.index(obs) not in index]

	return obstacle, cluster_id_list