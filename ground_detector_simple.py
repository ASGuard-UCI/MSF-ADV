from __future__ import division
import numpy as np
import random
import math

def my_ransac_v3(data,
			  distance_threshold=0.3,
			  P=0.99,
			  sample_size=3,
			  max_iterations=10000,
			  ):
	"""

	:param data:  n*3
	:param sample_size:
	:param P :
	:param distance_threshold:
	:param max_iterations:
	:return:
	"""
	# np.random.seed(12345)
	random.seed(12345)
	max_point_num = -999
	i = 0
	K = 10
	L_data = len(data)
	R_L = range(L_data)


	while i < K:

		# s3 = np.random.choice(L_data, sample_size, replace=False)
		s3 = random.sample(R_L, sample_size)

		if abs(data[s3[0],1] - data[s3[1],1]) < 3:
			continue




		coeffs = estimate_plane(data[s3,:], normalize=False)
		if coeffs is None:
			continue

		r = np.sqrt(coeffs[0]**2 + coeffs[1]**2 + coeffs[2]**2 )

		d = np.divide(np.abs(np.matmul(coeffs[:3], data.T) + coeffs[3]) , r)
		# d = abs(np.matmul(coeffs[:3], data.T) + coeffs[3]) / r
		d_filt = np.array(d < distance_threshold)

		near_point_num = np.sum(d_filt,axis=0)


		if near_point_num > max_point_num:
			max_point_num = near_point_num

			best_model = coeffs
			best_filt = d_filt

			w = near_point_num / L_data

			wn = np.power(w, 3)
			p_no_outliers = 1.0 - wn
			# sd_w = np.sqrt(p_no_outliers) / wn
			K = (np.log(1-P) / np.log(p_no_outliers)) #+ sd_w
		# print('# K:', i, K, near_point_num)

		i += 1

		if i > max_iterations:
			print(' RANSAC reached the maximum number of trials.')
			break

	print('took iterations:', i+1, 'best model:', best_model,
		  'explains:', max_point_num)
	return np.argwhere(best_filt).flatten(), best_model



def estimate_plane(xyz, normalize=True):
	"""
	:param xyz:  3*3 array
	x1 y1 z1
	x2 y2 z2
	x3 y3 z3
	:return: a b c d

	  model_coefficients.resize (4);
	  model_coefficients[0] = p1p0[1] * p2p0[2] - p1p0[2] * p2p0[1];
	  model_coefficients[1] = p1p0[2] * p2p0[0] - p1p0[0] * p2p0[2];
	  model_coefficients[2] = p1p0[0] * p2p0[1] - p1p0[1] * p2p0[0];
	  model_coefficients[3] = 0;
	  // Normalize
	  model_coefficients.normalize ();
	  // ... + d = 0
	  model_coefficients[3] = -1 * (model_coefficients.template head<4>().dot (p0.matrix ()));

	"""
	vector1 = xyz[1,:] - xyz[0,:]
	vector2 = xyz[2,:] - xyz[0,:]

	if not np.all(vector1):
		# print('will divide by zero..', vector1)
		return None
	dy1dy2 = vector2 / vector1

	if  not ((dy1dy2[0] != dy1dy2[1])  or  (dy1dy2[2] != dy1dy2[1])):
		return None


	a = (vector1[1]*vector2[2]) - (vector1[2]*vector2[1])
	b = (vector1[2]*vector2[0]) - (vector1[0]*vector2[2])
	c = (vector1[0]*vector2[1]) - (vector1[1]*vector2[0])
	# normalize
	if normalize:
		# r = np.sqrt(a ** 2 + b ** 2 + c ** 2)
		r = math.sqrt(a ** 2 + b ** 2 + c ** 2)
		a = a / r
		b = b / r
		c = c / r
	d = -(a*xyz[0,0] + b*xyz[0,1] + c*xyz[0,2])
	# return a,b,c,d
	return np.array([a,b,c,d])

def my_ransac(data,
			  distance_threshold=0.1,
			  P=0.99,
			  sample_size=3,
			  max_iterations=1000,
			  lidar_height=-1.73+0.4,
			  lidar_height_down=-1.73-0.2,
			  first_line_num=2000,
			  alpha_threshold=0.05, # 0.05
			  use_all_sample=False,
			  y_limit=4,
			  ):
	"""

	:param data:  N * 4
	:param sample_size:
	:param P :
	:param distance_threshold:
	:param max_iterations:
	:return:
	"""
	# np.random.seed(12345)
	random.seed(12345)
	max_point_num = -999
	best_model = None
	best_filt = None
	alpha = 999
	i = 0
	K = max_iterations 

	if not use_all_sample:

		z_filter = data[:,2] < lidar_height  
		z_filter_down = data[:,2] > lidar_height_down  
		filt = np.logical_and(z_filter_down, z_filter)  

		first_line_filtered = data[filt,:]
		print('first_line_filtered number.' ,first_line_filtered.shape,data.shape)
	else:
		first_line_filtered = data

	if data.shape[0] < 900 or first_line_filtered.shape[0] < 180:
		print(' RANSAC point number too small.')
		return None, None, None, None

	L_data = data.shape[0]
	R_L = range(first_line_filtered.shape[0])


	while i < K:

		# s3 = np.random.choice(L_data, sample_size, replace=False)
		s3 = random.sample(R_L, sample_size)


		coeffs = estimate_plane(first_line_filtered[s3,:], normalize=False)
		if coeffs is None:
			continue

		r = np.sqrt(coeffs[0]**2 + coeffs[1]**2 + coeffs[2]**2 )

		alphaz = math.acos(abs(coeffs[2]) / r)

		# r = math.sqrt(coeffs[0]**2 + coeffs[1]**2 + coeffs[2]**2 )
		# d = np.abs(np.matmul(coeffs[:3], data.T) + coeffs[3]) / r
		d = np.divide(np.abs(np.matmul(coeffs[:3], data[:,:3].T) + coeffs[3]), r)
		d_filt = np.array(d < distance_threshold)
		d_filt_object = ~d_filt

		near_point_num = np.sum(d_filt,axis=0)

		if near_point_num > max_point_num and alphaz < alpha_threshold:
			max_point_num = near_point_num

			best_model = coeffs
			best_filt = d_filt
			best_filt_object = d_filt_object


			# alpha = math.acos(abs(coeffs[2]) / r)
			alpha = alphaz

			w = near_point_num / L_data

			wn = math.pow(w, 3)
			p_no_outliers = 1.0 - wn
			# sd_w = np.sqrt(p_no_outliers) / wn
			K = (math.log(1-P) / math.log(p_no_outliers)) #+ sd_w

		i += 1


		if i > max_iterations:
			print(' RANSAC reached the maximum number of trials.')
			return None,None,None,None

	print('took iterations:', i+1, 'best model:', best_model,
		  'explains:', max_point_num)

	return np.argwhere(best_filt).flatten(),np.argwhere(best_filt_object).flatten(), best_model, alpha



# @profile
def my_ransac_segment(data,segment_x=20):
	"""


	:param data:  N * 4
	:param sample_size:
	:param P :
	:param distance_threshold:
	:param max_iterations:
	:return:
	"""
	data0_20 = data[data[:,0] <= segment_x, :]
	data20plus = data[data[:,0] > segment_x, :]

	indices0_20, indices20, model20, alpha_z0 = my_ransac(data0_20,
														   lidar_height=-1.73+0.2,
														   )

	indices20plus, indices2, model2, alpha_z = my_ransac(data20plus)



	if (indices20plus is not None) and (indices0_20 is not None):
		return np.vstack((
			data0_20,
			data20plus)), np.hstack(( indices0_20, indices20plus+data0_20.shape[0]))
	elif (indices20plus is not None) and (indices0_20 is None):
		return np.vstack((
			data20plus,
			data0_20)), indices20plus
	elif (indices20plus is None) and (indices0_20 is not None):
		return np.vstack((
			data0_20,
			data20plus)), indices0_20
	else:
		return None, None