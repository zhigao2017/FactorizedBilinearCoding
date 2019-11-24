#coding:utf-8
import warnings
warnings.filterwarnings('ignore')

class DefaultConfig(object):
	gpu_device = '0'
	#dataset 			#class 	train 	test
	DTD = True			#47 	3760 	1880
	CUB = False			#200 	5994 	5794
	INDOOR = False			#23 	51750 	5750
	MINC2500 = False	#	670		330

	RANK_ATOMS = 1
	NUM_CLUSTER = 2048
	BETA =  0.001
	model_name_pre = 'model_name'
	model_path = None  ## the path of the pretrained model
	save_low_bound = 79  ##when the accuracy achieves save_low_bound, the model is saved

	res_plus = 512			
	res = 448				
	train_print_freq = 256	
	
	lr = 0.01
	lr_scale = 0.1
	lr_freq_list = [40,80]

	train_bs = 16
	down_chennel = 512
	test_bs = 4
	test_epoch = 1
	pretrained = True
	pre_path = 'data/vgg16-397923af.pth'

	model_name = 'FBC'
	use_gpu = True

	if MINC2500:
		data_path = 'data/minc-2500/'
		train_txt_path = 'data/mincTrainImages.txt'
		test_txt_path = 'data/mincTestImages.txt'
		class_num = 23
	else:
		print('data error')

	max_epoches = 500


opt = DefaultConfig()