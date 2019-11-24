This repository is the implementation of AAAI 2020 paper: "Revisiting Bilinear Pooling: A Coding Perspective".

Our code requires PyTorch v1.0 and Python 2.
Download the MINC dataset, and put it into the 'data' folder.
Download the pretrained vgg-16 model, put it into the 'data' folder., and we name it as 'vgg16-397923af'

Two-step training model:

Step 1:
	python last.py -MINC True -data_path 'data/minc-2500/' -train_txt_path 'data/mincTrainImages.txt' -test_txt_path 'data/mincTestImages.txt' -rank 1 -k 2048 -beta 0.001 -pre_model_path 'data/vgg16-397923af.pth' -save_low_bound 99
REMARK: you'd better chose a moderate 'save_low_bound' to save the model for fineturn fhase.


Step 2:
	python finetune.py -MINC True -data_path 'data/minc-2500/' -train_txt_path 'data/mincTrainImages.txt' -test_txt_path 'data/mincTestImages.txt' -rank 1 -k 2048 -beta 0.001 -model_path 'data/vgg16-397923af.pth'

Or you can directly modify 'config.py' to set more detailed hyper-parameters.
