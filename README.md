This repository is the implementation of AAAI 2020 paper: "Revisiting Bilinear Pooling: A Coding Perspective".

Prerequisites
-------
Our code requires PyTorch v1.0 and Python 2.

Download the MINC dataset, and put it into the 'data' folder.

Download the pretrained vgg-16 model, put it into the 'data' folder, and we name it as 'vgg16-397923af.pth'

Training our model includes two steps.
-------

Step 1:

We train the new added layers.
```
python last.py -MINC True -data_path 'data/minc-2500/' -train_txt_path 'data/mincTrainImages.txt' -test_txt_path 'data/mincTestImages.txt' -rank 1 -k 2048 -beta 0.001 -pre_model_path 'data/vgg16-397923af.pth' -save_low_bound 99
```


Step 2:

We train the whole network.
```
python finetune.py -MINC True -data_path 'data/minc-2500/' -train_txt_path 'data/mincTrainImages.txt' -test_txt_path 'data/mincTestImages.txt' -rank 1 -k 2048 -beta 0.001 -model_path 'data/vgg16-397923af.pth'
```

You can modify 'config.py' to set more detailed hyper-parameters.


If this code is helpful, we'd appreciate it if you could cite our paper

```
@inproceedings{zhi2020revisiting,
  title={Revisiting Bilinear Pooling: A Coding Perspective},
  author={Gao, Zhi and Wu, Yuwei and Zhang, Xiaoxun and Dai, Jindou and Jia, Yunde and Harandi, Mehrtash},
  booktitle={Proceedings of AAAI Conference on Artificial Intelligence (AAAI)},
  year={2020}
}
```
