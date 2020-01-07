# WaveletCNN for Texture Classification

This is a Caffe implementation of a paper, [Wavelet Convolutional Neural Networks for Texture Classification](https://arxiv.org/abs/1707.07394) (arXiv, 2017).

## Requirements

- Python 3+
- Caffe

This code was tested with NVIDIA GeForce GTX 1080 on Windows 10.

When you set up the environment of Caffe on Windows, we recommend you to use [this version of Caffe](https://github.com/willyd/caffe/tree/windows), instead of [the windows branch of the official Caffe](https://github.com/BVLC/caffe/tree/windows).

## Pre-trained Model

Please go to `models` directory and follow the instructions.

## Usage

### Train with your own dataset

```
python run_waveletcnn.py --phase train --gpu 0 --dataset path/to/dataset_lmdb
```

To run this code, you have to prepare your dataset as LMDB format. (Otherwise, you have to rewrite `models/WaveletCNN_4level.prototxt`)

And you might need to rewrite some settings in `models/solver_WaveletCNN_4level.prototxt`, such as **test_iter**, **base_lr** and **max_iter**.

### Test with an image

```
python run_waveletcnn.py --phase test --gpu 0 --initmodel models/ImageNet_waveletCNN_level4.caffemodel --target_image path/to/image
```

Using the pre-trained model with ImageNet 2012 dataset, please download it following the instructions in `models` directory.

When you use your own trained model, path to the model is used instead as the argument of **--initmodel**. Additionally, you need to rewrite the path to the file, which contains label names, in `run_waveletcnn.py` (l.71).

### Test on a dataset
*Modified by Hubert Leterme*

```
python run_waveletcnn.py --phase test --gpu -1 --initmodel models/ImageNet_waveletCNN_level4.caffemodel --target_path path/to/dataset [--target_label n0XXXXXX]
```
The path to the ImageNet dataset should be something like `/home/username/dataset/val/`. The directory contains one folder for each class, named `n0XXXXXX`. Each of these folders contain images belonging to the given class.

If no target label is given, then the model will be tested on the entire dataset.


## Citation

If you find this code useful for your research, please cite our paper:

```
@article{Fujieda2017,
  author    = {Shin Fujieda and Kohei Takayama and Toshiya Hachisuka},
  title     = {Wavelet Convolutional Neural Networks for Texture Classification},
  journal   = {arXiv:1707.07394 [cs.CV]},
  year      = {2017},
}
```
