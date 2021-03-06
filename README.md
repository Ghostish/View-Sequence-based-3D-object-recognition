# View-Sequence-based-3D-object-recognition

### Introduction
This project contains the codes for our paper ["**Deeply Exploiting Long-Term View Dependency for 3D Shape Recognition**"](https://ieeexplore.ieee.org/document/8794586/).

--------
### Dependencies
+ python 3.6 (or higher)
+ mxnet 1.5.0 (or higher)
+ tqdm

To train the model on GPUs, please make sure [CUDA](https://developer.nvidia.com/cuda-toolkit) is correctly installed on your device. The codes are well tested with mxnet 1.5.1/1.6.0 under CUDA 9.2.

-----------
### Dataset preparation (ModelNet40/10)
1. Download ModelNet40 and ModelNet10 from their [official site](https://modelnet.cs.princeton.edu/).

2. Render  2d images from all .off files using the matlab codes provided by [MVCNN](https://github.com/suhangpro/mvcnn/tree/master/utils)

3. Follow the folder structure to re-arrange the rendered image files:

       ├── test  
       │   ├── airplane  
       │   │   ├── airplane_0627_0001.jpg  
       │   │   ├── airplane_0627_0002.jpg  
       │   │   ├── ......  
       │   │   ├── airplane_0627_0012.jpg  
       │   │   ├── ......  
       │   ├── bathtub  
       │   │   ├── ......  
       ......  
       ├── train  
       ...... 

You can also directly download our preprocessed datasets from [here](https://drive.google.com/drive/folders/1pWHjDgg2f393wpjQzGezZoTChh6rLWTk?usp=sharing).

---
### Usage

#### Classification

##### Training:

`python train_models.py --batch_size 4 --batch_update_period 32  --num_views 12  --num_classes 10 --dataset_path /xxxxx/xxxx/modelnet10`

Notes: Due to limited GPU memory, we can only feed the network with a small batch size (about 2~4). To prevent unstable training with a small batch size, we manually aggregate the gradient and do back propagation every **batch_update_period**.

##### Testing:
`python test.py --batch_size 4  --num_views 12 --num_classes 10 --dataset_path /xxxxx/xxxx/modelnet10 --checkpoint /xxxx/xxxx/xxx.params`

To reproduce the results in our paper, use the corresponding checkpoints to evaluate the models. Please download the checkpoint files from [here](https://drive.google.com/drive/folders/1v1CfqucWkqEvV-kHPg2NlhVCl67PCT5g?usp=sharing). 


### Citation
if you find our work useful in your research, please considering citing:
>@ARTICLE{8794586,  author={Y. {Xu} and C. {Zheng} and R. {Xu} and Y. {Quan}},  journal={IEEE Access},   title={Deeply Exploiting Long-Term View Dependency for 3D Shape Recognition},   year={2019},  volume={7},  number={},  pages={111678-111691},}

### LICENSE
The repo is released under the MIT License. See the [LICENSE file](./LICENSE) for more details.