# Introduction

This repository contains the source code for the paper titled 'No Escape: Deep-feature Reconstruction Disentanglement Cloth-Changing Person Re-Identification.'

If you encounter any issues with the code or if the provided link becomes inaccessible, please inform us by creating a GitHub issue. Your collaboration and acknowledgment are greatly appreciated. Thank you for citing our work!"



# Dataset used in our paper

-  The pre-processed datasets are available in this [link](https://pan.baidu.com/s/1QUEDns5o51byDEDtp4xZSw) (password: dhdx)  
- If the link is no longer valid, please let me know through GitHub issues. Thank you!
- Then unzip them to ./datasets folders.



# Dependencies

- Python 3.9
- torchvision>=0.15.0
- h5py>=3.9.0
- numpy>=1.21.5
- pillow>=9.4.0
- pyyaml>=6.0
- yacs>=0.1.8
- opencv-python>=4.7.0.72
- tqdm>=4.65.0



# How to Use

- Replace `_C.DATA.ROOT` and `_C.OUTPUT` in `configs/default_img.py`with your own `data path` and `output path`, respectively.

- Start training by executing the following commands.

1. For LTCC dataset: `python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset ltcc --cfg configs/res50_cels_cal.yaml --gpu 0,1 --spr 0 --sacr 0.05 --rr 1.0`
2. For PRCC dataset: `python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset prcc --cfg configs/res50_cels_cal.yaml --gpu 2,3 --spr 1.0 --sacr 0.05 --rr 1.0`



# Visual Results



![可视化](fig\2.svg)



![可视化](fig\3.svg)

If you have any questions, feel free to contact us through github.

