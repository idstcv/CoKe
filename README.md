# CoKe
PyTorch Implementation for Our CVPR'22 Paper: "Unsupervised Visual Representation Learning by Online Constrained K-Means"

## Requirements
* Python 3.8
* PyTorch 1.6

## Usage:
CoKe with single view
```
sh run_single_view.sh
```

CoKe with two views
```
sh run_double_view.sh
```

CoKe with multi-view
```
sh run_multi_view.sh
```


## Models:

model | epochs | crop | acc on ImageNet | link
-- | -- | -- | -- | --
CoKe | 1000 | 1x224 | 72.5 | [Google Drive](https://drive.google.com/file/d/1VZQlGInbSwytn5Y6KpYTQj5eQUqfSPTR/view?usp=sharing)
CoKe | 1000 | 2x224 | 74.9 | [Google Drive](https://drive.google.com/file/d/1PMBwwSlqa0qrDCR_40XzptHBv8Wsat9p/view?usp=sharing)
CoKe | 800 | 2x224+6x96 | 76.4 | [Google Drive](https://drive.google.com/file/d/11z7UHtshY3USR0MsVICKubYzPwfFd7bV/view?usp=sharing)

## Citation
If you use the package in your research, please cite our paper:
```
@inproceedings{qian2022coke,
  author    = {Qi Qian and
               Yuanhong Xu and
               Juhua Hu and
               Hao Li and
               Rong Jin},
  title     = {Unsupervised Visual Representation Learning by Online Constrained K-Means},
  booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR} 2022},
  year      = {2022}
}
```
