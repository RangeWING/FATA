# Feature Augmentation based Test-Time Adaptation
The code of

> Feature Augmentation based Test-Time Adaptation

## Environments  

`conda env create --file environment.yaml `

## Dataset
Download [ImageNet-C](https://zenodo.org/record/2235448) and modify `configs/data.yaml`.

## Experiment
After modifying `configs/config.yaml`,

`python main.py -c config`


## Acknowledgment
The code is based on the [DeYO](https://github.com/Jhyun17/DeYO), [Tent](https://github.com/DequanWang/tent), [EATA](https://github.com/mr-eggplant/EATA), and [SAR](https://github.com/mr-eggplant/SAR).
