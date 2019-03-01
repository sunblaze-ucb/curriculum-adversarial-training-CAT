# Curriculum Adversarial Training

This repository contains code to reproduce results from the paper:

**Curriculum Adversarial Training** <br>
*Qi-Zhi Cai, Min Du, Chang Liu, Dawn Song* <br>
Link: https://arxiv.org/abs/1805.04807

#### REQUIREMENTS

The code was tested with Python 2.7.12, Pytorch 0.3.1.

## Usages

#### TRAINING
```
python advtrain_all.py --args
```
#### TESTING
Get per-sample test results for each attack type and store them into files:
```
python test_all_worstcaseacc.py --model_dir ./checkpoint/*** --args
```
Sum total test accuracy from above per-sample statistics:
```
python sumResults.py result_dirs
```

## "models" folder is inherited from [here](https://github.com/kuangliu/pytorch-cifar).
