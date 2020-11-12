# Attention-based_Atrous_CNN
Pytorch code for the paper 'Attention-based Atrous Convolutional Neural Networks: Visualisation and Understanding Perspectives of Acoustic Scenes', by Zhao Ren, Qiuqiang Kong, Jing Han, Mark Plumbley, Björn Schuller.

# Data
DCASE 2018 Task 1 - Acoustic Scene Classification, containing:

subtask A: data from device A

subtask B: data from device A, B, and C

# Preparation
channels:
  - pytorch
dependencies:
  - matplotlib=2.2.2
  - numpy=1.14.5
  - h5py=2.8.0
  - pytorch=0.4.0
  - pip:
    - audioread==2.1.6
    - librosa==0.6.1
    - scikit-learn==0.19.1
    - soundfile==0.10.2

# Run 
sh runme.sh

In runme.sh, please run the following files for different tasks:
1. feature extraction: utils/features.py
2. training a model, and evaluation: main_pytorch.py

# Cite
If the user referred the code, please cite our paper,

@InProceedings{ren2019attention,

  Title                    = {{Attention-based atrous convolutional neural networks: Visualisation and understanding perspectives of acoustic scenes}},
  
  Author                   = {Ren, Zhao and Kong, Qiuqiang and Han, Jing and Plumbley, Mark and Schuller, Bj\"orn},
  
  Booktitle                = {Proc.\ ICASSP},
  
  Year                     = {2019},
  
  Address                  = {Brighton, UK},
  
  Pages                    = {56--60}
  
}





Zhao Ren

chair of Embedded Intelligence for Health Care and Wellbeing

University of Augsburg

07.08.2019
