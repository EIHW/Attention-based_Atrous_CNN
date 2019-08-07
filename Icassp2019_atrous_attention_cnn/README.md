Zhao Ren
ZD.B chair of Embedded Intelligence for Health Care and Wellbeing
University of Augsburg
01.08.2019


# Data: 
DCASE 2018 Task 1 - Acoustic Scene Classification

# Code: 
Attention-based atrous convolutional neural networks

# preparation

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

# run 
sh runme.sh



If the user referred or modified the code, please cite our paper,

@InProceedings{ren2019attention,
  Title                    = {{Attention-based atrous convolutional neural networks: Visualisation and understanding perspectives of acoustic scenes}},
  Author                   = {Ren, Zhao and Kong, Qiuqiang and Han, Jing and Plumbley, Mark and Schuller, Bj\"orn},
  Booktitle                = {Proc.\ ICASSP},
  Year                     = {2019},
  Address                  = {Brighton, UK},
  Pages                    = {56--60}
}



