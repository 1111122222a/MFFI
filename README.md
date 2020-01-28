This project contains the officail code of our Modality-free Face Identification (MFFI).
## Usage

1. Install pytorch

+ The code is tested on python3.6 and official [Pytorch@commitfd25a2a](https://github.com/pytorch/pytorch/tree/fd25a2a86c6afa93c7062781d013ad5f41e0504b#from-source), please install PyTorch from source.

2. Dataset
- Download the [CelabA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and LFWA (https://drive.google.com/drive/u/0/folders/0B7EVK8r0v71pQ3NzdzRhVUhSams)
- Please put dataset in folder './dataset/CelebA/' and './dataset/LFWA/' respectively.
- We have upload the split index of probe and gallery set of celebA, and the split index of
train and test set of LFWA.

3. Pretrained model

- Download the pretrain [ir-se50](https://pan.baidu.com/s/12BUjjwy1uUTEF9HCx5qvoQ) 
- Please put it into folder './pretrain/'.

4. Training

- There are four main python files.
- datareader.py contains Celeba and LFWA dataset classes.
- resnet.py contains the architecture of ir-se50
- model.py contains three modules. CNN, Attribute_Network, DCM_Logit and SAM
- train_celeba.py is the main python file, please run this file.

The results will be save into folder './results'.
