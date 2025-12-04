# MGKD
This repo provides a demo for the paper "Multi-Granularity Knowledge Distillation via Subspace-aware Representation" on the CIFAR-100 dataset.
![image](https://github.com/wwoww1/MGKD/blob/main/fig1.jpg)
# Requirements
python 3.9.12 (Anaconda version >=5.3.1 is recommended)

torch (torch version >= 2.8.0 is recommended)

torchaudio (torchaudio version >= 2.6.0 is recommended)

torchvision (torchvision version >= 0.22.0 is recommended)

pandas

numpy

NVIDIA GPU + CUDA CuDNN

# Running
1. Fetch the pretrained teacher models by:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models`
   
2. Run distillation by following commands. An example of running our TeKAP logit level (we use KL divergence as the base KD) is given by:
    ```
    python train_student.py
    ``` 
