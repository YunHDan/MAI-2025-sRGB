# Environment

First you need to create a new virtual environment.

```raw
conda create -n Enhanceformer python=3.7
conda activate Enhanceformer
```

Install Dependencies:

```raw
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

# Datasets

If you need to training, you need to unzip dped.zip.
Please place the training set and testing set in the Enhanceformer root directory (same level as README.md).

# Training

```raw
python basicsr/train.py --opt Options/Enhanceformer_MAI.yml
```

# Testing

Make sure your test set path is modified in Enhanceformer_MAI.yml. Then use the code in shell:

```raw
python basicsr/test.py --opt Options/Enhanceformer_MAI.yml
```

# Pretraining Model

Our trained model is net_g_latest.pth.