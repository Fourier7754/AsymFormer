# AsymFormer: Asymmetrical Cross-Modal Representation Learning for Mobile Platform Real-Time RGB-D Semantic Segmentation [[Paper](https://arxiv.org/abs/2309.14065)] [[Pre-trained Model](https://drive.google.com/file/d/1Pg6r3eJ245GaKbHfZob0Ek0CVhiS7VaR/view?usp=drive_link)] [[TensorRT Model](https://drive.google.com/file/d/1Z57x6e_YSroMCh3p9ttwKB7P7VLfa81k/view?usp=sharing)]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/asymformer-asymmetrical-cross-modal/real-time-semantic-segmentation-on-nyu-depth-1)](https://paperswithcode.com/sota/real-time-semantic-segmentation-on-nyu-depth-1?p=asymformer-asymmetrical-cross-modal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/asymformer-asymmetrical-cross-modal/semantic-segmentation-on-nyu-depth-v2)](https://paperswithcode.com/sota/semantic-segmentation-on-nyu-depth-v2?p=asymformer-asymmetrical-cross-modal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/asymformer-asymmetrical-cross-modal/semantic-segmentation-on-sun-rgbd)](https://paperswithcode.com/sota/semantic-segmentation-on-sun-rgbd?p=asymformer-asymmetrical-cross-modal) [![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/yourusername/repo/blob/main/LICENSE)

<p align="center">
  <img src="https://github.com/Fourier7754/AsymFormer/blob/main/Image/Overall%20Framework%20of%20AsymFormer.png" width="600" height="450">
</p>

This repository contains the official implementation of AsymFormer, a novel network for real-time RGB-D semantic segmentation.

- Achieves efficient and precise RGB-D semantic segmentation
- Allows effective fusion of multimodal features at low computational cost
- Minimizes superfluous parameters by optimizing computational resource distribution
- Enhances network accuracy through feature selection and multi-modal self-similarity features
- Utilizes Local Attention-Guided Feature Selection (LAFS) module for selective fusion
- Introduces Cross-Modal Attention-Guided Feature Correlation Embedding (CMA) module for cross-modal representations

## Results

AsymFormer achieves competitive results on the following datasets:
- NYUv2: 54.1% mIoU
- NYUv2 (Multi-Scale): 55.3% mIoU
- SUNRGBD: 49.1% mIoU

Notably, it also provides impressive inference speeds:
- Inference speed of 65 FPS on RTX3090
- Inference speed of 79 FPS on RTX3090 (FP16)
- Inference speed of 29 FPS on Tesla T4 (FP16)

## Installation

To run this project, we suggest using Ubuntu 20.04, PyTorch 2.0.1, and CUDA version higher than 12.0.

Other necessary package for running the evaluation and TensorRT FP16 quantization inference:
```
pip install timm
pip install scikit-image
pip install opencv-python-headless==4.5.5.64
pip install thop
pip install onnx
pip install oonnxruntime
pip install tensorrt==8.6.0
pip install pycuda
```

## Data Preparation
~~We used the same data source as the ACNet. The processed NYUv2 data (.npy) can be downloaded by [Google Drive](https://drive.google.com/file/d/1YgcBRCjmkLlVukjmvkNu1A7O8bRd14Ek/view?usp=sharing).~~
We find the former NYUv2 data has some mistakes. So we re-generate training data from original NYUv2 matlab .mat file: [Google Drive](https://drive.google.com/file/d/1c18pTIsMX1SJvVPBFpqWa7QILn1NPxTY/view?usp=drive_link).

## Train
To train the AsymFormer on NYUv2 dataset, you need to download the processed png format dataset [Google Drive](https://drive.google.com/file/d/1c18pTIsMX1SJvVPBFpqWa7QILn1NPxTY/view?usp=drive_link). and unzip the file to current folder. After that, the folder should be like:

```
├── data
│   ├── images
│   ├── depths
│   ├── labels
│   ├── train.txt
│   └── test.txt
├── utils
│   ├── __init__.py
│   └── utils.py
├── src
│   └── model files
├── NYUv2_dataloader.py
├── train.py
└── eval.py
```

Then run the train.py script.
```
python train.py
```

**Note:** The training process with batch size 8 requires 19GB GPU VRAM. We will release mixed-precision training script soon wihch will require about 12GB of VRAM. However, the mixed-precision training will only work on Linux platform.

## Eval

Run the eval.py script to evaluate AsymFormer on NYUv2 Dataset.
```
python eval.py
```

If you wish to run evaluation in multi-scale inference strategy, run the MS5_eval.py script:
```
python MS5_eval.py
```

## Model Exporting and Quantization
Currently, we have provided ONNX model and TensorRT FP16 model for evaluation and inference. 

### FP16 Inference (RTX3090 Platform)
The TensorRT inference notebook can be found in [Folder](https://github.com/Fourier7754/AsymFormer/tree/main/Inference). You can test AsymFormer on your local environment by:
- Downlaod the folder 'Inference'
- Downlaod the TensorRT FP 16 model, which generated and optimized for RTX 3090 platform. [[AsymFormer FP16 TensorRT Model](https://drive.google.com/file/d/1Z57x6e_YSroMCh3p9ttwKB7P7VLfa81k/view?usp=sharing)]
- Download the NYUv2 Dataset [NYUv2](https://drive.google.com/file/d/1YgcBRCjmkLlVukjmvkNu1A7O8bRd14Ek/view?usp=sharing).
- Put the 'AsymFormer.engine' in the 'Inference' folder.
- Modify the dataset path to your own path ↓
```
val_data = Data.RGBD_Dataset(transform=torchvision.transforms.Compose([scaleNorm(),
                                                                       ToTensor(),
                                                                       Normalize()]),
                             phase_train=False,
                             data_dir='Your Own Path', ← The file path of the NYUv2 dataset
                             txt_name='test.txt'    
                             )
```
- Run the Jupyter Notebook


### Optimize the AsymFormer for your own platform
You can generate your own TensorRT engine from the ONNX model.
We provide the original ONNX model and corresponding notebook to help you genrate the TensorRT model
- The ONNX model is exported on v17 operation, and it can be downloaded from [[AsymFormer ONNX Model](https://drive.google.com/file/d/1YA1t6IEFvtJSkT6jliWlfYdAVwSHV0Po/view?usp=drive_link)]
- The jupyter notebook contains loading ONNX model, checking numeric overflow and generating mixed-precision TensorRT model, which can be downloaded from [Generate TensorRT](https://github.com/Fourier7754/AsymFormer/blob/main/Notebooks/Generate_TensorRT_Model.ipynb). 

### Training
The The souce code of AsymFormer will be released soon.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Part of the code implementation was adapted from [ACNet's repository](https://github.com/anheidelonghu/ACNet).

If you find this repository useful in your research, please consider citing:

```
@misc{du2023asymformer,
      title={AsymFormer: Asymmetrical Cross-Modal Representation Learning for Mobile Platform Real-Time RGB-D Semantic Segmentation}, 
      author={Siqi Du and Weixi Wang and Renzhong Guo and Shengjun Tang},
      year={2023},
      eprint={2309.14065},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact

For any inquiries, please contact siqi.du1014@outlook.com.
Home page of the author: [Siqi.DU's ResearchGate](https://www.researchgate.net/profile/Siqi-Du-4)
