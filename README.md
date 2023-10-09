# AsymFormer: Asymmetrical Cross-Modal Representation Learning for Mobile Platform Real-Time RGB-D Semantic Segmentation (Submitted to ICRA 2024） [[Paper](https://arxiv.org/abs/2309.14065)]

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
- NYUv2: 52.0% mIoU
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
We used the same data source as the ACNet. The processed NYUv2 data (.npy) can be downloaded by [Google Drive](https://drive.google.com/file/d/1YgcBRCjmkLlVukjmvkNu1A7O8bRd14Ek/view?usp=sharing).

## Usage
Currently, we provided ONNX model and TensorRT FP16 model for evaluation and inference. The souce code will be released soon.

- The ONNX model is exported on v17 operation: [[AsymFormer ONNX Model](https://drive.google.com/file/d/1dSqmv19ErpxYcjl95I8buYeQi37qBlXQ/view?usp=sharing)]
- The TensorRT FP 16 model is generated and optimized for RTX 3090 platform. [[AsymFormer FP16 TensorRT Model](https://drive.google.com/file/d/1Z57x6e_YSroMCh3p9ttwKB7P7VLfa81k/view?usp=sharing)]
- If you need run FP16 model on other platform, we provide a jupyter notebook for checking numerical overflow and generating TensorRT engine on your own platform.

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
