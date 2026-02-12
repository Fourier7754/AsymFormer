# AsymFormer: Asymmetrical Cross-Modal Representation Learning for Mobile Platform Real-Time RGB-D Semantic Segmentation (CVPR 2024 - USM Workshop ) [[Paper](https://openaccess.thecvf.com/content/CVPR2024W/USM/papers/Du_AsymFormer_Asymmetrical_Cross-Modal_Representation_Learning_for_Mobile_Platform_Real-Time_RGB-D_CVPRW_2024_paper.pdf)] [[Pre-trained Model](https://drive.google.com/file/d/1Pg6r3eJ245GaKbHfZob0Ek0CVhiS7VaR/view?usp=drive_link)] [[TensorRT Model](https://drive.google.com/file/d/1Z57x6e_YSroMCh3p9ttwKB7P7VLfa81k/view?usp=sharing)]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/asymformer-asymmetrical-cross-modal/real-time-semantic-segmentation-on-nyu-depth-1)](https://paperswithcode.com/sota/real-time-semantic-segmentation-on-nyu-depth-1?p=asymformer-asymmetrical-cross-modal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/asymformer-asymmetrical-cross-modal/semantic-segmentation-on-nyu-depth-v2)](https://paperswithcode.com/sota/semantic-segmentation-on-nyu-depth-v2?p=asymformer-asymmetrical-cross-modal) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/asymformer-asymmetrical-cross-modal/semantic-segmentation-on-sun-rgbd)](https://paperswithcode.com/sota/semantic-segmentation-on-sun-rgbd?p=asymformer-asymmetrical-cross-modal) [![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/yourusername/repo/blob/main/LICENSE)

<p align="center">
  <img src="https://github.com/Fourier7754/AsymFormer/blob/8c5cee55f38123d958e26d5fb053b30a26ebdae5/Image/Overall%20Framework%20of%20AsymFormer.png" width="600" height="450">
</p>

This repository contains the official implementation of AsymFormer, a novel network for real-time RGB-D semantic segmentation.

- Achieves efficient and precise RGB-D semantic segmentation
- Allows effective fusion of multimodal features at low computational cost
- Minimizes superfluous parameters by optimizing computational resource distribution
- Enhances network accuracy through feature selection and multi-modal self-similarity features
- Utilizes Local Attention-Guided Feature Selection (LAFS) module for selective fusion
- Introduces Cross-Modal Attention-Guided Feature Correlation Embedding (CMA) module for cross-modal representations

## 📊 Results

AsymFormer achieves competitive results on the following datasets:
- NYUv2: 54.1% mIoU
- NYUv2 (Multi-Scale): 55.3% mIoU
- SUNRGBD: 49.1% mIoU

Notably, it also provides impressive inference speeds:
- Inference speed of 65 FPS on RTX3090
- Inference speed of 79 FPS on RTX3090 (FP16)
- Inference speed of 29 FPS on Tesla T4 (FP16)

---

## 🔄 Changelog (2026-02-12 Update)

> **Performance Update (Feb 12, 2026)**: CMA (Cross-Modal Attention) module has been optimized using PyTorch's native `F.scaled_dot_product_attention` (SDPA) for improved inference speed on Apple Silicon MPS.

### ⚡ SDPA Optimization for CMA Module

The `Cross_Atten_Lite_split` class in `src/AsymFormer.py` has been updated to use `F.scaled_dot_product_attention` instead of manual `matmul + softmax` operations. This provides:

- **Better memory access patterns** (similar to Flash Attention)
- **Fused kernel optimization** on supported backends
- **Weight compatibility** - original pretrained weights can still be loaded without modification

### 📊 MPS Benchmark Results (Apple M3 Max)

Testing device: Apple M3 Max (MPS backend)

#### Full Model Inference Speed (B0_T)

| Resolution | Latency (ms) | FPS |
|------------|--------------|-----|
| 224×224 | 15.96 | 62.66 |
| 320×320 | 20.30 | 49.26 |
| 480×640 | 29.27 | 34.16 |

#### CMA Module Speed Comparison (Original vs SDPA)

| Resolution | Original (ms) | SDPA (ms) | Speedup |
|------------|---------------|-----------|---------|
| 60×80 | 5.08 | 4.42 | **1.15x** |
| 120×160 | 74.39 | 58.24 | **1.28x** |

#### Accuracy Verification

| Metric | Original | SDPA |
|--------|----------|------|
| mIoU (NYUv2) | 54.0% | 54.0% ✓ |
| Accuracy | 78.0% | 78.0% ✓ |

**Conclusion**: SDPA provides **15-28% speedup** on CMA module with **zero accuracy loss**.

### 🔧 Technical Details

The SDPA implementation handles the dimension mismatch between V (`midc1`) and Q/K (`midc2`) by:
1. Padding V to match Q/K dimensions
2. Applying SDPA with fused kernel
3. Slicing output back to original V dimension

This maintains weight compatibility while gaining performance benefits.

---

## 🔄 Changelog (2026-02-11 Update)

> **Important Update (Feb 2026)**: This repository has been comprehensively updated to ensure compatibility with the latest PyTorch APIs and to improve code maintainability. All deprecated APIs have been replaced, and modern best practices have been implemented.

### 📦 Core Updates

#### 1. API Compatibility Improvements
- **Device Detection**: Updated device detection to support multiple platforms (CUDA, MPS for Apple Silicon, CPU)
- **Weight Loading**: Added `map_location='cpu'` parameter to `torch.load()` calls for CPU compatibility
- **CUDA Synchronization**: Wrapped `torch.cuda.synchronize()` calls with `torch.cuda.is_available()` checks to prevent errors on non-CUDA systems
- **Import Handling**: Added try-except blocks for `timm.models.registry.register_model` import to handle different timm versions

#### 2. Data Loading Enhancements
- **Modern Transforms**: Migrated to `torchvision.transforms.v2` API for improved performance and functionality
- **Advanced Data Augmentation**: Added support for:
  - CutMix augmentation
  - MixUp augmentation
  - Mosaic augmentation
- **Preloading Option**: Added optional data preloading for faster training
- **Thread Pool Loading**: Implemented concurrent file loading using `ThreadPoolExecutor`
- **Progress Display**: Added tqdm progress bars for better visibility during data loading

#### 3. Training Pipeline Improvements
- **GPU Augmentation**: Implemented `GPUAugmentation` class for on-the-fly augmentation on GPU
- **Automatic Evaluation**: Added automatic evaluation after training completion
- **Checkpoint Management**: Improved checkpoint saving and loading logic with fallback mechanisms
- **Learning Rate Scheduler**: Implemented warmup-based learning rate scheduler

#### 4. Code Structure Updates
- **AsymFormer.py**:
  - Enhanced `load_pretrain2()` function with multiple search paths for pretrain weights
  - Added automatic download fallback if pretrain weights not found locally
  - Improved error handling and logging
- **convnext.py**:
  - Updated import compatibility for newer timm versions
  - Fixed LayerNorm implementation
- **mix_transformer.py**:
  - Maintained compatibility with latest PyTorch and timm APIs
- **MLPDecoder.py**:
  - Kept stable implementation with no breaking changes

#### 5. Evaluation Script Updates
- **eval.py**:
  - Added `--save-json` flag for saving evaluation results to JSON format
  - Added `--json-path` parameter to specify output JSON file location
  - Improved error handling and logging
  - Fixed CUDA Event timing for CPU compatibility
- **MS5_eval.py**:
  - Similar updates as eval.py for multi-scale evaluation

#### 6. Utility Functions
- **utils/utils.py**:
  - Maintained existing utility functions
  - Fixed CrossEntropyLoss2d compatibility with newer PyTorch versions
- **dataloader.py** (both NYUv2 and SUNRGBD):
  - Updated to use modern torchvision transforms
  - Added debug logging for troubleshooting
  - Improved error handling for missing files

### 🚀 New Features

#### Automatic Evaluation After Training
- Training script now automatically evaluates the trained model after completion
- Evaluation uses the latest saved checkpoint by default
- Results can be saved to JSON format for easy analysis

### Bug Fixes

- Fixed CUDA-related errors when running on CPU-only systems
- Fixed import errors for newer timm library versions
- Fixed file path issues in dataset loading
- Fixed checkpoint loading with incorrect keys
- Fixed synchronization issues in timing code

### Backward Compatibility

The updated code maintains backward compatibility with the original 2023 implementation:
- Original command-line arguments still work
- Checkpoint format remains unchanged
- Dataset format (PNG) remains unchanged
- Evaluation output format remains the same (with optional JSON output)

### Migration Guide

If you have existing code from the 2023 version, the main changes you need to be aware of:

1. **Device Handling**: Replace hardcoded `cuda:0` with automatic device detection
2. **Data Loading**: Update to use `torchvision.transforms.v2` if you want new augmentation features
3. **Checkpoint Loading**: Add `map_location='cpu'` if you might load on CPU
4. **CUDA Operations**: Wrap CUDA-specific operations with `torch.cuda.is_available()` checks

---

## 🛠️ Installation

To run this project, we suggest using Ubuntu 20.04, PyTorch 2.0.1, and CUDA version higher than 12.0.

Other necessary packages for running evaluation and TensorRT FP16 quantization inference:
```
pip install timm
pip install scikit-image
pip install opencv-python-headless==4.5.5.64
pip install thop
pip install onnx
pip install onnxruntime
pip install tensorrt==8.6.0
pip install pycuda
```

## 📁 Data Preparation

~~We used the same data source as the ACNet. The processed NYUv2 data (.npy) can be downloaded by [Google Drive](https://drive.google.com/file/d/1YgcBRCjmkLlVukjmvkNu1A7O8bRd14Ek/view?usp=sharing).~~

We found the former NYUv2 data has some mistakes. So we re-generated the training data from the original NYUv2 matlab .mat file: [Google Drive](https://drive.google.com/file/d/1c18pTIsMX1SJvVPBFpqWa7QILn1NPxTY/view?usp=drive_link).

SUNRGBD Dataset: [Google Drive](https://drive.google.com/file/d/1CcbUuLi0QdN7LwbieHVmutRyxaKyczpy/view?usp=sharing)

## 🏋️ Train

To train AsymFormer on the NYUv2 dataset, you need to download the processed png format dataset [Google Drive](https://drive.google.com/file/d/1c18pTIsMX1SJvVPBFpqWa7QILn1NPxTY/view?usp=drive_link) and unzip the file to the current folder. After that, the folder should look like:

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

Then run the train.py script:
```bash
python train.py
```

**Note:** The training process with batch size 8 requires 19GB GPU VRAM. We will release a mixed-precision training script soon which will require about 12GB of VRAM. However, mixed-precision training will only work on Linux platform.

After training completes, the model will automatically be evaluated using the latest checkpoint. Results will be saved to a JSON file in the checkpoint directory.

## Eval

Run the eval.py script to evaluate AsymFormer on the NYUv2 Dataset:
```bash
python eval.py
```

If you wish to run evaluation in multi-scale inference strategy, run the MS5_eval.py script:
```bash
python MS5_eval.py
```

## Model Exporting and Quantization

Currently, we have provided ONNX model and TensorRT FP16 model for evaluation and inference.

### FP16 Inference (RTX3090 Platform)

The TensorRT inference notebook can be found in the [Inference folder](https://github.com/Fourier7754/AsymFormer/tree/main/Inference). You can test AsymFormer on your local environment by:

- Download the 'Inference' folder
- Download the TensorRT FP16 model, which was generated and optimized for RTX 3090 platform. [[AsymFormer FP16 TensorRT Model](https://drive.google.com/file/d/1Z57x6e_YSroMCh3p9ttwKB7P7VLfa81k/view?usp=sharing)]
- Download the NYUv2 Dataset [NYUv2](https://drive.google.com/file/d/1YgcBRCjmkLlVukjmvkNu1A7O8bRd14Ek/view?usp=sharing)
- Put 'AsymFormer.engine' in the 'Inference' folder
- Modify the dataset path to your own path:

```python
val_data = Data.RGBD_Dataset(transform=torchvision.transforms.Compose([scaleNorm(),
                                                                       ToTensor(),
                                                                       Normalize()]),
                             phase_train=False,
                             data_dir='Your Own Path',  # The file path of the NYUv2 dataset
                             txt_name='test.txt'
                             )
```
- Run the Jupyter Notebook

### Optimize AsymFormer for Your Own Platform

You can generate your own TensorRT engine from the ONNX model.
We provide the original ONNX model and a corresponding notebook to help you generate the TensorRT model:

- The ONNX model is exported on v17 operation, and it can be downloaded from [[AsymFormer ONNX Model](https://drive.google.com/file/d/1YA1t6IEFvtJSkT6jliWlfYdAVwSHV0Po/view?usp=drive_link)]
- The Jupyter notebook contains loading ONNX model, checking numeric overflow and generating mixed-precision TensorRT model, which can be downloaded from [Generate TensorRT](https://github.com/Fourier7754/AsymFormer/blob/main/Notebooks/Generate_TensorRT_Model.ipynb).

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
