# AsymFormer
# AsymFormer: Real-time RGB-D Semantic Segmentation

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/yourusername/repo/blob/main/LICENSE)

This repository contains the official implementation of AsymFormer, a novel network for real-time RGB-D semantic segmentation.

- Achieves efficient and precise RGBD semantic segmentation
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
- Inference speed of 79 FPS on RTX3090 after implementing 16-bit quantization

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/repo.git
```

2. Install the dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Download the NYUv2 and SUNRGBD datasets.

2. Preprocess the datasets according to the instructions provided in the repository.

3. Train the AsymFormer model:
```
python train.py --dataset dataset_folder --epochs 50
```

4. Evaluate the trained model:
```
python evaluate.py --dataset dataset_folder --model trained_model.pth
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Part of the code implementation was adapted from [xyz's repository](https://github.com/xyz/repo).
- The AsymFormer network architecture was inspired by [abc's work](https://arxiv.org/abs/1234.56789).

If you find this repository useful in your research, please consider citing:

```
@article{yourcitationhere,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2022}
}
```

## Contact

For any inquiries, please contact yourname@gmail.com.
