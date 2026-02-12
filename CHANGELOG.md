# Changelog

All notable changes to the AsymFormer project will be documented in this file.

---

## [2026-02-12] SDPA Optimization for CMA Module

> **Performance Update**: CMA (Cross-Modal Attention) module has been optimized using PyTorch's native `F.scaled_dot_product_attention` (SDPA) for improved inference speed on Apple Silicon MPS.

### SDPA Optimization Details

The `Cross_Atten_Lite_split` class in `src/AsymFormer.py` has been updated to use `F.scaled_dot_product_attention` instead of manual `matmul + softmax` operations. This provides:

- **Better memory access patterns** (similar to Flash Attention)
- **Fused kernel optimization** on supported backends
- **Weight compatibility** - original pretrained weights can still be loaded without modification

### MPS Benchmark Results (Apple M3 Max)

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

### Technical Details

The SDPA implementation handles the dimension mismatch between V (`midc1`) and Q/K (`midc2`) by:
1. Padding V to match Q/K dimensions
2. Applying SDPA with fused kernel
3. Slicing output back to original V dimension

This maintains weight compatibility while gaining performance benefits.

---

## [2026-02-11] PyTorch API Compatibility Update

> **Important Update**: This repository has been comprehensively updated to ensure compatibility with the latest PyTorch APIs and to improve code maintainability. All deprecated APIs have been replaced, and modern best practices have been implemented.

### Core Updates

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

### New Features

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

## Release History

| Date | Version | Summary |
|------|---------|---------|
| 2026-02-12 | SDPA Update | CMA module optimized with SDPA for 15-28% speedup on Apple Silicon |
| 2026-02-11 | Compatibility Update | PyTorch API modernization, new augmentation features, bug fixes |
| 2023 | Initial Release | CVPR 2024 USM Workshop paper release |
