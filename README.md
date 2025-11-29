# Hansa

Deep learning-based binary classifier for detecting AI-generated images. Achieves 88%+ accuracy on 32x32 RGB images using a lightweight CNN architecture.

## Why Hansa?

**Hansa** (Sanskrit: हंस) refers to the sacred swan in Hindu mythology, renowned for its mythical ability to separate milk from water when mixed together. This symbolizes the power of discernment and the ability to distinguish truth from falsehood—precisely what this model does by separating real images from AI-generated ones.

## Performance

- **Test Accuracy**: 88.34%
- **Precision**: ~87.44%
- **Recall**: ~89.56%
- **F1-Score**: ~88.48%
- **AUC**: 0.95+
- **Inference Time**: <10ms per image

## Architecture

```
Input (32x32x3)
    ↓
Data Augmentation (flip, rotation, zoom)
    ↓
3 Conv Blocks (32→64→128 filters)
    ↓
Global Average Pooling
    ↓
Dense (128) + Dropout (0.5)
    ↓
Output (sigmoid)
```

**Key Features:**
- Batch Normalization after each conv layer
- L2 Regularization (0.001)
- Dropout (0.2-0.5)
- Adam optimizer (LR: 1e-5)

## Requirements

```bash
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd hansa

# Create virtual environment
python -m venv hansa_env
source hansa_env/bin/activate  # On Windows: hansa_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

```
data/
├── train/
│   ├── REAL/     # 50,000 real images
│   └── FAKE/     # 50,000 AI-generated images
└── test/
    ├── REAL/     # 10,000 real images
    └── FAKE/     # 10,000 AI-generated images
```

**Image Specifications:**
- Format: JPG/PNG
- Size: 32x32 pixels
- Channels: RGB (3)
- Classes: Binary (REAL/FAKE)

## Training

```bash
jupyter notebook hansa.ipynb
```

**Training Configuration:**
- Epochs: 20 (with early stopping)
- Batch Size: 32
- Validation Split: 20%
- Callbacks: Early stopping, model checkpoint, LR reduction

**Outputs:**
- `hansa_best.keras` - Best model checkpoint
- `hansa.keras` - Final model
- `training_config.json` - Training parameters and metrics
- `training_history.png` - Loss/accuracy curves
- `confusion_matrix.png` - Confusion matrix + ROC curve
- `sample_predictions.png` - Prediction examples

## Inference

### Single Image

```bash
python predict.py image.jpg
```

Output:
```
✅ REAL IMAGE DETECTED
   Confidence: 87.34%
```

### Verbose Mode

```bash
python predict.py image.jpg --verbose
```

Output:
```
✅ REAL IMAGE DETECTED
   Confidence: 87.34%
   Real Score: 87.34%
   Fake Score: 12.66%
   Threshold: 0.5
   Inference Time: 8.42ms
   Image: image.jpg
```

### JSON Output

```bash
python predict.py image.jpg --json
```

Output:
```json
{
  "image_path": "image.jpg",
  "predicted_class": "REAL",
  "confidence": 0.8734,
  "real_score": 0.8734,
  "fake_score": 0.1266,
  "threshold": 0.5,
  "inference_time_ms": 8.42
}
```

### Batch Processing

```bash
python predict.py ./images --batch
```

Output:
```
==================================================
BATCH PREDICTION SUMMARY
==================================================
Total Images: 100
Successfully Processed: 100
Errors: 0

✅ Real Images: 52 (52.0%)
⚠️  Fake Images: 48 (48.0%)

Average Confidence: 83.45%
==================================================
```

### Custom Threshold

```bash
python predict.py image.jpg --threshold 0.7
```

### All Options

```bash
python predict.py <path> [options]

Options:
  --json              Output in JSON format
  --batch             Process directory of images
  --threshold FLOAT   Classification threshold (default: 0.5)
  --verbose           Show detailed output
  --quiet             Minimal output
  --model PATH        Path to model file (default: hansa.keras)
```

## Model Files

- **hansa.keras** (3.2 MB) - Trained model
- **training_config.json** - Model metadata and metrics

## Evaluation Metrics

The model provides:
- **Accuracy**: Overall correctness
- **Precision**: Of detected fakes, how many are actually fake
- **Recall**: Of all fakes, how many were detected
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

## Project Structure

```
hansa/
├── Hansa.ipynb                   # Training notebook
├── predict.py                    # Inference script
├── hansa.keras                   # Trained model
├── hansa_best.keras              # Best trained model
├── training_config.json          # Training metadata
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
└── data/                         # Dataset (not included)
    ├── train/
    └── test/
```

## Hardware Requirements

**Training:**
- RAM: 8GB minimum
- GPU: Optional (Apple M1/M2, NVIDIA CUDA)
- Storage: 5GB for dataset

**Inference:**
- RAM: 2GB minimum
- CPU: Any modern processor
- GPU: Not required

## Limitations

- Trained on 32x32 images only
- Binary classification (no multi-class)
- Performance degrades on high-resolution images
- Limited to RGB images
- Dataset-specific (may not generalize to all AI generators)

## Future Improvements

- [ ] Support for higher resolutions (128x128, 256x256)
- [ ] Multi-class detection (identify AI model used)
- [ ] Transfer learning from larger models
- [ ] Ensemble methods
- [ ] Real-time video processing
- [ ] Web API deployment

## License

MIT License - See LICENSE file for details

## Citation

If you use this model in your research, please cite:

```bibtex
@software{hansa2025,
  title={Hansa},
  author={tyrobro},
  year={2025},
  url={https://github.com/tyrobro/Hansa}
}
```

## Contact

For issues, questions, or contributions:
- GitHub Issues: [Create Issue](https://github.com/tyrobro/Hansa/issues)
---

**Last Updated:** November 2025  
**Model Version:** 1.0  
**Framework:** TensorFlow 2.x
