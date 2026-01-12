# Age Classification System - Project Summary

**Date:** December 24, 2025  
**Status:** Development Complete - Domain Adaptation Needed

---

## System Overview

Multi-task age classification system combining:
- **Classification head**: 8 age classes (0-12, 13-17, 18-25, 26-35, 36-45, 46-60, 61-74, 75+)
- **Regression head**: Continuous age estimation
- **Hybrid approach**: Uses regression when classification confidence is low

---

## Implemented Features

### 1. Architecture
- **Backbone**: EfficientNetB1 (pretrained on ImageNet)
- **Input**: 224×224 RGB images
- **Multi-task head**: Softmax classification + Linear regression
- **Regularization**: L2 (0.001), Dropout (0.5/0.6), Label smoothing (0.1)
- **Optimizer**: AdamW with two-stage training (freeze/fine-tune)

### 2. Training Improvements
✅ Balanced class sampling (800 samples per class)  
✅ Gaussian soft labels for boundary smoothing  
✅ Moderate class weights (1.2-1.4× for weak classes)  
✅ Advanced augmentation pipeline (RandomResizedCrop, ColorJitter, etc.)  
✅ EarlyStopping (patience=6) + ReduceLROnPlateau (patience=3)  
✅ Test-time augmentation (TTA): original/flip/center-crop

### 3. Inference Enhancements
✅ **Boundary nudging**: Regression adjusts softmax near class boundaries  
✅ **Hybrid prediction**: Falls back to regression-binning when confidence < 35%  
✅ **Enhanced reject policy**: Flags low confidence OR regression near boundary  
✅ **Temperature calibration**: T=5.187 (saved, but not applied due to shape issues)

### 4. Evaluation Metrics
✅ Softmax accuracy + confusion matrix  
✅ Regression-binning (exact + ±1 class)  
✅ Per-class accuracy breakdown  
✅ Data inspection tool for weak classes

### 5. Tools & Scripts
- `main.py` / `src/age_prediction.py`: Interactive menu system
- `test_prediction.py`: Quick single-image test
- `test_webcam.py`: Auto-capture 3 webcam frames
- `calibrate_and_save.py`: Temperature scaling calibration
- `src/data_inspection.py`: Visual inspection of weak classes

---

## Current Results

### Test Set Performance (UTKFace + Facial Age)
```
Dataset: 6400 images (balanced, 800 per class)
Split: 5120 train / 1280 test

Softmax Accuracy: 64.77% (829/1280)
Regression-binning (exact): 60.39%
Regression-binning (±1 class): 88.44%  ← Key metric!

Per-class accuracy:
  0-12:   74.59% ✓ Strong
  13-17:  79.01% ✓ Strong
  18-25:  64.15% ⚠ Weak (boundary confusion with 26-35)
  26-35:  44.10% ⚠ WEAK (main problem class)
  36-45:  58.33% ⚠ Weak
  46-60:  50.98% ⚠ Weak (boundary confusion with 36-45 and 61-74)
  61-74:  72.55% ✓ Strong
  75+:    72.85% ✓ Strong
```

### Key Insight
**Within-one-class: 88.44%** — Most "errors" are adjacent classes (e.g., 25 vs 26), which is acceptable for real-world use.

---

## Known Limitations

### 🔴 Critical: Domain Gap
**Problem**: Model performs well on test set (UTKFace/Facial Age) but fails on webcam/selfie images.

**Evidence**:
- Test image (professional photo): Predicted 13-17, Reg: 26.0 (actual: 23)
- Webcam images: Inconsistent predictions (2-year variance on identical poses)
- Softmax confidence consistently low (~22-26%) on out-of-domain images

**Root Cause**:
- Training data (UTKFace/Facial Age): Studio photos, professional lighting, specific camera characteristics
- Real-world data: Webcam, selfies, variable lighting, different compression

**Impact**: System is **not production-ready** for retail/webcam scenarios without domain adaptation.

---

## Recommendations

### Short-term (Production-Ready)
1. **Fine-tune on webcam/selfie dataset**:
   - Option A: IMDB-WIKI (500k+ celebrity selfies) — 2 hours
   - Option B: AgeDB or AFAD (real-world images) — 1-2 hours
   - Fine-tune for 5-10 epochs with high augmentation

2. **Use pretrained selfie model**:
   - FairFace (pretrained on selfies)
   - DEX (state-of-the-art age estimation)
   - Integrate as alternative backbone — 30 min

3. **Collect custom dataset**:
   - 50-100 webcam images from target environment (butikk)
   - Label with ground-truth ages
   - Fine-tune on domain-specific data

### Long-term (Research)
- **Ensemble**: Combine multiple models (UTKFace + IMDB-WIKI + FairFace)
- **Semi-supervised**: Use unlabeled webcam data with pseudo-labels
- **Active learning**: Iteratively label hard cases
- **Better preprocessing**: Test MediaPipe Face Mesh vs OpenCV DNN

---

## How to Use

### 1. Train Model (if starting fresh)
```powershell
python main.py
# Select option 1
```

### 2. Evaluate Model
```powershell
python main.py
# Select option 4
```

### 3. Test on Single Image
```powershell
python test_prediction.py
# Edit image path in script first
```

### 4. Test with Webcam
```powershell
python test_webcam.py
# Auto-captures 3 frames, saves to webcam_predictions/
```

### 5. Calibrate Confidence (Optional)
```powershell
python calibrate_and_save.py
```

---

## Technical Details

### Datasets
- **UTKFace**: ~23k images, age 0-116
- **Facial Age**: ~14k images, age-labeled folders
- **Cache**: `data/processed_data_cache.npz` (6400 balanced images)

### Model Files
- `models/age_model.keras`: Trained multi-task model
- `models/age_model_testdata.npz`: Test split + temperature parameter

### Configuration
- `IMAGE_SIZE`: (224, 224)
- `CONFIDENCE_THRESHOLD`: 0.35
- `BATCH_SIZE`: 32
- `LEARNING_RATE`: 3e-4 (stage 1), 2e-4 (stage 2)

---

## Conclusion

**What works:**
- Strong performance on test set (88.44% within-one-class)
- Robust boundary handling with regression + nudging
- Good infrastructure for production (reject policy, hybrid approach)

**What doesn't work:**
- Generalization to webcam/selfie images (domain gap)
- Confidence scores on out-of-domain data

**Next step:** Fine-tune on IMDB-WIKI or similar selfie dataset before production deployment.

---

## Contact & Future Work

For production deployment in retail/butikk scenario:
1. Download IMDB-WIKI dataset
2. Fine-tune model on selfie images (5-10 epochs)
3. Re-evaluate on webcam test set
4. Deploy with confidence threshold = 0.35

Estimated effort: **2-3 hours** to production-ready system.
