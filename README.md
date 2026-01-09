# Biometric Age Prediction

CNN-based age prediction using UTKFace and Facial Age datasets, with dataset preprocessing, augmentation, training, evaluation, and visualization utilities.

## Features
- Automatic dataset download from Kaggle via `kagglehub` and organized preprocessing (face crop placeholders, enhancement, resizing).
- Two-stage EfficientNetB0 training pipeline with fine-tuning and mixed precision.
- Disk-based generators with augmentation (Albumentations) to keep memory use low.
- Evaluation with confusion matrix, demographic analysis, ROC/DET curves, and policy-focused error summaries.
- Visualization helpers for age/gender/ethnicity distributions and per-image prediction display.

## Project Structure
- `main.py` – CLI entrypoint that launches the interactive menu.
- `age_prediction.py` – High-level application flow (train, load, predict, evaluate, save).
- `cnn_model.py` – Model definition, training loop, saving/loading, evaluation hook.
- `dataset_downloader.py` – Kaggle dataset fetcher; manages local dataset folders.
- `dataset_processor.py` – Preprocessing (crop, enhance, resize), metadata, and generator creation.
- `data_generator.py` – Disk-based Keras Sequence with augmentation support.
- `image_processing.py` – Face detector (OpenCV DNN) and image enhancement utilities.
- `evaluator.py` – Metrics, plots, demographic analysis, and policy checks.
- `visualization.py` – Dataset distribution plots and per-sample prediction display.
- `constants.py` – Shared age class ranges and image size.


## Datasets
- UTKFace (`jangedoo/utkface-new`) and Facial Age (`frabbisw/facial-age`) are downloaded automatically into `data/` on first run via kagglehub.
- Preprocessed faces (cropped/enhanced/resized) are stored under `data/preprocessed_faces/` with accompanying `metadata.json`.


## Running
```bash
python main.py
```

CLI options:
1. Train a new model (downloads + preprocesses data, builds generators, trains EfficientNetB0 in two stages).
2. Load an existing model (`models/age_model.keras` by default).
3. Predict on an image (performs preprocessing and shows prediction overlay).
4. Evaluate on test data (accuracy, confusion matrix, demographic breakdowns, ROC/DET).
5. Save current model (default path `models/age_model.keras`).
6. Quit.

## Key Notes
- Image size is fixed at 224x224; normalization happens in the preprocessing pipeline.
- Age classes are defined in `constants.py` (0-9, 10-17, 18-25, 26-35, 36-45, 46-60, 61-74, 75+).
- Mixed precision is enabled by default; ensure your GPU supports it or disable if troubleshooting.
