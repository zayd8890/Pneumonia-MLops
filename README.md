# Pneumonia Detection MLOps Pipeline

Production-ready modular deep learning codebase for chest X-ray pneumonia classification, supporting both **DenseNet** and **ResNet** architectures.

The model implementations strictly match the architectures and layer naming used during original notebook experiments, guaranteeing 100% backward-compatibility when loading pre-trained `.pth` checkpoints.

---

## 📁 Project Architecture

```
image-classification-mlops/
│
├── data/                      # Dataset splits (train/val/test)
├── notebooks/                 # Exploratory & research Jupyter notebooks
├── src/
│   ├── data/                  # Dataset, dataloader, and transforms
│   │   ├── dataset.py
│   │   └── dataloader.py
│   ├── models/                # DenseNet and ResNet exact definitions
│   │   ├── densenet.py
│   │   ├── resnet.py
│   │   └── model_factory.py
│   ├── training/              # Training loop, loss, checkpointing
│   │   └── trainer.py
│   ├── evaluation/            # Metrics, confusion matrix, predictions CSV
│   │   └── evaluator.py
│   └── inference/             # Single and batch image predictor
│       └── predictor.py
│
├── tests/                     # Architecture and pipeline unit tests
│   ├── test_models.py
│   └── test_data.py
│
├── configs/
│   └── config.yaml            # Hyperparameters and experiment configurations
│
├── train.py                   # Main training CLI
├── evaluate.py                # Evaluation CLI (alias: eval.py)
├── eval.py                    # Evaluation CLI wrapper
├── predict.py                 # Inference CLI
│
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation
└── .gitignore                 # Version control exclusions
```

---

## 🚀 Getting Started

### 1. Requirements & Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

Place your images in `data/` using standard folder structure:

```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

*(Alternatively, splits named `train_split`, `val_split`, and `test_split` are also automatically recognized).*

---

## 🏋️‍♂️ Training (`train.py`)

Train with either **DenseNet** or **ResNet** by passing `--model`:

### Train DenseNet-201 (Default)
```bash
python train.py --model densenet201 --data_dir data --epochs 20 --batch_size 32 --lr 0.001
```

### Train ResNet-18
```bash
python train.py --model resnet18 --data_dir data --epochs 20 --batch_size 32 --lr 0.001
```

### Train Using Configuration File
```bash
python train.py --config configs/config.yaml
```

**Supported Model Names:**
- DenseNet: `densenet121`, `densenet169`, `densenet201`, `densenet264`
- ResNet: `resnet18`, `resnet34`, `resnet101`

Checkpoints and training history JSON are saved automatically into `models/`.

---

## 📊 Evaluation (`evaluate.py` / `eval.py`)

Evaluate a trained model checkpoint against the test dataset:

```bash
python evaluate.py --model densenet201 --weights models/densenet201.pth --data_dir data
```
or using `eval.py`:
```bash
python eval.py --model resnet18 --weights models/resnet.pth --test_dir data/test
```

### Evaluation Outputs
- **Metrics Report:** Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix printed and saved to `results/metrics_<model>.json`.
- **Predictions CSV:** Matches the original notebook format:
  `True_Labels`, `Predicted_Labels`, `Probability_Class_0`, `Probability_Class_1` saved to `results/test_predictions_<model>.csv`.

---

## 🔮 Inference (`predict.py`)

### Predict Single Image
```bash
python predict.py --model densenet201 --weights models/densenet201.pth --image path/to/xray.jpeg
```

**Sample Output:**
```text
================ Prediction Result ================
Image:            path/to/xray.jpeg
Predicted Class:  PNEUMONIA (ID: 1)
Confidence:       98.42%
Probabilities:
  - NORMAL: 1.58%
  - PNEUMONIA: 98.42%
===================================================
```

### Batch Prediction on Directory
```bash
python predict.py --model resnet18 --weights models/resnet.pth --image_dir data/test/PNEUMONIA --output results/batch_preds.csv
```

---

## 🧪 Running Tests

Verify architecture shapes and transformations:

```bash
pytest tests/
```
