# Dogs vs. Cats: Image Classification with Transfer Learning

Can a model trained on millions of ImageNet photos learn to tell a cat from a dog with near-perfect accuracy? This project finds out — and benchmarks the answer on Kaggle's public leaderboard.

---

## The Task

Binary image classification on 25,000 labeled training images (12,500 cats, 12,500 dogs) from the [Kaggle Dogs vs. Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) competition. The evaluation metric is log loss — lower is better, with the target set at < 0.05.

---

## Approach

Two models were built and compared to isolate the value of transfer learning:

**Model 1 — Custom CNN Baseline**
A 3-block convolutional network trained from scratch. Establishes a performance floor before applying pretrained weights.

**Model 2 — ResNet50 with Transfer Learning + Fine-Tuning**
Two-phase training strategy:
1. **Warmup phase** — freeze the ResNet50 base, train only the classification head
2. **Fine-tuning phase** — unfreeze the top 30 layers and retrain at a very low learning rate

Augmentation (horizontal flip, rotation, zoom) and regularization (Dropout 0.5, EarlyStopping, ReduceLROnPlateau) applied throughout to prevent overfitting.

---

## Results

| Model | Val Accuracy | Val Loss |
|-------|-------------|----------|
| Custom CNN Baseline | 80.8% | 0.4113 |
| ResNet50 Transfer Learning | **98.5%** | **0.0447** |

**Kaggle Public Leaderboard: 0.05899 log loss** ✓ *(target: < 0.05 — achieved)*

Transfer learning delivered an 18-percentage-point accuracy gain over the baseline, confirming that pretrained ImageNet features transfer strongly to animal classification tasks.

---

## Key Technical Decisions

- Used `preprocess_input` instead of manual rescaling for ResNet50 — critical for correct normalization
- Chose `GlobalAveragePooling2D` over `Flatten` to reduce parameters and improve generalization
- Loaded cached Kaggle weights to avoid internet dependency during training
- Validated using a fixed 80/20 split with `seed=42` for reproducibility

---

## Files

| File | Description |
|------|-------------|
| `Dogs-vs-Cats-Classification.ipynb` | Full notebook: data pipeline, CNN baseline, ResNet50 training, Kaggle submission |
| `Classification-Summary.pdf` | Results summary with training curves and model comparison |

---

## Tools

Python · TensorFlow / Keras · ResNet50 · NumPy · Matplotlib · Kaggle
