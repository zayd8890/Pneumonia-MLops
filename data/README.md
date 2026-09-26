# Data Directory

Organize your chest X-ray image dataset using the standard `ImageFolder` structure:

```
data/
├── train/
│   ├── NORMAL/
│   │   ├── normal_001.jpeg
│   │   └── ...
│   └── PNEUMONIA/
│       ├── pneumonia_001.jpeg
│       └── ...
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Alternatively, dataset splits can also be named `train_split`, `val_split`, and `test_split` (matching the naming convention in the Colab notebooks).
