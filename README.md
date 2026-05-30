Micro‑Gesture Classification using ResNet‑18
===========================================

Purpose
-------
This project builds a deep‑learning classifier for 32 micro‑gesture classes using a pretrained ResNet‑18 model. 
The pipeline includes dataset preparation, augmentation, balanced sampling, training, validation, testing, and 
CSV prediction generation for an external unlabeled dataset.

Dataset
-------
- 32 gesture classes stored in separate folders.
- Dataset is split into:
  • 80% training
  • 10% validation
  • 10% testing
- External test set (GroupT/) contains 20,439 unlabeled images.

Main Steps
----------
1. Environment setup (PyTorch, Torchvision, GPU check)
2. Mount Google Drive for dataset and model storage
3. Copy and unzip dataset inside Colab RAM
4. Verify dataset structure and count images per class
5. Visualize random sample images
6. Split dataset into train/val/test folders
7. Create PyTorch Datasets and Dataloaders
8. Apply data augmentation (train) and normalization (val/test)
9. Handle class imbalance using WeightedRandomSampler
10. Build model using pretrained ResNet‑18 and replace final layer with 32‑class output
11. Train model for 10 epochs and save:
    • model for each epoch
    • best model based on validation accuracy
12. Evaluate best model on internal test set
13. Generate confusion‑matrix summary and per‑class accuracy
14. Run predictions on external test set and generate CSV:
    filename,class

Model Details
-------------
- Base model: ResNet‑18 pretrained on ImageNet
- Final layer replaced with: Linear(512 → 32)
- Loss: Weighted CrossEntropyLoss (handles class imbalance)
- Optimizer: Adam (lr=0.0001, weight_decay=1e‑4)
- Scheduler: ReduceLROnPlateau (monitors validation accuracy)
- Balanced sampling used during training

Outputs
-------
- Saved model files for all epochs
- best_model.pth (highest validation accuracy)
- Internal test accuracy
- Confusion‑matrix summary table
- Per‑class accuracy report
- Final CSV file for external test set:
  GroupT_T.csv

Technologies Used
-----------------
- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- scikit‑learn
- Google Colab GPU

Author
------
Muhammad Monowarul Sabbir
MSc Data‑Centric Engineering, LUT University
Email: mmsabbir.finland@gmail.com
LinkedIn: linkedin.com/in/muhammad-monowarul-sabbir-024119107
