# Pest Identification Model

## Overview
This project is a deep learning-based **pest identification model** that classifies pest species using **transfer learning** with a **ResNet50 backbone**. It integrates **prototype learning** and **distance-based entropy estimation** to enhance model robustness and interpretability.

## Features
- **Transfer Learning:** Uses a **pre-trained ResNet50** model to extract high-level features, fine-tuned for pest classification.
- **Data Augmentation:** Enhances generalization through **rotation, zooming, flipping, and shifting** transformations.
- **Prototype Learning:** Computes **class prototypes** (average feature representation) for improved decision-making.
- **Uncertainty Estimation:** Implements:
  - **Prototype Distance Entropy (Proto-DE)** to measure confidence in classification.
  - **Decision Boundary Distance Estimation (Bound-DE)** to detect uncertain predictions.
- **Scalability:** Supports large datasets via **ImageDataGenerator**, enabling memory-efficient training.

## Model Architecture
- **Feature Extractor:** ResNet50 (pre-trained on ImageNet, with top layers removed)
- **Custom Classifier:** Dense layers with softmax activation for multi-class classification
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy

## Dataset Preparation
1. Store training and validation images in respective directories.
2. Use `ImageDataGenerator` to preprocess images:
   ```python
   train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                      height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                      horizontal_flip=True, fill_mode='nearest')
   ```
3. Define dataset paths in Google Drive or local storage.

## Training
- The model is trained using **batch-wise data streaming** with a generator.
- Example training command:
   ```python
   history = model.fit(train_generator, steps_per_epoch=steps, epochs=100,
                       validation_data=validation_generator, validation_steps=val_steps)
   ```
- Supports **early stopping and learning rate scheduling** (can be integrated for better optimization).

## Prototype-Based Classification
- **Feature embeddings** are extracted from the trained model.
- Prototypes (mean embedding per class) are computed:
   ```python
   def calculate_prototypes(embedding_model, data_generator, num_classes):
       embeddings = []
       labels = []
       for images, label in data_generator:
           emb = embedding_model.predict(images)
           embeddings.append(emb)
           labels.append(label)
       return np.array(prototypes)
   ```
- Unknown samples are classified based on their **distance from prototypes**.

## Decision Confidence Estimation
- Uses **distance entropy** to measure classification certainty.
- Formula:
   ```python
   softmax_distances = np.exp(-distances) / np.exp(-distances).sum(axis=1, keepdims=True)
   entropy = -np.sum(softmax_distances * np.log(softmax_distances + 1e-5), axis=1)
   ```
- **Lower entropy** → Higher confidence in classification.

## Applications
- **Agriculture:** Automated pest detection for smart farming.
- **Food Safety:** Identifying pests in food storage facilities.
- **Research:** Large-scale image classification of insect species.

## Future Enhancements
- Fine-tuning with **additional pest datasets**.
- Integrating **semi-supervised learning** to improve performance on unseen pest species.
- Deploying the model as a **web API** for real-time pest classification.

## Contributors
- **Muhammad Anwar** (Lead Developer & Data Engineer)
- Open for contributions! Feel free to fork and submit PRs.

## License
This project is licensed under the **MIT License**.

---

🚀 **Star this repo if you find it useful!** 🌟
