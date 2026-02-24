# Task 4-4 — Fashion Multi-Label Classification REST API

This project implements a **multi-label image classification system** using TensorFlow and Keras, deployed as a RESTful API with Flask.

Unlike single-label classification, this model predicts **multiple attributes simultaneously** for each image.

---

## Project Overview

Each fashion image may contain multiple labels, including:

- Gender (Men / Women)
- Base Color (Black, Blue, etc.)
- Article Type (Tshirts, Jeans, Shoes, etc.)

The model predicts all relevant labels for a given image using a sigmoid output layer.

---

## Key Technical Differences (vs Single-Label)

| Single-Label | Multi-Label |
|--------------|-------------|
| Softmax activation | Sigmoid activation |
| Categorical Crossentropy | Binary Crossentropy |
| One class per image | Multiple labels per image |

---

## Dataset

Fashion product dataset containing images and structured metadata.

Each image is associated with:
- Gender
- Base Colour
- Article Type

Images and labels were processed to create multi-hot encoded targets.

> Dataset is not included in this repository.

---

## Model Architecture

- Pretrained Keras Application backbone (transfer learning)
- GlobalAveragePooling2D
- Dense hidden layer
- Final Dense layer with **sigmoid activation**
- Loss: **Binary Cross-Entropy**

### Why Sigmoid?

Because each output label is independent and can be simultaneously active.

### Why Binary Cross-Entropy?

Because each label represents an independent binary classification problem.

---

## REST API Features

The API returns:

- Best prediction per attribute group
- All active labels above a configurable threshold
- Top-k highest probability labels
- Adjustable threshold parameter