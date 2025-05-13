# 🌳 Street Tree Detection and Species Classification in Taipei City

This project focuses on detecting individual street trees and identifying their species in Taipei City using high-resolution satellite imagery and deep learning.

---

## 📌 Project Objective

- Automatically locate individual **street trees** from RGB satellite images.
- **Classify the tree species** of each detected tree.
- Focused on **urban settings**, where trees are often occluded or surrounded by buildings.

---

## 🧠 Methodology

The system consists of three main components:

1. **Tree Localization**  
   - Model: `HR-SFANet`  
   - Input: RGB satellite images  
   - Output: Tree center coordinates

2. **Background Removal**  
   - Model: `U²-Net`  
   - Purpose: Remove urban background, isolate tree crown image for classification

3. **Species Classification**  
   - Model: `ResNet` (or CNN variant)  
   - Input: Cropped tree crown image  
   - Output: Predicted species

---

## 🗂️ Dataset Sources

| Source | Description |
|--------|-------------|
| [Taipei City Government Open Data](https://data.gov.tw/) | Tree location + species |
| Google Maps Static API | Satellite RGB imagery |
| [NEON Tree Crown Dataset](https://zenodo.org/record/6598391) | Pretraining on tree detection |
| [TreeSatAI](https://zenodo.org/record/6780578) | Pretraining on tree species |

---

## 🧪 Model Training Flow

```text
[Input image (satellite)] → HR-SFANet → Tree Points
                                ↓
                    Cropped tree patches
                                ↓
                          U²-Net → isolated crown
                                ↓
                      ResNet → tree species
                                ↓
               [Final output: coordinates + species]