# EEG-based Stress Detection

This repository contains the code and resources for the paper:

> **Rakesh Kumar Rai, Dushyant Kumar Singh**  
> *Stress Detection through Wearable EEG Technology: A Signal-Based Approach*  
> Computers and Electrical Engineering, Elsevier, 2025.  
> [DOI: 10.1016/j.compeleceng.2025.110478](https://doi.org/10.1016/j.compeleceng.2025.110478)

---

## 📑 Abstract
Accurate and non-invasive stress detection is critical for mental health monitoring and early intervention. This work proposes a novel EEG-based stress detection framework that integrates:
- **FA-FCN**: Fuzzy Attention-based Fully Convolutional Network for segmentation  
- **HW-STFT**: Hybrid Wavelet–Short-Time Fourier Transform for feature extraction  
- **RLASSO**: Recursive LASSO for optimal feature selection  
- **AOS-GAN**: Adam-Optimized Sequential GAN for robust classification  

The model achieves **94% accuracy**, outperforming state-of-the-art baselines.

---
## 🗂 Repository Structure
- `code/` → Implementation (preprocessing, segmentation, feature extraction, GAN model)  
- `paper/` → Published paper PDF  


---

## ⚙️ Installation
```bash
git clone https://github.com/<your-username>/eeg-stress-detection.git
cd eeg-stress-detection
pip install -r requirements.txt
