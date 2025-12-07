# 🛡️ Hybrid DWT-DCT-SIFT Image Watermarking

A robust digital image watermarking application built with **Python** and **Streamlit**. 

This project implements a hybrid watermarking scheme based on **Discrete Wavelet Transform (DWT)**, **Discrete Cosine Transform (DCT)**, and **Scale-Invariant Feature Transform (SIFT)**. It is designed to be robust against both signal processing attacks (noise, compression) and geometric attacks (rotation, scaling).

## 📄 Methodology

This implementation follows the techniques described in the research: *"A Hybrid Robust Image Watermarking Method Based on DWT-DCT and SIFT for Copyright Protection"*.

1.  **Embedding:** 
    *   The host image is decomposed using **DWT (Haar)**.
    *   The watermark is embedded into the **HL1** sub-band using **DCT** coefficients.
    *   **SIFT keypoints** of the original image are calculated and saved to correct geometric distortions later.
2.  **Attacking:**
    *   The application includes a testing lab to simulate attacks: Rotation, Cropping, Salt & Pepper noise, Gaussian noise, and JPEG compression.
3.  **Extraction:**
    *   **SIFT** is used to calculate the affine transformation matrix between the original and attacked image.
    *   The image is geometrically corrected (un-rotated/un-scaled).
    *   The watermark is extracted from the corrected DWT-DCT coefficients.

## 🚀 Features

*   **GUI Dashboard:** User-friendly interface using Streamlit.
*   **Geometric Correction:** Uses SIFT to recover watermarks even after the image has been rotated or scaled.
*   **Attack Simulation:** Built-in tools to test robustness immediately.
*   **Blind/Semi-Blind:** Requires the secret key and SIFT feature data (small .pkl file) but not the full original image for extraction.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/Watermark-DWT-DCT-SIFT.git
    cd Watermark-DWT-DCT-SIFT
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

Run the Streamlit app:

```bash
streamlit run app.py