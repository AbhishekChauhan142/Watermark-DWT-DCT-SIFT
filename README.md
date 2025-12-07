# 🛡️ Hybrid DWT-DCT-SIFT Watermarking Lab

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://watermarkingtools-dwt-dct-sift.streamlit.app/)

### 🔴 **Live Demo:** [Click here to use the App](https://watermarkingtools-dwt-dct-sift.streamlit.app/)

A robust digital image watermarking application built with **Python** and **Streamlit**. 

This project implements a hybrid watermarking scheme based on **Discrete Wavelet Transform (DWT)**, **Discrete Cosine Transform (DCT)**, and **Scale-Invariant Feature Transform (SIFT)**. It is designed to withstand geometric attacks (Rotation, Scaling) and signal processing attacks (Noise, Compression).

## 🌟 Key Features

*   **1️⃣ Robust Embedding:** Hides watermarks in the DWT-DCT frequency domain, making them invisible to the human eye.
*   **2️⃣ Attack Laboratory (New!):** Simulate attacks directly in the app to test robustness.
    *   **Rotation:** Rotate images by specific degrees.
    *   **Noise:** Add Gaussian or Salt & Pepper noise.
    *   **Cropping:** Simulate data loss by cropping corners.
    *   **Compression:** Apply JPEG compression at various quality levels.
*   **3️⃣ Downloadable Datasets:** Download the *Watermarked* image, the *SIFT Data*, and even the *Attacked* versions for external testing.
*   **4️⃣ Geometric Correction:** Uses **SIFT** features to "undo" rotation and scaling before extraction.
*   **5️⃣ Noise Filtering:** Includes a post-processing denoise filter to clean up extracted watermarks.

## 📄 Methodology

This implementation follows the research paper approach:

1.  **Embedding:** 
    *   Image is decomposed using **DWT (Haar)**.
    *   Watermark is embedded into the **HL1** sub-band using **DCT** coefficients.
    *   **SIFT keypoints** are saved to a `.pkl` file (required for geometric recovery).
2.  **Attacking:**
    *   The app allows you to apply real-time distortions to test if the watermark survives.
3.  **Extraction:**
    *   The app matches SIFT features between the original (saved data) and the attacked image.
    *   It calculates an **Affine Transformation Matrix** to un-rotate/un-scale the image.
    *   The watermark is extracted from the corrected frequency coefficients.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AbhishekChauhan142/Watermark-DWT-DCT-SIFT.git
    cd Watermark-DWT-DCT-SIFT
    ```

2.  **Install dependencies:**
    *Note: We use specific versions to ensure compatibility with SciPy/PyWavelets.*
    ```bash
    pip install -r requirements.txt
    ```
    *(If you face binary incompatibility errors, run: `pip install "numpy<2" scipy PyWavelets`)*

## 💻 Usage

Run the Streamlit app locally:

```bash
streamlit run app.py