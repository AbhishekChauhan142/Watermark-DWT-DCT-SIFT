import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def convert_upload_to_cv2(uploaded_file):
    """Converts a Streamlit uploaded file to an OpenCV Grayscale image."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return img


def calculate_metrics(original, modified):
    """Calculates PSNR and SSIM between two images."""
    # Resize modified to match original if needed (for metrics only)
    if original.shape != modified.shape:
        modified = cv2.resize(modified, (original.shape[1], original.shape[0]))

    try:
        score_psnr = psnr(original, modified)
        score_ssim = ssim(original, modified, data_range=modified.max() - modified.min())
        return score_psnr, score_ssim
    except Exception as e:
        print(f"Metric calculation error: {e}")
        return 0.0, 0.0