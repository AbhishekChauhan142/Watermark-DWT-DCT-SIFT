import streamlit as st
import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct
import pickle
import io
from PIL import Image
import random


# ==========================================
# HELPER: CONVERSIONS & UTILS
# ==========================================
def to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def convert_to_display(img_array):
    """Converts array to uint8 for display/saving"""
    return np.clip(img_array, 0, 255).astype(np.uint8)


def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)


# ==========================================
# CLASS 1: ATTACK ENGINE
# ==========================================
class AttackEngine:
    @staticmethod
    def apply_gaussian_noise(image, var=10):
        row, col = image.shape[:2]
        mean = 0
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        if len(image.shape) == 3:
            gauss = gauss.reshape(row, col, 1)
        noisy = image + gauss
        return convert_to_display(noisy)

    @staticmethod
    def apply_salt_pepper(image, prob=0.01):
        output = np.copy(image)
        # Salt
        num_salt = np.ceil(prob * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        output[tuple(coords)] = 255
        # Pepper
        num_pepper = np.ceil(prob * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        output[tuple(coords)] = 0
        return output

    @staticmethod
    def apply_rotation(image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    @staticmethod
    def apply_cropping(image, percentage=0.2):
        h, w = image.shape[:2]
        h_crop = int(h * percentage)
        w_crop = int(w * percentage)
        cropped = image.copy()
        # Crop top-left corner by making it black
        cropped[0:h_crop, 0:w_crop] = 0
        return cropped

    @staticmethod
    def apply_jpeg_compression(image, quality=50):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        if len(image.shape) == 2:
            return cv2.cvtColor(decimg, cv2.COLOR_BGR2GRAY)
        return decimg


# ==========================================
# CLASS 2: WATERMARK ENGINE (DWT-DCT-SIFT)
# ==========================================
class WatermarkEngine:
    def __init__(self):
        self.alpha = 5.0
        self.sift = cv2.SIFT_create()

    def apply_dct(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def apply_idct(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    def get_pn_sequences(self, key, shape):
        np.random.seed(key)
        return np.random.randn(*shape), np.random.randn(*shape)

    def embed(self, original_img, watermark_img, key):
        img_gray = to_gray(original_img)
        wm_gray = to_gray(watermark_img)

        # 1. SIFT Extraction (Preserve features of original image)
        kp, des = self.sift.detectAndCompute(img_gray, None)

        # Serialize Keypoints
        kp_store = []
        for point in kp:
            kp_store.append((point.pt, point.size, point.angle, point.response, point.octave, point.class_id))

        sift_data = {'kp': kp_store, 'des': des, 'shape': img_gray.shape}
        sift_bytes = pickle.dumps(sift_data)

        # 2. DWT
        coeffs = pywt.dwt2(img_gray, 'haar')
        LL, (LH, HL, HH) = coeffs

        # 3. Resize Watermark
        h, w = HL.shape
        wm_resized = cv2.resize(wm_gray, (w // 8, h // 8))
        _, wm_bin = cv2.threshold(wm_resized, 128, 1, cv2.THRESH_BINARY)

        # 4. Embed in DCT domain of HL band
        HL_embedded = HL.copy()
        pn0, pn1 = self.get_pn_sequences(key, (8, 8))

        rows, cols = wm_bin.shape
        for i in range(rows):
            for j in range(cols):
                r, c = i * 8, j * 8
                block = HL[r:r + 8, c:c + 8]
                dct_block = self.apply_dct(block)

                seq = pn1 if wm_bin[i, j] == 1 else pn0
                dct_embedded = dct_block + (self.alpha * seq)

                HL_embedded[r:r + 8, c:c + 8] = self.apply_idct(dct_embedded)

        # 5. Inverse DWT
        watermarked = pywt.idwt2((LL, (LH, HL_embedded, HH)), 'haar')
        return convert_to_display(watermarked), sift_bytes

    def extract(self, attacked_img, sift_bytes, key):
        img_gray = to_gray(attacked_img)

        # 1. Load SIFT & Recover Geometry
        sift_data = pickle.loads(sift_bytes)
        orig_des = sift_data['des']
        orig_shape = sift_data['shape']

        # Reconstruct keypoints
        orig_kp = [cv2.KeyPoint(x=p[0][0], y=p[0][1], size=p[1], angle=p[2], response=p[3], octave=p[4], class_id=p[5])
                   for p in sift_data['kp']]

        kp_att, des_att = self.sift.detectAndCompute(img_gray, None)
        img_corrected = None

        # Geometric Correction Logic
        if des_att is not None and len(des_att) > 2:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(orig_des, des_att, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good) > 4:
                src_pts = np.float32([orig_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_att[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, _ = cv2.estimateAffinePartial2D(dst_pts, src_pts)
                if M is not None:
                    img_corrected = cv2.warpAffine(img_gray, M, (orig_shape[1], orig_shape[0]))

        if img_corrected is None:
            img_corrected = cv2.resize(img_gray, (orig_shape[1], orig_shape[0]))

        # 2. Extract
        coeffs = pywt.dwt2(img_corrected, 'haar')
        _, (_, HL_star, _) = coeffs

        h, w = HL_star.shape
        rows, cols = h // 8, w // 8
        extracted = np.zeros((rows, cols))
        pn0, pn1 = self.get_pn_sequences(key, (8, 8))

        for i in range(rows):
            for j in range(cols):
                r, c = i * 8, j * 8
                block = HL_star[r:r + 8, c:c + 8]
                dct_block = self.apply_dct(block)

                # Correlation
                corr0 = np.sum(dct_block * pn0)
                corr1 = np.sum(dct_block * pn1)
                extracted[i, j] = 255 if corr1 > corr0 else 0

        return convert_to_display(extracted)


# ==========================================
# STREAMLIT UI
# ==========================================
def main():
    st.set_page_config(page_title="Watermark Lab", layout="wide", page_icon="🔐")

    # Session State Initialization
    if 'watermarked_image' not in st.session_state:
        st.session_state.watermarked_image = None
    if 'sift_data' not in st.session_state:
        st.session_state.sift_data = None
    if 'attacked_image' not in st.session_state:
        st.session_state.attacked_image = None

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    secret_key = st.sidebar.number_input("Secret Key", value=1234,
                                         help="Must be the same for embedding and extraction.")

    st.title("🔐 Hybrid DWT-DCT-SIFT Watermarking")
    st.markdown("Robust image watermarking that survives geometric attacks (Rotation, Scaling) and signal processing.")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["1️⃣ Embed", "2️⃣ Attack (Simulate)", "3️⃣ Extract"])

    # --- TAB 1: EMBEDDING ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            host_file = st.file_uploader("Upload Host Image", type=['jpg', 'png'])
        with col2:
            wm_file = st.file_uploader("Upload Watermark (B&W)", type=['jpg', 'png'])

        if host_file and wm_file:
            host_img = load_image(host_file)
            wm_img = load_image(wm_file)

            st.image(host_img, caption="Host Image", width=250)

            if st.button("🚀 Embed Watermark"):
                engine = WatermarkEngine()
                with st.spinner("Embedding..."):
                    res_img, sift_data = engine.embed(host_img, wm_img, secret_key)

                    # Store in session state
                    st.session_state.watermarked_image = res_img
                    st.session_state.sift_data = sift_data

                    st.success("Embedding Successful!")
                    st.image(res_img, caption="Watermarked Result", width=250)

                    # Downloads
                    buf = cv2.imencode('.png', res_img)[1].tobytes()
                    st.download_button("💾 Download Image", buf, "watermarked.png", "image/png")
                    st.download_button("📂 Download SIFT Data", sift_data, "data.pkl")

        # --- TAB 2: ATTACK SIMULATION (UPDATED) ---
        with tab2:
            st.header("💥 Attack Laboratory")
            st.markdown("Simulate what happens if a hacker tries to destroy the watermark.")

            # Determine source image
            input_img = None
            if st.session_state.watermarked_image is not None:
                st.info("✅ Using the Watermarked Image generated in Step 1.")
                input_img = st.session_state.watermarked_image
            else:
                up_att = st.file_uploader("Or upload an image to attack", type=['png', 'jpg'],
                                          key="upload_attack_input")
                if up_att: input_img = load_image(up_att)

            if input_img is not None:
                # Layout: Controls on left, Image on right
                col_controls, col_display = st.columns([1, 2])

                with col_controls:
                    st.subheader("Choose Attack")
                    attack_type = st.selectbox("Type",
                                               ["None", "Gaussian Noise", "Salt & Pepper", "Rotation",
                                                "Cropping (Top-Left)", "JPEG Compression"])

                    # Default to input
                    attacked_res = input_img.copy()

                    # Attack Parameters UI
                    if attack_type == "Gaussian Noise":
                        var = st.slider("Variance", 0, 100, 20)
                        attacked_res = AttackEngine.apply_gaussian_noise(input_img, var)
                    elif attack_type == "Salt & Pepper":
                        prob = st.slider("Noise Prob", 0.0, 0.1, 0.02)
                        attacked_res = AttackEngine.apply_salt_pepper(input_img, prob)
                    elif attack_type == "Rotation":
                        angle = st.slider("Degrees", -45, 45, 10)
                        attacked_res = AttackEngine.apply_rotation(input_img, angle)
                    elif attack_type == "Cropping (Top-Left)":
                        perc = st.slider("Crop %", 0.0, 0.5, 0.2)
                        attacked_res = AttackEngine.apply_cropping(input_img, perc)
                    elif attack_type == "JPEG Compression":
                        qual = st.slider("Quality (1=Worst)", 1, 100, 30)
                        attacked_res = AttackEngine.apply_jpeg_compression(input_img, qual)

                with col_display:
                    st.image(attacked_res, caption=f"Result after {attack_type}", width=400)

                    # ACTION BUTTONS
                    st.divider()
                    btn_col1, btn_col2 = st.columns(2)

                    # 1. Send to Extraction Tab (Internal Memory)
                    with btn_col1:
                        if st.button("➡️ Use for Extraction", help="Send this image directly to Tab 3"):
                            st.session_state.attacked_image = attacked_res
                            st.success("Sent to Tab 3!")

                    # 2. Download to Computer (File)
                    with btn_col2:
                        # Convert numpy array to PNG bytes
                        is_success, buffer = cv2.imencode(".png", attacked_res)
                        io_buf = io.BytesIO(buffer)

                        st.download_button(
                            label="💾 Download Image",
                            data=io_buf,
                            file_name=f"attacked_{attack_type}.png",
                            mime="image/png"
                        )

        # ================= TAB 3: EXTRACTION (UPDATED) =================
        with tab3:
            st.header("🕵️ Extraction Lab")
            st.markdown("Recover the watermark even if the image was rotated, scaled, or noised.")

            col1, col2 = st.columns(2)

            # 1. Image Source
            img_to_extract = None
            with col1:
                st.subheader("1. Input Image")
                if st.session_state.attacked_image is not None:
                    st.info("Using image from Attack Lab")
                    img_to_extract = st.session_state.attacked_image
                    st.image(img_to_extract, width=200, caption="Attacked Input")
                else:
                    u_ext = st.file_uploader("Upload Attacked Image", type=['png', 'jpg'])
                    if u_ext:
                        img_to_extract = load_image(u_ext)
                        st.image(img_to_extract, width=200, caption="Uploaded Input")

            # 2. SIFT Source
            sift_to_use = None
            with col2:
                st.subheader("2. Recovery Data")
                if st.session_state.sift_data is not None:
                    st.info("Using SIFT data from Embed Lab")
                    sift_to_use = st.session_state.sift_data
                else:
                    u_sift = st.file_uploader("Upload .pkl File", type=['pkl'],
                                              help="Required to undo rotation/scaling")
                    if u_sift: sift_to_use = u_sift.read()

            # 3. Extraction Controls
            st.divider()
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                key_input = st.number_input("Secret Key", value=1234, key="key_ext_final")
            with col_ex2:
                use_denoise = st.checkbox("Apply Noise Filter", value=True,
                                          help="Removes white noise dots from extracted watermark")

            if img_to_extract is not None and sift_to_use is not None:
                if st.button("🔍 Analyze & Extract"):
                    engine = WatermarkEngine()
                    with st.spinner("1. Matching SIFT Features... 2. Correcting Geometry... 3. Decrypting..."):
                        try:
                            # We need to modify extract to return the corrected image for visualization
                            # NOTE: This requires a slight tweak to the Class method below,
                            # but for now we run the standard extract.

                            # Run Extraction
                            raw_wm = engine.extract(img_to_extract, sift_to_use, int(key_input))

                            # Post-Processing (Denoise)
                            final_wm = raw_wm
                            if use_denoise:
                                # Median Blur is great for removing salt-and-pepper noise from binary images
                                final_wm = cv2.medianBlur(raw_wm, 3)

                            # Display Results
                            st.success("Extraction Complete")

                            res_col1, res_col2 = st.columns(2)
                            with res_col1:
                                st.image(raw_wm, caption="Raw Extracted Signal", width=250, clamp=True)
                            with res_col2:
                                st.image(final_wm, caption="Cleaned Watermark", width=250, clamp=True)

                            # Download
                            buf = cv2.imencode('.png', final_wm)[1].tobytes()
                            st.download_button("💾 Download Watermark", buf, "extracted_watermark.png", "image/png")

                        except Exception as e:
                            st.error(f"Extraction failed. The attack might be too severe. Error: {str(e)}")
            else:
                st.warning("Please provide both the Attacked Image and the .pkl file.")


if __name__ == "__main__":
    main()