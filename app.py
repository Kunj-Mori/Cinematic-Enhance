import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tempfile
import os
import shutil
import time


class CinematicFilter:
    def __init__(self):
        self.params = {
            "contrast": 1.4,
            "brightness": 1.05,
            "saturation": 1.3,
            "tint": 12,
            "vignette": 0.75,
            "grain": 0.025
        }

        self.presets = {
            "Classic Cinema": {
                "contrast": 1.4,
                "brightness": 1.05,
                "saturation": 1.3,
                "tint": 12,
                "vignette": 0.75,
                "grain": 0.025
            },
            "Vintage": {
                "contrast": 1.2,
                "brightness": 0.9,
                "saturation": 0.8,
                "tint": 20,
                "vignette": 0.8,
                "grain": 0.04
            },
            "Modern": {
                "contrast": 1.3,
                "brightness": 1.1,
                "saturation": 1.4,
                "tint": 5,
                "vignette": 0.6,
                "grain": 0.01
            },
            "Noir": {
                "contrast": 1.5,
                "brightness": 0.85,
                "saturation": 0.3,
                "tint": 0,
                "vignette": 0.9,
                "grain": 0.03
            },
            "Warm": {
                "contrast": 1.2,
                "brightness": 1.1,
                "saturation": 1.3,
                "tint": 25,
                "vignette": 0.5,
                "grain": 0.02
            }
        }

    def apply(self, image):
        # Convert BGR (OpenCV) to RGB (PIL)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Apply contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(self.params["contrast"])

        # Apply brightness
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(self.params["brightness"])

        # Apply saturation
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(self.params["saturation"])

        processed = np.array(pil_image)

        # Apply tint, vignette, and grain effects
        processed = self.apply_tint(processed)
        processed = self.apply_vignette(processed)
        processed = self.add_grain(processed)

        # Convert back to BGR for OpenCV usage/display
        return cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

    def apply_tint(self, image):
        tint_amount = self.params["tint"] / 100.0
        image = image.astype(float)
        # Boost red channel, slightly reduce blue channel for warm tint
        image[:, :, 0] *= (1 + tint_amount)  # Red channel
        image[:, :, 2] *= (1 - tint_amount / 2)  # Blue channel
        return np.clip(image, 0, 255).astype(np.uint8)

    def apply_vignette(self, image):
        height, width = image.shape[:2]
        X_center = width / 2
        Y_center = height / 2
        X, Y = np.meshgrid(np.linspace(0, width - 1, width), np.linspace(0, height - 1, height))
        dist = np.sqrt((X - X_center) ** 2 + (Y - Y_center) ** 2)
        max_dist = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
        dist = dist / max_dist
        vignette = 1 - (dist * (1 - self.params["vignette"]))
        vignette = np.dstack([vignette] * 3)
        return (image * vignette).astype(np.uint8)

    def add_grain(self, image):
        grain = np.random.normal(0, self.params["grain"] * 255, image.shape).astype(np.float32)
        noisy = cv2.add(image.astype(np.float32), grain)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def update_params(self, new_params):
        self.params.update(new_params)

    def apply_preset(self, preset_name):
        if preset_name in self.presets:
            self.params = self.presets[preset_name].copy()


def process_video(uploaded_file, filter):
    temp_dir = tempfile.mkdtemp()
    try:
        input_path = os.path.join(temp_dir, "input_video.mp4")
        output_path = os.path.join(temp_dir, "output_video.mp4")

        with open(input_path, 'wb') as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = filter.apply(frame)
            out.write(processed_frame)

            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count} of {total_frames}")

        cap.release()
        out.release()

        with open(output_path, 'rb') as f:
            processed_video = f.read()

        return processed_video

    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def main():
    st.set_page_config(page_title="Cinematic Image Filter", layout="wide")
    st.title("🎬 Cinematic Image Filter")
    st.write("Transform your images and videos with a cinematic look!")

    filter = CinematicFilter()

    st.sidebar.title("Filter Parameters")
    preset = st.sidebar.selectbox(
        "Select Filter Style",
        ["Custom"] + list(filter.presets.keys()),
        key='filter_preset'
    )

    if preset != "Custom":
        filter.apply_preset(preset)
        st.sidebar.write("Current Filter Parameters:")
        for param, value in filter.params.items():
            st.sidebar.text(f"{param}: {value:.2f}")
    else:
        new_params = {
            "contrast": st.sidebar.slider("Contrast", 0.5, 2.0, filter.params["contrast"], key='contrast_slider'),
            "brightness": st.sidebar.slider("Brightness", 0.5, 2.0, filter.params["brightness"], key='brightness_slider'),
            "saturation": st.sidebar.slider("Saturation", 0.5, 2.0, filter.params["saturation"], key='saturation_slider'),
            "tint": st.sidebar.slider("Tint", 0, 30, filter.params["tint"], key='tint_slider'),
            "vignette": st.sidebar.slider("Vignette", 0.0, 1.0, filter.params["vignette"], key='vignette_slider'),
            "grain": st.sidebar.slider("Film Grain", 0.0, 0.1, filter.params["grain"], key='grain_slider')
        }
        filter.update_params(new_params)

    mode = st.radio("Select Mode", ["Image", "Video", "Webcam"], key='mode_selection')

    if mode == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key='image_uploader')

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            result = filter.apply(image)

            col1, col2 = st.columns(2)
            with col1:
                st.header("Original")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            with col2:
                st.header("Processed")
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

            if st.button("Save Processed Image", key='save_image_button'):
                processed_img = cv2.imencode('.jpg', result)[1].tobytes()
                st.download_button(
                    label="Download Image",
                    data=processed_img,
                    file_name="cinematic_image.jpg",
                    mime="image/jpeg",
                    key='download_image_button'
                )

    elif mode == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'], key='video_uploader')

        if uploaded_file is not None:
            try:
                processed_video = process_video(uploaded_file, filter)
                st.download_button(
                    label="Download Processed Video",
                    data=processed_video,
                    file_name="cinematic_video.mp4",
                    mime="video/mp4",
                    key='download_video_button'
                )
            except Exception as e:
                st.error(f"An error occurred while processing the video: {str(e)}")

    elif mode == "Webcam":
        st.write("📷 Use your device camera.")

        live_filter = st.checkbox("Start Live Cinematic Filter", key="live_filter_checkbox")

        if live_filter:
            FRAME_WINDOW = st.image([])

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("⚠️ Cannot open webcam. Please make sure your camera is connected and not in use by another application.")
                return

            try:
                while live_filter:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("⚠️ Failed to grab frame from webcam.")
                        break

                    filtered_frame = filter.apply(frame)
                    FRAME_WINDOW.image(cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB))

                    # Allow checkbox to update (stop live streaming)
                    live_filter = st.checkbox("Start Live Cinematic Filter", value=True, key="live_filter_checkbox")

                    time.sleep(0.03)
            except Exception as e:
                st.error(f"Error during webcam streaming: {e}")
            finally:
                cap.release()
                st.write("Webcam streaming stopped.")

        else:
            camera_file = st.camera_input("Take a picture")

            if camera_file is not None:
                image = Image.open(camera_file)
                image_np = np.array(image.convert("RGB"))
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                result = filter.apply(image_bgr)
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

                col1, col2 = st.columns(2)
                with col1:
                    st.header("Original Image")
                    st.image(image)
                with col2:
                    st.header("Cinematic Filter Applied")
                    st.image(result_rgb)

                processed_img = cv2.imencode('.jpg', result)[1].tobytes()
                st.download_button(
                    label="Download Filtered Image",
                    data=processed_img,
                    file_name="cinematic_image.jpg",
                    mime="image/jpeg"
                )


if __name__ == "__main__":
    main()
