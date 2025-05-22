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
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(self.params["contrast"])

            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(self.params["brightness"])

            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(self.params["saturation"])

            processed = np.array(pil_image)

            processed = self.apply_tint(processed)
            processed = self.apply_vignette(processed)
            processed = self.add_grain(processed)

            return cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        except Exception as e:
            return image

    def apply_tint(self, image):
        tint_amount = self.params["tint"] / 100.0
        image = image.astype(float)
        image[:, :, 0] *= (1 + tint_amount)  # Red
        image[:, :, 2] *= (1 - tint_amount / 2)  # Blue
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


def find_working_camera():
    """Find the first working camera"""
    # Try common camera indices with different backends
    backends_to_try = []
    
    if os.name == 'nt':  # Windows
        backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    else:  # Linux/Mac
        backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY]
    
    for backend in backends_to_try:
        for i in range(5):  # Try indices 0-4
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.release()
                        return i, backend
                cap.release()
            except:
                continue
    
    # Fallback - try default method
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    return i, cv2.CAP_ANY
            cap.release()
        except:
            continue
    
    return None, None


def main():
    st.set_page_config(page_title="Cinematic Image Filter", layout="wide")
    st.title("🎬 Cinematic Image Filter")
    st.write("Transform your images and videos with a cinematic look!")

    # Initialize filter
    filter = CinematicFilter()

    # Sidebar for filter controls
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
            "contrast": st.sidebar.slider("Contrast", 0.5, 2.0, filter.params["contrast"]),
            "brightness": st.sidebar.slider("Brightness", 0.5, 2.0, filter.params["brightness"]),
            "saturation": st.sidebar.slider("Saturation", 0.5, 2.0, filter.params["saturation"]),
            "tint": st.sidebar.slider("Tint", 0, 30, filter.params["tint"]),
            "vignette": st.sidebar.slider("Vignette", 0.0, 1.0, filter.params["vignette"]),
            "grain": st.sidebar.slider("Film Grain", 0.0, 0.1, filter.params["grain"])
        }
        filter.update_params(new_params)

    # Mode selection
    mode = st.radio("Select Mode", ["Image", "Video", "Webcam"])

    if mode == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

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

            processed_img = cv2.imencode('.jpg', result)[1].tobytes()
            st.download_button(
                label="Download Image",
                data=processed_img,
                file_name="cinematic_image.jpg",
                mime="image/jpeg"
            )

    elif mode == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])

        if uploaded_file is not None:
            try:
                processed_video = process_video(uploaded_file, filter)
                st.download_button(
                    label="Download Processed Video",
                    data=processed_video,
                    file_name="cinematic_video.mp4",
                    mime="video/mp4"
                )
            except Exception as e:
                st.error(f"An error occurred while processing the video: {str(e)}")

    elif mode == "Webcam":
        st.write("📷 Live Camera with Cinematic Filter")
        
        # Initialize session state
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False
        if 'camera_found' not in st.session_state:
            st.session_state.camera_found = False
            # Try to find camera on startup
            cam_idx, backend = find_working_camera()
            if cam_idx is not None:
                st.session_state.camera_index = cam_idx
                st.session_state.camera_backend = backend
                st.session_state.camera_found = True

        # Live webcam toggle
        webcam_active = st.checkbox("🎥 Start Live Filter", value=st.session_state.webcam_active)
        
        if webcam_active != st.session_state.webcam_active:
            st.session_state.webcam_active = webcam_active
            st.rerun()

        # Live webcam stream
        if st.session_state.webcam_active:
            if not st.session_state.camera_found:
                st.error("❌ No camera found! Please check your camera connection.")
            else:
                try:
                    # Initialize camera
                    cap = cv2.VideoCapture(st.session_state.camera_index, st.session_state.camera_backend)
                    
                    if not cap.isOpened():
                        st.error("Failed to open camera")
                    else:
                        # Set camera properties for better performance
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # Create placeholder for video
                        video_placeholder = st.empty()
                        
                        # Stream frames
                        frame_count = 0
                        max_frames = 50  # Limit frames before refresh to prevent infinite loop
                        
                        while frame_count < max_frames and st.session_state.webcam_active:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Failed to capture frame")
                                break
                            
                            # Apply cinematic filter
                            filtered_frame = filter.apply(frame)
                            
                            # Convert to RGB for display
                            display_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB)
                            
                            # Display frame
                            video_placeholder.image(
                                display_frame, 
                                caption="Live Cinematic Filter",
                                use_column_width=True
                            )
                            
                            frame_count += 1
                            time.sleep(0.03)  # ~30 FPS
                        
                        cap.release()
                        
                        # Auto-refresh for continuous streaming
                        if st.session_state.webcam_active:
                            time.sleep(0.1)
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"Camera error: {str(e)}")
                    st.write("Try refreshing the page or check if another app is using the camera.")

        # Divider
        st.write("---")
        
        # Static camera capture
        st.write("📸 Or capture a single photo:")
        camera_file = st.camera_input("Take a picture")

        if camera_file is not None:
            # Process the captured image
            image = Image.open(camera_file)
            image_np = np.array(image.convert("RGB"))
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Apply filter
            result = filter.apply(image_bgr)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.header("Original")
                st.image(image)
            with col2:
                st.header("With Cinematic Filter")
                st.image(result_rgb)

            # Download button
            processed_img = cv2.imencode('.jpg', result)[1].tobytes()
            st.download_button(
                label="Download Filtered Image",
                data=processed_img,
                file_name="cinematic_photo.jpg",
                mime="image/jpeg"
            )


if __name__ == "__main__":
    main()
