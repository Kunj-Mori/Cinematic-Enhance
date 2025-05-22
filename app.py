import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tempfile
import os
import shutil
import time
import threading
import queue


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


def find_working_camera(max_index=10):
    """
    Try to find a working camera index by checking up to max_index.
    Returns index if found, else None.
    """
    working_cameras = []
    
    # Try different backends and indices
    backends = [cv2.CAP_ANY]
    if os.name == 'nt':  # Windows
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:  # Linux/Mac
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    
    for backend in backends:
        for idx in range(max_index):
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap is not None and cap.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        working_cameras.append((idx, backend))
                        cap.release()
                        return idx, backend
                cap.release()
            except Exception:
                continue
    
    return None, None


def initialize_camera(camera_index, backend):
    """Initialize camera with proper error handling"""
    try:
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            return None
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return cap
    except Exception as e:
        st.error(f"Failed to initialize camera: {e}")
        return None


class CameraStream:
    def __init__(self, camera_index, backend):
        self.camera_index = camera_index
        self.backend = backend
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
    
    def start(self):
        if self.running:
            return False
        
        self.cap = initialize_camera(self.camera_index, self.backend)
        if self.cap is None:
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def _capture_frames(self):
        while self.running:
            if self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    # Keep only the latest frame
                    if not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
            time.sleep(0.01)
    
    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
        if self.cap is not None:
            self.cap.release()
        self.cap = None


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

        # Initialize session state for camera stream
        if 'camera_stream' not in st.session_state:
            st.session_state.camera_stream = None
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False

        # Find available camera
        if st.button("🔍 Detect Camera", key="detect_camera"):
            with st.spinner("Searching for cameras..."):
                cam_idx, backend = find_working_camera()
                
                if cam_idx is not None:
                    st.success(f"✅ Camera found at index {cam_idx}")
                    st.session_state.camera_index = cam_idx
                    st.session_state.camera_backend = backend
                else:
                    st.error("❌ No accessible camera found. Please check:")
                    st.write("- Camera is connected properly")
                    st.write("- Camera is not being used by another application")
                    st.write("- Camera permissions are granted")

        # Camera controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start Live Filter", key="start_camera") and hasattr(st.session_state, 'camera_index'):
                if st.session_state.camera_stream is None:
                    st.session_state.camera_stream = CameraStream(
                        st.session_state.camera_index, 
                        st.session_state.camera_backend
                    )
                
                if st.session_state.camera_stream.start():
                    st.session_state.camera_active = True
                    st.rerun()
                else:
                    st.error("Failed to start camera")
        
        with col2:
            if st.button("⏹️ Stop Camera", key="stop_camera"):
                if st.session_state.camera_stream is not None:
                    st.session_state.camera_stream.stop()
                    st.session_state.camera_stream = None
                st.session_state.camera_active = False
                st.rerun()

        # Live camera feed
        if st.session_state.camera_active and st.session_state.camera_stream is not None:
            FRAME_WINDOW = st.empty()
            
            # Display live feed for a few seconds then auto-refresh
            start_time = time.time()
            frame_count = 0
            
            while st.session_state.camera_active and time.time() - start_time < 5:
                frame = st.session_state.camera_stream.get_frame()
                
                if frame is not None:
                    try:
                        filtered_frame = filter.apply(frame)
                        FRAME_WINDOW.image(
                            cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB),
                            caption="Live Cinematic Filter"
                        )
                        frame_count += 1
                    except Exception as e:
                        st.error(f"Error processing frame: {e}")
                        break
                
                time.sleep(0.1)  # Control frame rate
            
            # Auto refresh to continue the stream
            if st.session_state.camera_active:
                st.rerun()

        # Static camera capture
        st.write("---")
        st.write("📸 Or take a single photo:")
        
        camera_file = st.camera_input("Take a picture", key="camera_input")

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
                mime="image/jpeg",
                key="download_camera_image"
            )


if __name__ == "__main__":
    main()
