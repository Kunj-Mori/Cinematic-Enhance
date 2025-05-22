import os
import platform
import sys
import cv2
import streamlit as st
import time

def find_working_camera_linux(max_index=5):
    for idx in range(max_index):
        device_path = f"/dev/video{idx}"
        if os.path.exists(device_path):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.release()
                return idx
            cap.release()
    return None

def suppress_stderr(func, *args, **kwargs):
    """Temporarily suppress stderr while running func."""
    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        result = func(*args, **kwargs)
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(devnull)
        os.close(saved_stderr)
    return result

def get_camera_index():
    if platform.system() == "Linux":
        return find_working_camera_linux()
    else:
        # For other OSes, try index 0 directly
        cap = suppress_stderr(cv2.VideoCapture, 0)
        if cap.isOpened():
            cap.release()
            return 0
        return None

# Then inside your Streamlit app where webcam is handled:
def webcam_streaming(filter):
    st.write("📷 Use your device camera.")
    live_filter = st.checkbox("Start Live Cinematic Filter", key="live_filter_checkbox")

    if live_filter:
        cam_idx = get_camera_index()
        if cam_idx is None:
            st.error("⚠️ No accessible webcam found on your device.")
            return

        FRAME_WINDOW = st.image([])

        # Open capture with stderr suppressed
        cap = suppress_stderr(cv2.VideoCapture, cam_idx)
        if not cap.isOpened():
            st.error("⚠️ Cannot open webcam. Please check your device or permissions.")
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
        # Non-live webcam capture (take a snapshot)
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
