# Cinematic Image Filter

A Streamlit application that applies cinematic filters to images and videos, featuring real-time parameter adjustment and webcam support.

## Features

- Image processing with cinematic effects
- Video processing with progress tracking
- Real-time webcam filtering
- Adjustable parameters:
  - Contrast
  - Brightness
  - Saturation
  - Tint
  - Vignette
  - Film grain
- Side-by-side comparison for images
- Download processed media

## Installation

1. Clone this repository or download the files
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. The application will open in your default web browser

3. Choose your mode:
   - Image: Upload and process still images
   - Video: Upload and process video files
   - Webcam: Apply filters to your webcam feed in real-time

4. Adjust the filter parameters in the sidebar to customize the look

5. Download your processed media using the provided buttons

## Supported Formats

- Images: JPG, JPEG, PNG
- Videos: MP4, AVI, MOV

## Requirements

- Python 3.7+
- OpenCV
- Streamlit
- NumPy
- Pillow 