# Rusty-Pipe 🦀🐍

A high-performance hybrid Computer Vision pipeline leveraging **Python** for machine learning inference (YOLOv11) and **Rust** for CPU-bound downstream processing.

## 🚀 Overview

`Rusty-Pipe` solves the Python GIL bottleneck in computer vision tasks by offloading heavy post-processing logic (filtering, tracking, geometric calculations) to a native Rust extension.

- **Driver (Python):** Manages the video stream, executes GPU-accelerated YOLO inference via `ultralytics`, and handles visualization.
- **Engine (Rust):** A compiled native module (`rust_cv_core`) using `PyO3` that processes detection data at native speeds.

## 🛠 Prerequisites

- **Python 3.10+**
- **Rust (Stable)** and `cargo`
- **C Compiler** (gcc, clang, or MSVC)

## 📦 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/crybo-rybo/rusty-pipe.git
   cd rusty-pipe
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install maturin ultralytics opencv-python
   ```

## 🔨 Building the Rust Core

The Rust engine must be compiled and installed into your Python environment. We use `maturin` to bridge the two languages.

```bash
cd rust_cv_core
maturin develop
cd ..
```

*Note: `maturin develop` builds the crate and installs it directly into the active virtual environment.*

## 🏃 Running the Application

Ensure your webcam is connected and run:

```bash
python app.py
```

- **Controls:** Press `q` to exit the video stream.
- **Logic:** The current implementation filters the YOLO stream to only display "Person" detections processed via the Rust core.

## 🌍 Cross-Platform Execution

`Rusty-Pipe` is designed to be cross-platform (macOS, Linux, Windows), but there are a few environment-specific considerations:

### **1. Hardware Acceleration**
- **macOS:** YOLO will automatically attempt to use **Metal (MPS)** on Apple Silicon.
- **Linux/Windows:** If an NVIDIA GPU is present, ensure **CUDA** is installed for maximum performance. Otherwise, it will default to CPU.

### **2. Camera Access**
- **macOS:** You may need to grant your Terminal/IDE "Camera" permissions in *System Settings > Privacy & Security*.
- **Linux:** Ensure your user is in the `video` group (e.g., `sudo usermod -aG video $USER`).

### **3. Building for Distribution**
To build a shareable wheel (`.whl`) for a specific platform rather than just developing locally:
```bash
cd rust_cv_core
maturin build --release
```

## 📜 License
See `rusty_pipe_core_prd.md` for project specifications and architectural goals.
