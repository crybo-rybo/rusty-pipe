# Product Requirements Document (PRD): Rust-Python Hybrid CV Pipeline

**Project Name:** Rusty-Pipe-Core
**Version:** 1.0
**Status:** Draft
**Architecture Pattern:** Python Interface / Rust Engine (PyO3 + Maturin)

---

## 1. Executive Summary
The goal is to engineer a high-performance Computer Vision (CV) application that leverages **Python** for machine learning inference (specifically YOLOv8/v11) and **Rust** for CPU-bound downstream processing. This hybrid architecture mitigates Python's Global Interpreter Lock (GIL) bottlenecks during post-processing tasks such as spatial filtering, object tracking, or complex geometric calculations.

## 2. System Architecture

The system follows a "Driver-Extension" model:
1.  **The Driver (Python):** Manages the video stream, executes the GPU-accelerated YOLO inference, and handles visualization.
2.  **The Engine (Rust):** Compiled as a native Python extension module. It receives raw detection data, performs heavy computation, and returns structured results.



### 2.1 Data Flow
1.  **Input:** `cv2` captures frame -> `ultralytics` generates `Results` object.
2.  **Bridge:** Python iterates results -> Instantiates Rust `Detection` structs.
3.  **Process:** Rust receives `Vec<Detection>` -> Releases GIL -> Calculates Logic.
4.  **Output:** Rust returns processed data (e.g., `ProcessingResult` struct) to Python.

---

## 3. Technical Stack Specifications

| Component | Technology | Version / Requirement |
| :--- | :--- | :--- |
| **Language** | Python | 3.10+ |
| **Language** | Rust | Stable (Latest) |
| **FFI / Binding** | PyO3 | `0.23+` (with `extension-module` feature) |
| **Build System** | Maturin | `1.4+` |
| **ML Framework** | Ultralytics | `yolo` (v8 or v11) |
| **Computer Vision** | OpenCV | `opencv-python` |

---

## 4. Functional Requirements

### 4.1 Rust Module (`rust_cv_core`)
The Rust extension must expose the following to Python:

#### 4.1.1 Data Structures (`structs`)
Must define a Python-compatible class `Detection` with the following fields (read/write):
* `id`: `int` (Class ID)
* `conf`: `float` (Confidence score 0.0 - 1.0)
* `bbox`: `(float, float, float, float)` (x, y, width, height)

#### 4.1.2 Core Logic (`functions`)
Must expose a function `process_frame` that:
1.  Accepts a list of detections: `detections: Vec<Detection>`.
2.  **Performance Constraint:** Logic must run in a separate thread or explicitly release the GIL if computation exceeds 1ms.
3.  **Business Logic (MVP):** Filter detections that are:
    * Low confidence (< 0.5).
    * Located in a specific "danger zone" (e.g., center point x > 500).
4.  Returns a list of resulting alert strings or a structured result object.

### 4.2 Python Application (`app.py`)
The Python driver must:
1.  Initialize the YOLO model (e.g., `yolov8n.pt`).
2.  Open a video stream (Webcam 0).
3.  Loop through frames:
    * Run inference.
    * Map YOLO `result.boxes` to `rust_cv_core.Detection`.
    * Call `rust_cv_core.process_frame()`.
    * Draw bounding boxes and overlay Rust-generated alerts on the frame.
    * Display frame via `cv2.imshow`.

---

## 5. Implementation Roadmap (Step-by-Step)

### Phase 1: Project Initialization
* **Action:** Create a standard directory structure for a mixed Python/Rust project.
* **Tooling:** Use `maturin new rust_cv_core --binding pyo3`.
* **Config:** Update `Cargo.toml` to include `pyo3` dependencies.

### Phase 2: Rust Implementation
* **Action:** Implement `lib.rs`.
* **Details:** Define the `#[pyclass]` structs and the `#[pyfunction]` logic. Ensure `#[pymodule]` entry point is correct.
* **Build:** Run `maturin develop` to build and install the library into the local virtual environment.

### Phase 3: Python Integration
* **Action:** Create `main.py`.
* **Details:** Import the new compiled module. Implement the video loop and data conversion (Tensor -> Rust Struct).

### Phase 4: Optimization & Testing
* **Action:** Benchmark the `process_frame` function.
* **Criteria:** Rust processing time must be negligible (< 2ms) compared to model inference time.

---

## 6. Development Configuration

### 6.1 `Cargo.toml` Reference
The coding agent should use the following dependency configuration:

```toml
[package]
name = "rust_cv_core"
version = "0.1.0"
edition = "2021"

[lib]
name = "rust_cv_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.23", features = ["extension-module"] }
