# AICamera: Real-Time High-Speed Person Detection & Tracking

<p align="center">
  <img src="assets/a9JxPqMwTs7VdFnL4oYe.gif" alt="AICamera Demo" width="720"/>
</p>

## 🚀 Overview

**AICamera** is a high-performance, real-time object detection and tracking system built as part of a larger Computer Vision pipeline during a project at **ExaWizards Inc.** It focuses on accurate **person detection and tracking**, optimized for deployment on NVIDIA GPUs using **TensorRT**. The system leverages the speed and accuracy of **YOLOv8** for detection and combines it with **DeepSORT** for robust multi-object tracking.

This submodule was designed to act as a core engine in downstream applications such as real-time surveillance, retail analytics, and smart camera systems — with a strong emphasis on efficiency, modularity, and real-world deployability.

---

## 🧠 Key Technologies

* **🔍 Detection:** [YOLOv8](https://docs.ultralytics.com/models/yolov8/) — state-of-the-art real-time object detection model by Ultralytics.
* **🧭 Tracking:** [DeepSORT](https://arxiv.org/abs/1703.07402) — combines motion (Kalman filter) and appearance (ReID) features for reliable tracking.
* **⚡ Acceleration:** [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) — significantly reduces inference time by optimizing ONNX models to run on NVIDIA GPUs.

---

## ⚙️ How It Works

1. **Input Capture:** Reads video frames from a file or webcam using OpenCV.
2. **Preprocessing:** Resizes and normalizes images for inference.
3. **Detection (YOLOv8 + TensorRT):** Outputs class labels, bounding boxes, and confidence scores.
4. **Tracking (DeepSORT + TensorRT ReID):**

   * ReID crops extracted from detected persons.
   * TensorRT-optimized ReID model generates embeddings.
   * Kalman filter + cosine distance + IoU matching for identity preservation.
5. **Visualization:** Annotated video with object IDs and bounding boxes.

---

## ✅ Features

* 🚶‍♂️ **Focused on Person Tracking**
* ⚡ **Blazing Fast** due to TensorRT acceleration
* 🔁 **Modular Pipeline** with clean interfaces for detection, tracking, and I/O
* 🧪 **Easily Configurable:** Adjust thresholds, engine paths, and target classes
* 🖥️ **Supports Webcam & Video Input**
* 📂 **Out-of-the-box Setup:** Includes helper scripts for model downloading and engine conversion

---

## 🖥️ System Requirements

* **OS:** Ubuntu 22.04 LTS (or compatible)
* **GPU:** NVIDIA CUDA-enabled GPU (Compute Capability ≥ 6.1)
* **NVIDIA Stack:**

  * NVIDIA Driver (latest)
  * CUDA Toolkit ≥ 12.1
  * cuDNN ≥ 9.0
  * TensorRT ≥ 10.x (Ensure `trtexec` is available)
* **Python:** 3.10.x

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/abdur75648/AI-Camera.git
cd AI-Camera
```

### 2. Ensure NVIDIA Stack is Installed

Ensure you have the NVIDIA driver, CUDA, cuDNN, and TensorRT installed. You can check if they are installed correctly by running:

```bash
nvidia-smi
```
This should show your GPU information.

Follow [NVIDIA's official documentation](https://docs.nvidia.com/) for installing the driver, CUDA, cuDNN, and TensorRT, if not already installed.

Ensure:

```bash
nvcc --version
trtexec --help
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download ONNX Models

```bash
bash scripts/download_models.sh
```

### 5. Export TensorRT Engines

```bash
bash scripts/export_trt_engines.sh
```

This creates:

* `models/detection/yolov8n.engine`
* `models/reid/deepsort_reid.engine`

---

## ▶️ Usage

Run on a video file:

```bash
python3 -m src.aicamera_tracker --input sample_input/sample_video.mp4 --show_display
```

Run on webcam:

```bash
python3 -m src.aicamera_tracker --webcam_id 0 --output_filename outputs/webcam_run.mp4 --show_display
```

Run with custom confidence threshold:

```bash
python3 -m src.aicamera_tracker --input video.mp4 --conf_thresh 0.4
```

### 🔧 Common CLI Options

| Argument            | Description                                 |
| ------------------- | ------------------------------------------- |
| `--input`           | Path to input video file                    |
| `--webcam_id`       | Webcam ID (default: 0)                      |
| `--output_dir`      | Output directory (default: `outputs/`)      |
| `--output_filename` | Output filename (auto-generated if not set) |
| `--show_display`    | Show live OpenCV display                    |
| `--no_save`         | Skip saving output video                    |
| `--yolo_engine`     | Path to YOLOv8 TensorRT engine              |
| `--reid_engine`     | Path to ReID TensorRT engine                |
| `--conf_thresh`     | Confidence threshold for detections         |
| `--device`          | Inference device (default: `cuda:0`)        |

---

## ⚡ Performance

| Component           | Raw Engine Speed (GTX 1660Ti) | Notes                          |
| ------------------- | ----------------------------- | ------------------------------ |
| YOLOv8n (TRT)       | \~400+ FPS                    | Highly optimized inference     |
| ReID (TRT)          | \~600+ FPS                    | Fast identity embedding        |
| End-to-End Pipeline | \~30 FPS                   | Varies by resolution, #objects |

> ⚠️ Enabling `--show_display` may reduce FPS due to rendering overhead. Disable for benchmarking.

---

## ⚙️ Configuration

Update `src/config.py` to modify:

* **Paths:**

  * `YOLO_ENGINE_PATH`, `REID_ENGINE_PATH`
* **YOLO Settings:**

  * `YOLO_INPUT_SHAPE`, `YOLO_CONF_THRESHOLD`
* **DeepSORT Settings:**

  * `DEEPSORT_MAX_DIST`, `DEEPSORT_MAX_AGE`, etc.
* **Classes to Track:**

  * `CLASSES_TO_TRACK = {'person'}`

---

## 🧱 Project Structure

```
AICamera/
├── assets/              # Demo assets
├── models/              # Pretrained models (.onnx, .engine)
│   ├── detection/
│   └── reid/
├── scripts/             # Scripts for model setup
├── sample_input/        # Sample videos/images
├── src/
│   ├── aicamera_tracker.py
│   ├── config.py
│   ├── detector/        # YOLOv8 wrapper
│   ├── tracker/         # DeepSORT tracker
│   │   └── core/        # Core DeepSORT logic
│   ├── utils/           # Helper utilities
│   └── trt_utils/       # TensorRT engine handling
└── requirements.txt
```

---

## 🔮 Future Enhancements

* Support for other YOLOv8 sizes (s, m, l, x)
* Integration with other tracking algorithms (e.g., ByteTrack, OC-SORT)
* Smarter gallery management in ReID
* Asynchronous pipeline for faster I/O
* MOT evaluation metrics (MOTA, MOTP)
* Batch-mode frame processing

---

## 📄 License

Licensed under the [MIT License](LICENSE). You're free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [DeepSORT](https://github.com/nwojke/deep_sort)
* [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)