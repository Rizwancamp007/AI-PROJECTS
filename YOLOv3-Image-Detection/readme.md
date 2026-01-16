# YOLOv3 Image Detection using OpenCV

## 📌 Project Overview

This project implements **object detection in images** using the **YOLOv3 (You Only Look Once)** deep learning model and **OpenCV’s DNN module** in Python. The system detects multiple objects in a single image, draws bounding boxes, and labels them with confidence scores.

---

## 🧠 Key Features

* YOLOv3 pre-trained model
* OpenCV DNN-based implementation
* Multi-object detection
* Non-Maximum Suppression (NMS)
* Bounding box visualization
* Output image saving

---

## 🛠️ Technologies Used

* Python
* OpenCV
* NumPy
* Matplotlib
* YOLOv3
* COCO Dataset

---

## 📂 Project Structure

```
YOLOv3-Image-Detection/
│
├── src/
│   └── imageDetection.py
├── data/
│   ├── image.jpg
│   └── output_image.jpg
├── model/
│   ├── yolov3.cfg
│   ├── yolov3.weights
│   └── coco.names
├── requirements.txt
├── README.md
└── demo.html
```

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/AI.git
cd AI/YOLOv3-Image-Detection
```

### 2. Create Environment (Optional)

```bash
conda create -n yolo python=3.10
conda activate yolo
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
cd src
python imageDetection.py
```

---

## 📊 Output

* Detected objects with bounding boxes
* Labeled classes with confidence scores
* Output image saved in `data/output_image.jpg`

---

## 🚀 Future Improvements

* Real-time webcam detection
* GPU acceleration (CUDA)
* Upgrade to YOLOv5/YOLOv8
* Web-based interface

---

## 👤 Author

**Rizwan Khan**
BS Computer Science
AI & Computer Vision Enthusiast

