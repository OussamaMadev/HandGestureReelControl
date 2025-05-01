# HandGestureReelControl
A real-time reel controller powered by deep convolutional neural networks (CNNs) that recognizes hand gestures to intuitively navigate and control social media reels.


**HandGestureReelControler** is a real-time hand gesture recognition system powered by deep convolutional neural networks (CNNs). It captures hand movements using a webcam and maps them to customizable actions — such as navigating reels, liking content, scrolling through comments, or triggering app-specific commands — providing a smooth, touchless control experience for social media and multimedia platforms.

---

## 🧠 What It Does

- 🎥 Detects **hand gestures live** from webcam video.
- 🧠 Uses a CNN model trained on real gesture data.
- ⚙️ Maps each gesture to a **user-defined action**.

---

## 🎯 Dataset

Trained on the [HG14 Hand Gesture Dataset](https://www.kaggle.com/datasets/gulerosman/hg14-handgesture14-dataset) by **gulerosman**.

- 14 gesture classes
- 22,000+ labeled images
- Multiple hand types, angles, and lighting conditions

---

## 🧱 Features

- Real-time video processing with OpenCV
- Custom gesture-action assignment (JSON or GUI)
- Modular CNN for gesture classification
- Extensible control targets: reels, apps, games, interfaces
- **Predefined Actions**:
  - Scroll next/previous
  - Click like ❤️
  - Open / navigate comments 💬
  - Any keyboard or mouse emulation

---

## 📂 Project Structure

```
HandGestureReelControler/
├── model/             # CNN model training and saved weights
├── app/               # Real-time app for gesture recognition
├── data/              # Dataset and loading utilities
├── actions.json       # Gesture-to-action mapping config
├── requirements.txt   # Project dependencies
└── README.md
```

---

## 🛠 Tech Stack

- **Python 3**
- **TensorFlow / Keras**
- **OpenCV** for real-time video
- **PyAutoGUI / pynput** for simulating keyboard/mouse events
- (Optional) **Flask** or **PyQt** for UI

---

## 🚀 Quick Start

### 1. Clone the project and install dependencies

```bash
git clone https://github.com/OussamaMadev/HandGestureReelControl
cd HandGestureReelControler
pip install -r requirements.txt
```

### 2. Download the dataset

Download the [HG14 dataset](https://www.kaggle.com/datasets/gulerosman/hg14-handgesture14-dataset) and place it inside the `data/` directory.

### 3. Train the model

```bash
python model/train.py
```

### 4. Launch the controller app

```bash
python app/main.py
```

You can customize gesture actions in `actions.json`.

---

## 🔄 Example Gesture Actions

| Gesture         | Action                  |
|----------------|--------------------------|
| Palm open       | Scroll to next reel      |
| Fist            | Scroll to previous reel  |
| Two fingers     | Click like ❤️            |
| Thumbs up       | Open comments            |
| Swipe left      | Close comments           |

---

## 📸 Screenshots & Demo

Coming soon...

---

## 📚 References

- [arXiv:2309.11610](https://arxiv.org/pdf/2309.11610)
- [HG14 Dataset on Kaggle](https://www.kaggle.com/datasets/gulerosman/hg14-handgesture14-dataset)

