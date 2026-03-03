# 🚗 Driver Drowsiness Detection System 😴⚠️

An AI-powered **Driver Drowsiness Detection System** built using Deep Learning (CNN) and deployed with a Flask Web Application.

This system detects whether a driver is:

- ✅ Active (Alert)
- 😴 Fatigued (Drowsy)

It supports:

- 🎥 Live Webcam Detection
- 🖼️ Image Upload
- 🎬 Video Upload

---

# 📌 Project Overview

Driver drowsiness is one of the major causes of road accidents.  
This project uses a Convolutional Neural Network (CNN) to classify driver faces into two categories:

- **Active Subjects** – Alert / Awake Drivers  
- **Fatigue Subjects** – Drowsy / Sleepy Drivers  

The trained model is integrated into a web application for real-time monitoring.

---

# 🧠 Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- Flask  
- Pillow  
- NumPy  

---

# 📂 Project Structure

```
Driver-Drowsiness-Detection/
│
├── 0 FaceImages/              # Dataset (Not uploaded to GitHub)
│   ├── Active Subjects/
│   └── Fatigue Subjects/
│
├── saved_models/              # Trained model (Not uploaded)
│   └── drowsiness_cnn.h5
│
├── train_model.py             # Model training script
├── app.py                     # Flask web application
├── requirements.txt           # Project dependencies
└── README.md
```

---

# 📊 Dataset

This project uses the Driver Drowsiness Detection dataset from Kaggle:

Dataset Link:  
https://www.kaggle.com/datasets/amaldev007/driver-drowsiness-detection-system-dataset

## Dataset Folder Structure

```
0 FaceImages/
├── Active Subjects/
└── Fatigue Subjects/
```

- Active Subjects → Alert driver images  
- Fatigue Subjects → Drowsy driver images  

⚠️ Note: The dataset is not included in this repository to keep the project lightweight.

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Sanjay83174/Driver-Drowsiness-Detection-System.git
cd driver-drowsiness-detection
```


---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Download Dataset

Download from Kaggle and place inside:

```
0 FaceImages/
```

Make sure the folder structure matches exactly as shown above.

---

# 🏋️ Train the Model

Run:

```bash
python train_model.py
```

During training, the script will:

- Analyze dataset
- Split into 80% training and 20% validation
- Apply data augmentation
- Train CNN model
- Save best model to:

```
saved_models/drowsiness_cnn.h5
```

---

# 🌐 Run the Web Application

After training is complete:

```bash
python app.py
```

Open your browser and go to:

```
http://localhost:5000
```

---

# 🎥 Application Features

## 🔴 Live Camera Detection
- Real-time webcam monitoring
- Instant classification result

## 🖼️ Image Upload
- Upload a driver image
- Get prediction immediately

## 🎬 Video Upload
- Upload recorded driving video
- Frame-by-frame drowsiness detection

---

# 🧪 Model Details

- Convolutional Neural Network (CNN)
- Binary Classification
- Data Augmentation
- Dropout Regularization
- Binary Crossentropy Loss
- Adam Optimizer

---

# 🚀 Future Improvements

- Add alarm sound when drowsiness detected
- Deploy to cloud (AWS / Render)
- Improve accuracy with larger dataset
- Mobile application integration
- Eye Aspect Ratio based hybrid detection

---

# 💡 Real-World Applications

- Smart Vehicles
- Truck Driver Monitoring
- Public Transport Safety
- Industrial Operator Monitoring

---

# 👨‍💻 Author

Sanjay HL  
Deep Learning & Computer Vision Enthusiast

If you found this project useful, please ⭐ star the repository!

---

# 📜 License

This project is developed for educational and research purposes.
