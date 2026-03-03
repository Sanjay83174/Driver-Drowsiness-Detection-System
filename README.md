🚗 Driver Drowsiness Detection System 😴⚠️

An AI-powered Driver Drowsiness Detection System built using Deep Learning (CNN) and deployed with a Flask Web Application.

This project detects whether a driver is Alert (Active) or Drowsy (Fatigued) using:

🎥 Live Webcam Detection

🖼️ Image Upload

🎬 Video Upload

The goal is to help reduce road accidents caused by driver fatigue.

📌 Project Overview

Driver drowsiness is one of the leading causes of road accidents worldwide.
This project uses a Convolutional Neural Network (CNN) to classify driver faces into two categories:

✅ Active Subjects (Alert / Awake Drivers)

😴 Fatigue Subjects (Drowsy / Sleepy Drivers)

The trained model is integrated into a user-friendly web application for real-time detection.

🧠 Tech Stack

🐍 Python

🤖 TensorFlow / Keras

📷 OpenCV

🌐 Flask

🖼️ Pillow

🔢 NumPy

📂 Project Structure

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

📊 Dataset Information

This project uses the Driver Drowsiness Detection Dataset from Kaggle:

🔗 Dataset Link:
https://www.kaggle.com/datasets/amaldev007/driver-drowsiness-detection-system-dataset

Dataset Structure

After downloading, organize it like this:

0 FaceImages/
   ├── Active Subjects/
   └── Fatigue Subjects/

   Active Subjects → Alert driver images

Fatigue Subjects → Drowsy driver images

Labels are automatically inferred from folder names.

⚠️ The dataset is not included in this repository to keep the project lightweight and professional.

⚙️ Installation & Setup
1️⃣ Clone the Repository

git clone https://github.com/your-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection

2️⃣ Install Required Packages

pip install -r requirements.txt

3️⃣ Download the Dataset

Download the dataset from the Kaggle link above.

Place it inside:

0 FaceImages/

Make sure the folder structure matches exactly.

🏋️ Train the CNN Model

Run:
python train_model.py

🔍 During Training, the Script Will:

Analyze class distribution

Split dataset into training and validation (80/20)

Apply data augmentation

Train CNN model

Save best model to:

saved_models/drowsiness_cnn.h5

🌐 Run the Web Application

After training is complete:

python app.py

Then open your browser:

http://localhost:5000

🎥 Application Features
🔴 Live Camera Detection

Uses webcam

Real-time face detection

Instant prediction (Active / Drowsy)

🖼️ Image Upload

Upload a driver image

Get classification result instantly

🎬 Video Upload

Upload recorded driving video

Frame-by-frame drowsiness detection

🧪 Model Details

Convolutional Neural Network (CNN)

Binary Classification

Data Augmentation

Dropout Regularization

Loss Function: Binary Crossentropy

Optimizer: Adam

🚀 Future Improvements

🔊 Add alarm sound when drowsiness detected

📱 Convert into mobile application

🌍 Deploy to cloud (AWS / Render / Heroku)

😎 Hybrid detection using Eye Aspect Ratio

🎯 Improve accuracy using larger dataset

💡 Real-World Applications

🚛 Truck Driver Monitoring Systems

🚗 Smart Vehicles

🚌 Public Transport Safety

🏭 Industrial Operator Monitoring

👨‍💻 Author

Sanjay HL
Deep Learning & Computer Vision Enthusiast

If you found this project useful, please ⭐ star the repository!

📜 License

This project is developed for educational and research purposes.

⭐ Why This Project Stands Out

✔ Real-world safety problem
✔ Deep Learning + Computer Vision
✔ End-to-end pipeline (Training → Deployment)
✔ Interactive Web Application
✔ Clean and professional ML project structure
