# PRODIGY_ML_04
Hand Gesture Recognition 🤖✋ ML system recognizing 10 hand gestures from LeapGestRecog dataset. Implements SVM (85% accuracy) and CNN (95% accuracy) models. Includes image upload, real-time prediction, and comprehensive evaluation. Perfect for HCI applications, sign language translation, and gesture-based control systems.


🎯 Project Overview
A complete machine learning pipeline for recognizing 10 hand gestures from the LeapGestRecog dataset. Implements both traditional SVM and deep learning CNN approaches with comprehensive evaluation and deployment-ready features.

✨ Key Features
✅ Dual Model Architecture - SVM (85-90% accuracy) & CNN (92-95% accuracy)
✅ Real-Time Image Upload - Instant prediction with confidence scores
✅ Complete Evaluation Suite - Confusion matrices, metrics, visualizations
✅ Optimized for Google Colab - Uses T4 GPU for fast training
✅ Production Ready - Model saving/loading, batch prediction support

📊 Dataset Information
Dataset: LeapGestRecog

Gestures: 10 classes (01_palm, 02_l, 03_fist, etc.)

Images: 20,000 grayscale (240×320 pixels)

Subjects: 10 individuals

Images/Class: 200 per subject per gesture



hand-gesture-recognition/
├── ML_Task4.ipynb              # Main Colab notebook (complete solution)
├── gesture_cnn_model.h5        # Trained CNN model (95% accuracy)
├── gesture_svm_model.pkl       # Trained SVM model (88% accuracy)
├── results.json               # Performance metrics & statistics
├── requirements.txt           # Python dependencies
├── README.md                 # This documentation
└── examples/                 # Sample hand gesture images
    ├── test_palm.png
    ├── test_fist.png
    └── test_l_shape.png




🔧 Technical Implementation
Preprocessing Pipeline
python
1. Grayscale conversion
2. Hand region detection (contour-based)
3. Square padding & resizing (64×64)
4. Normalization (0-1 range)
5. Data augmentation (optional)
CNN Architecture
python
Input(64,64,1) → Conv2D(32) → BatchNorm → Conv2D(32) → MaxPooling
→ Conv2D(64) → BatchNorm → Conv2D(64) → MaxPooling
→ Conv2D(128) → BatchNorm → Conv2D(128) → MaxPooling
→ Flatten → Dense(256) → Dropout → Dense(128) → Dropout
→ Output(10) [Softmax]
