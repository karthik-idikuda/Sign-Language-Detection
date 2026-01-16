# Sign Language Detection

## Overview
A real-time computer vision system that translates sign language gestures into text and speech. This project aims to bridge the communication gap for the deaf and hard-of-hearing community by providing an accessible, AI-powered interpreter.

## Features
-   **Real-time Prediction**: Instant translation of hand signs to text.
-   **Alphabet Support**: Recognition of American Sign Language (ASL) alphabets.
-   **Sentence Formation**: Ability to string together words into sentences.
-   **Text-to-Speech**: Vocalization of the translated text.
-   **Adaptive Learning**: (Optional) Feature to fine-tune the model on user-specific variations.

## Technology Stack
-   **Vision**: OpenCV, MediaPipe for hand landmarks.
-   **Model**: LSTM / CNN (TensorFlow/Keras) for sequence classification.
-   **Language**: Python.

## Usage Flow
1.  **Capture**: Webcam records the user's hand signs.
2.  **Extract**: Keypoints (joints) are extracted from the frames.
3.  **Predict**: The model classifies the sequence of keypoints.
4.  **Output**: Recognized text is displayed and spoken aloud.

## Quick Start
```bash
# Clone the repository
git clone https://github.com/Nytrynox/Sign-Language-Detection.git

# Install requirements
pip install -r requirements.txt

# Start detection
python app.py
```

## License
MIT License

## Author
**Karthik Idikuda**
