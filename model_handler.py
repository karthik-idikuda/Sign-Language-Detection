import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import mediapipe as mp
import pickle
import os

class SignLanguageModel:
    def __init__(self):
        self.model = self._create_model()
        self.mp_hands = mp.solutions.hands
        self.labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # Create model directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
            
        # Load model if exists
        if os.path.exists('models/sign_language_model.h5'):
            try:
                self.model.load_weights('models/sign_language_model.h5')
                print("Model loaded successfully!")
            except:
                print("Training new model...")
                self._train_model()
        else:
            print("Training new model...")
            self._train_model()

    def _create_model(self):
        model = models.Sequential([
            layers.Input(shape=(21, 3)),  # 21 landmarks, 3 coordinates each
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(26, activation='softmax')  # 26 classes for A-Z
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _train_model(self):
        # Generate synthetic data for training
        # This is a simplified version - in production, you should use real data
        X_train = []
        y_train = []
        
        # Generate synthetic hand landmark data
        for i in range(1000):  # 1000 samples
            sample = np.random.rand(21, 3)  # 21 landmarks, 3 coordinates each
            label = np.random.randint(0, 26)  # 26 classes (A-Z)
            X_train.append(sample)
            y_train.append(label)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train the model
        self.model.fit(X_train, y_train, epochs=10, batch_size=32)
        
        # Save the model
        self.model.save_weights('models/sign_language_model.h5')

    def preprocess_landmarks(self, hand_landmarks):
        # Convert landmarks to relative coordinates
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        base_z = hand_landmarks.landmark[0].z
        
        landmarks = []
        for landmark in hand_landmarks.landmark:
            # Calculate relative positions to make prediction invariant to hand position
            landmarks.append([
                landmark.x - base_x,
                landmark.y - base_y,
                landmark.z - base_z
            ])
        return np.array([landmarks])

    def calculate_angles(self, hand_landmarks):
        # Calculate angles between fingers for better gesture recognition
        angles = []
        finger_tips = [4, 8, 12, 16, 20]  # Indices of fingertips
        finger_bases = [2, 5, 9, 13, 17]   # Indices of finger bases
        
        for tip, base in zip(finger_tips, finger_bases):
            tip_point = np.array([
                hand_landmarks.landmark[tip].x,
                hand_landmarks.landmark[tip].y,
                hand_landmarks.landmark[tip].z
            ])
            base_point = np.array([
                hand_landmarks.landmark[base].x,
                hand_landmarks.landmark[base].y,
                hand_landmarks.landmark[base].z
            ])
            wrist_point = np.array([
                hand_landmarks.landmark[0].x,
                hand_landmarks.landmark[0].y,
                hand_landmarks.landmark[0].z
            ])
            
            vector1 = tip_point - base_point
            vector2 = wrist_point - base_point
            
            angle = np.arccos(
                np.dot(vector1, vector2) / 
                (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            )
            angles.append(angle)
            
        return np.array(angles)

    def get_gesture_meaning(self, hand_landmarks):
        """Get meaning of common gestures with advanced detection"""
        angles = self.calculate_angles(hand_landmarks)
        landmarks = hand_landmarks.landmark
        
        # Helper function to check if a finger is extended
        def is_finger_extended(finger_tip_idx, finger_base_idx):
            return landmarks[finger_tip_idx].y < landmarks[finger_base_idx].y

        # Get finger states
        thumb_extended = landmarks[4].x > landmarks[3].x if landmarks[5].x < landmarks[17].x else landmarks[4].x < landmarks[3].x
        index_extended = is_finger_extended(8, 6)
        middle_extended = is_finger_extended(12, 10)
        ring_extended = is_finger_extended(16, 14)
        pinky_extended = is_finger_extended(20, 18)
        
        # Get extended finger count
        extended_fingers = [index_extended, middle_extended, ring_extended, pinky_extended]
        extended_count = sum(extended_fingers)
        
        # Advanced gesture detection
        
        # Open palm / Hello - all fingers extended
        if all(extended_fingers) and thumb_extended:
            return "Hello", 0.98
        
        # Closed fist / No - no fingers extended
        if not any(extended_fingers) and not thumb_extended:
            return "No", 0.98
            
        # Thumbs up - only thumb extended
        if thumb_extended and not any(extended_fingers):
            if landmarks[4].y < landmarks[3].y:
                return "Good", 0.98
            else:
                return "Bad", 0.98
        
        # OK sign - thumb and index finger touching
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        index_tip = np.array([landmarks[8].x, landmarks[8].y])
        distance = np.linalg.norm(thumb_tip - index_tip)
        if distance < 0.05 and middle_extended and ring_extended and pinky_extended:
            return "OK", 0.98
            
        # Victory/Peace - index and middle extended only
        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "Peace", 0.98
            
        # Rock on - index and pinky extended only
        if index_extended and pinky_extended and not middle_extended and not ring_extended:
            return "Rock", 0.98
            
        # Call me - thumb and pinky extended only
        if thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]):
            return "Call", 0.98
            
        # Stop sign - all fingers extended, no thumb
        if all(extended_fingers) and not thumb_extended:
            return "Stop", 0.98
            
        # I Love You - thumb, index, and pinky extended
        if thumb_extended and index_extended and pinky_extended and not middle_extended and not ring_extended:
            return "Love", 0.98
            
        # Silence/Shush - index finger to lips gesture
        if index_extended and not any([middle_extended, ring_extended, pinky_extended]):
            if landmarks[8].y > landmarks[0].y:  # finger pointing down/forward
                return "Quiet", 0.95
                
        # Direction gestures - single index finger pointing
        if index_extended and not any([middle_extended, ring_extended, pinky_extended]):
            # Point up
            if landmarks[8].y < landmarks[5].y - 0.15:
                return "Up", 0.95
            # Point down  
            elif landmarks[8].y > landmarks[5].y + 0.15:
                return "Down", 0.95
            # Point right
            elif landmarks[8].x > landmarks[5].x + 0.15:
                return "Right", 0.95
            # Point left
            elif landmarks[8].x < landmarks[5].x - 0.15:
                return "Left", 0.95
            else:
                return "Point", 0.90
                
        # Number gestures 1-5
        if extended_count > 0 and extended_count <= 5:
            # One finger (index only)
            if extended_count == 1 and index_extended:
                return "One", 0.98
            # Two fingers (index and middle)
            elif extended_count == 2 and index_extended and middle_extended:
                return "Two", 0.98
            # Three fingers (index, middle, ring)
            elif extended_count == 3 and index_extended and middle_extended and ring_extended:
                return "Three", 0.98
            # Four fingers (all except pinky)
            elif extended_count == 4 and not pinky_extended:
                return "Four", 0.98
            # Five fingers (all extended)
            elif extended_count == 4 and pinky_extended:  # Note: checking 4 because thumb is separate
                return "Five", 0.98
        
        # More advanced gestures
        # Gun gesture - thumb up, index extended, others folded
        if thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]):
            if abs(landmarks[4].y - landmarks[8].y) < 0.05:  # thumb and index roughly aligned
                return "Gun", 0.95
                
        # Hang loose - thumb and pinky extended, hand tilted
        if thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]):
            return "Hang Loose", 0.95
            
        # Thinking gesture - index finger to temple
        if index_extended and not any([middle_extended, ring_extended, pinky_extended]):
            if landmarks[8].x > landmarks[0].x + 0.1:  # finger positioned to the side
                return "Think", 0.90
                
        # Money gesture - thumb rubbing against fingers
        if thumb_extended and index_extended and middle_extended:
            return "Money", 0.90
            
        # Come here gesture - all fingers curled in and out motion
        if not any(extended_fingers):
            return "Come", 0.85
            
        # Applause gesture - clapping motion detection would need temporal analysis
        # For now, we'll detect open palms facing each other
        if all(extended_fingers):
            return "Clap", 0.80
        
        return None, None

    def predict(self, hand_landmarks):
        if hand_landmarks is None:
            return None, None
        
        # First check for common gestures
        gesture, confidence = self.get_gesture_meaning(hand_landmarks)
        if gesture:
            return gesture, confidence
        
        # If no common gesture detected, try ASL recognition
        processed_landmarks = self.preprocess_landmarks(hand_landmarks)
        prediction = self.model.predict(processed_landmarks)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        if confidence > 0.7:  # Confidence threshold
            return self.labels[predicted_class], confidence
        return None, None
