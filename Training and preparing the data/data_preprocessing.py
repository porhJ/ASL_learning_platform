import cv2
import mediapipe as mp
import os
import numpy as np
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Gesture labels
gesture_labels = ["A", "B", "C", "D","E", "F"]

# Prepare dataset list
dataset = []

# Process images from each gesture folder
for label in gesture_labels:
    gesture_folder = os.path.join("dataset", label)  # Path to gesture folder
    for filename in os.listdir(gesture_folder):
        # Build the full path to the image
        image_path = os.path.join(gesture_folder, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Could not read image {image_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)  # x-coordinate
                    landmarks.append(lm.y)  # y-coordinate
                    landmarks.append(lm.z)  # z-coordinate
                
                # Flatten landmarks to 1D array
                landmarks = np.array(landmarks).flatten()
                
                # Append the landmarks and the gesture label
                dataset.append((landmarks, label))

# Save the dataset to CSV
with open('hand_gesture_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write headers if needed (optional)
    writer.writerow([f'Landmark_{i+1}' for i in range(63)] + ['Gesture'])  # 63 for 21 landmarks * 3 coordinates
    for data in dataset:
        writer.writerow([*data[0], data[1]])  # Write landmarks + label

print("Dataset has been saved to hand_gesture_data.csv.")
