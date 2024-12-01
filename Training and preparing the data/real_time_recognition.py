import cv2
import mediapipe as mp
import pickle
import numpy as np
import random

# Load MediaPipe Hands
mp_hands = mp.solutions.hands

glist = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
tgesture = random.choice(glist)

# Load the trained model, label encoder, and scaler
with open('gesture_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)
    label_encoder = pickle.load(model_file)
    scaler = pickle.load(model_file)  # Load scaler for real-time normalization

# Initialize MediaPipe Hands for hand tracking
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize OpenCV video capture (webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.putText(frame, "Do: " + tgesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Convert the frame to RGB (MediaPipe requires RGB input)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                # Append each landmark's x, y, z coordinates to the landmarks list
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            # Convert landmarks list to a flattened NumPy array and reshape for prediction
            landmarks = np.array(landmarks).flatten().reshape(1, -1)

            # Normalize landmarks using the scaler (to match training conditions)
            landmarks = scaler.transform(landmarks)

            # Predict the gesture using the trained model
            prediction = clf.predict(landmarks)
            gesture = label_encoder.inverse_transform(prediction)

            # Display the predicted gesture on the frame
            cv2.putText(frame, f'Gesture: {gesture[0]}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if gesture == tgesture :
                cv2.putText(frame, "Correct !", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else :
                cv2.putText(frame, "Incorrect !", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        # If no hand landmarks are detected, display a "No Gesture Detected" message
        cv2.putText(frame, 'Wrong Gesture: No Gesture Detected', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with the gesture label or "wrong gesture"
    cv2.imshow("Real-Time Hand Gesture Recognition", frame)
    


    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
