import cv2
import mediapipe as mp
import pickle
import numpy as np
import io
from flask import Flask, render_template
from flask_socketio import SocketIO, emit # type: ignore
from PIL import Image
import time
from threading import Thread, Event

app = Flask(__name__)
socketio = SocketIO(app)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)


with open('gesture_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)
    label_encoder = pickle.load(model_file)
    scaler = pickle.load(model_file)  

# Gesture assignment
global tgesture
glist = ["A", "B", "C", "D", "E"]
tgesture = np.random.choice(glist)
stop_event = Event()  


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def on_connect():
    print("Client connected")
    socketio.emit('AssignedGesture', {'gesture': tgesture})

@socketio.on('next')
def next_gesture():
    global tgesture
    tgesture = np.random.choice(glist)

def assigned_gesture():
    global tgesture
    while not stop_event.is_set():
        time.sleep(1)
        socketio.emit('AssignedGesture', {'gesture': tgesture})

@socketio.on('message')
def handle_video_frame(data):
    # Convert the received blob data to a NumPy array
    image = Image.open(io.BytesIO(data))
    frame = np.array(image)

    # Process the frame using MediaPipe and the model
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)
            landmarks = np.array(landmarks).flatten().reshape(1, -1)

            # Normalize landmarks using the scaler
            landmarks = scaler.transform(landmarks)

            
            prediction = clf.predict(landmarks)
            gesture = label_encoder.inverse_transform(prediction)[0]

           
            emit('current_gesture', {'current': gesture})
            if gesture == tgesture:
                emit('correctness', {'correctness': "correct"})
            else:
                emit('correctness', {'correctness': "incorrect"})
    else:
        emit('current_gesture', {'current': "nothing"})

    
    _, encoded_frame = cv2.imencode('.jpg', frame)
    emit('video_feed', encoded_frame.tobytes())


if __name__ == '__main__':
    thread = Thread(target=assigned_gesture)
    thread.start() 

    try:
        socketio.run(app, debug=True)
    finally:
        stop_event.set()  
        thread.join()  
