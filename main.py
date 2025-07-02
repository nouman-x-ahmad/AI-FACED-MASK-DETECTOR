#Nouman Ahmad



import cv2
import numpy as np
from flask import Flask, render_template, Response
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)


MODEL_PATH = "MobileNetV2_finetuned.h5"
model = load_model(MODEL_PATH)

# same image size as used in training
IMAGE_SIZE = (224, 224)

# Label mapping 
labels_dict = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}

# Loadin.. Haar Cascade for face detection 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def preprocess_face(face):
   
    face = cv2.resize(face, IMAGE_SIZE)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face


def generate_frames():
    # Usin OpenCV to capture from the webcam
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert the frame to gray scale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            for (x, y, w, h) in faces:
                # Extract the face ROI
                face_roi = frame[y:y + h, x:x + w]
                # Preprocess the face ROI
                face_input = preprocess_face(face_roi)
                # Predict mask status
                preds = model.predict(face_input)
                class_idx = np.argmax(preds, axis=1)[0]
                label = labels_dict[class_idx]
                # Set bounding box color: green if wearing mask correctly, red otherwise
                if label == "with_mask":
                    color = (0, 255, 0)  # green
                elif label == "mask_weared_incorrect":
                    color = (255, 0, 0)   # yellow
                else:
                    color = (0, 0, 255)  # red
                # Draw the bounding box and label on the frame
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    # Main page that displays the video stream
    return render_template("index.html")


@app.route('/video_feed')
def video_feed():
    # Video streaming route. .
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # Runingg Flask app on localhost
    app.run(host="0.0.0.0", port=5000, debug=True)
