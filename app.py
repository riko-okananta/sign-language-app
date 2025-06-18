from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='SLD_ResNet50V2_FT.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize webcam
camera = cv2.VideoCapture(0)

# Define class labels (A-Z for ASL alphabet)
labels = [chr(i) for i in range(65, 91)]  # A-Z

def preprocess_frame(frame):
    # Resize frame to match model input (e.g., 224x224)
    frame = cv2.resize(frame, (224, 224))
    # Normalize pixel values
    frame = frame / 255.0
    # Convert to FLOAT32 for TFLite
    frame = frame.astype(np.float32)
    # Expand dimensions for model input
    frame = np.expand_dims(frame, axis=0)
    return frame

def predict_sign(frame):
    # Preprocess frame
    processed_frame = preprocess_frame(frame)
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_frame)
    # Run inference
    interpreter.invoke()
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Get predicted class
    predicted_class = labels[np.argmax(output_data)]
    return predicted_class

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Predict sign
            predicted_sign = predict_sign(frame)
            # Draw prediction on frame
            cv2.putText(frame, f'Sign: {predicted_sign}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    success, frame = camera.read()
    if success:
        predicted_sign = predict_sign(frame)
        # Optional: Convert prediction to speech
        tts = gTTS(text=predicted_sign, lang='en')
        tts.save('static/output.mp3')
        return predicted_sign
    return "Error reading frame"

if __name__ == '__main__':
    app.run(debug=True)