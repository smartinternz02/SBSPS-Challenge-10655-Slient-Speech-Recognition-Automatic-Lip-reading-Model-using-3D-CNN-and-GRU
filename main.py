import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from flask import Flask, render_template, request, send_file

app = Flask(__name__)

def preprocess_frames(frames):
    frames = frames.astype(np.float32)
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


@app.route("/", methods=["POST","GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    
    model_path = "./static/lipnet_model.tflite"
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()

    # Load the StringLookups
    saved_vocab = ['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '?', '!', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ']
    char_to_num = tf.keras.layers.StringLookup(vocabulary=saved_vocab, oov_token="")

    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )
    file = request.files["demo1"]
    video_path = "uploaded_video.mpg"
    file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[190:236, 80:220]
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    #frames = np.expand_dims(frames, axis=0) 
    frames = preprocess_frames(frames)
    frames = np.expand_dims(frames, axis=-1)
    
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(frames, axis=0))
    
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    decoder = tf.keras.backend.ctc_decode(predictions, input_length=[75], greedy=True)[0][0].numpy()
    converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')

    return converted_prediction

if __name__ == "__main__":
    app.run(debug=True)