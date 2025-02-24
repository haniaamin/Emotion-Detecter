import cv2 # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from keras import models # type: ignore

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your emotion detection model
emotion_model = tf.keras.models.load_model('./models/notebok_model_13.h5')  # Replace with the actual path to your model

# Define the emotions that your model predicts
emotion_labels = ['Anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

# Load an image from file or capture from a webcam
img = cv2.imread('EX_photo\elon.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

# Iterate over detected faces
for (x, y, w, h) in faces:
    # Extract the region of interest (ROI) for each face
    face_roi = gray[y:y+h, x:x+w]

    # Resize the face ROI to match the input size expected by the emotion detection model
    face_roi_resized = cv2.resize(face_roi, (48, 48))
    
    face_roi_resized = cv2.cvtColor(face_roi_resized, cv2.COLOR_GRAY2BGR)

    # Normalize the face ROI to match the preprocessing of your emotion detection model
    face_roi_normalized = face_roi_resized / 255.0
    

    # Reshape the image to match the input shape expected by the model
    face_roi_input = np.reshape(face_roi_normalized, (1,48, 48, 3))

    # Perform emotion detection using the TensorFlow model
    emotion_prediction = emotion_model.predict(face_roi_input)

    # Get the predicted emotion label index
    predicted_emotion_index = np.argmax(emotion_prediction)

    # Get the predicted emotion label
    predicted_emotion_label = emotion_labels[predicted_emotion_index]

    # Overlay the emotion label on the original image
    cv2.putText(img, f'Emotion: {predicted_emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Draw a rectangle around the face
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the result
cv2.imshow('Emotion Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


