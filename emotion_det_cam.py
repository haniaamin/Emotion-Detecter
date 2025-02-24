import cv2# type: ignore
import numpy as np# type: ignore
import tensorflow as tf# type: ignore

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your emotion detection model
emotion_model = tf.keras.models.load_model('model9')  # Replace with the actual path to your model

# Define the emotions that your model predicts
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Open a connection to the laptop camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for each face
        face_roi = gray[y:y+h, x:x+w]

        # Resize the face ROI to match the input size expected by the emotion detection model
        face_roi_resized = cv2.resize(face_roi, (48, 48))

        # Normalize the face ROI to match the preprocessing of your emotion detection model
        face_roi_normalized = face_roi_resized / 255.0

        # Convert the grayscale image to three channels (RGB) if your model expects it
        face_roi_normalized_rgb = cv2.merge([face_roi_normalized] * 3)

        # Reshape the image to match the input shape expected by the model
        face_roi_input = np.reshape(face_roi_normalized_rgb, (1, 48, 48, 3))

        # Perform emotion detection using the TensorFlow model
        emotion_prediction = emotion_model.predict(face_roi_input)

        # Get the predicted emotion label index
        predicted_emotion_index = np.argmax(emotion_prediction)

        # Get the predicted emotion label
        predicted_emotion_label = emotion_labels[predicted_emotion_index]

        # Overlay the emotion label on the original frame
        cv2.putText(frame, f'Emotion: {predicted_emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
