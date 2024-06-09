import cv2
import numpy as np
from PIL import Image
from numpy import asarray, expand_dims
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer, LabelEncoder
import pickle

# Function to preprocess the face image for FaceNet model
def preprocess_face(face_img):
    face_img = face_img.astype('float32')
    mean, std = face_img.mean(), face_img.std()
    face_img = (face_img - mean) / std
    face_img = expand_dims(face_img, axis=0)
    return face_img

# Load the MTCNN detector and FaceNet model
detector = MTCNN()
facenet = FaceNet()

# Load the SVM model
model_filename = 'finalized_model.sav'
model = pickle.load(open(model_filename, 'rb'))

# Load the Label Encoder
label_encoder_filename = 'label_encoder.pkl'
label_encoder = pickle.load(open(label_encoder_filename, 'rb'))

# Normalize input vectors
in_encoder = Normalizer(norm='l2')

# Function to predict the label of a given image
def predict_image(image_path):
    gbr1 = cv2.imread(image_path)
    
    if gbr1 is None:
        print(f"Error loading image {image_path}")
        return

    # Detect faces using MTCNN
    faces = detector.detect_faces(gbr1)
    
    if faces:
        for face in faces:
            x1, y1, width, height = face['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
            gbr_array = asarray(gbr)
            
            face_region = gbr_array[y1:y2, x1:x2]
            face_img = Image.fromarray(face_region)
            face_img = face_img.resize((160, 160))
            face_img = asarray(face_img)
            
            face_img = preprocess_face(face_img)
            
            # Ensure the embedding extraction method is correct
            embedding = facenet.embeddings(face_img)
            embedding = in_encoder.transform(embedding)

            # Predict
            yhat_class = model.predict(embedding)
            yhat_prob = model.predict_proba(embedding)

            # Get label
            predicted_label = label_encoder.inverse_transform(yhat_class)
            print(f'Predicted: {predicted_label[0]} (Probability: {yhat_prob[0][yhat_class[0]]:.2f})')

            # Draw a rectangle around the face
            cv2.rectangle(gbr1, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Annotate the label and probability
            label = f'{predicted_label[0]}'
            cv2.putText(gbr1, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Display the image
        cv2.imshow('Image', gbr1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No faces detected in the image.")

# Path to the unseen image
unseen_image_path = 'media/solo_face.jpg'
predict_image(unseen_image_path)
