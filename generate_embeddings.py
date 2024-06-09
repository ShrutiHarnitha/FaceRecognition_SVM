import cv2
import os
import numpy as np
from os import listdir
from PIL import Image
from numpy import asarray, expand_dims
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet

# Function to preprocess the face image for FaceNet model
def preprocess_face(face_img):
    face_img = face_img.astype('float32')
    mean, std = face_img.mean(), face_img.std()
    face_img = (face_img - mean) / std
    face_img = expand_dims(face_img, axis=0)
    return face_img

# Directory of the dataset
folder = 'dataset/'  
embeddings = []
labels = []

# Initialize the MTCNN detector
detector = MTCNN()

# Initialize FaceNet model
facenet = FaceNet()

# Loop through the images in the dataset folder
for filename in listdir(folder):
    path = os.path.join(folder, filename)
    gbr1 = cv2.imread(path)
    
    if gbr1 is None:
        print(f"Error loading image {path}")
        continue

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
            
            # Flatten the embedding if it has more than 1 dimension
            if embedding.ndim > 1:
                embedding = embedding.flatten()

            # Extract label from the filename 
            label = os.path.splitext(filename)[0].split('_')[0]
            
            # Append embedding and label
            embeddings.append(embedding)
            labels.append(label)

# Convert lists to arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

# Save embeddings and labels to .npz file
np.savez_compressed('faces-embeddings.npz', embeddings, labels)
