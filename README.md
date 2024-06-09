# Face Recognition using SVM

This project performs face recognition using the MTCNN for face detection and the FaceNet model for generating face embeddings. It then classifies these embeddings using a pretrained SVM model. 

The following components are used:-

MTCNN: Multi-task Cascaded Convolutional Networks for face detection. 

FaceNet: A deep learning model that outputs 128-dimensional embeddings for face images. 

SVM (Support Vector Machine): A classifier trained on face embeddings for recognizing faces. 

# Dependencies
pip install numpy opencv-python pillow mtcnn keras-facenet scikit-learn

# Project Structure
generate_embeddings.py: Generates embeddings for the face images in the dataset and checks their uniqueness.

train_classifier.py: Trains an SVM classifier on the generated embeddings.

predict_image.py: Predicts the labels for faces in a given image and displays the results.

finalized_model.sav: The pretrained SVM model for classification.

label_encoder.pkl: The label encoder used to encode and decode class labels.

# How It Works
Face Detection: MTCNN detects faces in the input image.

Face Embeddings: FaceNet converts the detected face into a 128-dimensional embedding.

Normalization: The embeddings are normalized to have unit length.

Classification: The pretrained SVM classifier predicts the label of each face embedding.

Visualization: The predicted labels and bounding boxes are drawn on the image for visualization.
