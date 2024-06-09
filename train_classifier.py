from numpy import load
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = load('faces-embeddings.npz')
embeddings, labels = data['arr_0'], data['arr_1']

# Print the shape of embeddings and labels for debugging
print('Embeddings shape:', embeddings.shape)
print('Labels shape:', labels.shape)

# Ensure embeddings array is 2D
n_samples = embeddings.shape[0]
n_features = embeddings.shape[1] if len(embeddings.shape) > 1 else 1
print('Number of samples:', n_samples)
print('Number of features:', n_features)

# Reshape embeddings if necessary
if len(embeddings.shape) == 3:
    embeddings = embeddings.reshape((n_samples, n_features))

# Normalize input vectors
in_encoder = Normalizer(norm='l2')
embeddings = in_encoder.transform(embeddings)

# Label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(labels)
labels = out_encoder.transform(labels)

# Fit model
model = SVC(kernel='linear', probability=True)
model.fit(embeddings, labels)

# Save the model and the label encoder
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

label_encoder_filename = 'label_encoder.pkl'
pickle.dump(out_encoder, open(label_encoder_filename, 'wb'))

# Evaluate model (optional)
yhat_train = model.predict(embeddings)
score_train = accuracy_score(labels, yhat_train)
print(f'Accuracy: train={score_train*100:.3f}')
