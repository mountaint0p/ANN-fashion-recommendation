import streamlit as st
import numpy as np
from keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import faiss
from PIL import Image

@st.cache_resource
def loadResources():
    # Load model + feature extractor
    model = load_model('model-data/model.h5')
    featureExtractor = Model(inputs=model.input, outputs=model.get_layer('dense_4').output)
    
    # Load features, labels, and image paths
    embedding_data = np.load("model-data/features-data.npz", mmap_mode='r')
    features = embedding_data["features"]
    labels = embedding_data["labels"]
    paths = embedding_data["paths"]
    
    # Set up FAISS index + normalize features
    d = features.shape[1]
    index = faiss.IndexFlatL2(d)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalizedFeatures = features / norms
    index.add(normalizedFeatures)
    
    return model, featureExtractor, features, labels, paths, index

model, featureExtractor, features, labels, paths, index = loadResources()

def preprocessImage(image):
    # issue with image dimensionality 
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Resize the image
    img = image.resize((256, 256))
    # Convert the image to a numpy array 
    imgArray = img_to_array(img)  
    imgArray = np.expand_dims(imgArray, axis=0) 
    return imgArray

def extractFeatures(image, featureExtractor):
    # Create an ImageDataGenerator for preprocessing
    input_datagen = ImageDataGenerator(rescale=1./255)
    imgArray = preprocessImage(image)
    input_gen = input_datagen.flow(imgArray, batch_size=1, shuffle=False)
    feats = featureExtractor.predict(input_gen)
    # Normalize features
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    normalizedFeats = feats / norms
    return normalizedFeats

def getRecommendations(queryFeatures, index, paths):
    # Search for k nearest neighbors
    k = 5
    distances, indices = index.search(queryFeatures, k)
    neighborIndices = indices[0]  
    recommendedPaths = [paths[i] for i in neighborIndices]
    return recommendedPaths

# UI
st.title("Fashion Recommendation Demo")

# Upload image
uploadedFile = st.file_uploader("Upload an outfit picture for some recommendations!", type=["jpg", "jpeg", "png"])

if uploadedFile is not None:
    # Display the inputted image
    userImage = Image.open(uploadedFile)
    st.image(userImage, caption="Uploaded Image", use_container_width=True)

    # Extract features
    inputFeatures = extractFeatures(userImage, featureExtractor)

    # Get top 5 recommendations
    recommendationPaths = getRecommendations(inputFeatures, index, paths)

    # Load and display recommended image
    st.subheader("Top 5 Recommendations:")
    for recPath in recommendationPaths:
        recImg = Image.open(recPath)
        st.image(recImg, caption=recPath, use_container_width=True)
