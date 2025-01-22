# Tala Flaifel 
# run code with  
# python predict.py ./test_images/wild_pansy.jpg classification_model.h5 
# python predict.py ./test_images/wild_pansy.jpg classification_model.h5 --category_names label_map.json --top_k 3



import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import tensorflow_hub as hub
from PIL import Image
import json

# custom Keras layer
get_custom_objects()['KerasLayer'] = hub.KerasLayer

# process the image
def process_image(image: np.ndarray) -> np.ndarray:
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224)) #resize image
    image = image / 255.0 #normalize pixels
    image = image.numpy() #convert to numpy
    
    if image.shape[0] == 1: #remove any extra batch 
        image = image[0]
    
    return image

# load model
def load_model_with_custom_objects(model_path: str) -> tf.keras.Model:
    return load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

# predict function
def predict(image_path: str, model: tf.keras.Model, top_k: int) -> tuple:
    image = Image.open(image_path) #load and process img 
    image = np.asarray(image)
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    
    predictions = model.predict(processed_image) # predict 
    
    top_k_indices = np.argsort(predictions[0])[-top_k:][::-1] #find top k
    top_k_probs = predictions[0][top_k_indices]    
    top_k_classes = [str(idx) for idx in top_k_indices]
    
    return top_k_probs, top_k_classes

# get classes names (more user friendly)
def load_class_names(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('--top_k', type=int, default=1, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping labels to flower names')

    args = parser.parse_args()
    
    # load the model with custom objects
    model = load_model_with_custom_objects(args.model_path)
    
    # predictions
    top_k_probs, top_k_classes = predict(args.image_path, model, args.top_k)
    
    # get class names
    if args.category_names:
        class_names = load_class_names(args.category_names)
        top_k_class_names = [class_names.get(cls, "Unknown") for cls in top_k_classes]
    else:
        top_k_class_names = top_k_classes
    
    # results
    print(f"Top {args.top_k} predictions:")
    for i in range(len(top_k_class_names)):
        print(f"{top_k_class_names[i]}: {top_k_probs[i]:.4f}")

if __name__ == '__main__':
    main()
