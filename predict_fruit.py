import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load trained model
model = tf.keras.models.load_model("fruit_classification_model.keras")
print("Model loaded successfully!")

# Class names in same order as training
class_names = ['apple', 'banana', 'mango', 'orange', 'strawberry']

# Get image path from user
image_path = input("\nEnter image path: ")

# Check if file exists
if not os.path.exists(image_path):
    print("Error: File not found!")
    exit()

# Load and preprocess image
original_image = image.load_img(image_path)
processed_image = image.load_img(image_path, target_size=(128, 128))
image_array = image.img_to_array(processed_image)
image_array = np.expand_dims(image_array, axis=0)
image_array = image_array / 255.0

# Make prediction
predictions = model.predict(image_array, verbose=0)
predicted_index = np.argmax(predictions)
confidence = np.max(predictions) * 100
predicted_fruit = class_names[predicted_index]

# Display result
plt.figure(figsize=(6,5))
plt.imshow(original_image)
plt.title(f"Predicted: {predicted_fruit}\nConfidence: {confidence:.2f}%", 
          fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()

# Print results
print("\n" + "="*40)
print("PREDICTION RESULT")
print("="*40)
print(f"Predicted Fruit: {predicted_fruit}")
print(f"Confidence: {confidence:.2f}%")
print("="*40)

# Show all class probabilities
print("\nClass-wise Probabilities:")
print("-"*35)
for i, (name, prob) in enumerate(zip(class_names, predictions[0])):
    percentage = prob * 100
    bar = "|" * int(percentage / 5)
    print(f"{i+1}. {name:12} : {percentage:5.2f}%  {bar}")