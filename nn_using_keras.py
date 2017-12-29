# to run: python nn_using_keras.py --image images/African_Bush_Elephant.jpg

# Ilya: this script works for ImageNet model ONLY!

# https://github.com/fchollet/keras/tree/master/examples

# Keras is an open source neural network library written in Python.
# https://keras.io/

# Theano is a numerical computation library for Python.
# In Theano, computations are expressed using a NumPy-esque syntax and compiled to run efficiently
# on either CPU or GPU architectures.

import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.models import load_model

model = ResNet50(weights='imagenet')

target_size = (224, 224)


def predict(model, img, target_size, top_n=5):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
    top_n: # of top predictions to return
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return decode_predictions(preds, top=top_n)[0]


def plot_preds(image, preds):
  plt.imshow(image)
  plt.axis('off')
  plt.figure()
  order = list(reversed(range(len(preds))))
  bar_preds = [pr[2] for pr in preds]
  labels = (pr[1] for pr in preds)
  plt.barh(order, bar_preds, alpha=0.5)
  plt.yticks(order, labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()


if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image")
  args = a.parse_args()

  if args.image is not None:
    img = Image.open(args.image)
    preds = predict(model, img, target_size)
    plot_preds(img, preds)

