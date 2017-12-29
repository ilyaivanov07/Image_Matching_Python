# to run: python nn_using_keras.py --image images/African_Bush_Elephant.jpg

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

# model = ResNet50(weights='imagenet')

target_size = (150, 150)


def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  # return preds[0]
  return decode_predictions(preds, top=3)


# def predict(model, img, target_size, top_n=3):
#   """Run model prediction on image
#   Args:
#     model: keras model
#     img: PIL format image
#     target_size: (w,h) tuple
#     top_n: # of top predictions to return
#   Returns:
#     list of predicted labels and their probabilities
#   """
#   if img.size != target_size:
#     img = img.resize(target_size)
#
#   x = image.img_to_array(img)
#   x = np.expand_dims(x, axis=0)
#   x = preprocess_input(x)
#   preds = model.predict(x)
#   return decode_predictions(preds, top=top_n)[0]

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


# def plot_preds(image, preds):
#   plt.imshow(image)
#   plt.axis('off')
#   plt.figure()
#   order = list(reversed(range(len(preds))))
#   # labels = ("cat", "dog")
#   labels = (pr[1] for pr in preds)
#   plt.barh([0, 1], preds, alpha=0.5)
#   plt.yticks(order, labels)
#   plt.xlabel('Probability')
#   plt.xlim(0, 1.01)
#   plt.tight_layout()
#   plt.show()


if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image")
  a.add_argument("--image_url", help="url to image")
  args = a.parse_args()

  if args.image is None and args.image_url is None:
    a.print_help()
    sys.exit(1)

  if args.image is not None:
    # img = Image.open(args.image)
    model = load_model("examples/first_try.h5")
    img = Image.open("images/2_scale.jpg")
    preds = predict(model, img, target_size)
    plot_preds(img, preds)

  # if args.image_url is not None:
  # args.image_url = "https://www.google.com/imgres?imgurl=https%3A%2F%2Fwww.humanrights.gov.au%2Fsites%2Fdefault%2Ffiles%2Fstyles%2Flarge%2Fpublic%2Fnews%2Fimage%2FTimWilson20160215_0.jpg%3Fitok%3DjUa84LIC&imgrefurl=https%3A%2F%2Fwww.humanrights.gov.au%2Fnews%2Fstories%2Fhuman-rights-commissioner-resigns&docid=_yA4coIS0MmUiM&tbnid=z2mV3H7kyDjtaM%3A&vet=10ahUKEwjD262ajpnYAhWFPN8KHR9ZAJwQMwiVASgGMAY..i&w=760&h=425&bih=854&biw=1280&q=human&ved=0ahUKEwjD262ajpnYAhWFPN8KHR9ZAJwQMwiVASgGMAY&iact=mrc&uact=8"
  # response = requests.get(args.image_url)
  # img = Image.open(BytesIO(response.content))
  # preds = predict(model, img, target_size)
  # plot_preds(img, preds)