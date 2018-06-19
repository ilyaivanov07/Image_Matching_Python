import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib
#import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot

target_size = (150, 150) #fixed size for InceptionV3 architecture; the size needs to be the same as in the trained model


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
  y_classes = preds.argmax(axis=-1) # this returns the index in the class array
  print (y_classes)

  return preds[0]


def plot_preds(image, preds):
  plt = matplotlib.pyplot
  plt.imshow(image)
  plt.axis('off')
  plt.figure()
  labels = ("cat", "dog")
  plt.barh([0, 1], preds, alpha=0.5)
  plt.yticks([0, 1], labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()


if __name__=="__main__":
  a = argparse.ArgumentParser()
  args = a.parse_args()

  model = load_model("examples/first_try.h5")
  img = Image.open("images/train/dogs/dog.2.jpg")
  preds = predict(model, img, target_size)
  for i in preds:
      print(i)

  plot_preds(img, preds)

