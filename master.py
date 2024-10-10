import os

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import opendatasets as od
import plotly.express as px

dataset = 'https://www.kaggle.com/datasets/kunalgupta2616/dog-vs-cat-images-data'
od.download(dataset)

class_names = ["Cat", "Dog"]
n_dogs = len(os.listdir('dog-vs-cat-images-data/dogcat/train/dogs'))
n_cats = len(os.listdir('dog-vs-cat-images-data/dogcat/train/cats'))
n_images = [n_cats, n_dogs]
fig = px.pie(names=class_names, values=n_images)
fig.show()