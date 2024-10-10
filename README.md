# Cats vs Dogs Classification Project

## Project Overview
This project demonstrates how to classify images of cats and dogs using a Convolutional Neural Network (CNN) in TensorFlow and Keras. The dataset used includes images of cats and dogs split into training, validation, and test sets. The model is trained on a dataset containing 25,000 images and uses CNN layers to extract features from the images.

## Project Structure

- **Training Set**: 25,000 images of cats and dogs
- **Validation Set**: 8,000 images of cats and dogs
- **Test Set**: 12,500 images

All images are resized to 150x150 pixels to match the input shape expected by the CNN.

## Requirements

The project uses the following libraries:

- TensorFlow and Keras for model creation and training
- Numpy for numerical computations
- Matplotlib and Seaborn for plotting
- Sklearn for confusion matrix generation
- Plotly for interactive data visualization
- Open Datasets and SciPy for data manipulation

To install the required libraries, you can use pip:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn plotly scipy

