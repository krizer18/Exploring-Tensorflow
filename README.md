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
```
##  Model Architecture


The model is built using a sequential CNN architecture with three convolutional layers followed by max-pooling layers.

## Model Performance
The model's performance improved over the course of training:

- Training Duration: 50s
- Steps per Epoch: 3s/step
- Final Accuracy: 0.7181
- Final Loss: 1.0715

The training process showed improvement in both training and validation accuracy over 3 epochs, with the validation accuracy slightly outperforming the training accuracy.


## Future Improvements


1. Address the model's bias towards cats by:
   - Balancing the dataset
   - Adjusting class weights
   - Fine-tuning the model architecture
2. Increase the number of training epochs to potentially improve accuracy
3. Implement data augmentation techniques to enhance model generalization
4. Experiment with different CNN architectures or transfer learning from pre-trained models

## What I have learnt 

Through this project, which serves as a stepping stone into my machine learning journey, I have learned several important techniques necessary for improving a model's accuracy. This project deepened my understanding of the role batch sizes play in model performance.

Using the Epoch vs. Accuracy graph, I observed the difference in noise between larger and smaller batch sizes. When using a stochastic gradient descent approach, the graph appeared quite noisy, making it difficult to extract clear conclusions. However, with a batch gradient descent approach, I quickly realized that my computer's memory could not handle an entire batch of over 25,000 photos, causing it to crash. After experimenting, I found that setting the batch size to around 750 to 1000 photos worked well.

Additionally, I learned that choosing the right number of epochs is crucial. Too few epochs prevent the model from reaching an acceptable accuracy, while too many can cause the model to overfit and incorrectly predict cats and dogs.

Furthermore, I realized the importance of APIs as powerful tools for innovation and collaboration. Thanks to Kaggle’s API, I was able to readily access images for training, validation, and testing.

There have been many other valuable lessons throughout the process of building this project, and I look forward to expanding my knowledge through future projects.
