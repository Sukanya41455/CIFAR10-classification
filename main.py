import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix

class Cifar10:
    def __init__(self):
        """
        Initialize model path, target labels, model and testset
        """
        self.model_path = "cifar_cnn50.h5"
        self.labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.X_test = None
        self.y_test = None
        self.y_categorical = None
        self.model = None

    def load_testset(self):
        """
        Loads the CIFAR-10 dataset, normalizes the image data, and
        prepare the labels in categorical format for model evaluation.
        """

        # Load CIFAR-10 dataset
        (X_train, y_train), (self.X_test, self.y_test) = cifar10.load_data()
        print("Testset loaded successfully!")

        # Normalize the images (values between 0 and 1)
        self.X_test = self.X_test / 255.0

        # Convert the labels to categorical format (one-hot encoding)
        self.y_categorical = to_categorical(self.y_test, 10)
        return


    def load_model(self):
        """
        Load the trained model from the specified file path.
        """
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
        return

    def predict(self):
        """
        Make predictions on the test set using the loaded model,
        evaluate the model's accuracy,
        and return the predicted labels.
        Returns:
            y_pred (numpy array): The predicted labels for the test set.
        """

        # Predict the classes of the test set
        y_pred = self.model.predict(self.X_test)
        # Convert predicted probabilities to class labels
        y_pred = np.argmax(y_pred, axis=1)
        print("Prediction completed!")
        # Evaluate the model on the test set
        evaluation = self.model.evaluate(self.X_test, self.y_categorical)
        print(f'Accuracy on Testset : {evaluation[1] * 100:.2f}%')
        return y_pred


    def evaluate(self, predictions):
        """
        Evaluate the model's predictions by plotting a confusion matrix.
        Args:
            predictions (numpy array): The predicted class labels for the test set.
        """

        # Generate the confusion matrix
        cm = confusion_matrix(self.y_test, predictions)
        # Create a ConfusionMatrixDisplay object
        cm = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=self.labels)
        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = cm.plot(xticks_rotation='vertical', ax=ax)
        plt.show()
        print("Confusion Matrix generated!")
        return

    def report(self, predictions):
        """
        Print a detailed classification report, including precision, recall, and F1-score.
        Args:
            predictions (numpy array): The predicted class labels for the test set.
        """
        print("Classification Report:")
        print(classification_report(self.y_test, predictions))
        return


if __name__ == '__main__':
    # Initialize the cifar10 prediction class
    obj = Cifar10()
    # Load the testset
    obj.load_testset()
    # Load the trained model
    obj.load_model()
    # Make predictions
    predictions = obj.predict()
    # Run evaluations
    obj.evaluate(predictions)
    # Generate classification report
    obj.report(predictions)
