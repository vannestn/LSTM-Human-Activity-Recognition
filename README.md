## Introduction:
This work expands upon the efforts of Guillaume Chevalier:
> Guillaume Chevalier, LSTMs for Human Activity Recognition, 2016,
> https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

Guillaume used a TensorFlow version 1.x model that achieved 91% accuracy to classify motion data. In this notebook I have reused code from their introduction and data wrangling efforts, before defining my own CNN-DNN model with TensorFlow version 2. Sections that used Guillaume's code are annotated for clarity.

My model achieved 92.1% accuracy with validation data. Reviewing graphs, you will se that training and validation accuracies remain relatively in sync, which suggests minimal overfitting.





## Setup Environment for Running in Google Colab
This step is done because some local environments may not have a GPU, and so running this notebook in colab allows users to mount a T4 GPU cheaply in order to speed up the training process


```python
import os

try:
    from google.colab import drive

    # Mount Google Drive
    drive.mount('/content/drive')

    # Change to what you expect to be the notebook's directory
    # based on your common directory structure
    GITHUB_FOLDER_PATH = '/content/drive/My Drive/Colab_Notebooks/github/'
    REPO_NAME = 'LSTM-Human-Activity-Recognition'
    os.chdir(os.path.join(GITHUB_FOLDER_PATH, REPO_NAME))
    print(f"Changed directory to {os.getcwd()}")

except ImportError:
    # Local environment (like running in VS Code)
    notebook_path = os.getcwd()
    print(f"Running on local machine, current directory is {notebook_path}")
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    Changed directory to /content/drive/My Drive/Colab_Notebooks/github/LSTM-Human-Activity-Recognition


### (Authored by Guillaume Chevalier)
# <a title="Activity Recognition" href="https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition" > LSTMs for Human Activity Recognition</a>

Human Activity Recognition (HAR) using smartphones dataset and an LSTM RNN. Classifying the type of movement amongst six categories:
- WALKING,
- WALKING_UPSTAIRS,
- WALKING_DOWNSTAIRS,
- SITTING,
- STANDING,
- LAYING.

Compared to a classical approach, using a Recurrent Neural Networks (RNN) with Long Short-Term Memory cells (LSTMs) require no or almost no feature engineering. Data can be fed directly into the neural network who acts like a black box, modeling the problem correctly. [Other research](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.names) on the activity recognition dataset can use a big amount of feature engineering, which is rather a signal processing approach combined with classical data science techniques. The approach here is rather very simple in terms of how much was the data preprocessed.

Let's use Google's neat Deep Learning library, TensorFlow, demonstrating the usage of an LSTM, a type of Artificial Neural Network that can process sequential data / time series.

## Video dataset overview

Follow this link to see a video of the 6 activities recorded in the experiment with one of the participants:

<p align="center">
  <a href="http://www.youtube.com/watch?feature=player_embedded&v=XOEN9W05_4A
" target="_blank"><img src="http://img.youtube.com/vi/XOEN9W05_4A/0.jpg"
alt="Video of the experiment" width="400" height="300" border="10" /></a>
  <a href="https://youtu.be/XOEN9W05_4A"><center>[Watch video]</center></a>
</p>

## Details about the input data

I will be using an LSTM on the data to learn (as a cellphone attached on the waist) to recognise the type of activity that the user is doing. The dataset's description goes like this:

> The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used.

That said, I will use the almost raw data: only the gravity effect has been filtered out of the accelerometer  as a preprocessing step for another 3D feature as an input to help learning. If you'd ever want to extract the gravity by yourself, you could fork my code on using a [Butterworth Low-Pass Filter (LPF) in Python](https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform) and edit it to have the right cutoff frequency of 0.3 Hz which is a good frequency for activity recognition from body sensors.

## What is an RNN?

As explained in [this article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), an RNN takes many input vectors to process them and output other vectors. It can be roughly pictured like in the image below, imagining each rectangle has a vectorial depth and other special hidden quirks in the image below. **In our case, the "many to one" architecture is used**: we accept time series of [feature vectors](https://www.quora.com/What-do-samples-features-time-steps-mean-in-LSTM/answer/Guillaume-Chevalier-2) (one vector per [time step](https://www.quora.com/What-do-samples-features-time-steps-mean-in-LSTM/answer/Guillaume-Chevalier-2)) to convert them to a probability vector at the output for classification. Note that a "one to one" architecture would be a standard feedforward neural network.

> <a href="https://www.dl-rnn-course.neuraxio.com/start?utm_source=github_lstm" ><img src="https://raw.githubusercontent.com/Neuraxio/Machine-Learning-Figures/master/rnn-architectures.png" /></a>
> [Learn more on RNNs](https://www.dl-rnn-course.neuraxio.com/start?utm_source=github_lstm)

## What is an LSTM?

An LSTM is an improved RNN. It is more complex, but easier to train, avoiding what is called the vanishing gradient problem. I recommend [this course](https://www.dl-rnn-course.neuraxio.com/start?utm_source=github_lstm) for you to learn more on LSTMs.

> [Learn more on LSTMs](https://www.dl-rnn-course.neuraxio.com/start?utm_source=github_lstm)



### (Authored by Nick Van Nest)
## Import Packages


```python
import sys
sys.executable
```




    '/usr/bin/python3'




```python
# All Includes
import sys
!{sys.executable} -m pip install scikit_learn


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
from tensorflow.keras import regularizers

from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import os
```

    Requirement already satisfied: scikit_learn in /usr/local/lib/python3.10/dist-packages (1.2.2)
    Requirement already satisfied: numpy>=1.17.3 in /usr/local/lib/python3.10/dist-packages (from scikit_learn) (1.23.5)
    Requirement already satisfied: scipy>=1.3.2 in /usr/local/lib/python3.10/dist-packages (from scikit_learn) (1.11.2)
    Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from scikit_learn) (1.3.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit_learn) (3.2.0)


### (Modified work taken from Guillaume Chevalier)
## Define Constants to Describe Input Data Features and Labels


```python
# Useful Constants

# Those are separate normalised input features for the neural network
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]
```

### (Modified work taken from Guillaume Chevalier)
## Let's start by downloading the data:


```python
DATA_PATH = "data/"

!pwd && ls
os.chdir(DATA_PATH)
!pwd && ls

!python download_dataset.py

!pwd && ls
os.chdir("..")
!pwd && ls

DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
print("\n" + "Dataset is now located at: " + DATASET_PATH)

```

    /content/drive/My Drive/Colab_Notebooks/github/LSTM-Human-Activity-Recognition
    data		     mae_and_loss_last_80_percent_model1.png
    LICENSE		     mae_and_loss_model1.png
    LSTM_files	     README.md
    LSTM_new.ipynb	     training_validation_metrics_model1.png
    LSTM_original.ipynb
    /content/drive/My Drive/Colab_Notebooks/github/LSTM-Human-Activity-Recognition/data
     download_dataset.py   source.txt	 'UCI HAR Dataset.zip'
     __MACOSX	      'UCI HAR Dataset'
    
    Downloading...
    Dataset already downloaded. Did not download twice.
    
    Extracting...
    Dataset already extracted. Did not extract twice.
    
    /content/drive/My Drive/Colab_Notebooks/github/LSTM-Human-Activity-Recognition/data
     download_dataset.py   source.txt	 'UCI HAR Dataset.zip'
     __MACOSX	      'UCI HAR Dataset'
    /content/drive/My Drive/Colab_Notebooks/github/LSTM-Human-Activity-Recognition
    data		     mae_and_loss_last_80_percent_model1.png
    LICENSE		     mae_and_loss_model1.png
    LSTM_files	     README.md
    LSTM_new.ipynb	     training_validation_metrics_model1.png
    LSTM_original.ipynb
    
    Dataset is now located at: data/UCI HAR Dataset/



### (Modified work taken from Guillaume Chevalier)
## Preparing dataset:



```python
# Load "X" (the neural network's training and testing inputs)
def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1

TRAIN = "train/"
TEST = "test/"

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)


# Load "y" (the neural network's training and testing outputs)

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

```

### (The code below was authored by Nick Van Nest)
## Define Utility Functions



```python
import tensorflow as tf

def one_hot(y_, n_classes):
    # This function was taken with modification from Guillaume Chevalier)
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

def prepare_dataset(features, labels, batch_size, shuffle_buffer_size=1000):
    """
    Prepare a shuffled and batched tf.data.Dataset given features and labels.

    Args:
    - features (ndarray): The feature data; shape should be (num_samples, 128, 9).
    - labels (ndarray): The label data; shape should be (num_samples, 5).
    - batch_size (int): Size of each batch.
    - shuffle_buffer_size (int): Size of shuffle buffer.

    Returns:
    - dataset (tf.data.Dataset): Shuffled and batched dataset.
    """

    # Create a tf.data.Dataset object from the features and labels
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Shuffle the dataset
    dataset = dataset.shuffle(shuffle_buffer_size)

    # Batch the data
    dataset = dataset.batch(batch_size)

    # Using prefetch to improve performance
    dataset = dataset.prefetch(1)

    return dataset

def plot_lr_vs_loss(epochs, lr_loss_logger):
    # Use learning rates logged by custom logger
    lrs = lr_loss_logger.learning_rates

    # Use losses logged by custom logger
    losses = lr_loss_logger.losses

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.semilogx(lrs, losses)
    plt.tick_params('both', length=10, width=1, which='both')
    plt.axis([min(lrs), max(lrs), min(losses), max(losses)])
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs Loss')
    plt.show()

    return #np.array([lrs, losses]).T

def plot_series_dual_y(x, y1, y2, title=None, xlabel=None, ylabel1=None, ylabel2=None, legend1=None, legend2=None, filename=None):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel(ylabel2, color=color)
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    if legend1 or legend2:
        ax1.legend([legend1], loc='upper left')
        ax2.legend([legend2], loc='upper right')

    plt.title(title)
    fig.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_training_validation_metrics(history, title, filename=None):
    # Inline plots
    %matplotlib inline

    # Font settings
    font = {
        'weight': 'bold',
        'size': 18
    }
    plt.rc('font', **font)

    # Figure size
    plt.figure(figsize=(12, 12))

    # Extract training and validation metrics from history object
    train_mae = history.history['mae']
    train_loss = history.history['loss']
    val_mae = history.history['val_mae']
    val_loss = history.history['val_loss']

    # Create epoch axis and plot training metrics
    epochs = range(1, len(train_mae) + 1)
    plt.plot(epochs, train_mae, 'b--', label='Training MAE')
    plt.plot(epochs, train_loss, 'r--', label='Training Loss')

    # Plot validation metrics
    plt.plot(epochs, val_mae, 'b-', label='Validation MAE')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')

    # Labels and title
    plt.title(title)
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Progress (MAE or Loss values)')
    plt.xlabel('Epoch')

    # Save the plot if filename is provided
    if filename:
        plt.savefig(filename)

    # Show the plot
    plt.show()

def print_classification_metrics(X, y_true_oh, model, average_method="weighted"):
    # Make predictions on the original ordered data
    predictions_prob = model.predict(X)
    predictions = np.argmax(predictions_prob, axis=1)

    # Converting one-hot encoded y_test to label encoded
    true_labels = np.argmax(y_true_oh, axis=1)

    # Calculating metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average=average_method)
    recall = recall_score(true_labels, predictions, average=average_method)
    f1 = f1_score(true_labels, predictions, average=average_method)

    print(f"Testing Accuracy: {accuracy * 100}%")
    print(f"Precision: {precision * 100}%")
    print(f"Recall: {recall * 100}%")
    print(f"F1 Score: {f1 * 100}%")

def plot_normalized_confusion_matrix(X, y_true_oh, model, labels, cmap=plt.cm.rainbow):
    # Make predictions on the original ordered data
    predictions_prob = model.predict(X)
    predictions = np.argmax(predictions_prob, axis=1)

    # Converting one-hot encoded y_test to label encoded
    true_labels = np.argmax(y_true_oh, axis=1)

    # Generate confusion matrix
    confusion_mtx = confusion_matrix(true_labels, predictions)
    normalised_confusion_matrix = np.array(confusion_mtx, dtype=np.float32)/np.sum(confusion_mtx)*100

    # Create a plot
    plt.figure(figsize=(8, 8))
    plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix (Normalized)")
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
```

## Define Callback Functions to Modify Training


```python
def lr_scheduler(epoch, lr, initial_lr=1e-6, final_lr=1e-2, epochs=10):
    factor = (final_lr / initial_lr) ** (1 / (epochs - 1))
    return initial_lr * (factor ** epoch)

class LearningRateLossLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.learning_rates = []
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.learning_rates.append(lr)
        self.losses.append(logs['loss'])

class StopAtThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(StopAtThresholdCallback, self).__init__()
        self.threshold = threshold  # Threshold for stopping the training

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy is not None and val_accuracy > self.threshold:
            print(f"\nReached {self.threshold * 100}% validation accuracy. Stopping training.")
            self.model.stop_training = True
```


```python
TRAINING_BATCH_SIZE = 512
TESTING_BATCH_SIZE = 32
NUMBER_OF_CLASSES = 6
WINDOW_SIZE = len(X_train[0])
TIME_STEP_PARAMETER_SIZE = len(X_train[0][0])

# Prepare the dataset
y_train_oh = one_hot(y_train, NUMBER_OF_CLASSES)
y_test_oh = one_hot(y_test, NUMBER_OF_CLASSES)
prepared_training = prepare_dataset(X_train, y_train_oh, TRAINING_BATCH_SIZE)
prepared_testing = prepare_dataset(X_test, y_test_oh, TESTING_BATCH_SIZE)

```


```python
# Build the Model
def create_model_v1():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                        strides=1,
                        activation="relu",
                        padding='causal',
                        kernel_regularizer=regularizers.l2(0.001),
                        input_shape=[WINDOW_SIZE, TIME_STEP_PARAMETER_SIZE]),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation="softmax"),
  ])
  return model

def create_model_v2():
  model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=[WINDOW_SIZE, TIME_STEP_PARAMETER_SIZE], activation='tanh'),
    tf.keras.layers.LSTM(64, activation='tanh'),
    tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation="softmax"),
  ])
  return model

# Print the model summary
model = create_model_v1()
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv1d_1 (Conv1D)           (None, 128, 64)           1792      
                                                                     
     lstm_2 (LSTM)               (None, 128, 128)          98816     
                                                                     
     lstm_3 (LSTM)               (None, 128)               131584    
                                                                     
     dense_3 (Dense)             (None, 64)                8256      
                                                                     
     dropout_2 (Dropout)         (None, 64)                0         
                                                                     
     dense_4 (Dense)             (None, 64)                4160      
                                                                     
     dropout_3 (Dropout)         (None, 64)                0         
                                                                     
     dense_5 (Dense)             (None, 6)                 390       
                                                                     
    =================================================================
    Total params: 244998 (957.02 KB)
    Trainable params: 244998 (957.02 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________


## Determine the optimal learning rate for the model


```python
EPOCHS = 30

# Initialize the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-8)  # Start with the initial learning rate

# Initialize the Model
model = create_model_v1()

# Compile the model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=optimizer,
              metrics=['accuracy']
              )

# Set the callbacks
lr_loss_logger = LearningRateLossLogger()
scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr_scheduler(epoch, lr, initial_lr=1e-5, final_lr=1e-1, epochs=EPOCHS))

# Train the model
history = model.fit(
    prepared_training,
    epochs=EPOCHS,
    callbacks=[scheduler_callback, lr_loss_logger],
    validation_data = prepared_testing
)

# Your existing plotting function should work fine now
plot_lr_vs_loss(epochs=EPOCHS, lr_loss_logger=lr_loss_logger)

```

    Epoch 1/30
    15/15 [==============================] - 13s 237ms/step - loss: 1.9558 - accuracy: 0.1508 - val_loss: 1.9522 - val_accuracy: 0.1218 - lr: 1.0000e-05
    Epoch 2/30
    15/15 [==============================] - 2s 142ms/step - loss: 1.9527 - accuracy: 0.1647 - val_loss: 1.9468 - val_accuracy: 0.2209 - lr: 1.3738e-05
    Epoch 3/30
    15/15 [==============================] - 2s 151ms/step - loss: 1.9450 - accuracy: 0.1817 - val_loss: 1.9388 - val_accuracy: 0.2090 - lr: 1.8874e-05
    Epoch 4/30
    15/15 [==============================] - 2s 143ms/step - loss: 1.9362 - accuracy: 0.2140 - val_loss: 1.9270 - val_accuracy: 0.2966 - lr: 2.5929e-05
    Epoch 5/30
    15/15 [==============================] - 2s 143ms/step - loss: 1.9232 - accuracy: 0.2387 - val_loss: 1.9088 - val_accuracy: 0.3560 - lr: 3.5622e-05
    Epoch 6/30
    15/15 [==============================] - 1s 91ms/step - loss: 1.9024 - accuracy: 0.2810 - val_loss: 1.8754 - val_accuracy: 0.3627 - lr: 4.8939e-05
    Epoch 7/30
    15/15 [==============================] - 1s 77ms/step - loss: 1.8594 - accuracy: 0.3098 - val_loss: 1.7994 - val_accuracy: 0.3787 - lr: 6.7234e-05
    Epoch 8/30
    15/15 [==============================] - 1s 79ms/step - loss: 1.7587 - accuracy: 0.3588 - val_loss: 1.5950 - val_accuracy: 0.4489 - lr: 9.2367e-05
    Epoch 9/30
    15/15 [==============================] - 2s 136ms/step - loss: 1.5631 - accuracy: 0.4246 - val_loss: 1.3652 - val_accuracy: 0.5280 - lr: 1.2690e-04
    Epoch 10/30
    15/15 [==============================] - 1s 93ms/step - loss: 1.3030 - accuracy: 0.5343 - val_loss: 1.0380 - val_accuracy: 0.6430 - lr: 1.7433e-04
    Epoch 11/30
    15/15 [==============================] - 1s 79ms/step - loss: 1.2218 - accuracy: 0.5807 - val_loss: 1.1466 - val_accuracy: 0.5789 - lr: 2.3950e-04
    Epoch 12/30
    15/15 [==============================] - 1s 79ms/step - loss: 1.1355 - accuracy: 0.5817 - val_loss: 0.9736 - val_accuracy: 0.6261 - lr: 3.2903e-04
    Epoch 13/30
    15/15 [==============================] - 1s 79ms/step - loss: 1.0223 - accuracy: 0.6279 - val_loss: 0.9890 - val_accuracy: 0.6420 - lr: 4.5204e-04
    Epoch 14/30
    15/15 [==============================] - 1s 80ms/step - loss: 0.9435 - accuracy: 0.6564 - val_loss: 0.9182 - val_accuracy: 0.6973 - lr: 6.2102e-04
    Epoch 15/30
    15/15 [==============================] - 1s 79ms/step - loss: 0.9408 - accuracy: 0.6598 - val_loss: 0.9524 - val_accuracy: 0.5982 - lr: 8.5317e-04
    Epoch 16/30
    15/15 [==============================] - 1s 88ms/step - loss: 0.8944 - accuracy: 0.6477 - val_loss: 0.9675 - val_accuracy: 0.6149 - lr: 0.0012
    Epoch 17/30
    15/15 [==============================] - 1s 79ms/step - loss: 0.8107 - accuracy: 0.6921 - val_loss: 0.8418 - val_accuracy: 0.6814 - lr: 0.0016
    Epoch 18/30
    15/15 [==============================] - 1s 89ms/step - loss: 0.7397 - accuracy: 0.7398 - val_loss: 0.8110 - val_accuracy: 0.7469 - lr: 0.0022
    Epoch 19/30
    15/15 [==============================] - 2s 135ms/step - loss: 0.6582 - accuracy: 0.7973 - val_loss: 0.7096 - val_accuracy: 0.7927 - lr: 0.0030
    Epoch 20/30
    15/15 [==============================] - 1s 81ms/step - loss: 1.0131 - accuracy: 0.7001 - val_loss: 0.9642 - val_accuracy: 0.5816 - lr: 0.0042
    Epoch 21/30
    15/15 [==============================] - 1s 79ms/step - loss: 0.9601 - accuracy: 0.5819 - val_loss: 0.9092 - val_accuracy: 0.5762 - lr: 0.0057
    Epoch 22/30
    15/15 [==============================] - 1s 78ms/step - loss: 0.8566 - accuracy: 0.6235 - val_loss: 0.8192 - val_accuracy: 0.5989 - lr: 0.0079
    Epoch 23/30
    15/15 [==============================] - 1s 78ms/step - loss: 1.1311 - accuracy: 0.5318 - val_loss: 1.0349 - val_accuracy: 0.5935 - lr: 0.0108
    Epoch 24/30
    15/15 [==============================] - 1s 80ms/step - loss: 1.0085 - accuracy: 0.5344 - val_loss: 1.0099 - val_accuracy: 0.4947 - lr: 0.0149
    Epoch 25/30
    15/15 [==============================] - 1s 80ms/step - loss: 0.9651 - accuracy: 0.6036 - val_loss: 1.1160 - val_accuracy: 0.5416 - lr: 0.0204
    Epoch 26/30
    15/15 [==============================] - 1s 79ms/step - loss: 0.9159 - accuracy: 0.6147 - val_loss: 0.9929 - val_accuracy: 0.5999 - lr: 0.0281
    Epoch 27/30
    15/15 [==============================] - 1s 78ms/step - loss: 0.8579 - accuracy: 0.6166 - val_loss: 0.9428 - val_accuracy: 0.5887 - lr: 0.0386
    Epoch 28/30
    15/15 [==============================] - 1s 88ms/step - loss: 1.0943 - accuracy: 0.5559 - val_loss: 1.4218 - val_accuracy: 0.3756 - lr: 0.0530
    Epoch 29/30
    15/15 [==============================] - 2s 136ms/step - loss: 1.4953 - accuracy: 0.4869 - val_loss: 1.4457 - val_accuracy: 0.5782 - lr: 0.0728
    Epoch 30/30
    15/15 [==============================] - 1s 81ms/step - loss: 1.8072 - accuracy: 0.4524 - val_loss: 1.5657 - val_accuracy: 0.4713 - lr: 0.1000



    
![png](LSTM_new_files/LSTM_new_20_1.png)
    


## Clear the session and use the learning rate to perform training in bulk


```python
tf.keras.backend.clear_session()

EPOCHS = 150

# Initialize the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=5.5e-4)  # Start with the initial learning rate

# Initialize the Model
model = create_model_v1()

# Compile the model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=optimizer,
              metrics=['accuracy','mae']
              )

# Set the callbacks
lr_loss_logger = LearningRateLossLogger()
stop_at_threshold = StopAtThresholdCallback(threshold=0.92)

# Train the model
history = model.fit(
    prepared_training,
    epochs=EPOCHS,
    callbacks=[lr_loss_logger, stop_at_threshold],
    validation_data = prepared_testing
)
```

    Epoch 1/150
    15/15 [==============================] - 7s 138ms/step - loss: 1.8765 - accuracy: 0.3224 - mae: 0.2722 - val_loss: 1.6116 - val_accuracy: 0.5124 - val_mae: 0.2459
    Epoch 2/150
    15/15 [==============================] - 1s 80ms/step - loss: 1.4877 - accuracy: 0.4687 - mae: 0.2212 - val_loss: 1.2460 - val_accuracy: 0.5131 - val_mae: 0.1959
    Epoch 3/150
    15/15 [==============================] - 2s 135ms/step - loss: 1.3253 - accuracy: 0.4973 - mae: 0.1974 - val_loss: 1.2015 - val_accuracy: 0.5528 - val_mae: 0.1886
    Epoch 4/150
    15/15 [==============================] - 1s 81ms/step - loss: 1.1307 - accuracy: 0.5724 - mae: 0.1796 - val_loss: 1.0059 - val_accuracy: 0.5843 - val_mae: 0.1638
    Epoch 5/150
    15/15 [==============================] - 1s 88ms/step - loss: 1.0385 - accuracy: 0.6062 - mae: 0.1581 - val_loss: 0.9957 - val_accuracy: 0.5626 - val_mae: 0.1589
    Epoch 6/150
    15/15 [==============================] - 1s 79ms/step - loss: 0.9858 - accuracy: 0.5841 - mae: 0.1586 - val_loss: 0.8722 - val_accuracy: 0.6566 - val_mae: 0.1474
    Epoch 7/150
    15/15 [==============================] - 1s 79ms/step - loss: 0.8858 - accuracy: 0.6432 - mae: 0.1421 - val_loss: 0.8918 - val_accuracy: 0.6529 - val_mae: 0.1395
    Epoch 8/150
    15/15 [==============================] - 1s 88ms/step - loss: 0.8860 - accuracy: 0.6691 - mae: 0.1360 - val_loss: 0.8526 - val_accuracy: 0.6590 - val_mae: 0.1321
    Epoch 9/150
    15/15 [==============================] - 1s 88ms/step - loss: 0.8318 - accuracy: 0.6872 - mae: 0.1312 - val_loss: 0.8283 - val_accuracy: 0.6810 - val_mae: 0.1289
    Epoch 10/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.7572 - accuracy: 0.7262 - mae: 0.1217 - val_loss: 0.7935 - val_accuracy: 0.6997 - val_mae: 0.1204
    Epoch 11/150
    15/15 [==============================] - 1s 92ms/step - loss: 0.7446 - accuracy: 0.7333 - mae: 0.1171 - val_loss: 0.8539 - val_accuracy: 0.6939 - val_mae: 0.1246
    Epoch 12/150
    15/15 [==============================] - 1s 91ms/step - loss: 0.6999 - accuracy: 0.7501 - mae: 0.1130 - val_loss: 0.8284 - val_accuracy: 0.6851 - val_mae: 0.1197
    Epoch 13/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.6505 - accuracy: 0.7786 - mae: 0.1036 - val_loss: 0.8052 - val_accuracy: 0.7075 - val_mae: 0.1131
    Epoch 14/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.6038 - accuracy: 0.7950 - mae: 0.0973 - val_loss: 0.8285 - val_accuracy: 0.7513 - val_mae: 0.1050
    Epoch 15/150
    15/15 [==============================] - 1s 79ms/step - loss: 0.6081 - accuracy: 0.7878 - mae: 0.0938 - val_loss: 0.8235 - val_accuracy: 0.7570 - val_mae: 0.1041
    Epoch 16/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.5845 - accuracy: 0.7886 - mae: 0.0926 - val_loss: 0.7390 - val_accuracy: 0.7482 - val_mae: 0.1045
    Epoch 17/150
    15/15 [==============================] - 1s 88ms/step - loss: 0.5749 - accuracy: 0.7916 - mae: 0.0945 - val_loss: 0.7790 - val_accuracy: 0.7771 - val_mae: 0.0936
    Epoch 18/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.5831 - accuracy: 0.7935 - mae: 0.0913 - val_loss: 0.7143 - val_accuracy: 0.7587 - val_mae: 0.1005
    Epoch 19/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.5337 - accuracy: 0.8274 - mae: 0.0843 - val_loss: 0.7322 - val_accuracy: 0.7757 - val_mae: 0.0918
    Epoch 20/150
    15/15 [==============================] - 1s 88ms/step - loss: 0.4900 - accuracy: 0.8459 - mae: 0.0774 - val_loss: 0.8259 - val_accuracy: 0.7835 - val_mae: 0.0897
    Epoch 21/150
    15/15 [==============================] - 2s 136ms/step - loss: 0.4669 - accuracy: 0.8611 - mae: 0.0720 - val_loss: 0.8083 - val_accuracy: 0.7849 - val_mae: 0.0856
    Epoch 22/150
    15/15 [==============================] - 1s 89ms/step - loss: 0.4648 - accuracy: 0.8630 - mae: 0.0704 - val_loss: 0.8592 - val_accuracy: 0.7750 - val_mae: 0.0892
    Epoch 23/150
    15/15 [==============================] - 1s 88ms/step - loss: 0.5571 - accuracy: 0.8271 - mae: 0.0802 - val_loss: 0.8469 - val_accuracy: 0.7784 - val_mae: 0.0945
    Epoch 24/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.5297 - accuracy: 0.8399 - mae: 0.0799 - val_loss: 0.6898 - val_accuracy: 0.7889 - val_mae: 0.0971
    Epoch 25/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.5728 - accuracy: 0.8447 - mae: 0.0836 - val_loss: 0.7633 - val_accuracy: 0.7920 - val_mae: 0.0967
    Epoch 26/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.4458 - accuracy: 0.8819 - mae: 0.0709 - val_loss: 0.7770 - val_accuracy: 0.7933 - val_mae: 0.0856
    Epoch 27/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.4604 - accuracy: 0.8670 - mae: 0.0693 - val_loss: 0.9816 - val_accuracy: 0.7754 - val_mae: 0.0894
    Epoch 28/150
    15/15 [==============================] - 1s 79ms/step - loss: 0.6198 - accuracy: 0.8292 - mae: 0.0799 - val_loss: 0.7156 - val_accuracy: 0.8022 - val_mae: 0.0849
    Epoch 29/150
    15/15 [==============================] - 2s 159ms/step - loss: 0.5220 - accuracy: 0.8479 - mae: 0.0787 - val_loss: 0.7014 - val_accuracy: 0.8083 - val_mae: 0.0937
    Epoch 30/150
    15/15 [==============================] - 2s 151ms/step - loss: 0.4287 - accuracy: 0.8940 - mae: 0.0696 - val_loss: 0.7311 - val_accuracy: 0.8096 - val_mae: 0.0813
    Epoch 31/150
    15/15 [==============================] - 1s 99ms/step - loss: 0.4063 - accuracy: 0.8964 - mae: 0.0608 - val_loss: 0.8839 - val_accuracy: 0.7876 - val_mae: 0.0877
    Epoch 32/150
    15/15 [==============================] - 1s 79ms/step - loss: 0.4043 - accuracy: 0.8958 - mae: 0.0590 - val_loss: 0.7878 - val_accuracy: 0.8144 - val_mae: 0.0758
    Epoch 33/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.4065 - accuracy: 0.8991 - mae: 0.0537 - val_loss: 0.7751 - val_accuracy: 0.8069 - val_mae: 0.0765
    Epoch 34/150
    15/15 [==============================] - 1s 79ms/step - loss: 0.4170 - accuracy: 0.8896 - mae: 0.0581 - val_loss: 0.7802 - val_accuracy: 0.8300 - val_mae: 0.0729
    Epoch 35/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.4041 - accuracy: 0.8898 - mae: 0.0584 - val_loss: 0.6219 - val_accuracy: 0.8646 - val_mae: 0.0634
    Epoch 36/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.3935 - accuracy: 0.9030 - mae: 0.0551 - val_loss: 0.7617 - val_accuracy: 0.8144 - val_mae: 0.0794
    Epoch 37/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.3796 - accuracy: 0.9010 - mae: 0.0552 - val_loss: 0.6744 - val_accuracy: 0.8453 - val_mae: 0.0723
    Epoch 38/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.3253 - accuracy: 0.9282 - mae: 0.0467 - val_loss: 0.7372 - val_accuracy: 0.8392 - val_mae: 0.0637
    Epoch 39/150
    15/15 [==============================] - 2s 136ms/step - loss: 0.3485 - accuracy: 0.9105 - mae: 0.0468 - val_loss: 0.5014 - val_accuracy: 0.8711 - val_mae: 0.0583
    Epoch 40/150
    15/15 [==============================] - 1s 96ms/step - loss: 0.3337 - accuracy: 0.9019 - mae: 0.0483 - val_loss: 0.4757 - val_accuracy: 0.8935 - val_mae: 0.0507
    Epoch 41/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.3460 - accuracy: 0.9151 - mae: 0.0471 - val_loss: 0.6437 - val_accuracy: 0.8551 - val_mae: 0.0598
    Epoch 42/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.3026 - accuracy: 0.9340 - mae: 0.0383 - val_loss: 0.4579 - val_accuracy: 0.9013 - val_mae: 0.0425
    Epoch 43/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.2592 - accuracy: 0.9421 - mae: 0.0347 - val_loss: 0.4553 - val_accuracy: 0.9087 - val_mae: 0.0409
    Epoch 44/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.2548 - accuracy: 0.9429 - mae: 0.0313 - val_loss: 0.4442 - val_accuracy: 0.9036 - val_mae: 0.0415
    Epoch 45/150
    15/15 [==============================] - 1s 79ms/step - loss: 0.2457 - accuracy: 0.9438 - mae: 0.0321 - val_loss: 0.4568 - val_accuracy: 0.8985 - val_mae: 0.0414
    Epoch 46/150
    15/15 [==============================] - 1s 88ms/step - loss: 0.2571 - accuracy: 0.9366 - mae: 0.0321 - val_loss: 0.4346 - val_accuracy: 0.9016 - val_mae: 0.0402
    Epoch 47/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.2467 - accuracy: 0.9380 - mae: 0.0321 - val_loss: 0.4238 - val_accuracy: 0.9080 - val_mae: 0.0415
    Epoch 48/150
    15/15 [==============================] - 1s 94ms/step - loss: 0.2384 - accuracy: 0.9467 - mae: 0.0298 - val_loss: 0.4695 - val_accuracy: 0.8972 - val_mae: 0.0401
    Epoch 49/150
    15/15 [==============================] - 2s 138ms/step - loss: 0.2502 - accuracy: 0.9427 - mae: 0.0302 - val_loss: 0.4472 - val_accuracy: 0.9023 - val_mae: 0.0427
    Epoch 50/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.2379 - accuracy: 0.9442 - mae: 0.0313 - val_loss: 0.4449 - val_accuracy: 0.9026 - val_mae: 0.0398
    Epoch 51/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.2243 - accuracy: 0.9495 - mae: 0.0278 - val_loss: 0.4198 - val_accuracy: 0.9036 - val_mae: 0.0386
    Epoch 52/150
    15/15 [==============================] - 1s 79ms/step - loss: 0.2316 - accuracy: 0.9463 - mae: 0.0310 - val_loss: 0.4412 - val_accuracy: 0.9057 - val_mae: 0.0416
    Epoch 53/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.2232 - accuracy: 0.9497 - mae: 0.0282 - val_loss: 0.4324 - val_accuracy: 0.9067 - val_mae: 0.0369
    Epoch 54/150
    15/15 [==============================] - 1s 82ms/step - loss: 0.2153 - accuracy: 0.9506 - mae: 0.0261 - val_loss: 0.4514 - val_accuracy: 0.9043 - val_mae: 0.0390
    Epoch 55/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.2143 - accuracy: 0.9509 - mae: 0.0270 - val_loss: 0.4450 - val_accuracy: 0.9067 - val_mae: 0.0375
    Epoch 56/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.2156 - accuracy: 0.9497 - mae: 0.0264 - val_loss: 0.4550 - val_accuracy: 0.9030 - val_mae: 0.0389
    Epoch 57/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.2095 - accuracy: 0.9490 - mae: 0.0272 - val_loss: 0.4541 - val_accuracy: 0.9084 - val_mae: 0.0382
    Epoch 58/150
    15/15 [==============================] - 2s 136ms/step - loss: 0.2242 - accuracy: 0.9490 - mae: 0.0261 - val_loss: 0.4783 - val_accuracy: 0.8979 - val_mae: 0.0392
    Epoch 59/150
    15/15 [==============================] - 1s 82ms/step - loss: 0.2251 - accuracy: 0.9446 - mae: 0.0293 - val_loss: 0.4456 - val_accuracy: 0.9084 - val_mae: 0.0394
    Epoch 60/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.2176 - accuracy: 0.9490 - mae: 0.0277 - val_loss: 0.4493 - val_accuracy: 0.9063 - val_mae: 0.0388
    Epoch 61/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.2146 - accuracy: 0.9471 - mae: 0.0286 - val_loss: 0.4274 - val_accuracy: 0.9074 - val_mae: 0.0391
    Epoch 62/150
    15/15 [==============================] - 1s 79ms/step - loss: 0.2043 - accuracy: 0.9525 - mae: 0.0264 - val_loss: 0.4783 - val_accuracy: 0.9063 - val_mae: 0.0398
    Epoch 63/150
    15/15 [==============================] - 1s 79ms/step - loss: 0.2471 - accuracy: 0.9324 - mae: 0.0339 - val_loss: 0.5030 - val_accuracy: 0.8965 - val_mae: 0.0413
    Epoch 64/150
    15/15 [==============================] - 1s 79ms/step - loss: 0.2552 - accuracy: 0.9464 - mae: 0.0278 - val_loss: 0.4762 - val_accuracy: 0.8928 - val_mae: 0.0400
    Epoch 65/150
    15/15 [==============================] - 1s 79ms/step - loss: 0.2333 - accuracy: 0.9406 - mae: 0.0308 - val_loss: 0.4211 - val_accuracy: 0.8972 - val_mae: 0.0409
    Epoch 66/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.2110 - accuracy: 0.9478 - mae: 0.0301 - val_loss: 0.3416 - val_accuracy: 0.9084 - val_mae: 0.0382
    Epoch 67/150
    15/15 [==============================] - 1s 88ms/step - loss: 0.2026 - accuracy: 0.9516 - mae: 0.0277 - val_loss: 0.3924 - val_accuracy: 0.9104 - val_mae: 0.0358
    Epoch 68/150
    15/15 [==============================] - 1s 94ms/step - loss: 0.1950 - accuracy: 0.9544 - mae: 0.0255 - val_loss: 0.4224 - val_accuracy: 0.9111 - val_mae: 0.0360
    Epoch 69/150
    15/15 [==============================] - 1s 83ms/step - loss: 0.1949 - accuracy: 0.9524 - mae: 0.0256 - val_loss: 0.4251 - val_accuracy: 0.9131 - val_mae: 0.0351
    Epoch 70/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.1944 - accuracy: 0.9527 - mae: 0.0253 - val_loss: 0.4577 - val_accuracy: 0.9118 - val_mae: 0.0358
    Epoch 71/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.1929 - accuracy: 0.9540 - mae: 0.0253 - val_loss: 0.4227 - val_accuracy: 0.9175 - val_mae: 0.0337
    Epoch 72/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.1916 - accuracy: 0.9525 - mae: 0.0250 - val_loss: 0.4578 - val_accuracy: 0.9135 - val_mae: 0.0354
    Epoch 73/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.1939 - accuracy: 0.9516 - mae: 0.0253 - val_loss: 0.4773 - val_accuracy: 0.9060 - val_mae: 0.0377
    Epoch 74/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.1921 - accuracy: 0.9514 - mae: 0.0265 - val_loss: 0.4037 - val_accuracy: 0.9179 - val_mae: 0.0349
    Epoch 75/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.1858 - accuracy: 0.9557 - mae: 0.0242 - val_loss: 0.3783 - val_accuracy: 0.9091 - val_mae: 0.0347
    Epoch 76/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.1931 - accuracy: 0.9523 - mae: 0.0256 - val_loss: 0.4023 - val_accuracy: 0.9196 - val_mae: 0.0347
    Epoch 77/150
    15/15 [==============================] - 1s 95ms/step - loss: 0.2376 - accuracy: 0.9514 - mae: 0.0267 - val_loss: 0.3614 - val_accuracy: 0.9114 - val_mae: 0.0347
    Epoch 78/150
    15/15 [==============================] - 2s 139ms/step - loss: 0.1876 - accuracy: 0.9538 - mae: 0.0262 - val_loss: 0.4354 - val_accuracy: 0.9077 - val_mae: 0.0373
    Epoch 79/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.1887 - accuracy: 0.9536 - mae: 0.0247 - val_loss: 0.3902 - val_accuracy: 0.9145 - val_mae: 0.0354
    Epoch 80/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.2021 - accuracy: 0.9490 - mae: 0.0290 - val_loss: 0.4266 - val_accuracy: 0.9057 - val_mae: 0.0398
    Epoch 81/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.2257 - accuracy: 0.9436 - mae: 0.0287 - val_loss: 0.5277 - val_accuracy: 0.8945 - val_mae: 0.0411
    Epoch 82/150
    15/15 [==============================] - 1s 82ms/step - loss: 0.2118 - accuracy: 0.9459 - mae: 0.0287 - val_loss: 0.5147 - val_accuracy: 0.9070 - val_mae: 0.0398
    Epoch 83/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.1981 - accuracy: 0.9486 - mae: 0.0284 - val_loss: 0.3371 - val_accuracy: 0.9169 - val_mae: 0.0340
    Epoch 84/150
    15/15 [==============================] - 1s 88ms/step - loss: 0.2201 - accuracy: 0.9429 - mae: 0.0309 - val_loss: 0.4065 - val_accuracy: 0.9145 - val_mae: 0.0413
    Epoch 85/150
    15/15 [==============================] - 1s 81ms/step - loss: 0.1957 - accuracy: 0.9506 - mae: 0.0307 - val_loss: 0.4394 - val_accuracy: 0.9091 - val_mae: 0.0386
    Epoch 86/150
    15/15 [==============================] - 1s 80ms/step - loss: 0.1982 - accuracy: 0.9426 - mae: 0.0275 - val_loss: 0.5543 - val_accuracy: 0.8897 - val_mae: 0.0432
    Epoch 87/150
    14/15 [===========================>..] - ETA: 0s - loss: 0.2228 - accuracy: 0.9393 - mae: 0.0309
    Reached 92.0% validation accuracy. Stopping training.
    15/15 [==============================] - 1s 96ms/step - loss: 0.2238 - accuracy: 0.9389 - mae: 0.0312 - val_loss: 0.3636 - val_accuracy: 0.9209 - val_mae: 0.0389


## Plot Mean Absolute Error (MAE) and Loss During Training


```python
# Get mae and loss from history log
mae = history.history['mae']
loss = history.history['loss']

# Get number of epochs
epochs = range(len(loss))

# Plotting the full graph
plot_series_dual_y(
    x=epochs,
    y1=mae,
    y2=loss,
    xlabel='Epochs',
    ylabel1='MAE',
    ylabel2='Loss',
    legend1='MAE',
    legend2='Loss',
    title='MAE and Loss across Epochs',
    filename='mae_and_loss_model1.png')

"""Plotting the last 80% of epochs. This is done because plotting the first 20%
of epochs reduces the resolution of the graph for the latter 80%"""
zoom_split = int(len(epochs) * 0.2)
epochs_zoom = epochs[zoom_split:]
mae_zoom = mae[zoom_split:]
loss_zoom = loss[zoom_split:]

plot_series_dual_y(
    x=epochs_zoom,
    y1=mae_zoom,
    y2=loss_zoom,
    xlabel='Epochs',
    ylabel1='MAE',
    ylabel2='Loss',
    legend1='MAE',
    legend2='Loss',
    title='MAE and Loss across the last 80% of Epochs',
    filename='mae_and_loss_last_80_percent_model1.png'
)
```


    
![png](LSTM_new_files/LSTM_new_24_0.png)
    



    
![png](LSTM_new_files/LSTM_new_24_1.png)
    


## Monitor MAE and Loss in Training and Validation Data for any discrepancies


```python
#%matplotlib inline
plot_training_validation_metrics(history, "Training and Validation progress over epochs", filename="training_validation_metrics_model1.png")
```


    
![png](LSTM_new_files/LSTM_new_26_0.png)
    



```python
# # Extract the metrics from the history object
# train_mae = np.array(history.history['mae'])
# train_loss = np.array(history.history['loss'])
# val_mae = np.array(history.history['val_mae'])
# val_loss = np.array(history.history['val_loss'])

# # Stack them into a 2D array
# metrics_array = np.vstack([train_mae, train_loss, val_mae, val_loss])

# # If you want to transpose the array so that each row corresponds to an epoch and each column to a metric, you can do:
# metrics_array = metrics_array.T

# # print(metrics_array)

```

## Generate Classification Metrics to Determine Performance and Graph Confusion Matrix


```python
# Test the functions
print_classification_metrics(X_test, y_test_oh, model)
plot_normalized_confusion_matrix(X_test, y_test_oh, model, LABELS)
```

    93/93 [==============================] - 1s 6ms/step
    Testing Accuracy: 92.09365456396336%
    Precision: 92.16791653094502%
    Recall: 92.09365456396336%
    F1 Score: 92.073437392551%
    93/93 [==============================] - 1s 5ms/step



    
![png](LSTM_new_files/LSTM_new_29_1.png)
    


### (Authored by Guillaume Chevalier discussing his model)
## Conclusion

Outstandingly, **the final accuracy is of 91%**! And it can peak to values such as 93.25%, at some moments of luck during the training, depending on how the neural network's weights got initialized at the start of the training, randomly.

This means that the neural networks is almost always able to correctly identify the movement type! Remember, the phone is attached on the waist and each series to classify has just a 128 sample window of two internal sensors (a.k.a. 2.56 seconds at 50 FPS), so it amazes me how those predictions are extremely accurate given this small window of context and raw data. I've validated and re-validated that there is no important bug, and the community used and tried this code a lot. (Note: be sure to report something in the issue tab if you find bugs, otherwise [Quora](https://www.quora.com/), [StackOverflow](https://stackoverflow.com/questions/tagged/tensorflow?sort=votes&pageSize=50), and other [StackExchange](https://stackexchange.com/sites#science) sites are the places for asking questions.)

I specially did not expect such good results for guessing between the labels "SITTING" and "STANDING". Those are seemingly almost the same thing from the point of view of a device placed at waist level according to how the dataset was originally gathered. Thought, it is still possible to see a little cluster on the matrix between those classes, which drifts away just a bit from the identity. This is great.

It is also possible to see that there was a slight difficulty in doing the difference between "WALKING", "WALKING_UPSTAIRS" and "WALKING_DOWNSTAIRS". Obviously, those activities are quite similar in terms of movements.

I also tried my code without the gyroscope, using only the 3D accelerometer's 6 features (and not changing the training hyperparameters), and got an accuracy of 87%. In general, gyroscopes consumes more power than accelerometers, so it is preferable to turn them off.


## Improvements

In [another open-source repository of mine](https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs), the accuracy is pushed up to nearly 94% using a special deep LSTM architecture which combines the concepts of bidirectional RNNs, residual connections, and stacked cells. This architecture is also tested on another similar activity dataset. It resembles the nice architecture used in "[Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf)", without an attention mechanism, and with just the encoder part - as a "many to one" architecture instead of a "many to many" to be adapted to the Human Activity Recognition (HAR) problem. I also worked more on the problem and came up with the [LARNN](https://github.com/guillaume-chevalier/Linear-Attention-Recurrent-Neural-Network), however it's complicated for just a little gain. Thus the current, original activity recognition project is simply better to use for its outstanding simplicity.

If you want to learn more about deep learning, I have also built a list of the learning ressources for deep learning which have revealed to be the most useful to me [here](https://github.com/guillaume-chevalier/Awesome-Deep-Learning-Resources).


## References

The [dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) can be found on the UCI Machine Learning Repository:

> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.

## Citation

Copyright (c) 2016 Guillaume Chevalier. To cite my code, you can point to the URL of the GitHub repository, for example:

> Guillaume Chevalier, LSTMs for Human Activity Recognition, 2016,
> https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition

My code is available for free and even for private usage for anyone under the [MIT License](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/LICENSE), however I ask to cite for using the code.

Here is the BibTeX citation code:
```
@misc{chevalier2016lstms,
  title={LSTMs for human activity recognition},
  author={Chevalier, Guillaume},
  year={2016}
}
```

## Extra links

### Connect with me

- [LinkedIn](https://ca.linkedin.com/in/chevalierg)
- [Twitter](https://twitter.com/guillaume_che)
- [GitHub](https://github.com/guillaume-chevalier/)
- [Quora](https://www.quora.com/profile/Guillaume-Chevalier-2)
- [YouTube](https://www.youtube.com/c/GuillaumeChevalier)
- [Dev/Consulting](http://www.neuraxio.com/en/)

### Liked this project? Did it help you? Leave a [star](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/stargazers), [fork](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/network/members) and share the love!

This activity recognition project has been seen in:

- [Hacker News 1st page](https://news.ycombinator.com/item?id=13049143)
- [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow#tutorials)
- [TensorFlow World](https://github.com/astorfi/TensorFlow-World#some-useful-tutorials)
- And more.

---



```python
# Let's convert this notebook to a README automatically for the GitHub project's title page:
!jupyter nbconvert --to markdown LSTM.ipynb
!mv LSTM.md README.md
```

    [NbConvertApp] WARNING | pattern 'LSTM.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        overwrite base name use for output files.
                    can only be used when converting one notebook at a time.
        Default: ''
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    
    mv: cannot stat 'LSTM.md': No such file or directory

