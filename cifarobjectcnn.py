import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm


def create_model(learning_rate):
    model = models.Sequential()

    # Define the first hidden layer with 20 nodes.
    # model.add(tf.keras.layers.Dense(units=10,
    #                                 activation='softmax',
    #                                 name='Hidden1',
    #                                 activity_regularizer=regularizers.l2(0.01)
    #                                 ))

    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(256, kernel_constraint=maxnorm(3)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(128, kernel_constraint=maxnorm(3)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(10))
    model.add(layers.Activation('softmax'))

    # model.add(layers.Dense(10))

    # checkpoint = ModelCheckpoint('best_model_improved.h5',
    #                              monitor='val_loss',
    #                              verbose=0,
    #                              save_best_only=True,
    #                              mode='auto')
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def train_model(model, train_images, train_labels, test_images, test_labels, epochs, batch_size=None):
    # Split the dataset into features and label.

    history = model.fit(train_images, train_labels, batch_size=batch_size,
                        epochs=epochs, shuffle=True, validation_data=(test_images, test_labels))

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # To track the progression of training, gather a snapshot
    # of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)

    return epochs, hist


def plot_curve(epochs, hist):
    """Plot a curve of one or more classification metrics vs. epoch."""
    # list_of_metrics should be one of the names shown in:
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    # for m in list_of_metrics:
    x = hist["accuracy"]
    plt.plot(epochs[1:], x[1:], label="accuracy")

    plt.show()


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

learning_rate = 0.001
epochs = 10
batch_size = 32

my_model = create_model(learning_rate)

epochs, hist = train_model(my_model, train_images, train_labels, test_images, test_labels, epochs, batch_size)
