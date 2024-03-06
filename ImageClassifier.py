import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.layers import BatchNormalization, Dense, MaxPooling2D, Flatten, Activation, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam
from kerastuner import HyperModel, RandomSearch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

# Load dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Flatten the labels
train_labels = train_labels.flatten()
test_labels = test_labels.flatten()

# Plot the first 25 images from the training set
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(train_labels[i])
plt.show()

# Set up data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
datagen.fit(train_images)

# Define HyperModel for tuning
class CNNHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        model.add(Conv2D(filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=32),
                         kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))

        model.add(Conv2D(filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=32),
                         kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
                         activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))

        model.add(Flatten())
        model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=512, step=32),
                        activation='relu'))
        model.add(Dense(100, activation='softmax'))

        model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])

        return model

hypermodel = CNNHyperModel(input_shape=(32, 32, 3))

# Hyperparameter tuning
tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='hyperparam_tuning',
    project_name='cifar100_cnn'
)

tuner.search(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Get the best model and train
best_model = tuner.get_best_models(num_models=1)[0]
custom_history = best_model.fit(
    datagen.flow(train_images, train_labels, batch_size=64),
    epochs=20,
    validation_data=(test_images, test_labels)
)

# Print training and validation accuracy for custom model
print('Custom Model Training Accuracy:', custom_history.history['accuracy'][-1])
print('Custom Model Validation Accuracy:', custom_history.history['val_accuracy'][-1])

# Generate predictions and classification report for custom model
custom_predictions = best_model.predict(test_images)
custom_predictions_classes = np.argmax(custom_predictions, axis=1)
custom_report = classification_report(test_labels, custom_predictions_classes, target_names=['Class'+str(i) for i in range(100)])
print('Custom CNN Classification Report:\n', custom_report)

# ResNet50 model setup and training
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
resnet_model = Sequential()
resnet_model.add(resnet_base)
resnet_model.add(GlobalAveragePooling2D())
resnet_model.add(Dense(100, activation='softmax'))

resnet_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
resnet_history = resnet_model.fit(
    train_images, 
    train_labels, 
    epochs=10, 
    validation_data=(test_images, test_labels)
)

# Print training and validation accuracy for ResNet50 model
print('ResNet50 Model Training Accuracy:', resnet_history.history['accuracy'][-1])
print('ResNet50 Model Validation Accuracy:', resnet_history.history['val_accuracy'][-1])

# Generate predictions and classification report for ResNet50 model
resnet_predictions = resnet_model.predict(test_images)
resnet_predictions_classes = np.argmax(resnet_predictions, axis=1)
resnet_report = classification_report(test_labels, resnet_predictions_classes, target_names=['Class'+str(i) for i in range(100)])
print('ResNet50 Model Classification Report:\n', resnet_report)
