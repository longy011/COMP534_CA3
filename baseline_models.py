""" This program composes basic transfer models which act as the performance bassline
they only have a single softmax output layer added and no training to the original weights , as per the specification

in testing they each get around 50-60% accuracy and perform poorly on imbalanced the dataset (i.e. they are hopeless at 
differentiating between covid and pneumonia)

"""

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.applications import vgg16, resnet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# default paths for image generators
data_path = 'Data/'
# 3 classes: Covid, healthy, Pneumonia
num_classes = 3


def build_transfer_model(Model, num_classes, opt=Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy') -> Sequential:
    """builds a model from one of the Model types in tensorflow.python.keras.applications"""
    model = Sequential()
    model.add(Model(include_top=False, weights='imagenet', pooling='avg'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.layers[0].trainable = False
    model.compile(optimizer=opt, loss=loss, metrics=['sparse_categorical_accuracy'])
    return model


def build_generator(data_path, preprocessing_function, train, img_size=224):
    """Builds a generator 
    if for the purpose of training this returns a tuple: training and validation sets,
    if for the purpose of testing this returns a single generator: testing set
    """
    if train:
        generator_instance = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            validation_split=0.2)
        train_generator = generator_instance.flow_from_directory(
            data_path,
            target_size=(img_size, img_size),
            class_mode='sparse',
            subset='training',
            batch_size=8
        )
        val_generator = generator_instance.flow_from_directory(
            data_path,
            target_size=(img_size, img_size),
            class_mode='sparse',
            subset='validation',
            batch_size=8
        )
        return train_generator, val_generator
    else:
        generator_instance = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            validation_split=0.2)
        test_generator = generator_instance.flow_from_directory(
            data_path,
            target_size=(img_size, img_size),
            class_mode='sparse',
            batch_size=16
        )
        
        return test_generator

def train_model(model, preprocess_function, data_path, model_name=''):
    """train a model using an instance of a model, its preprocessor, the path to the data and a name for outputs"""
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')

    X_train,X_val = build_generator(train_path, preprocess_function,train=True)

    X_test = build_generator(test_path, preprocess_function,train=False)

    y_test = X_test.classes
    history = model.fit(X_train, validation_data=X_val, verbose=1)

    y_pred = np.argmax(model.predict(X_test, verbose=1),axis=1)
    print('Classification Report of ', model_name)
    report = classification_report(y_test, y_pred)
    # need to add labels, but had some problems here before
    print(report)
    return history

def main():
    resnet50 =build_transfer_model(resnet.ResNet50,num_classes)
    vgg = build_transfer_model(vgg16.VGG16,num_classes)

    resnet_history = train_model(resnet50, resnet.preprocess_input, data_path, "resnet50")
    vgg_history = train_model(vgg, vgg16.preprocess_input, data_path, "vgg16")

    #add a learning curve

if __name__ == '__main__':
    main()