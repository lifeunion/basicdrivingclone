import pandas
from sklearn.model_selection import train_test_split
import json

import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Lambda, Activation, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D
import os
import csv, cv2
import matplotlib.pyplot as plt

np.random.seed(0)

# parameters to tune at
number_of_epochs = 15
learning_rate = 1e-4
keep_prob = 0.3
the_batch_size = 256
data_dir = "data/"
test_size_ratio = 0.2
randomness_begin = 4564

def load_logfile():
    # this method takes care of fetching all the data and store it for training purposes
    lines = []

    with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images=[]
    measurements=[]

    for line in lines:
        source_path = line[0]
        
        #filename=source_path.split('/')[-1]
        #current_path = 'data/IMG/' + filename
        image = cv2.imread(source_path, cv2.IMREAD_COLOR)
        #height, width, channels = image.shape
        images.append(np.array(image))
        measurement = float(line[3])
        measurements.append(measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)
    
    #X_train, X_valid, y_train, y_valid = train_test_split(images, measurements, test_size=test_size_ratio, random_state=randomness_begin)
    #return X_train, X_valid, y_train, y_valid
    return X_train, y_train

def build_model():
    # The model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
    # Source:  https://arxiv.org/pdf/1604.07316.pdf

    model = Sequential()

    # normalization
    # model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))

     # cropping2D layer
    model.add(Cropping2D(cropping=((60,20), (0,0))))
    model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    
    model.add(Dropout(keep_prob))
    
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    return model


def train_model(model, X_train, y_train, the_batch_size, number_of_epochs):
    # after storing data and building the model, now we can run the data through the model here
    model.compile(loss='mse', optimizer='adam')
    history_objects = model.fit(X_train, y_train, validation_split=0.2,
                    batch_size=the_batch_size, 
                    nb_epoch=number_of_epochs,
                    verbose=1, 
                    shuffle = True)
    model.save('model.h5')
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def main():
    data = load_logfile()
    model = build_model()

    json_string = model.to_json()
    with open('./model.json', 'w') as f:
        f.write(json_string)

    #train_model(model, *data, batch_size, number_of_epochs)
    model.compile(loss='mse', optimizer='adam')
    history_objects = model.fit(*data, validation_split=0.2,
                    batch_size=the_batch_size, 
                    nb_epoch=number_of_epochs,
                    verbose=1, 
                    shuffle = True)
    model.save('model.h5')

if __name__ == '__main__':
    main()