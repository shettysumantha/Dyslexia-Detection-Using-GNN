import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tqdm import tqdm       
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models


def show_history_graph(history):
    # Plotting accuracy
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
    elif 'acc' in history.history:
        plt.plot(history.history['acc'], label='acc')
        plt.plot(history.history['val_acc'], label='val_acc')
    
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

    # Plotting loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

    
data2=[]
data=[]
featurematrix=[]
label=[]
label2=[]
cw_directory = os.getcwd()
#cw_directory='D:/Hand gesture/final_code'
folder='C:\sumantha\project\Data\eye dataset'
for filename in os.listdir(folder):
    
    sub_dir=(folder+'/' +filename)
    for img_name in os.listdir(sub_dir):
        img_dir=str(sub_dir+ '/' +img_name)
        print(int(filename),img_dir)
        img = cv2.imread(img_dir)
        # Resize image
        img = cv2.resize(img,(128,128))
        if len(img.shape)==3:
            img2 = cv2.resize(img,(32,32))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img2=img2.flatten()
            data2.append(img2/255.0)
            label2.append(int(filename))
            
        data11=np.array(img)
        data.append(data11/255.0)
        label.append(int(filename))
 

#target1=train_targets[label]
##

def train_CNN(data, label):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(36))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2)

    history = model.fit(np.array(X_train), np.array(Y_train), epochs=20, 
                    validation_data=(np.array(X_test), np.array(Y_test)))


    show_history_graph(history)
    
    test_loss, test_acc = model.evaluate(np.array(X_test), np.array(Y_test), verbose=2)
    print("Testing Accuracy:", test_acc)
    print("Testing Loss:", test_loss)

    model.save('eye_movement_trained.h5')
    return model

# CNN Training
model_CNN = train_CNN(data, label)


# CNN Training
model_CNN = train_CNN(data,label)
Y_CNN=model_CNN.predict(np.array(data))