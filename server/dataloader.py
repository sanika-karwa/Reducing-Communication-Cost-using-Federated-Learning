import PIL
from PIL import Image
import numpy as np
import json
import tensorflow as tf

def imageretreival(path):
    inputfile = open(path,'r')
    jsondecode = json.load(inputfile)
    train_images = []
    for i in range(len(jsondecode["x"])):
        name = jsondecode["x"][i]
        #image_path = 'C:\\Users\\vaish\\Downloads\\capstone\\data\\celeba\\data\\raw\\img_align_celeba\\%s' %name
        image_path = 'C:\\Users\\amart\\OneDrive\\Desktop\\img_align_celeba\\%s' %name
        train_images.append(image_path)
    img_array = np.zeros((len(train_images),28,28), dtype = np.uint8)
    for i in range(len(train_images)):
        image = PIL.Image.open(train_images[i])
        image = image.convert('L')
        image = image.resize((28,28), Image.BICUBIC)
        img_array[i,:,:]= np.asarray(image)
    train_labels = jsondecode["y"]
    train_labels = np.asarray(train_labels).astype('float64').reshape((-1,1))
    m1 = int(len(train_images)*0.7)
    #spilit to test,train
    train_l = train_labels[:m1]
    test_labels = train_labels[m1:]
    img_array = img_array.reshape([len(train_images),28,28,1])
    train_images = img_array.astype('float64')
    train_img = train_images[:m1]
    test_images = train_images[m1:]
    return(train_img,train_l,test_images,test_labels)


def create_model():
    model = tf.keras.Sequential()
    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[5,5], padding='same', activation=tf.nn.relu, input_shape=(28,28,1))) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[2,2],strides=2))
    #model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[5,5], padding='same', activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=[2,2],strides = 2))
    #model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units = 1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(units = 2))
    
    #Compiling the model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
             optimizer='adam',
             metrics=['accuracy'])
    return model

def create_vgg_model():
    NUM_CLASSES = 10
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(input_shape=(28, 28, 1), filters=64,
                                     kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=4096, activation="relu"))
    model.add(tf.keras.layers.Dense(units=4096, activation="relu"))
    model.add(tf.keras.layers.Dense(units=NUM_CLASSES, activation="softmax"))
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

def fedavg(global_weights,weights,client_datapoints,global_datapoints):
    weights = [i*client_datapoints for i in weights]
    weights = [i/(int(global_datapoints)+int(client_datapoints)) for i in weights]
    global_weights = [x+y for x,y in zip(global_weights,weights)]
    return global_weights,(global_datapoints+client_datapoints)
    
def regen(struct,mask,pos):
    for i in range(len(pos)):
        if len(pos[i]) == 5:
            struct[pos[i][0]][pos[i][1]][pos[i][2]][pos[i][3]][pos[i][4]] = mask[i]
        elif len(pos[i]) == 3:
            struct[pos[i][0]][pos[i][1]][pos[i][2]] = mask[i]
        elif len(pos[i]) == 2:
            struct[pos[i][0]][pos[i][1]] = mask[i]
    return(struct)

def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'bytes'