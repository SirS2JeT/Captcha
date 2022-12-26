import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import cv2
import string
import math
import tensorflow as tf
import random
import keras
from keras.models import Model
from keras import layers
import pandas as pd
from sklearn import metrics

label = []
for dirname, _, filenames in os.walk('Data/'):
    for filename in filenames:
        tmp=os.path.join(dirname, filename)
        label.append(tmp[tmp.rfind("\\")+1:len(tmp)])

dir_name = 'samples/'
num_samples = len(label)
print("Total images: ",num_samples)

symbols = string.ascii_lowercase + "0123456789" # All symbols captcha can contain
num_symbols = len(symbols)
img_shape = (50, 200, 1)
len_captcha = 5
BATCH_SIZE = 32
EPOCHS = 100

print(num_symbols)

X = np.zeros((num_samples, 50, 200, 1)) #1070*50*200
Y = np.zeros((num_samples, len_captcha, num_symbols)) #1070*5*36

for i in range(num_samples):
    img = cv2.imread(os.path.join("Data\\" + dir_name, label[i]), cv2.IMREAD_GRAYSCALE)
    #plt.imshow(img, cmap=plt.get_cmap('gray'))
    img = img / 255
    img = np.reshape(img, img_shape)
    tmp = np.zeros((len_captcha, num_symbols))
    for j in range(len_captcha):
        t = symbols.find(label[i][j])
        tmp[j,t] = 1
    X[i] = img
    Y[i] = tmp

num_train = math.ceil(0.8*num_samples)
num_test = num_samples - num_train

X_train = X[:num_train]
Y_train = Y[:num_train]

X_test = X[num_train:]
Y_test = Y[num_train:]

print("Total training captcha: ",num_train)
print("Total test captcha: ",num_test)


def create_model():
    inp = layers.Input(shape=img_shape)  # 50x200
    conv1 = layers.Conv2D(16, (3, 3), activation='relu')(inp)
    maxp1 = layers.MaxPooling2D(2, 2)(conv1)
    conv2 = layers.Conv2D(32, (3, 3), activation='relu')(maxp1)
    maxp2 = layers.MaxPooling2D(2, 2)(conv2)
    conv3 = layers.Conv2D(64, (3, 3), activation='relu')(maxp2)
    maxp3 = layers.MaxPooling2D(2, 2)(conv3)
    flat = layers.Flatten()(maxp3)
    outputs = []
    for _ in range(5):
        dense = layers.Dense(100, activation='relu')(flat)
        drop = layers.Dropout(0.5)(dense)
        res = layers.Dense(num_symbols, activation='softmax')(drop)

        outputs.append(res)

    model = Model(inp, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

model = create_model()

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

# Model compile
early_stopping = keras.callbacks.EarlyStopping(
    patience=100,    # time for epoch
    min_delta=0.001,    # weight delta
    restore_best_weights=True,  # weight transferring
)

history = model.fit(
    X_train, [Y_train[:,0], Y_train[:,1], Y_train[:,2], Y_train[:,3], Y_train[:,4]],
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)

# history_df = pd.DataFrame(history.history)
#
#
# # Start the plot at epoch 5
# graphs = plt.figure()
# g1 = graphs.add_subplot(1, 2, 1)
# g1.plot(history_df.loc[5:, ['loss', 'val_loss']])
# g2 = graphs.add_subplot(1, 2, 2)
# g2.plot(history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']])


def predict(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error")
    else:
        # plt.imshow(img, cmap=plt.get_cmap('gray'))
        img = img / 255
    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))
    ans = np.reshape(res, (len_captcha, num_symbols))
    capt = ''
    probs = 1
    for a in ans:
        capt += symbols[np.argmax(a)]
        probs *= np.max(a)

    return capt, probs

count=0

for i in range(num_train,num_samples):
    tmp=label[i]
    Captcha, Probs = predict(os.path.join("Data\\" + dir_name, tmp))
    if (Captcha==tmp[:len_captcha]):
        count = count + 1
test_acc = count / num_test * 100
print("Test accuracy: ", np.round(test_acc, 2))

def check_captcha(number=25):
    plt.figure(figsize=(15, 15))
    count = 0
    for i in range(number):
        tmp=label[random.randint(num_train, num_samples-1)]
        capt, probs = predict(os.path.join("Data\\" + dir_name, tmp))
        img = cv2.imread(os.path.join("Data\\" + dir_name, tmp), cv2.IMREAD_GRAYSCALE)
        plt.subplot(5, 5, i+1)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.xlabel(capt)
        plt.xticks([])
        plt.yticks([])
        if capt==tmp[:5]:
            count = count + 1
    print("Number of true predicted captchas: ", count, "/25")

check_captcha()

plt.show()
