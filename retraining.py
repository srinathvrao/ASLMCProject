import os
import cv2
import numpy as np
import tensorflow as tf

base_dir = "letter_images_cropped/"
imnames = os.listdir(base_dir)

keras = tf.keras
load_model = keras.models.load_model
Model = keras.models.Model

# from keras_efficientnets import EfficientNetB0
# effnet = EfficientNetB0(include_top=False, weights='imagenet')


# print(hsfe.summary())
images = []
outputs = []
for imname in imnames:
	print(imname)
	img = cv2.imread(base_dir+imname)
	# img = cv2.cvtColor(cv2.imread(base_dir+imname), cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (200, 200))
	# img = np.array(img) / 255.0
	# img = img.reshape(200, 200, 1)
	images.append(img)
	outputs.append(int(ord(imname[-5])-65))
images = np.stack(images,axis=0)
outputs = np.array(outputs)
# images = np.concatenate((images,img), axis=0)
print(images.shape,outputs.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, outputs, stratify=outputs, test_size=0.25)
print(X_train.shape, X_test.shape)
ytr=[]
for i in y_train:
	y_tr = [ 0 for k in range(27)]
	y_tr[i] = 1
	ytr.append(y_tr)
ytr = np.array(ytr)
ops=[]
for i in outputs:
	y_tr = [ 0 for k in range(27)]
	y_tr[i] = 1
	ops.append(y_tr)
ops = np.array(ops)

yte=[]
for i in y_test:
	y_te = [ 0 for k in range(27)]
	y_te[i] = 1
	yte.append(y_te)
yte = np.array(yte)
print(y_train.shape, y_test.shape)
# hsfe = load_model('model.h5')
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model = keras.Sequential()

model.add(keras.layers.Conv2D(32,(3,3),activation="relu",padding="same",input_shape=(200,200,3)))
model.add(keras.layers.Conv2D(32,(3,3),activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D(3,3))

model.add(keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D(3,3))

model.add(keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
model.add(keras.layers.MaxPooling2D(3,3))

model.add(keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))
model.add(keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(512,activation="relu"))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(29,activation="softmax"))

# model.load_weights('modelk.h5')
model.load_weights('modelt.h5')

opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt,loss="sparse_categorical_crossentropy",metrics=['accuracy'])

# hsfe = load_model('modelk.h5')
# model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
# hsfe.fit(X_train, y_train, epochs=5, batch_size=32) - 15% test acc, modelz
model.fit(images, outputs, epochs=5, batch_size=32)
model.save("modelt.h5")
# op1 = model.evaluate(X_train, y_train)
# op = model.evaluate(X_test, y_test)
# y_pred = model.predict(X_test).argmax(1)
# from sklearn import metrics
# print(metrics.classification_report(y_test, y_pred, digits=3))

'''
Epoch 10/10
78/78 [==============================] - 7s 83ms/sample - loss: 0.4695 - acc: 0.9487
26/26 [==============================] - 0s 16ms/sample - loss: 3.2787 - acc: 0.1538

   accuracy                          0.154        26
   				precision  recall		f1        support
   macro avg      0.135     0.154     0.141        26
weighted avg      0.135     0.154     0.141        26

37 165
Accuracy: 0.22
{'CRIME': 2, 'FUND': 1, 'LOW': 2, 'SET': 1, 'TRUST': 1, 'LANE': 1, 'HARSH': 1, 'LUNG': 1, 'MIND': 1, 'VIDEO': 2, 'Buy': 1, 'CAR': 1, 'Glare': 1, 'Novel': 1, 'Say': 1, 'Trail': 1, 'Wash': 1, 'agent': 1, 'fate': 2, 'fun': 1, 'gold': 2, 'iron': 2, 'lamb': 2, 'mild': 1, 'sail': 2, 'throw': 3, 'war': 1}
{'CRIME': ['U', 'C', 'U', 'T', 'T'], 'ADOPT': ['N', 'R', 'T', 'O', 'Q'], 'BOAT': ['U', 'T', 'T', 'X'], 'CAP': ['T', 'M', 'T'], 'CHIEF': ['G', 'G', 'W', 'B', 'T'], 'FUND': ['W', 'U', 'G', 'X'], 'LINK': ['R', 'M', 'T', 'U'], 'LOW': ['X', 'N', 'W'], 'SET': ['N', 'T', 'S'], 'TRUST': ['T', 'U', 'U', 'M', 'E'], 'LANE': ['L', 'U', 'T', 'X'], 'ANGLE': ['T', 'X', 'Y', 'X', 'X'], 'ANT': ['X', 'B', 'U'], 'FRAME': ['X', 'U', 'A', 'A', 'X'], 'HARSH': ['S', 'X', 'R', 'W', 'N'], 'LEASE': ['T', 'X', 'X', 'T', 'L'], 'LUNG': ['L', 'U', 'T', 'X'], 'MIND': ['M', 'D', 'A', 'L'], 'NAP': ['E', 'X', 'T'], 'VIDEO': ['V', 'I', 'B', 'X', 'B'], 'Glare': ['B', 'K', 'K', 'M', 'R'], 'Buy': ['U', 'F', 'Y'], 'CAR': ['U', 'B', 'R'], 'Dump': ['R', 'R', 'F', 'J'], 'Monk': ['I', 'F', 'J', 'M'], 'Mouse': ['F', 'F', 'U', 'X', 'X'], 'Novel': ['B', 'F', 'W', 'B', 'I'], 'Say': ['B', 'I', 'Y'], 'Trail': ['R', 'B', 'B', 'I', 'I'], 'Wash': ['W', 'I', 'I', 'D'], 'lamb': ['R', 'T', 'M', 'U'], 'agent': ['T', 'Z', 'T', 'T', 'T'], 'fate': ['W', 'T', 'T', 'T'], 'fun': ['F', 'U', 'T'], 'gold': ['S', 'C', 'L', 'R'], 'iron': ['I', 'R', 'C', 'T'], 'mild': ['T', 'I', 'L', 'R'], 'sail': ['S', 'M', 'I', 'D'], 'throw': ['T', 'H', 'U', 'C', 'W'], 'war': ['W', 'T', 'R']}
'''


# print(op1)
# print(op)
'''
29 165
Accuracy: 0.18
{'LOW': 1, 'TRUST': 1, 'FRAME': 2, 'HARSH': 1, 'LANE': 1, 'LEASE': 1, 'VIDEO': 2, 'Buy': 1, 'CAR': 1, 'Glare': 1, 'Say': 1, 'Trail': 1, 'Wash': 1, 'agent': 1, 'fate': 1, 'fun': 1, 'iron': 2, 'lamb': 1, 'mild': 1, 'sail': 2, 'throw': 3, 'war': 2}

Train on 78 samples
Epoch 10/10
78/78 [==============================] - 6s 77ms/sample - loss: 0.0013 - acc: 0.9995 - true_positives: 77.0000 - false_positives: 0.0000e+00 - true_negatives: 2028.0000 - false_negatives: 1.0000 
26/26 [==============================] - 0s 9ms/sample - loss: 0.0996 - acc: 0.9815 - true_positives: 18.0000 - false_positives: 5.0000 - true_negatives: 671.0000 - false_negatives: 8.0000
[0.09955135732889175, 0.98148155, 18.0, 5.0, 671.0, 8.0]
accuracy: 98.15%
precision: 0.7826
recall: 0.6923
f1 score: 0.7347
'''