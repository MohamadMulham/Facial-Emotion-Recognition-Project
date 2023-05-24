import keras
import os
import time
import cv2
import tensorflow as tf
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
from keras.layers.core import Dense, Flatten
from keras.layers import Input, Dense, BatchNormalization, Flatten
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint




def create_data():
   for category in labels:
      path = os.path.join(directory, category)
      class_num = labels.index(category)
      for img in os.listdir(path):
         try:
            img_array = cv2.imread(os.path.join(path, img), 1)
            new_array= cv2.resize(img_array, (img_size, img_size))
            data.append([new_array, class_num])
         except Exception as e:
            pass

epo = 35
img_size= 224
batch_size = 128
FROZEN_LAYER_NUM = 19
classNum= 6
shape = (img_size,img_size, 1)
directory = 'Path to Merged Dataset'
sdir= 'Path to file to save the results'
labels = ['angry', 'fear', 'happy','neutral','sad','surprise']
data =[]


create_data()
np.random.shuffle(data)
x=[]#examples
y=[]#labels


for features, label in data:
   x.append(features)
   y.append(label)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)



x_train = np.array(x_train)
x_test = np.array(x_test)
x_val = np.array(x_val)

y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)




checkpoint = ModelCheckpoint(
    filepath= 'Path to save model',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)





emotion_model = tf.keras.applications.VGG19()
base_input = emotion_model.layers[0].input
base_output = emotion_model.layers[-2].output
final_output = Flatten(name='flatten2')(base_output)

final_output = BatchNormalization()(final_output)
final_output = Dense(4096, activation='relu', name='fc6')(final_output)
final_output = Dense(1024, activation='relu', name='fc7')(final_output)


for i in range(FROZEN_LAYER_NUM):
    emotion_model.layers[i].trainable = False

final_output = Dense(classNum, activation='softmax', name='classifier')(final_output)
emotion_model = keras.Model(inputs = base_input, outputs= final_output)
optim = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


emotion_model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

emotion_model.summary()
model_info = emotion_model.fit(x_train, y_train,validation_data=(x_val, y_val) , epochs=epo, callbacks= [checkpoint])


train_loss = model_info.history['loss']
val_loss = model_info.history['val_loss']
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Path to save the figure')
plt.clf()

train_accuracy = model_info.history['accuracy']
val_accuracy = model_info.history['val_accuracy']
epochs = range(1, len(train_accuracy) + 1)
plt.plot(epochs, train_accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Path to save the figure')
plt.clf()


# save model structure in jason file
model_json = emotion_model.to_json()
model_path =''
with open(model_path+".json", "w") as json_file:
    json_file.write(model_json)


#********************Testing Section********************


#Loading the model structure and weights
modelFile = open(model_path+".json", 'r')
loaded_model_json = modelFile.read()
modelFile.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights(model_path+".h5")


#Predicting the labels of the testing section
Y_pred = emotion_model.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)
print('Classification Report')
print(np.mean(y_pred == y_test))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.savefig(sdir+'Confusion Matrix.png')
plt.clf()

