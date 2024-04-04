import tensorflow as tf 
import os 
import cv2
from matplotlib import pyplot as plt
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')
ust_dizin = os.path.abspath(os.path.join(os.getcwd(), ".."))

data_path = os.path.join(ust_dizin,"data")
breast_benign = os.path.join(ust_dizin, "data","breast_benign")

breast_malignant = os.path.join(ust_dizin, "data" , "breast_malignant")

image_exts = ['jpeg','jpg', 'bmp', 'png']

img = cv2.imread(os.path.join(ust_dizin,"data","breast_benign","breast_benign_0001.jpg"))
print(os.path.join(ust_dizin,"data","breast_benign","breast_benign_0001.jpg"))

# print(img)
# print(img.shape)

plt.imshow(img)
plt.show()


data = tf.keras.utils.image_dataset_from_directory(data_path)

data_iterator = data.as_numpy_iterator()
batch  = data_iterator.next()
#  for visualtion 

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

plt.show()

data = data.map(lambda x,y: (x/255, y))

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.summary()

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


import cv2

img = cv2.imread('breast_benign_0001.jpg')


resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()



prediction = model.predict(np.expand_dims(resize/255, 0))



if prediction > 0.5: 
    print(f'Belirlenen sinif kötü huylu')
else:
    print(f'belirlenen sinif iyi huylu')