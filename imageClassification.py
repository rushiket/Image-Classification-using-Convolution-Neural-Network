#Importing all necessery libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout
from keras import backend as K
from keras.models import load_model
import argparse
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras.optimizers import RMSprop


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/predict")
a = ap.parse_args()
mode = a.mode

class myCallback(keras.callbacks.Callback):
	def on_epoch_end(self, epochs,logs={}):
		if(logs.get('acc')>0.90):
			print('\nReached 90% accuracy so cancelling traning!')
			self.model.stop_training = True
	

#Every image in the dataset is of size 224*224
img_width, img_height = 224, 224

train_data_dri = 'v_data/train'
validation_data_dir = 'v_data/test'
nb_train_sample = 400
nb_validation_samples = 100
epochs = 20
batch_size = 16

if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)


train_datagen = ImageDataGenerator(rescale=1./255,
									rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
									shear_range = 0.2,
									zoom_range=0.2,
									horizontal_flip=True)

test_datagen = ImageDataGenerator(1./255)

train_generator = train_datagen.flow_from_directory(
													train_data_dri,
													target_size = (img_width, img_height),
													batch_size=batch_size,
													class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
														validation_data_dir,
														target_size=(img_width,img_height),
														batch_size= batch_size,
														class_mode='binary')


label_map = train_generator.class_indices
print(label_map)



model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

if mode =='train':
	model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.0001),metrics=['acc'])
	callbacks = myCallback()
	history = model.fit_generator(train_generator,
					steps_per_epoch=nb_train_sample / batch_size,
					epochs = epochs,
					validation_data = validation_generator,
					verbose = 1,
					callbacks = [callbacks])



	model.save('model_save1.h5')

elif mode == 'predict':
	img_np=[]
	model = load_model('model_save.h5')
	img_path = 'D:\Personal Folder\project\Machine Learning Projects\Image_Classifier\plane.jpg'
	img = cv2.imread(img_path)
	img = cv2.resize(img,(224,224))
	img = img.reshape((1,224,224,3))
	img_np = np.array(img)
	class1 = model.predict(img)
	# lable_predict = class1.argmax(axis=-1)
	predict_class = model.predict_classes(img_np)


	if predict_class == [[0]]:
		print("Car")
	elif predict_class == [[1]] :
		print("Plane")



	


