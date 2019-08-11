# convolutional Neural Network
# Relu Layer
# pooling 
# Flattening
# Full Connection
# softmax & Cross-Entropy
 # Installing Theano
 # Installing Tensorflow
 # Installing Keras

#part -1 Building the CNN
from keras.models import Sequential
from Keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#initialising the CNN
classifier=Sequential()

#step-1 -convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

#step -2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step3-Flattening
classifier.add(Flatten())

#step-4 Full connection
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

#compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# part -2 Fitting the CNN to the image
#copyt nad paster from keras.io image preprocessing

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=50,
        validation_data=test_set,
        validation_steps=800)

#part 3 - Making new Predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',taget_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)#cannot expect single directly we must put it in batch
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
	prediction='dog'
else:
	prediction='cat'


print(prediction)


















