#keras implementation of a CNN for image classification.


#IMPORTING LIBRARIES
import os 
import glob
import pandas as pd
import numpy as np 
#IMPORTING THE KERAS LIBRARIES AND PACKAGES
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator
#initaializing the NN
classifier=Sequential()

#step- 1 CONVOLUTION
classifier.add(Convolution2D(32,3,3,input_shape=(128,128,3),activation='relu'))
#step-2 POOLING
classifier.add(MaxPooling2D(pool_size=(2,2)))



classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#step-3 FLATTEN
classifier.add(Flatten())


#step-4 FULLY CONNECTED
classifier.add(Dense(output_dim=64,activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#augmenting the images fir better results
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)



training_set = train_datagen.flow_from_directory(
        r'F:\kaggle\dogcat\sorted\train',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')
#fitting the cnn to the images
classifier.fit_generator(
        training_set,
        steps_per_epoch=1000,
        epochs=6)
#it gave an accuracy of nearly 83% at the end

#saving the model so that we can use it again 
classifier.save('F:\kaggle\dogcat\model.h5')
classifier.save_weights('F:\kaggle\dogcat\model_weights.h5')

imglist=[]
count=0;

#reading in all test images
testpath=r'F:\kaggle\dogcat\test'
for img in glob.glob(os.path.join(testpath,'*')):
       img = image.load_img(img, target_size=(128, 128))
       x = img_to_array(img)
       ax=np.array(x)
       af=ax.astype('float64')
       af/=255 
       imglist.append(af)
       

count
imglist=np.array(imglist)
imglist.shape

#loading our saved MODEL as I to quit it for a while 
model = load_model('F:\kaggle\dogcat\model.h5')

#predicting on the test images
scores = model.predict(imglist)
print(scores[4])
scores.shape

index=np.arange(12500)+1
print(index.shape)
sc=np.reshape(scores, (12500))
sc.shape
ind=np.reshape(index, (12500))
dat=pd.DataFrame()
dat['id']=ind
dat['label']=sc
dat.head()

#writing the predictions on a csv file for submission
dat.to_csv(r'F:\kaggle\dogcat\pred.csv',index=False)
