import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import scipy
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input

DATA_LIST = 'two/train'
DATASET_PATH  = 'two/train'
TEST_DIR =  'two/test'
IMAGE_SIZE    = (224, 224)
NUM_CLASSES   = len(DATA_LIST)
BATCH_SIZE    = 10  
NUM_EPOCHS    = 1
LEARNING_RATE = 0.0005 # start off with high rate first 0.001 and experiment with reducing it gradually 

# Generate Training and Validation batches
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=50,featurewise_center = True,
                                   featurewise_std_normalization = True,width_shift_range=0.2,
                                   height_shift_range=0.2,shear_range=0.25,zoom_range=0.1,
                                   zca_whitening = True,channel_shift_range = 20,
                                   horizontal_flip = True,vertical_flip = True,
                                   validation_split = 0.2,fill_mode='constant')

train_batches = train_datagen.flow_from_directory(DATASET_PATH,target_size=IMAGE_SIZE,
                                                  shuffle=True,batch_size=BATCH_SIZE,
                                                  subset = "training",seed=42,
                                                  class_mode="binary")

valid_batches = train_datagen.flow_from_directory(DATASET_PATH,target_size=IMAGE_SIZE,
                                                  shuffle=True,batch_size=BATCH_SIZE,
                                                  subset = "validation",seed=42,
                                                  class_mode="binary")

# Build model
m = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet',input_shape=(224, 224, 3))
m.trainable = False
conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu", name='conv1')
flatten = tf.keras.layers.Flatten(name='flatten')
dp1 = tf.keras.layers.Dropout(0.2, name='dropout1')
dense = tf.keras.layers.Dense(256, activation='relu',name='dense1')
dp2 = tf.keras.layers.Dropout(0.2, name='dropout2')
output = tf.keras.layers.Dense(1, name = 'output',activation='sigmoid')
    
model = tf.keras.Sequential([m, conv1, flatten, dp1, dense, dp2, output])
model.summary()

# Train model

STEP_SIZE_TRAIN=train_batches.n//train_batches.batch_size
STEP_SIZE_VALID=valid_batches.n//valid_batches.batch_size

model.compile(loss="binary_crossentropy",
              optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
              metrics=['accuracy'])

output = model.fit(train_batches,
                 validation_data = valid_batches,
                 validation_steps = STEP_SIZE_VALID,
                 batch_size = BATCH_SIZE,
                 steps_per_epoch =STEP_SIZE_TRAIN,
                 epochs= NUM_EPOCHS)

# Save model
model.save('saved_model.h5')

# Prediction
def getPrediction(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    pred = model.predict(image)
    for index, probability in enumerate(pred):
        if probability > 0.5:
            prob = round(probability[0]*100, 2)
            # print("prob = ", prob)
            return (str(prob) + "% Normal")
        else:
            prob = round((1-probability[0])*100, 2)
            # print("prob = ", prob)
            return (str(prob) + "% COVID19")
