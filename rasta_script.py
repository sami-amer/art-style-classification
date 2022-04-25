import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

# MUST BE IN RASTA FOLDER
os.chdir("rasta/")
tf.keras.backend.clear_session()

print(tf.version.VERSION)
print(tf.test.is_gpu_available())

rasta_model = tf.keras.models.load_model('models/default/model.h5')
rasta_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['categorical_accuracy',tf.keras.metrics.TopKCategoricalAccuracy(k=5)])

batch_size = 32
img_height = 224
img_width = 224

d_size = "small" # or full

train_dir = f"data/wikipaintings_{d_size}/wikipaintings_train"
val_dir = f"data/wikipaintings_{d_size}/wikipaintings_val"
test_dir = f"data/wikipaintings_{d_size}/wikipaintings_test"

# normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode='categorical')

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  val_dir,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode='categorical')

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  label_mode='categorical')

class_names = train_ds.class_names

test_ds = test_ds.map(lambda image,label: (tf.keras.applications.imagenet_utils.preprocess_input(image),label))
train_ds = train_ds.map(lambda image,label: (tf.keras.applications.imagenet_utils.preprocess_input(image),label))
val_ds = val_ds.map(lambda image,label: (tf.keras.applications.imagenet_utils.preprocess_input(image),label))

# Test the model
# rasta_model.evaluate(test_ds,verbose=1)

# Freeze weights everywhere except for input and output
for layer in rasta_model.layers[1:-2]:
    layer.trainable = False

# rasta_model.summary()

x = rasta_model.layers[-2].output 
x = tf.keras.layers.Dense(len(class_names),name="dense_end",activation='softmax')(x)
# # predictions = tf.keras.layers.Dense(25, activation = "softmax")(x)
rasta_trainable = tf.keras.Model(inputs = rasta_model.input, outputs = x)
rasta_trainable.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['categorical_accuracy',tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
rasta_trainable.fit(train_ds, validation_data=val_ds,batch_size=batch_size, epochs=25)

rasta_trainable.evaluate(test_ds)

rasta_trainable.save("rasta_trained")