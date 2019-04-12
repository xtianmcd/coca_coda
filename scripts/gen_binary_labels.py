import os
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential

base_dir = '/Volumes/ElementsExternal/coca_coda'

train_dir = os.path.join(base_dir,'train','ims')
validation_dir = os.path.join(base_dir, 'validation','ims')

train_coke_dir = os.path.join(train_dir,'coke')
print ('Total training coke images:', len(os.listdir(train_coke_dir)))

train_not_dir = os.path.join(train_dir,'not_coke')
print ('Total training not_coke images:', len(os.listdir(train_not_dir)))

validation_coke_dir = os.path.join(validation_dir,'coke')
print ('Total validation coke images:', len(os.listdir(validation_coke_dir)))

validation_not_dir = os.path.join(validation_dir,'not_coke')
print ('Total validation not_coke images:', len(os.listdir(validation_not_dir)))

image_size = 224
batch_size = 32

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                    train_dir,target_size=(image_size, image_size),
                    batch_size=batch_size,class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
                    validation_dir,target_size=(image_size, image_size),
                    batch_size=batch_size,class_mode='binary')

IMG_SHAPE = (image_size, image_size, 3)

base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                                include_top=True,
                                                weights='imagenet')

base_model.trainable=False
base_model.summary()

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(1, activation='sigmoid')
])



model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

len(model.trainable_variables)

epochs=5
steps_per_epoch = train_generator.n
validation_steps = validation_generator.n

history = model.fit_generator(train_generator,
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator,
                              validation_steps=validation_steps)
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

model.save_weights('resnet50_weights.h5')
model.save('resnet50_model.h5')

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.imsave('training.png')
