import os
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
import numpy as np
import imageio

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

train=[]
for i in os.listdir('/Volumes/ElementsExternal/coca_coda/train/ims/coke/'):
    if i.endswith('png'):
        train.append(['/Volumes/ElementsExternal/coca_coda/train/ims/coke/'+i,1])
for i in os.listdir('/Volumes/ElementsExternal/coca_coda/train/ims/not_coke/'):
    if i.endswith('png'):
        train.append(['/Volumes/ElementsExternal/coca_coda/train/ims/not_coke/'+i,0])
np.random.shuffle(train)

val=[]
for i in os.listdir('/Volumes/ElementsExternal/coca_coda/validation/ims/coke/'):
    if i.endswith('png'):
        val.append(['/Volumes/ElementsExternal/coca_coda/validation/ims/coke/'+i,1])
for i in os.listdir('/Volumes/ElementsExternal/coca_coda/validation/ims/not_coke/'):
    if i.endswith('png'):
        val.append(['/Volumes/ElementsExternal/coca_coda/validation/ims/not_coke/'+i,0])
np.random.shuffle(val)

trn_ims = np.array([np.array(imageio.imread(im[0])) for im in train])
trn_lab = [im[1] for im in train]

val_ims = np.array([np.array(imageio.imread(im[0])) for im in val])
val_lab = [im[1] for im in val]

np.save('trn_arri',trn_ims)
np.save('trn_arrl',trn_lab)
np.save('val_arri',val_ims)
np.save('val_arrl',val_lab)
# with open('trn_arr.txt','w') as t:
#     t.write(str(trn_ims))
#     t.wwrite(str(trn_lab))
#     # trn_ims=t[0].readline()
#     # trn.lab=t[1].readline()
#
# with open('val_arr.txt','w') as v:
#     v.write(str(val_ims))
#     v.write(str(val_lab))
#     # val_ims=v[0].readline()
#     # val.lab=v[1].readline()

ts=list(set([im.shape for im in trn_ims]))
vs=list(set([im.shape for im in val_ims]))

trn_ims1 = np.array([im for im in trn_ims if im.shape==ts[0]])
trn_ims2 = np.array([im for im in trn_ims if im.shape==ts[1]])
trn_ims3 = np.array([im for im in trn_ims if im.shape==ts[2]])
trn_ims4 = np.array([im for im in trn_ims if im.shape==ts[3]])
print(len(trn_ims1),len(trn_ims2),len(trn_ims3),len(trn_ims4))

val_ims1 = np.array([im for im in val_ims if im.shape==vs[0]])
val_ims2 = np.array([im for im in val_ims if im.shape==vs[1]])
val_ims3 = np.array([im for im in val_ims if im.shape==vs[2]])
val_ims4 = np.array([im for im in val_ims if im.shape==vs[3]])
print(len(val_ims1),len(val_ims2),len(val_ims3),len(val_ims4))

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
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

epochs=3
# steps_per_epoch = train_generator.n
# validation_steps = validation_generator.n

# history = model.fit_generator(train_generator,
#                               steps_per_epoch = steps_per_epoch,
#                               epochs=epochs,
#                               workers=4,
#                               validation_data=validation_generator,
#                               validation_steps=validation_steps)
history = model.fit(trn_ims,trn_lab,
                              batch_size=100,
                              epochs=epochs,
                              workers=4,
                              validation_data=(val_ims,val_lab))
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

model.save_weights('mnv2_wts.h5')
model.save('mnv2_model.h5')

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
plt.imshow()
plt.imsave('training.png',plt.figure)
