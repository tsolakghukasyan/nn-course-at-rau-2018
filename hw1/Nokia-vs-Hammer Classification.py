
# coding: utf-8

# In[46]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from skimage import io
import os
import numpy as np


# ### hyperparameters

# In[91]:


batch_size = 5
num_classes = 2
epochs = 5


# ### train, test data

# In[92]:


def read_images(path, label=1):
    data = []
    for img in os.listdir(path):
        data.append(io.imread(path + '/' + img, as_grey=True).reshape(1024))
    return data, [label] * len(data)


# In[93]:


def compile_set(path, label):    
    x_train, y_train = read_images(path + '/train', label)
    x_dev, y_dev = read_images(path + '/dev', label)
    x_test, y_test = read_images(path + '/test', label)
    return {'train': (x_train, y_train),
            'dev': (x_dev, y_dev),
            'test': (x_test, y_test)}


# In[94]:


# the data, split between train and test sets
phone_set = compile_set('phone_dataset', label=0)
hammer_set = compile_set('hammer_dataset', label=1)

x_train = np.array(phone_set['train'][0] + hammer_set['train'][0])
y_train = np.array(phone_set['train'][1] + hammer_set['train'][1])

x_dev = np.array(phone_set['dev'][0] + hammer_set['dev'][0])
y_dev = np.array(phone_set['dev'][1] + hammer_set['dev'][1])

x_test = np.array(phone_set['test'][0] + hammer_set['test'][0])
y_test = np.array(phone_set['test'][1] + hammer_set['test'][1])

print(x_train.shape[0], 'train samples')
print(x_dev.shape[0], 'development samples')
print(x_test.shape[0], 'test samples')


# ### neural network classifier

# In[96]:


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(1024,)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


# ### training

# In[97]:


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_dev, y_dev))


# In[98]:


score = model.evaluate(x_dev, y_dev, verbose=0)
print('Dev loss:', score[0])
print('Dev accuracy:', score[1])


# ### accuracy

# In[99]:


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

