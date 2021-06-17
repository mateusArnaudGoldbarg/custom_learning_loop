#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tensorflow as tf


# In[3]:


print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


# In[4]:


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#x_train = np.reshape(x_train, (-1, 784))
#x_test = np.reshape(x_test, (-1, 784))
#train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))


# In[5]:


x_train.shape


# In[6]:


x_test.shape


# In[7]:


x_train.shape[0]


# In[8]:


def plot_sample(x,y,index):
    plt.figure(figsize = (20,5))
    plt.imshow(x[index])
    plt.xlabel(y[index])


# In[9]:


for i in range(5):
    plot_sample(x_train,y_train,i)


# In[10]:


'''NORMALIZE'''

x_train = x_train/255
x_test = x_test/255


x_train=x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test=x_test.reshape((x_test.shape[0],28,28,1))


# In[11]:


'''MODEL'''
model = models.Sequential([
    layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[12]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[13]:


model.summary()


# In[22]:


"""JUMP THIS!!!"""
history = model.fit(x_train, y_train, epochs=15,validation_data=(x_test,y_test))


# In[23]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


#CRIANDO O OTIMIZADOR
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
#CRIANDO A FUNÇÃO DE PERDA
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#x_train = np.reshape(x_train, (-1, 784))
#x_test = np.reshape(x_test, (-1, 784))

x_train=x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test=x_test.reshape((x_test.shape[0],28,28,1))
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)


# In[76]:





# In[16]:


'''LEARNING LOOP'''
epochs = 15
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print("Training loss (for one batch) at step %d: %.4f" %(step, float(loss_value)))
            print("Seen so far: %s samples" % ((step + 1) * 64))


# In[15]:


test_accuracy = tf.keras.metrics.Accuracy()
logits = model(x_test, training=False)
prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
test_accuracy(prediction, y_test)
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


# In[16]:


tf.stack([y_test,prediction],axis=1)


# In[17]:


weight = model.get_weights()
np.savetxt('weight.csv' , weight , fmt='%s', delimiter=',')


# In[ ]:





# In[ ]:





# In[ ]:




