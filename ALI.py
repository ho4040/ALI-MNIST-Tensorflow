
# coding: utf-8

# # ALI MNIST Tensorfle 5th
# 

# In[1]:


import os, time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:


batch_size = 128
z_size = 100
train_epoch = 2000
w_init = tf.contrib.layers.xavier_initializer()
training = True

def generator(z, reuse=False):
    with tf.variable_scope('generator', reuse=reuse), tf.device('/device:GPU:0'):
        s_size = 4
        initializer = tf.random_normal_initializer(mean=0, stddev=0.02)
        z = tf.convert_to_tensor(z)
        with tf.variable_scope('reshape'): # z vector convert into tensor
            outputs = tf.layers.dense(z, 1024 * s_size * s_size)
            outputs = tf.reshape(outputs, [-1, s_size, s_size, 1024])
            outputs = tf.layers.batch_normalization(outputs, training=training)
            outputs = tf.nn.relu(outputs, name='outputs') # shape (batch_size, 4, 4, 1024)

        with tf.variable_scope('tconv1'): # Transposed conv 1
            tconv1 = tf.layers.conv2d_transpose(outputs, filters=512, kernel_size=[5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            tconv1 = tf.layers.batch_normalization(tconv1, training=training)
            tconv1 = tf.nn.relu(tconv1) # shape = (batch_size, 8, 8, 512)

        with tf.variable_scope('tconv2'): # Transposed conv 2
            tconv2 = tf.layers.conv2d_transpose(tconv1, filters=256, kernel_size=[5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            tconv2 = tf.layers.batch_normalization(tconv2, training=training)
            tconv2 = tf.nn.relu(tconv2) # output shape = (batch_size, 16, 16, 256)

        with tf.variable_scope('tconv3'): # Transposed conv 3
            tconv3 = tf.layers.conv2d_transpose(tconv2, filters=128, kernel_size=[5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            tconv3 = tf.layers.batch_normalization(tconv3, training=training)
            tconv3 = tf.nn.relu(tconv3) # output shape = (batch_size, 32, 32, 128)

        with tf.variable_scope('tconv4'): # Transposed conv 4 Filters == RGB
            tconv4 = tf.layers.conv2d_transpose(inputs=tconv3, filters=1, kernel_size=[5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            #tconv4 = tf.layers.batch_normalization(tconv4, training=training)

        # tanh output
        g = tf.nn.tanh(tconv4) # output shape = (batch_size, 64, 64, 3)
        #g = tf.nn.sigmoid(tconv4)
        return g

def generator_reverse(x, reuse=False):
    with tf.variable_scope('generator_reverse', reuse=reuse), tf.device('/device:GPU:0'):
        initializer = tf.random_normal_initializer(mean=0, stddev=0.02)
        d_inputs = tf.convert_to_tensor(x)

        with tf.variable_scope('conv1'): # conv 1
            conv1 = tf.layers.conv2d(d_inputs, 64, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = tf.nn.leaky_relu(conv1) # shape (batch_size, 32, 32, 64)

        with tf.variable_scope('conv2'): # conv 2
            conv2 = tf.layers.conv2d(conv1, 128, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = tf.nn.leaky_relu(conv2) # shape (batch_size, 16, 16, 128)

        with tf.variable_scope('conv3'): # conv 3
            conv3 = tf.layers.conv2d(conv2, 256, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            conv3 = tf.layers.batch_normalization(conv3, training=training)
            conv3 = tf.nn.leaky_relu(conv3) # shape (batch_size, 4, 4, 256)

        # conv 4
        with tf.variable_scope('conv4'):
            conv4 = tf.layers.conv2d(conv3, 512, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            conv4 = tf.layers.batch_normalization(conv4, training=training)
            conv4 = tf.nn.leaky_relu(conv4) # shape (batch_size, 2, 2, 512)

        batch_size = conv4.get_shape()[0].value
        reshaped = tf.reshape(conv4, [batch_size, -1])
        g_r = tf.layers.dense(reshaped, z_size, name='d', activation=None)  
        return g_r
    
    
def discriminator( x, z, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse), tf.device('/device:GPU:0'):
        
        initializer = tf.random_normal_initializer(mean=0, stddev=0.02)
        d_inputs = tf.convert_to_tensor(x)
        
        z = tf.layers.dense(z, 64*64, activation=None)
        z = tf.reshape(z, (-1, 64, 64, 1))
        z = tf.concat([d_inputs, z], 3)

        with tf.variable_scope('conv1'): # conv 1
            conv1 = tf.layers.conv2d(z, 64, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            conv1 = tf.layers.batch_normalization(conv1, training=training)
            conv1 = tf.nn.leaky_relu(conv1) # shape (batch_size, 32, 32, 64)

        with tf.variable_scope('conv2'): # conv 2
            conv2 = tf.layers.conv2d(conv1, 128, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            conv2 = tf.layers.batch_normalization(conv2, training=training)
            conv2 = tf.nn.leaky_relu(conv2) # shape (batch_size, 16, 16, 128)

        with tf.variable_scope('conv3'): # conv 3
            conv3 = tf.layers.conv2d(conv2, 256, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            conv3 = tf.layers.batch_normalization(conv3, training=training)
            conv3 = tf.nn.leaky_relu(conv3) # shape (batch_size, 4, 4, 256)

        # conv 4
        with tf.variable_scope('conv4'):
            conv4 = tf.layers.conv2d(conv3, 512, [5, 5], strides=(2, 2), padding='SAME', kernel_initializer=initializer)
            conv4 = tf.layers.batch_normalization(conv4, training=training)
            conv4 = tf.nn.leaky_relu(conv4) # shape (batch_size, 2, 2, 512)

        batch_size = conv4.get_shape()[0].value
        reshaped = tf.reshape(conv4, [batch_size, -1])
        logits = tf.layers.dense(reshaped, 1, name='d', activation=None) # TODO : remove dense
        d = tf.nn.sigmoid(logits)
        return d, logits


# In[3]:


z = tf.random_normal(shape=(batch_size, z_size), mean=0.0, stddev=1.0, dtype=tf.float32)
#z = tf.random_uniform(shape=(batch_size, z_size), dtype=tf.float32, minval=-1.0, maxval=1.0)
x = tf.placeholder(shape=[batch_size, 784], dtype=tf.float32) # for mnist

reshaped = tf.reshape(x, (batch_size, 28, 28, 1)) # convert raw data to valid image data
resized_x = tf.image.resize_images(reshaped, (64,64,)) # change size as D input  

z_fake = generator_reverse( resized_x, reuse=False )
x_fake = generator(z, reuse=False)
x_reconstructed = generator(z_fake, reuse=True)


y_fake, logits_fake = discriminator(x_fake, z, False)
y_real, logits_real = discriminator(resized_x, z_fake, True)

#loss_d = -(tf.reduce_mean(tf.log(y_real)) + tf.reduce_mean(tf.log(1-y_fake)))
#loss_g = tf.reduce_mean(tf.log(1-y_fake))

# use logits and sigmoid_cross_entropy to remove sigmoid from backpropagation process, 
label_one = tf.ones_like(logits_real)
label_zero = tf.zeros_like(logits_fake)
loss_d = tf.losses.sigmoid_cross_entropy(multi_class_labels=label_zero, logits=logits_fake) + tf.losses.sigmoid_cross_entropy(multi_class_labels=label_one, logits=logits_real)
loss_g = tf.losses.sigmoid_cross_entropy(multi_class_labels=label_one, logits=logits_fake) + tf.losses.sigmoid_cross_entropy(multi_class_labels=label_zero, logits=logits_real)


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):

    d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
    g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
    g_r_vars = [var for var in tf.trainable_variables() if 'generator_reverse' in var.name]

    # optimize D
    d_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)
    d_train = d_opt.minimize(loss_d, var_list=d_vars)

    # optimize G
    g_opt = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)
    g_train = d_opt.minimize(loss_g, var_list=g_vars+g_r_vars)


# In[4]:


fake_images = tf.reshape(x_fake, [-1, 64, 64, 1])
reconstructed_images = tf.reshape(x_reconstructed, [-1, 64, 64, 1]) 
tf.summary.image('fake_images', fake_images, 10)
tf.summary.image('reconstructed_images', reconstructed_images, 10)
tf.summary.image('sample_images', resized_x, 10)
tf.summary.histogram('y_real', y_real)
tf.summary.histogram('y_fake', y_fake)


# In[5]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1~1 from 0~1


# In[6]:


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
writer = tf.summary.FileWriter('./ali_dcgan_1st', sess.graph)
merged = tf.summary.merge_all()


# In[7]:


np.random.seed(int(time.time()))
start_time = time.time()
print('training start at {}'.format(start_time))

for epoch in range(train_epoch):
    
    # update discriminator
    for i in range(len(train_set) // batch_size):
        
        x_data = train_set[i * batch_size:(i + 1) * batch_size]

        _ = sess.run(d_train, {x: x_data})
        _ = sess.run(g_train, {x: x_data})
    
    #print(np.shape(_con_0), np.shape(_con_1))
    if epoch % 1 == 0:
        _summ = sess.run(merged, {x: x_data})
        writer.add_summary(_summ, epoch)
        
        _resized_x, _reconstructed_images = sess.run([resized_x, reconstructed_images], {x: x_data})
        
        plt.figure(figsize=(9,1))    
        for k in range(1, 8, 2):
            imgIdx = k // 2
            graphIdx1 = ''.join(str(x) for x in [1,8,k])
            graphIdx2 = ''.join(str(x) for x in [1,8,k+1])
            #print(imgIdx, graphIdx1, graphIdx2)
            plt.subplot(graphIdx1)
            plt.axis('off')
            plt.imshow(np.reshape(_resized_x[imgIdx], [64,64]), cmap="gray")
            plt.subplot(graphIdx2)
            plt.axis('off')
            plt.imshow(np.reshape(_reconstructed_images[imgIdx], [64,64]), cmap="gray")
            
        plt.title("epoch {}".format(epoch))
        plt.show()

