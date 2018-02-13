import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
single_image = mnist.train.images[1].reshape(28,28)
plt.imshow(single_image, cmap='gist_gray')
plt.show()

# Creating placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])

# Creating variables
W = tf.Variable(tf.zeros([784, 10])) #weights come from 784 neurons to 10 neurons at the output
b = tf.Variable(tf.zeros([10]))

# Creating Graph operations
y = tf.matmul(x,W) + b

# Defining loss function

#placeholder for true value of y
y_true = tf.placeholder(tf.float32, shape=[None, 10])

#Cross entropy loss function with softmax
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits= y))

# Optimizer(Gradient descent)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# Create session and run

#initialize variable
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x:batch_x, y_true:batch_y})

    # Test the Train Model
    matches = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))

    acc = tf.reduce_mean(tf.cast(matches, tf.float32)) #cast the boolean values between 0 or 1

    print(sess.run(acc, feed_dict={x: mnist.test.images, y_true: mnist.test.labels})) #prints accuracy of the model