# _*_coding:utf-8_*_
import numpy as np
import time
import tensorflow as tf

w1 = tf.Variable(1.)
w2 = tf.Variable(2.)
b = tf.Variable(0.)
tw = tf.Variable(0.0)

q = w1 * 2. + w2 * 3. + b

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print(sess.run(q))
sess.run(tf.assign(tw, w1.eval()))
sess.run(tf.assign(w1, 23.0))
print(sess.run(q))
sess.run(tf.assign(w1, tw))
print(sess.run(q))