import glob
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

path = 'picture/'
w = 224
h = 224
c = 3


class vgg(object):
    def __init__(self,batch_size=16):
        self.batch_size =batch_size

    def __call__(self, session, vars, ckpt='checkpoint/new_vgg-9277',ckpt_load=True):
        if ckpt_load:
            saver = tf.train.Saver(var_list=vars)
            saver.restore(sess=session, save_path=ckpt)


    def read_img(self, path):
        cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
        imgs = []
        labels = []
        for idx, folder in tqdm(enumerate(cate)):
            for im in glob.glob(folder + '/*.png'):
                # print('reading the image: %s' % (im))
                img = Image.open(im)
                img = img.convert('L')
                img = np.expand_dims(np.asarray(img, np.float32), 2) / 255.0
                imgs.append((img,idx))
        return imgs

    def make_batch(self,data):
        idx = np.random.choice(len(data), self.batch_size)

        x_batch = []
        y_batch =  []
        for i in idx:
            x_batch.append(data[i][0])
            y_batch.append(np.eye(100)[data[i][1]])

        return x_batch, y_batch

    def build_network(self,input=None, height=224, width=224, channel=1,reuse=False):
        self.y = tf.placeholder(tf.int64, shape=[self.batch_size, 100], name='labels_placeholder')

        def weight_variable(shape, name="weights"):
            initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
            return tf.get_variable(initializer=initial, name=name)

        def bias_variable(shape, name="biases"):
            initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
            return tf.get_variable(initializer=initial, name=name)

        def conv2d(input, w):
            return tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')

        def pool_max(input):
            return tf.nn.max_pool(input,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  name='pool1')

        def fc(input, w, b):
            return tf.matmul(input, w) + b

        with tf.variable_scope('vgg',reuse=reuse) as scope:
            # conv1
            if reuse:
                scope.reuse_variables()
            with tf.variable_scope('conv1_1',reuse=reuse) as scope:
                kernel = weight_variable([3, 3, 1, 64])
                biases = bias_variable([64])
                self.output_conv1_1 = tf.nn.relu(conv2d(input, kernel) + biases, name=scope.name)

            with tf.variable_scope('conv1_2',reuse=reuse) as scope:
                kernel = weight_variable([3, 3, 64, 64])
                biases = bias_variable([64])
                self.output_conv1_2 = tf.nn.relu(conv2d(self.output_conv1_1, kernel) + biases, name=scope.name)

            pool1 = pool_max(self.output_conv1_2)

            # conv2
            with tf.variable_scope('conv2_1',reuse=reuse) as scope:
                kernel = weight_variable([3, 3, 64, 128])
                biases = bias_variable([128])
                self.output_conv2_1 = tf.nn.relu(conv2d(pool1, kernel) + biases, name=scope.name)

            with tf.variable_scope('conv2_2',reuse=reuse) as scope:
                kernel = weight_variable([3, 3, 128, 128])
                biases = bias_variable([128])
                self.output_conv2_2 = tf.nn.relu(conv2d(self.output_conv2_1, kernel) + biases, name=scope.name)

            pool2 = pool_max(self.output_conv2_2)

            # conv3
            with tf.variable_scope('conv3_1',reuse=reuse) as scope:
                kernel = weight_variable([3, 3, 128, 256])
                biases = bias_variable([256])
                self.output_conv3_1 = tf.nn.relu(conv2d(pool2, kernel) + biases, name=scope.name)

            with tf.variable_scope('conv3_2',reuse=reuse) as scope:
                kernel = weight_variable([3, 3, 256, 256])
                biases = bias_variable([256])
                self.output_conv3_2 = tf.nn.relu(conv2d(self.output_conv3_1, kernel) + biases, name=scope.name)

            with tf.variable_scope('conv3_3',reuse=reuse) as scope:
                kernel = weight_variable([3, 3, 256, 256])
                biases = bias_variable([256])
                self.output_conv3_3 = tf.nn.relu(conv2d(self.output_conv3_2, kernel) + biases, name=scope.name)

            pool3 = pool_max(self.output_conv3_3)

            # conv4
            with tf.variable_scope('conv4_1',reuse=reuse) as scope:
                kernel = weight_variable([3, 3, 256, 512])
                biases = bias_variable([512])
                self.output_conv4_1 = tf.nn.relu(conv2d(pool3, kernel) + biases, name=scope.name)

            with tf.variable_scope('conv4_2',reuse=reuse) as scope:
                kernel = weight_variable([3, 3, 512, 512])
                biases = bias_variable([512])
                self.output_conv4_2 = tf.nn.relu(conv2d(self.output_conv4_1, kernel) + biases, name=scope.name)

            with tf.variable_scope('conv4_3',reuse=reuse) as scope:
                kernel = weight_variable([3, 3, 512, 512])
                biases = bias_variable([512])
                self.output_conv4_3 = tf.nn.relu(conv2d(self.output_conv4_2, kernel) + biases, name=scope.name)

            pool4 = pool_max(self.output_conv4_3)

            # conv5
            with tf.variable_scope('conv5_1',reuse=reuse) as scope:
                kernel = weight_variable([3, 3, 512, 512])
                biases = bias_variable([512])
                self.output_conv5_1 = tf.nn.relu(conv2d(pool4, kernel) + biases, name=scope.name)

            with tf.variable_scope('conv5_2',reuse=reuse) as scope:
                kernel = weight_variable([3, 3, 512, 512])
                biases = bias_variable([512])
                self.output_conv5_2 = tf.nn.relu(conv2d(self.output_conv5_1, kernel) + biases, name=scope.name)

            with tf.variable_scope('conv5_3',reuse=reuse) as scope:
                kernel = weight_variable([3, 3, 512, 512])
                biases = bias_variable([512])
                self.output_conv5_3 = tf.nn.relu(conv2d(self.output_conv5_2, kernel) + biases, name=scope.name)

            pool5 = pool_max(self.output_conv5_3)

            # fc6
            with tf.variable_scope('fc6',reuse=reuse) as scope:
                shape = int(np.prod(pool5.get_shape()[1:]))
                kernel = weight_variable([shape, 4096])
                biases = bias_variable([4096])
                pool5_flat = tf.reshape(pool5, [-1, shape])
                self.output_fc6 = tf.nn.relu(fc(pool5_flat, kernel, biases), name=scope.name)

            # fc7
            with tf.variable_scope('fc7',reuse=reuse) as scope:
                kernel = weight_variable([4096, 4096])
                biases = bias_variable([4096])
                self.output_fc7 = tf.nn.relu(fc(self.output_fc6, kernel, biases), name=scope.name)

            # fc8
            with tf.variable_scope('fc8',reuse=reuse) as scope:
                kernel = weight_variable([4096, 100])
                biases = bias_variable([100])
                self.output_fc8 = fc(self.output_fc7, kernel, biases)

            self.finaloutput = tf.nn.softmax(self.output_fc8, name="softmax")

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_fc8, labels=self.y))
        return self.output_conv2_2, self.output_conv3_3, self.output_conv4_2

    def optim(self):
        self.optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)

        prediction_labels = tf.argmax(self.finaloutput, axis=1, name="output")
        read_labels = self.y

        correct_prediction = tf.equal(prediction_labels, tf.argmax(read_labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))


    def train_network(self, num_epochs=2000):
        self.x = tf.placeholder(tf.float32, shape=[None,224,224,1])
        self.build_network(self.x)
        self.optim()
        data= self.read_img(path)
        num_example = len(data)
        np.random.shuffle(data)
        ratio = 0.8
        s = np.int(num_example * ratio)
        self.train_data = data[:s]
        self.val_data = data[s:]
        print('data size :', num_example)
        init = tf.global_variables_initializer()
        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver()
            # saver.restore(sess,'checkpoint/new_vgg-9277')

            for epoch_index in range(num_epochs):
                step_num = int(len(self.train_data) / self.batch_size)
                for step in range(step_num):
                    x_batch, y_batch = self.make_batch(self.train_data)

                    _,loss = sess.run([self.optimize, self.cost],
                                                 feed_dict={self.x: x_batch, self.y: y_batch})
                    x_val , y_val = self.make_batch(self.val_data)
                    accuracy = sess.run(self.accuracy,feed_dict={self.x:x_val,self.y:y_val})
                    if step % 10 == 0:
                        print (' epoch {} step {} accuracy {} loss {} ').format(epoch_index, step, accuracy, loss)
                saver.save(sess=sess, save_path='checkpoint/new_vgg', global_step=step)


if __name__ == '__main__':
    vgg = vgg(16)
    # vgg.build_network()
    # vgg(tf.Session(),tf.all_variables())
    vgg.train_network()