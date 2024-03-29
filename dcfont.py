import re
import argparse
import os
import matplotlib.pyplot as plt
import train_vgg
import util
from ops import *
import cv2
EPS = 1e-12
SIZE = 224
from PIL import Image

class DCFont():

    def __init__(self, num_class, learning_rate, batch_size,vgg_path):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, SIZE, SIZE, 1])
        self.num_class = num_class
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, SIZE, SIZE, 1])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, 20])
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.vgg_path = vgg_path

    def network(self):
        with tf.variable_scope('style_transfer_network'):
            conv_1 = RBC(self.x, num_output=64, kernel=[5, 5], stride=[2, 2])
            conv_2 = RBC(conv_1, num_output=128, kernel=[5, 5], stride=[2, 2])
            conv_3 = RBC(conv_2, num_output=256, kernel=[5, 5], stride=[2, 2])

            conv_4 = conv2d(conv_3, num_output=512, kernel=[5, 5], stride=[2, 2])

            concated = tf.concat([self.style_vector, conv_4, self.category], axis=3)

            res_1 = self.Res_block(concated)
            res_2 = self.Res_block(res_1)
            res_3 = self.Res_block(res_2)
            res_4 = self.Res_block(res_3)
            res_5 = self.Res_block(res_4)

            deconv_1 = RBD(res_5, 256, kernel=[5, 5], stride=[2, 2])
            deconv_2 = RBD(tf.concat([deconv_1, conv_3], axis=3), 128, kernel=[5, 5], stride=[2, 2])
            deconv_3 = RBD(tf.concat([deconv_2, conv_2], axis=3), 64, kernel=[5, 5], stride=[2, 2])
            deconv_4 = RBD(tf.concat([deconv_3, conv_1], axis=3), 1, kernel=[5, 5], stride=[2, 2])

            output = tf.nn.tanh(deconv_4)

        return output

    def Res_block(self, input):
        _, _, _, channel = input.get_shape().as_list()
        net = batch_norm(input)
        net = relu(net)
        net = conv2d(net, channel, [3, 3], stride=[1, 1])

        net = batch_norm(net)
        net = relu(net)
        net = conv2d(net, channel, [3, 3], stride=[1, 1])

        return input + net

    def vgg_net(self):

        self.vgg = train_vgg.vgg()
        self.vgg.build_network(self.x)
        self.vgg.optim()
        self.vgg_vars = tf.all_variables()

        conv5_3 = self.vgg.output_conv5_3
        with tf.variable_scope('reconstruct'):
            enc_1 = conv2d(conv5_3, num_output=256, kernel=[5, 5], stride=[1, 1])
            enc_2 = conv2d(enc_1, num_output=128, kernel=[5, 5], stride=[1, 1])
            self.category = conv2d(enc_2, num_output=64, kernel=[5, 5], stride=[1, 1])
            dec_2 = deconv2d(self.category, 128, kernel=[5, 5], stride=[1, 1])
            dec_1 = deconv2d(tf.concat([dec_2, enc_2], axis=3), 256, kernel=[5, 5], stride=[1, 1])
            self.style_vector = deconv2d(tf.concat([dec_1, enc_1], axis=3), 512, kernel=[5, 5], stride=[1, 1])
            self.L_style = tf.reduce_mean(tf.abs(self.style_vector - conv5_3))

    def discriminator(self, discrim_inputs, discrim_targets, reuse=False):
        df_dim = 64
        image = tf.concat([discrim_inputs, discrim_targets], axis=3)

        h0 = leaky_relu(conv2d(image, df_dim, kernel=[5, 5]))

        h1 = leaky_relu(conv2d(h0, df_dim * 2, kernel=[5, 5]))

        h2 = leaky_relu(conv2d(h1, df_dim * 4, kernel=[5, 5]))

        h3 = leaky_relu(conv2d(h2, df_dim * 8, kernel=[5, 5]))

        fc1 = tf.layers.dense(tf.layers.flatten(h3), 1)

        fc2 = tf.layers.dense(tf.layers.flatten(h3), 20)

        return tf.nn.sigmoid(fc1), fc1, fc2

    def mse(self, input_data, output):
        return tf.reduce_mean(tf.squared_difference(input_data, output))

    def build_model(self):

        self.vgg_net()
        self.output = self.network()
        self.L1_loss = tf.reduce_mean(tf.abs(self.target - self.output)) * 10
        layers = self.vgg.build_network(self.x, reuse=True)
        fake_layers = self.vgg.build_network(self.output, reuse=True)
        self.style_constancy = self.mse(layers[0], fake_layers[0]) + self.mse(layers[1], fake_layers[1]) + self.mse(
            layers[2], fake_layers[2])
        with tf.variable_scope('discriminator'):
            real, real_logits, self.real_class = self.discriminator(self.x, self.target)
        with tf.variable_scope('discriminator', reuse=True):
            fake, fake_logits, self.fake_class = self.discriminator(self.x, self.output)
        real_style = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real)))
        fake_style = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.zeros_like(fake)))

        self.real_category = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_class, labels=self.label))
        self.fake_category = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_class, labels=self.label))

        self.d_loss = real_style + fake_style + self.real_category
        self.g_loss = fake_style + self.fake_category + self.style_constancy + self.L1_loss

        reconstruct_vars = [var for var in tf.trainable_variables() if 'reconstruct' in var.name]
        style_vars = [var for var in tf.trainable_variables() if 'style_transfer_network' in var.name]
        disc_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
        self.reconstruct_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.L_style,var_list=reconstruct_vars)
        self.d_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss, var_list=disc_vars)
        self.g_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss, var_list=style_vars)

    def train(self, reference, train_root, epoch, step,save_path):
        self.build_model()
        all_saver = tf.train.Saver()
        file_writer = tf.summary.FileWriter('checkpoint', tf.get_default_graph())
        with tf.Session() as sess:
            self.vgg(sess, self.vgg_vars,ckpt=self.vgg_path)
            init_vars = [var for var in tf.all_variables() if var not in self.vgg_vars]
            sess.run(tf.initialize_variables(init_vars))
            # all_saver.restore(sess=sess,save_path=tf.train.latest_checkpoint('checkpoint'))
            for ep in range(epoch):

                for st in range(step):
                    input_x, input_y, label = util.make_batch(reference, train_root)
                    result, style_loss, g_loss, d_loss, L1_loss, _, _, _ = sess.run(
                        [self.output, self.style_constancy, self.g_loss, self.d_loss, self.L1_loss, self.d_opt,
                         self.reconstruct_opt, self.g_opt],
                        feed_dict={self.x: input_x, self.target: input_y,
                                   self.label: label})

                    print(
                        'epoch : {} step : {} constancy loss : {:.4} G_loss : {:.4} D_loss :{:.4} l1_loss : {:.4}'.format(
                        ep, st, style_loss, g_loss, d_loss, L1_loss))
                all_saver.save(sess=sess, save_path=save_path, global_step=ep)


    def train_part2(self, reference, target, epoch, step,save_path2):
        self.build_model()
        all_saver = tf.train.Saver()
        if not os.path.exists(save_path2):
            os.makedirs(save_path2)
        with tf.Session() as sess:
            self.vgg(sess, self.vgg_vars)
            init_vars = [var for var in tf.all_variables() if var not in self.vgg_vars]
            sess.run(tf.initialize_variables(init_vars))
            all_saver.restore(sess=sess,save_path= 'checkpoint/dcfont-999')

            for ep in range(epoch):
                for st in range(step):
                    input_x, input_y, label = util.make_batch2(reference,target=target)
                    result, style_loss, g_loss, d_loss, L1_loss, _, _, _ = sess.run(
                        [self.output, self.style_constancy, self.g_loss, self.d_loss, self.L1_loss, self.d_opt,
                         self.reconstruct_opt, self.g_opt],
                        feed_dict={self.x: input_x, self.target: input_y,
                                   self.label: label})

                    print(
                        'epoch : {} step : {} constancy loss : {:.4} G_loss : {:.4} D_loss :{:.4} l1_loss : {:.4}'.format(
                        ep, st, style_loss, g_loss, d_loss, L1_loss))
                all_saver.save(sess=sess, save_path=save_path2, global_step=st)
    def test(self,output_dir,reference, checkpoint):
        self.build_model()
        all_saver = tf.train.Saver()
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        with tf.Session() as sess:
            self.vgg(sess, self.vgg_vars)
            init_vars = [var for var in tf.all_variables() if var not in self.vgg_vars]
            sess.run(tf.initialize_variables(init_vars))
            all_saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(checkpoint))
            test_imgs = os.listdir(reference)

            for i in range(int(len(test_imgs)/self.batch_size)):
                loads = test_imgs[i*self.batch_size:(i+1)*self.batch_size]
                imgs = []
                for j in range(self.batch_size):
                    path = os.path.join(reference,loads[j])
                    imgs.append(util.image_load(path))
                result = sess.run([self.output],feed_dict={self.x:imgs})
                for j in range(self.batch_size):
                    plt.imsave(os.path.join(output_dir,loads[j]),result[0][j].reshape(SIZE,SIZE),cmap='binary_r',format='png')


    def pad_resize(self,im, target_size, fill_color=(255)):
        x, y = im.size
        target_size = int(target_size)
        new_im = Image.new('L', (SIZE, target_size), fill_color)
        new_im.paste(im, (int((SIZE - x) / 2), int((target_size - y) / 2)))
        return np.asarray(new_im)

    # def string(self):
    #
    #     self.build_model()
    #     calli_strings = ['새해복많이받으세요','생일축하합니다','몸건강히하세요','메리크리스마스','즐거운한가위']
    #     zoom_str = ['복','축','몸','스','한']
    #     zoom_rate =  1.8
    #     target_size = int(SIZE * zoom_rate)
    #     all_saver = tf.train.Saver()
    #     with tf.Session() as sess:
    #         self.vgg(sess, self.vgg_vars)
    #         init_vars = [var for var in tf.all_variables() if var not in self.vgg_vars]
    #         sess.run(tf.initialize_variables(init_vars))
    #         all_saver.restore(sess=sess, save_path=tf.train.latest_checkpoint('checkpoint'))
    #         for j,calli_string in enumerate(calli_strings):
    #             imgs = []
    #             for str in calli_string:
    #                 ref_img = os.path.join('reference', str + '.png')
    #                 ref_img = np.expand_dims(util.image_load(ref_img), 0)
    #                 imgs.append(sess.run(self.output, feed_dict={self.x: ref_img}))
    #             for i in range(len(calli_string)):
    #                 if i > 0:
    #                     if i is calli_string.index(zoom_str[j]):
    #                         img = imgs[i].squeeze()*255
    #                         img = Image.fromarray(img)
    #                         img = img.resize((target_size,target_size))
    #                         img = np.asarray(img)
    #                         temp = np.concatenate([temp, img], 1)
    #                     else:
    #                         img = imgs[i].squeeze() *255
    #                         img = Image.fromarray(img)
    #                         img = self.pad_resize(img,target_size)
    #                         temp = np.concatenate([temp, img], 1)
    #
    #
    #                 else:
    #                     img = imgs[i].squeeze()*255
    #                     img = Image.fromarray(img)
    #                     img = self.pad_resize(img,target_size)
    #                     temp = img
    #
    #             plt.imsave(calli_string + '.png', temp, cmap='binary_r')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning_rate", default=0.0002)
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--input_size", type=int, help="image input size ", default=224)
    parser.add_argument("--output_size", type=int, help="image output size", default=224)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=200)
    parser.add_argument("--step", type=int, help="how many roop in a epoch", default=1000)
    parser.add_argument("--mode", type=str, help="select mode training or test", default='train2')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint/',
                        help='models are saved here')
    parser.add_argument('--reference_root', type=str, default='reference')
    parser.add_argument('--num_class', type=int, default=20)
    parser.add_argument('--train_root', type=str, default='dataset')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--save_path2', type=str)
    parser.add_argument('--vgg_path', type=str,default='checkpoint/new_vgg-9277')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    model = DCFont(num_class=args.num_class,
                   learning_rate=args.lr,
                   batch_size=args.batch_size,
                   vgg_path = args.vgg_path
                   )
    # model.string()
    if args.mode == 'train':
        model.train(args.reference_root, args.train_root, args.epoch, args.step,args.save_path)
    elif args.mode == 'train2':
        model.train_part2(args.reference_root,args.target, args.epoch, args.step,args.save_path2)
    elif args.mode == 'test':
        model.test(args.output_dir, args.reference_root, checkpoint=args.checkpoint_dir)

