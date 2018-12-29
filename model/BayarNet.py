import numpy as np
import tensorflow as tf

from utils.Utils import normalize
from base.BaseModel import BaseModel


class BayarNet(BaseModel):
    def __init__(self, data_loader,
                 config, trainable=True, retrain='complete'):
        super(BayarNet, self).__init__(config)
        self.data_loader = data_loader
        self.kp = config.keep_prob
        self.n_classes = config.num_classes
        self.logits = None
        self.logits_argmax = None
        self.loss = None
        self.acc = None
        self.optimizer = None
        self.train_step = None
        self.build_model(config)
        self.init_saver()

    def build_model(self, config):
        self.global_step_tensor = tf.Variable(
            0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(
            self.global_step_tensor)

        self.global_epoch_tensor = tf.Variable(
            0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(
            self.global_epoch_tensor + 1)

        with tf.name_scope('inputs') as scope:
            self.x = tf.placeholder(
                'float32',
                shape=[
                    None,
                    config.image_shape,
                    config.image_shape,
                    config.image_channels
                    ],
                name='x'
                )
            self.y = tf.placeholder(
                'int32', shape=[None], name='y')
            self.tr = tf.placeholder(
                'bool', shape=None, name='trainable')

            tf.add_to_collection('inputs', self.x)
            tf.add_to_collection('inputs', self.y)

        with tf.name_scope('residual') as scope:
            self.convres_kernel = tf.get_variable(
                'convres_kernel',
                [5, 5, 1, 3],
                initializer=tf.contrib.layers.xavier_initializer())
            self.convres_biases = tf.get_variable(
                'convres_biases', [3],
                initializer=tf.random_normal_initializer())

        with tf.name_scope('network') as scope:
            normalized_k = tf.py_func(
                normalize,
                [self.convres_kernel],
                tf.float32,
                name='normalize_kernel')

            normalized_k.set_shape(self.convres_kernel.get_shape())

            self.convres_kernel.assign(normalized_k)

            convres = tf.nn.conv2d(
                self.x,
                self.convres_kernel,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='convres')
            convres = tf.nn.bias_add(
                convres, self.convres_biases)

            conv_1 = self.conv_layer(
                inputs=convres, filters=96,
                k_size=7, stride=2,
                padding='VALID',
                scope_name='conv_1',
                fabs=False, active=True,
                epsilon=1e-4, train=self.tr)
            pool_1 = self.maxpool(
                inputs=conv_1,
                k_size=5, stride=2,
                padding='SAME',
                scope_name='pool_1')

            conv_2 = self.conv_layer(
                inputs=pool_1, filters=64,
                k_size=5, stride=1,
                padding='VALID',
                scope_name='conv_2',
                fabs=False, active=True,
                epsilon=1e-4, train=self.tr)
            pool_2 = self.maxpool(
                inputs=conv_2,
                k_size=5, stride=2,
                padding='SAME',
                scope_name='pool_2')

            conv_3 = self.conv_layer(
                inputs=pool_2, filters=64,
                k_size=5, stride=1,
                padding='VALID',
                scope_name='conv_3',
                fabs=False, active=True,
                epsilon=1e-4, train=self.tr)
            pool_3 = self.maxpool(
                inputs=conv_3,
                k_size=5, stride=2,
                padding='SAME',
                scope_name='pool_3')

            conv_4 = self.conv_layer(
                inputs=pool_3, filters=128,
                k_size=1, stride=1,
                padding='VALID',
                scope_name='conv_4',
                fabs=False, active=True,
                epsilon=1e-4, train=self.tr)
            pool_4 = self.averagepool(
                inputs=conv_4,
                k_size=5, stride=2,
                padding='SAME',
                scope_name='pool_4')

            cur_dim = pool_4.get_shape()
            pool4_dim = cur_dim[1] * cur_dim[2] * cur_dim[3]
            pool4_flatten = tf.reshape(pool_4, shape=[-1, pool4_dim])

            fc5 = self.fully_connected(
                pool4_flatten, out_dim=1024,
                scope_name='fc5', activation=tf.nn.relu)
            fc5 = tf.layers.dropout(
                fc5,
                self.kp,
                training=self.tr,
                name='dropout_1')

            fc6 = self.fully_connected(
                fc5, out_dim=512,
                scope_name='fc6', activation=tf.nn.relu)
            fc6 = tf.layers.dropout(
                fc6,
                self.kp,
                training=self.tr,
                name='dropout_2')

            self.logits = self.fully_connected(
                fc6, out_dim=self.n_classes,
                scope_name='logits', activation=None)

            tf.add_to_collection('logits', self.logits)

        with tf.name_scope('logits_argmax') as scope:
            self.logits_argmax = tf.argmax(
                self.logits, axis=1,
                output_type=tf.int64, name='out_argmax')

        with tf.name_scope('loss') as scope:
            self.entropy = tf.losses.sparse_softmax_cross_entropy(
                labels=self.y, logits=self.logits)
            self.loss = tf.reduce_mean(self.entropy, name='loss')

        with tf.name_scope('learning_rate_decay') as scope:
            learning_rate = tf.train.cosine_decay_restarts(
                learning_rate=config.learning_rate,
                global_step=self.global_step_tensor,
                first_decay_steps=config.num_iter_per_epoch,
                t_mul=2.0,
                m_mul=0.9,
                alpha=0.0,
                name='learning_rate_with_restart')

        with tf.name_scope('train_step') as scope:
            self.optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=self.config.momentum)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(
                    self.loss, global_step=self.global_step_tensor)

        with tf.name_scope('accuracy'):
            prediction = tf.nn.softmax(self.logits, name='prediction')
            correct_prediction = tf.equal(
                tf.argmax(
                    prediction, axis=1), tf.cast(self.y, dtype=tf.int64))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('loss', self.loss)
        tf.add_to_collection('acc', self.accuracy)

    def init_saver(self):
        self.saver = tf.train.Saver(
            max_to_keep=self.config.max_to_keep,
            save_relative_paths=True)

    def conv_layer(self, inputs, filters, k_size,
                   stride, padding, scope_name,
                   epsilon, train, fabs=True, active=True):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            in_channels = inputs.shape[-1]
            kernel = tf.get_variable(
                'kernel',
                [k_size, k_size, in_channels, filters],
                initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(
                'biases', [filters],
                initializer=tf.random_normal_initializer())
            conv = tf.nn.bias_add(
                tf.nn.conv2d(
                    inputs,
                    kernel,
                    strides=[1, stride, stride, 1],
                    padding=padding),
                biases)

            if fabs:
                conv = tf.abs(conv)

            beta = tf.Variable(
                tf.constant(0.0, shape=[filters]),
                name='beta',
                trainable=False)
            gamma = tf.Variable(
                tf.constant(1.0, shape=[filters]),
                name='gamma',
                trainable=False)
            batch_mean, batch_var = tf.nn.moments(conv, [0, 1, 2])
            ema = tf.train.ExponentialMovingAverage(decay=0.1)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])

                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(
                train,
                mean_var_with_update,
                lambda: (ema.average(batch_mean), ema.average(batch_var)))

            bn_conv = tf.nn.batch_normalization(
                conv, mean, var, beta, gamma, epsilon)

            if active:
                f_conv = tf.nn.tanh(bn_conv)
            else:
                f_conv = tf.nn.relu(bn_conv)

        return f_conv

    def maxpool(self, inputs, k_size,
                stride, padding='VALID', scope_name='maxpool'):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            pool = tf.nn.max_pool(
                inputs, ksize=[1, k_size, k_size, 1],
                strides=[1, stride, stride, 1], padding=padding)
        return pool

    def averagepool(self, inputs, k_size,
                    stride, padding='VALID', scope_name='avgpool'):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            pool = tf.nn.avg_pool(
                inputs, ksize=[1, k_size, k_size, 1],
                strides=[1, stride, stride, 1], padding=padding)
        return pool

    def fully_connected(self, inputs, out_dim,
                        scope_name, activation=tf.nn.relu):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            in_dim = inputs.shape[-1]
            w = tf.get_variable(
                'weights', [in_dim, out_dim],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                'biases', [out_dim],
                initializer=tf.random_normal_initializer())
            out = tf.nn.bias_add(tf.matmul(inputs, w), b)

            if activation:
                out = activation(out)
            return out
