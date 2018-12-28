import numpy as np
import tensorflow as tf


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')

    learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
        np.pi * (tf.cast(
            global_step, tf.float32) - warmup_steps - hold_base_rate_steps
            ) / float(total_steps - warmup_steps - hold_base_rate_steps)))

    if hold_base_rate_steps > 0:
        learning_rate = tf.where(
            global_step > warmup_steps + hold_base_rate_steps,
            learning_rate, learning_rate_base)

    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to'
                             'warmup_learning_rate')

        slope = (learning_rate_base * warmup_learning_rate) / warmup_steps

        warmup_rate = slope * tf.cast(
            global_step, tf.float32) + warmup_learning_rate

        learning_rate = tf.where(
            global_step < warmup_steps,
            warmup_rate, learning_rate,
            name='learning_rate')

    return tf.where(
        global_step > total_steps,
        0.0, learning_rate,
        name='learning_rate')
