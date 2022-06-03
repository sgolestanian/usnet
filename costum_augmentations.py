import tensorflow as tf
import tensorflow.keras.backend as K


class SaltPepperNoise(tf.keras.layers.Layer):
    """
    Salt and pepper noise augmentation as a keras layer.
    """
    def __init__(self, noise_probability, pepper_probability, *args, **kwargs):
        super(SaltPepperNoise, self).__init__(*args, **kwargs)
        self.noise_probability = noise_probability
        self.pepper_probability = pepper_probability

    def build(self, input_shape):
        return super(SaltPepperNoise, self).build(input_shape)
    
    def call(self, x, training=False):
        x = tf.convert_to_tensor(x)
        input_shape = tf.shape(x)

        select_mask = K.random_bernoulli(input_shape, p=self.noise_probability)
        select_mask = tf.math.reduce_sum(select_mask, axis=-1, keepdims=True)
        select_mask = tf.clip_by_value(select_mask, clip_value_min=0, clip_value_max=1)
        select_mask = tf.repeat(select_mask, input_shape[-1], axis=-1)

        saltpepper_mask = K.random_bernoulli(input_shape, p=self.pepper_probability)
        saltpepper_mask = tf.math.reduce_sum(saltpepper_mask, axis=-1, keepdims=True)
        saltpepper_mask = tf.clip_by_value(saltpepper_mask, clip_value_min=0, clip_value_max=1)
        saltpepper_mask = tf.repeat(saltpepper_mask, input_shape[-1], axis=-1)

        y = tf.multiply(x, (1-select_mask)) + tf.multiply(saltpepper_mask, select_mask)

        return K.in_train_phase(y, x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "noise_probability":self.noise_probability,
            "pepper_probability":self.pepper_probability,
        })