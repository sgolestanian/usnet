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
    
    def __call__(self, x, training=False):
        x = tf.convert_to_tensor(x)
        select_mask = K.random_bernoulli(self.input_shape, p=self.noise_probability)
        saltpepper_mask = K.random_bernoulli(self.input_shape, p=self.pepper_probability)

        y = tf.matmul(x, (1-select_mask)) + tf.matmul(saltpepper_mask, select_mask)

        return K.in_train_phase(y, x, training=training)