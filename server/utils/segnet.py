"""
SegNet-like architecture.
Uses concatenation between encoder outputs and upsample outputs unlike the original paper.
"""

import tensorflow as tf

def segnet_model(width, height, output_channels):
    """
    SegNet-like model with VGG16 backbone.
    """
    base_model = tf.keras.applications.VGG16(input_shape=[width, height, 3], include_top=False)

    layer_names = [
        'block1_pool',  # 112x112
        'block2_pool',  # 56x56
        'block3_pool',  # 28x28
        'block4_pool',  # 14x14
        'block5_pool',  # 7x7
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False

    inputs = tf.keras.layers.Input(shape=[width, height, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)

    up_block_5 = tf.keras.Sequential()
    up_block_5.add(tf.keras.layers.UpSampling2D())
    up_block_5.add(tf.keras.layers.Convolution2D(512, (3, 3), padding="same"))
    up_block_5.add(tf.keras.layers.BatchNormalization())
    up_block_5.add(tf.keras.layers.Activation("relu"))
    up_block_5.add(tf.keras.layers.Convolution2D(512, (3, 3), padding="same"))
    up_block_5.add(tf.keras.layers.BatchNormalization())
    up_block_5.add(tf.keras.layers.Activation("relu"))
    up_block_5.add(tf.keras.layers.Convolution2D(512, (3, 3), padding="same"))
    up_block_5.add(tf.keras.layers.BatchNormalization())
    up_block_5.add(tf.keras.layers.Activation("relu"))

    up_block_5 = up_block_5(skips[4])
    up_block_5 = tf.keras.layers.Concatenate()([up_block_5, skips[3]])

    up_block_4 = tf.keras.Sequential()
    up_block_4.add(tf.keras.layers.UpSampling2D())
    up_block_4.add(tf.keras.layers.Convolution2D(512, (3, 3), padding="same"))
    up_block_4.add(tf.keras.layers.BatchNormalization())
    up_block_4.add(tf.keras.layers.Activation("relu"))
    up_block_4.add(tf.keras.layers.Convolution2D(512, (3, 3), padding="same"))
    up_block_4.add(tf.keras.layers.BatchNormalization())
    up_block_4.add(tf.keras.layers.Activation("relu"))
    up_block_4.add(tf.keras.layers.Convolution2D(256, (3, 3), padding="same"))
    up_block_4.add(tf.keras.layers.BatchNormalization())
    up_block_4.add(tf.keras.layers.Activation("relu"))

    up_block_4 = up_block_4(up_block_5)
    up_block_4 = tf.keras.layers.Concatenate()([up_block_4, skips[2]])

    up_block_3 = tf.keras.Sequential()
    up_block_3.add(tf.keras.layers.UpSampling2D())
    up_block_3.add(tf.keras.layers.Convolution2D(256, (3, 3), padding="same"))
    up_block_3.add(tf.keras.layers.BatchNormalization())
    up_block_3.add(tf.keras.layers.Activation("relu"))
    up_block_3.add(tf.keras.layers.Convolution2D(256, (3, 3), padding="same"))
    up_block_3.add(tf.keras.layers.BatchNormalization())
    up_block_3.add(tf.keras.layers.Activation("relu"))
    up_block_3.add(tf.keras.layers.Convolution2D(128, (3, 3), padding="same"))
    up_block_3.add(tf.keras.layers.BatchNormalization())
    up_block_3.add(tf.keras.layers.Activation("relu"))

    up_block_3 = up_block_3(up_block_4)
    up_block_3 = tf.keras.layers.Concatenate()([up_block_3, skips[1]])

    up_block_2 = tf.keras.Sequential()
    up_block_2.add(tf.keras.layers.UpSampling2D())
    up_block_2.add(tf.keras.layers.Convolution2D(128, (3, 3), padding="same"))
    up_block_2.add(tf.keras.layers.BatchNormalization())
    up_block_2.add(tf.keras.layers.Activation("relu"))
    up_block_2.add(tf.keras.layers.Convolution2D(64, (3, 3), padding="same"))
    up_block_2.add(tf.keras.layers.BatchNormalization())
    up_block_2.add(tf.keras.layers.Activation("relu"))

    up_block_2 = up_block_2(up_block_3)
    up_block_2 = tf.keras.layers.Concatenate()([up_block_2, skips[0]])

    up_block_1 = tf.keras.Sequential()
    up_block_1.add(tf.keras.layers.UpSampling2D())
    up_block_1.add(tf.keras.layers.Convolution2D(64, (3, 3), padding="same"))
    up_block_1.add(tf.keras.layers.BatchNormalization())
    up_block_1.add(tf.keras.layers.Activation("relu"))
    up_block_1.add(tf.keras.layers.Convolution2D(output_channels, (3, 3), padding="same"))
    up_block_1.add(tf.keras.layers.BatchNormalization())
    up_block_1.add(tf.keras.layers.Activation("softmax"))

    up_block_1 = up_block_1(up_block_2)

    return tf.keras.Model(inputs=inputs, outputs=up_block_1)
