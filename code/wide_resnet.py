from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten
from keras.layers import Dense, Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K


channel_axis = -1


def initial_conv(x, out_c=16):
    x = Conv2D(
            out_c,
            (3, 3),
            padding='same',
            kernel_initializer='he_normal')(x)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(
            axis=channel_axis,
            momentum=0.1,
            epsilon=1e-5,
            gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    return x


def residual_unit(x, out_c, dropout, is_first, compress):
    skip = x
    if is_first:
        skip = Conv2D(
                out_c,
                (1, 1),
                padding='same',
                strides=(2, 2) if compress else (1, 1),
                kernel_initializer='he_normal')(x)

    x = BatchNormalization(
            axis=channel_axis,
            momentum=0.1,
            epsilon=1e-5,
            gamma_initializer='uniform')(x)

    x = Activation('relu')(x)

    x = Conv2D(
            out_c//4,
            (1, 1),
            padding='same',
            strides=(1, 1),
            kernel_initializer='he_normal')(x)

    x = BatchNormalization(
            axis=channel_axis,
            momentum=0.1,
            epsilon=1e-5,
            gamma_initializer='uniform')(x)

    x = Activation('relu')(x)

    x = Conv2D(
            out_c//4,
            (3, 3),
            padding='same',
            strides=(2, 2) if compress else (1, 1),
            kernel_initializer='he_normal')(x)

    x = BatchNormalization(
            axis=channel_axis,
            momentum=0.1,
            epsilon=1e-5,
            gamma_initializer='uniform')(x)

    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Conv2D(
            out_c,
            (1, 1),
            padding='same',
            strides=(1, 1),
            kernel_initializer='he_normal')(x)

    return Add()([x, skip])


def residual_block(x, out_c, N, dropout, compress=True):
    x = residual_unit(x, out_c, dropout, is_first=True, compress=compress)
    for i in range(N-1):
        x = residual_unit(x, out_c, dropout, is_first=False, compress=False)
    return x


def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1):
    """
    Creates a Wide Residual Network with specified parameters
    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:
    """
    global channel_axis

    if (K.image_data_format() == 'channels_first'):
        channel_axis = 1
        image_width = input_dim[3]
    else:
        channel_axis = -1
        image_width = input_dim[2]

    ip = Input(batch_shape=input_dim)

    x = initial_conv(ip, 16)
    nb_conv = 1

    x = residual_block(x, 16*k, N, dropout, compress=False)
    nb_conv += N*3

    x = residual_block(x, 32*k, N, dropout, compress=True)
    nb_conv += N*3 + 1

    x = residual_block(x, 64*k, N, dropout, compress=True)
    nb_conv += N*3 + 1

    x = AveragePooling2D((image_width//4, image_width//4))(x)
    x = Flatten()(x)

    if nb_classes == 2:
        x = Dense(1, activation='sigmoid')(x)
    else:
        x = Dense(nb_classes, activation='softmax')(x)

    model = Model(ip, x)

    if verbose:
        print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model


if __name__ == "__main__":
    from keras.utils import plot_model

    shape = (None, 32, 32, 7)

    wrn_28_10 = create_wide_residual_network(
            shape,
            nb_classes=10,
            N=2,
            k=2,
            dropout=0.0)

    wrn_28_10.summary()

    plot_model(wrn_28_10, "WRN-16-2.png", show_shapes=True, show_layer_names=True)
