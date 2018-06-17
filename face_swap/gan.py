from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.initializers import RandomNormal
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
import keras.backend as K
from tensorflow.contrib.distributions import Beta
import tensorflow as tf
from keras.optimizers import Adam

from ast import literal_eval
from face_swap.PixelShuffler import PixelShuffler
from face_swap import gan_utils


conv_init = RandomNormal(0., 0.02)


# Difference between autoencoder block
# kernel initializer, use bias to false
def conv(filters, kernel_size=5, strides=2, leaky_relu=False, batch_norm=False):
    def block(x):
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides,
                   kernel_initializer=conv_init,
                   use_bias=False,
                   padding='same')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        if leaky_relu:
            x = LeakyReLU(0.2)(x)
        else:
            x = Activation("relu")(x)
        return x
    return block

# Standard feed-forward CNN with skip connections that bypass the convolution layers
# ref: http://torch.ch/blog/2016/02/04/resnets.html
def res_block(filters, kernel_size=3):
    def block(input_tensor):
        x = Conv2D(filters, kernel_size=kernel_size,
                   kernel_initializer=conv_init, use_bias=False, padding="same")(input_tensor)
        #x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters, kernel_size=kernel_size,
                   kernel_initializer=conv_init, use_bias=False, padding="same")(x)
        #x = BatchNormalization()(x)
        x = add([x, input_tensor])
        x = LeakyReLU(alpha=0.2)(x)
        return x
    return block


# Difference between autoencoder block
# kernel initializer, use bias to false
def upscale(filters, kernel_size=3):
    def block(x):
        x = Conv2D(filters*4, kernel_size=kernel_size, use_bias=False,
                   kernel_initializer=conv_init, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block


# Difference between autoencoder encoder is the initial conv2d
def Encoder(input_shape, hidden_dim=1024, init_filters=128, num_conv_blocks=4):
    model_input = Input(shape=input_shape)

    x = Conv2D(init_filters//2, kernel_size=5,
               kernel_initializer=conv_init, use_bias=False, padding="same")(model_input)

    for i in range(num_conv_blocks):
        x = conv(init_filters * (2 ** i))(x)

    x = Dense(hidden_dim)(Flatten()(x))
    x = Dense(4 * 4 * hidden_dim)(x)
    x = Reshape((4, 4, hidden_dim))(x)
    x = upscale(hidden_dim//2)(x)
    return Model(model_input, x)


# Difference between autoencoder encoder is additional res blocks
def Decoder(input_shape, init_filters=256, num_deconv_blocks=3, num_res_blocks=2, include_alpha=False):
    model_input = Input(shape=input_shape)
    x = model_input

    for i in range(num_deconv_blocks):
        x = upscale(init_filters // (1 if i == 0 else (2 ** i)))(x)

    for i in range(num_res_blocks):
        # TOFIX generalize num filter based on last number in deconv
        x = res_block(64)(x)

    if include_alpha:
        alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
        rgb = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
        out = concatenate([alpha, rgb])
        return Model(model_input, out)
    else:
        x = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)
        return Model(model_input, x)


def Discriminator(input_shape, init_filters=64, num_conv_blocks=3):
    model_input = Input(shape=input_shape)

    x = model_input
    for i in range(num_conv_blocks):
        x = conv(init_filters * (2 ** i), kernel_size=4, leaky_relu=True)(x)

    x = Conv2D(1, kernel_size=4,
               kernel_initializer=conv_init, use_bias=False, padding="same", activation="sigmoid")(x)
    return Model(model_input, x)

# use batchnorm in both generator and discriminator
# use ReLU in generator for all layers except for the output, which uses Tanh
# use LeakyReLU in the discriminator for all layers
# ref: http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf
def get_gan(cfg, load_discriminators=True):
    models_path = cfg.get('models_path', None)
    img_shape = literal_eval(cfg.get('img_shape'))
    encoder_dim = cfg.get('encoder_dim')
    decoder_input_shape = literal_eval(cfg.get('decoder_input_shape'))
    discriminator_input_shape = literal_eval(cfg.get('discriminator_input_shape'))
    include_alpha = cfg.get('masked', False)

    encoder = Encoder(img_shape, encoder_dim)
    decoder_a = Decoder(decoder_input_shape, include_alpha=include_alpha)
    decoder_b = Decoder(decoder_input_shape, include_alpha=include_alpha)

    x = Input(shape=img_shape)

    gen_a = Model(x, decoder_a(encoder(x)))
    gen_b = Model(x, decoder_b(encoder(x)))

    dis_a = Discriminator(discriminator_input_shape)
    dis_b = Discriminator(discriminator_input_shape)

    if models_path:
        print("Loading Models...")
        encoder.load_weights(models_path + '/encoder.h5')
        decoder_a.load_weights(models_path + '/decoder_A.h5')
        decoder_b.load_weights(models_path + '/decoder_B.h5')
        if load_discriminators:
            dis_a.load_weights(models_path + "/netDA.h5")
            dis_b.load_weights(models_path + "/netDB.h5")
        print("Models Loaded")

    return gen_a, gen_b, dis_a, dis_b


def build_training_functions(cfg, netGA, netGB, netDA, netDB, vggface=None):
    """
    Get generators and discriminators training functions.
    Cycles trough variables, defines the loss, gets updates from weights and defines training functions
    :param cfg: model configuration
    :param netGA:
    :param netGB:
    :param netDA:
    :param netDB:
    :param vggface:
    :return:
    """
    img_shape = literal_eval(cfg.get('img_shape'))
    lrD = cfg.get('discriminator_learning_rate')
    lrG = cfg.get('generator_learning_rate')

    distorted_A, fake_A, path_A = gan_utils.cycle_variables(netGA)
    distorted_B, fake_B, path_B = gan_utils.cycle_variables(netGB)
    real_A = Input(shape=img_shape)
    real_B = Input(shape=img_shape)

    vggface_feat = gan_utils.perceptual_loss_model(vggface)

    loss_DA, loss_GA = gan_utils.define_loss(netDA, real_A, fake_A, vggface_feat)
    loss_DB, loss_GB = gan_utils.define_loss(netDB, real_B, fake_B, vggface_feat)

    weightsDA = netDA.trainable_weights
    weightsGA = netGA.trainable_weights
    weightsDB = netDB.trainable_weights
    weightsGB = netGB.trainable_weights

    # Adam(..).get_updates(...)
    training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDA, [], loss_DA)
    netDA_train = K.function([distorted_A, real_A], [loss_DA], training_updates)
    training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsGA, [], loss_GA)
    netGA_train = K.function([distorted_A, real_A], [loss_GA], training_updates)

    training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDB, [], loss_DB)
    netDB_train = K.function([distorted_B, real_B], [loss_DB], training_updates)
    training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsGB, [], loss_GB)
    netGB_train = K.function([distorted_B, real_B], [loss_GB], training_updates)

    return netGA_train, netGB_train, netDA_train, netDB_train


def build_training_functions_masked(cfg, netGA, netGB, netDA, netDB, vggface=None,
                             use_mask_hinge_loss=False, m_mask=0.5, lr_factor=1):
    """
    Get generators and discriminators training functions.
    Cycles trough variables, defines the loss, gets updates from weights and defines training functions
    :param cfg: model configuration
    :param netGA:
    :param netGB:
    :param netDA:
    :param netDB:
    :param vggface:
    :param use_mask_hinge_loss:
    :param m_mask:
    :param lr_factor:
    :return:
    """
    img_shape = literal_eval(cfg.get('img_shape'))
    lrD = cfg.get('discriminator_learning_rate')
    lrG = cfg.get('generator_learning_rate')

    # Alpha mask regularizations
    # m_mask = 0.5 # Margin value of alpha mask hinge loss
    w_mask = 0.1  # hinge loss
    w_mask_fo = 0.01  # Alpha mask total variation loss

    distorted_A, fake_A, mask_A, path_A, fun_mask_A, fun_abgr = gan_utils.cycle_variables_masked(netGA)
    distorted_B, fake_B, mask_B, path_B, fun_mask_B, fun_abgr = gan_utils.cycle_variables_masked(netGB)
    real_A = Input(shape=img_shape)
    real_B = Input(shape=img_shape)

    vggface_feat = gan_utils.perceptual_loss_model(vggface)

    loss_DA, loss_GA = gan_utils.define_loss_masked(netDA, real_A, fake_A, distorted_A, vggface_feat,
                                                     mixup_alpha=cfg.get('mixup_alpha', None))
    loss_DB, loss_GB = gan_utils.define_loss_masked(netDB, real_B, fake_B, distorted_B, vggface_feat,
                                                    mixup_alpha=cfg.get('mixup_alpha', None))

    # Alpha mask loss
    if not use_mask_hinge_loss:
        loss_GA += 1e-3 * K.mean(K.abs(mask_A))
        loss_GB += 1e-3 * K.mean(K.abs(mask_B))
    else:
        loss_GA += w_mask * K.mean(K.maximum(0., m_mask - mask_A))
        loss_GB += w_mask * K.mean(K.maximum(0., m_mask - mask_B))

    # Alpha mask total variation loss
    loss_GA += w_mask_fo * K.mean(gan_utils.first_order(mask_A, axis=1))
    loss_GA += w_mask_fo * K.mean(gan_utils.first_order(mask_A, axis=2))
    loss_GB += w_mask_fo * K.mean(gan_utils.first_order(mask_B, axis=1))
    loss_GB += w_mask_fo * K.mean(gan_utils.first_order(mask_B, axis=2))

    weightsDA = netDA.trainable_weights
    weightsGA = netGA.trainable_weights
    weightsDB = netDB.trainable_weights
    weightsGB = netGB.trainable_weights

    # Adam(..).get_updates(...)
    training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDA, [], loss_DA)
    netDA_train = K.function([distorted_A, real_A], [loss_DA], training_updates)
    training_updates = Adam(lr=lrG * lr_factor, beta_1=0.5).get_updates(weightsGA, [], loss_GA)
    netGA_train = K.function([distorted_A, real_A], [loss_GA], training_updates)

    training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsDB, [], loss_DB)
    netDB_train = K.function([distorted_B, real_B], [loss_DB], training_updates)
    training_updates = Adam(lr=lrG * lr_factor, beta_1=0.5).get_updates(weightsGB, [], loss_GB)
    netGB_train = K.function([distorted_B, real_B], [loss_GB], training_updates)

    return netGA_train, netGB_train, netDA_train, netDB_train
