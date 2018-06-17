from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.optimizers import Adam

from ast import literal_eval
from face_swap.PixelShuffler import PixelShuffler


def conv(filters, kernel_size=5, strides=2):
    def block(x):
        x = Conv2D(filters, kernel_size=kernel_size,
                   strides=strides, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        return x
    return block


# deconvolution block used in the decoder
def upscale(filters, kernel_size=3):
    def block(x):
        x = Conv2D(filters * 4, kernel_size=kernel_size,
                   padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block


def Encoder(input_shape, hidden_dim, init_filters=128, num_conv_blocks=4):
    model_input = Input(shape=input_shape)

    x = model_input
    for i in range(num_conv_blocks):
        x = conv(init_filters * (2 ** i))(x)

    x = Dense(hidden_dim)(Flatten()(x))
    x = Dense(4 * 4 * hidden_dim)(x)
    x = Reshape((4, 4, hidden_dim))(x)
    x = upscale(hidden_dim//2)(x)
    return Model(model_input, x)


def Decoder(input_shape, init_filters=256, num_deconv_blocks=3):
    model_input = Input(shape=input_shape)
    x = model_input

    for i in range(num_deconv_blocks):
        x = upscale(init_filters // (1 if i == 0 else (2 ** i)))(x)

    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    return Model(model_input, x)


def get_autoencoders(cfg):
    models_path = cfg.get('models_path', None)
    IMAGE_SHAPE = literal_eval(cfg.get('img_shape'))
    ENCODER_DIM = cfg.get('encoder_dim')
    DECODER_INPUT_SHAPE = literal_eval(cfg.get('decoder_input_shape'))
    encoder_init_filters = cfg.get('encoder_init_filters')
    encoder_nb_conv_blocks = cfg.get('encoder_nb_conv_blocks')
    decoder_init_filters = cfg.get('decoder_init_filters')
    decoder_nb_conv_blocks = cfg.get('decoder_nb_conv_blocks')

    optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

    encoder = Encoder(IMAGE_SHAPE, ENCODER_DIM,
                      init_filters=encoder_init_filters,
                      num_conv_blocks=encoder_nb_conv_blocks)
    decoder_a = Decoder(DECODER_INPUT_SHAPE,
                        init_filters=decoder_init_filters,
                        num_deconv_blocks=decoder_nb_conv_blocks)
    decoder_b = Decoder(DECODER_INPUT_SHAPE,
                        init_filters=decoder_init_filters,
                        num_deconv_blocks=decoder_nb_conv_blocks)

    x = Input(shape=IMAGE_SHAPE)

    autoencoder_a = Model(x, decoder_a(encoder(x)))
    autoencoder_b = Model(x, decoder_b(encoder(x)))
    autoencoder_a.compile(optimizer=optimizer, loss='mean_absolute_error')
    autoencoder_b.compile(optimizer=optimizer, loss='mean_absolute_error')

    if models_path:
        print("Loading Models...")
        encoder.load_weights(models_path + '/encoder.h5')
        decoder_a.load_weights(models_path + '/decoder_A.h5')
        decoder_b.load_weights(models_path + '/decoder_B.h5')
        print("Models Loaded")

    return autoencoder_a, autoencoder_b

