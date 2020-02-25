"""
A smaller version of biggan
Seems to work better than stylegan on small, diverse datasets
"""
from tqdm import tqdm
import cv2
from PIL import Image
from keras.utils import plot_model
from functools import partial

from keras.layers import BatchNormalization, Dense, Reshape, Lambda, Multiply, Add, Layer
from keras.layers import Activation, UpSampling2D, AveragePooling2D, GlobalAveragePooling2D, Input
from keras.layers import Concatenate, Embedding, Flatten, LeakyReLU, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.engine.network import Network
from keras.utils import multi_gpu_model

from model.utils import *
from model.layers import *
from model.network_blocks import *

class StyleFewGAN():
    def __init__(
        self, 
        img_dim_x,
        img_dim_y,
        img_depth,
        z_len,
        n_classes,
        g_lr,
        d_lr,
        batch_size,
        save_freq,
        training_dir,
        validation_dir,
        checkpoint_dir,
        testing_dir,
        n_cpu,
        n_gpu,
        n_noise_samples=16
        ):
        self.img_dim_x = img_dim_x
        self.img_dim_y = img_dim_y
        self.img_depth = img_depth
        self.z_len = z_len
        self.n_noise_samples = n_noise_samples
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.n_cpu = n_cpu
        self.n_gpu = n_gpu
        self.training_dir = training_dir
        self.validation_dir = validation_dir
        self.checkpoint_dir = checkpoint_dir
        self.testing_dir = testing_dir
        for path in [self.validation_dir,
                     self.checkpoint_dir,
                     self.testing_dir]:
            if not os.path.isdir(path):
                os.makedirs(path)
        self.n_classes = n_classes
        self.n_projections = 8
        self.name = 'stylegan2'
        self.noise_samples = np.random.normal(0,0.8,size=(self.n_noise_samples, self.z_len))
        self.resolutions = [512, 256, 128, 64, 32, 16, 8, 4]
        self.build_generator()
        self.build_discriminator()

    ###############################
    ## All our architecture
    ###############################
    def build_generator(self):
        model_in = Input(shape=(self.n_projections, self.z_len, ))
        style = TimeDistributed(Lambda(
            lambda x: x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + K.epsilon())
            ))(model_in)
        for _ in range(1):
            style = TimeDistributed(Dense(units=self.z_len, kernel_initializer='he_uniform'))(style)
            style = TimeDistributed(LeakyReLU(0.2))(style)

        ch = 32
        x = LearnedConstantLatent()(model_in)
        style0 = Lambda(lambda x: x[:, 0, ...])(style)
        x = style2_generator_layer(x, style0, output_dim=ch) #4x32
        x = style2_generator_layer(x, style0, output_dim=ch) #4x32
        rgb_4x4 = to_rgb(x, style0) #4x3
        up_rgb_4x4 = UpSampling2D(2, interpolation='bilinear')(rgb_4x4)

        style1 = Lambda(lambda x: x[:, 1, ...])(style)
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(up_rgb_4x4)
        x = LeakyReLU(0.2)(x)
        x = style2_generator_layer(x, style1, output_dim=ch) #8x32
        x = style2_generator_layer(x, style1, output_dim=ch) #8x32
        rgb_8x8 = to_rgb(x, style1)
        x = Add()([rgb_8x8, up_rgb_4x4])
        up_rgb_8x8 = UpSampling2D(2, interpolation='bilinear')(x)

        style2 = Lambda(lambda x: x[:, 2, ...])(style)
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(up_rgb_8x8)
        x = LeakyReLU(0.2)(x)
        x = style2_generator_layer(x, style2, output_dim=ch) #16x32
        x = style2_generator_layer(x, style2, output_dim=ch) #16x32
        rgb_16x16 = to_rgb(x, style2)
        x = Add()([rgb_16x16, up_rgb_8x8])
        up_rgb_16x16 = UpSampling2D(2, interpolation='bilinear')(x)

        style3 = Lambda(lambda x: x[:, 3, ...])(style)
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(up_rgb_16x16)
        x = LeakyReLU(0.2)(x)
        x = style2_generator_layer(x, style3, output_dim=ch) #32x32
        x = style2_generator_layer(x, style3, output_dim=ch) #32x32
        rgb_32x32 = to_rgb(x, style3)
        x = Add()([rgb_32x32, up_rgb_16x16])
        up_rgb_32x32 = UpSampling2D(2, interpolation='bilinear')(x)

        style4 = Lambda(lambda x: x[:, 4, ...])(style)
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(up_rgb_32x32)
        x = LeakyReLU(0.2)(x)
        x = style2_generator_layer(x, style4, output_dim=ch) #64x32
        x = style2_generator_layer(x, style4, output_dim=ch) #64x32
        rgb_64x64 = to_rgb(x, style4)
        x = Add()([rgb_64x64, up_rgb_32x32])
        up_rgb_64x64 = UpSampling2D(2, interpolation='bilinear')(x)

        style5 = Lambda(lambda x: x[:, 5, ...])(style)
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(up_rgb_64x64)
        x = LeakyReLU(0.2)(x)
        x = style2_generator_layer(x, style5, output_dim=ch) #128x32
        x = style2_generator_layer(x, style5, output_dim=ch) #128x32
        rgb_128x128 = to_rgb(x, style5)
        x = Add()([rgb_128x128, up_rgb_64x64])
        up_rgb_128x128 = UpSampling2D(2, interpolation='bilinear')(x)

        style6 = Lambda(lambda x: x[:, 6, ...])(style)
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(up_rgb_128x128)
        x = LeakyReLU(0.2)(x)
        x = style2_generator_layer(x, style6, output_dim=ch) #256x32
        x = style2_generator_layer(x, style6, output_dim=ch) #256x32
        rgb_256x256 = to_rgb(x, style6)
        x = Add()([rgb_256x256, up_rgb_128x128])
        up_rgb_256x256 = UpSampling2D(2, interpolation='bilinear')(x)

        style7 = Lambda(lambda x: x[:, 7, ...])(style)
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(up_rgb_256x256)
        x = LeakyReLU(0.2)(x)
        x = style2_generator_layer(x, style7, output_dim=ch) #512x32
        x = style2_generator_layer(x, style7, output_dim=ch) #512x32
        rgb_512x512 = to_rgb(x, style7)

        self.generator = Model(
            model_in,
            [rgb_512x512, rgb_256x256, rgb_128x128, rgb_64x64, rgb_32x32, rgb_16x16, rgb_8x8, rgb_4x4]
            )   
        print(self.generator.summary())   


    def build_discriminator(self):
        model_ins = [Input(shape=(dim, dim, self.img_depth)) for dim in self.resolutions]
        
        ch = 32
        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(model_ins[0])
        x = LeakyReLU(0.2)(x)
        feat_256 = style2_discriminator_block(x, ch) # 256x32

        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(model_ins[1])
        x = LeakyReLU(0.2)(x)
        x = Add()([x, feat_256])
        feat_128 = style2_discriminator_block(x, ch) # 128x32

        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(model_ins[2])
        x = LeakyReLU(0.2)(x)
        x = Add()([x, feat_128])
        feat_64 = style2_discriminator_block(x, ch) # 64x32

        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(model_ins[3])
        x = LeakyReLU(0.2)(x)
        x = Add()([x, feat_64])
        feat_32 = style2_discriminator_block(x, ch) # 32x32

        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(model_ins[4])
        x = LeakyReLU(0.2)(x)
        x = Add()([x, feat_32])
        feat_16 = style2_discriminator_block(x, ch) # 16x32

        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(model_ins[5])
        x = LeakyReLU(0.2)(x)
        x = Add()([x, feat_16])
        feat_8 = style2_discriminator_block(x, ch) # 8x32

        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(model_ins[6])
        x = LeakyReLU(0.2)(x)
        x = Add()([x, feat_8])
        feat_4 = style2_discriminator_block(x, ch) # 4x32

        x = Conv2D(
            filters=ch,
            kernel_size=1,
            kernel_initializer='he_normal'
            )(model_ins[7])
        x = LeakyReLU(0.2)(x)
        x = Add()([x, feat_4])

        # 4x4
        x = MiniBatchStd()(x)
        x = Conv2D(
            filters=ch,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_normal'
            )(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(
            filters=ch,
            kernel_size=4,
            padding='valid',
            kernel_initializer='he_normal'
            )(x)
        x = LeakyReLU(0.2)(x)
        x = Flatten()(x)

        # architecture of tail stem
        model_out = Dense(units=1, kernel_initializer='he_normal')(x)

        self.discriminator = Model(model_ins, model_out)
        self.frozen_discriminator = Network(model_ins, model_out)

        print(self.discriminator.summary())

    def build_model(self):
        d_optimizer = Adam(lr=self.d_lr, beta_1=0.0, beta_2=0.99)
        g_optimizer = Adam(lr=self.g_lr, beta_1=0.0, beta_2=0.99)

        # build complete discriminator
        fake_in = [Input(shape=(dim, dim, self.img_depth)) for dim in self.resolutions]
        real_in = [Input(shape=(dim, dim, self.img_depth)) for dim in self.resolutions]
        fake_label = self.discriminator(fake_in)
        real_label = self.discriminator(real_in)

        self.discriminator_model = Model(
            real_in + fake_in,
            [real_label, fake_label]
            )
        if self.n_gpu > 1:
            self.discriminator_model = multi_gpu_model(self.discriminator_model, gpus=self.n_gpu)
        self.discriminator_model.compile(
            d_optimizer,
            loss=[
                nonsat_real_discriminator_loss,
                nonsat_fake_discriminator_loss
                ]
            )

        self.frozen_discriminator.trainable = False

        # build generator model
        z_in = Input(shape=(self.n_projections, self.z_len, ))
        fake_imgs = self.generator(z_in)
        frozen_fake_label = self.frozen_discriminator(fake_imgs)

        self.generator_model = Model(z_in, frozen_fake_label)
        self.generator_model.compile(g_optimizer, nonsat_generator_loss)
        
        print(self.discriminator_model.summary())
        print(self.generator_model.summary())

    ###############################
    ## All our training, etc
    ###############################               

    def train(self, epochs):
        #batch_size = self.batch_size*self.n_gpu
        img_generator = ImgGenerator(
            img_dir=self.training_dir,
            batch_size=self.batch_size,
            n_cpu=self.n_cpu,
            img_dim=self.img_dim_x,
            multiscale=True
            )
        n_batches = img_generator.n_batches
        for epoch in range(epochs):
            d_loss_accum = []
            g_loss_accum = []

            n_steps = 8196
            pbar = tqdm(total=n_steps)
            for batch_i in range(n_steps):
                real_batch, _ = img_generator.next()
                mix_prob = np.random.randint(10)
                noise = np.random.normal(0, 1, size=(self.batch_size, self.z_len))
                noise = np.stack([noise]*self.n_projections, axis=1)
                if mix_prob != 0:
                    break_point = np.random.randint(low=0, high=self.n_projections)
                    noise2 = np.random.normal(0, 1, size=(self.batch_size, self.z_len))
                    noise2 = np.stack([noise2]*self.n_projections, axis=1)
                    noise[:, break_point:, :] = noise2[:, break_point:, :]
                dummy = np.ones(shape=(self.batch_size, ))

                fake_batch = self.generator.predict(noise)
                
                d_loss = self.discriminator_model.train_on_batch(
                    real_batch + fake_batch,
                    [dummy, dummy]
                    )
                d_loss_accum.append(d_loss[0])

                g_loss = self.generator_model.train_on_batch(noise, dummy)
                g_loss_accum.append(g_loss)

                pbar.update()
            pbar.close()

            print('{}/{} ----> d_loss: {}, g_loss: {}'.format(
                epoch, 
                epochs, 
                np.mean(d_loss_accum), 
                np.mean(g_loss_accum)
                ))

            if epoch % self.save_freq == 0:
                self.noise_validation(epoch)
                self.save_model_weights(epoch, np.mean(d_loss_accum), np.mean(g_loss_accum))
        img_generator.end()                


    def noise_validation(self, epoch):
        print('Generating Images...')
        if not os.path.isdir(self.validation_dir):
            os.mkdir(self.validation_dir)
        predicted_imgs = self.generator.predict(np.stack([self.noise_samples]*self.n_projections, axis=1))
        predicted_imgs = [((img+1)*127.5).astype(np.uint8) for img in predicted_imgs[0]]

        # fill a grid
        grid_dim = int(np.sqrt(self.n_noise_samples))
        img_grid = np.zeros(shape=(self.img_dim_x*grid_dim, 
                                   self.img_dim_y*grid_dim,
                                   self.img_depth))


        positions = itertools.product(range(grid_dim), range(grid_dim))
        for (x_i, y_i), img in zip(positions, predicted_imgs):
            x = x_i * self.img_dim_x
            y = y_i * self.img_dim_y
            img_grid[y:y+self.img_dim_y, x:x+self.img_dim_x, :] = img

        img_grid = Image.fromarray(img_grid.astype(np.uint8))
        img_grid.save(os.path.join(self.validation_dir, "{}_validation_img_{}.png".format(self.name, epoch)))

    def save_model_weights(self, epoch, d_loss, g_loss):
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        generator_savename = os.path.join(
            self.checkpoint_dir,
            '{}_generator_weights_{}_{:.3f}.h5'.format(self.name, epoch, g_loss)
            )
        discriminator_savename = os.path.join(
            self.checkpoint_dir,
            '{}_discriminator_weights_{}_{:.3f}.h5'.format(self.name, epoch, d_loss)
            )
        self.generator.save_weights(generator_savename)
        self.discriminator.save_weights(discriminator_savename)

    def generate_samples(self, savedir, n_samples, batch_size=8, z_var=0.5):
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        n_batches = n_samples//batch_size
        pbar = tqdm(total=n_batches)
        for i in range(n_batches):
            noise_batch = np.random.normal(0, z_var, size=(batch_size, self.z_len))
            predicted_imgs = self.generator.predict(np.stack([noise_batch]*self.n_projections, axis=1))
            predicted_imgs = [((img+1)*127.5).astype(np.uint8) for img in predicted_imgs]
            for j, img in enumerate(predicted_imgs):
                img_name = os.path.join(
                    savedir,
                    "{}_{}.png".format(i, j)
                    )
                img = Image.fromarray(img.astype(np.uint8))
                img.save(img_name)
            pbar.update()
        pbar.close()
