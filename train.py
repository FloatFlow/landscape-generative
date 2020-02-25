import argparse
import keras.backend as K
#from model.stylegan import StyleGAN
#from model.minigan import MiniGAN
#from model.msgstylegan import MSGStyleGAN
from model.stylegan2 import StyleGAN2
import psutil
from tensorflow.python.client import device_lib

N_CPU = psutil.cpu_count()
N_GPU = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for GAN')

    # general parameters

    parser.add_argument(
        '--train',
        type=bool,
        default=True
        )
    parser.add_argument(
        '--training_dir',
        type=str,
        default='images'
        )
    parser.add_argument(
        '--validation_dir',
        type=str,
        default='logging/validation_images'
        )
    parser.add_argument(
        '--testing_dir',
        type=str,
        default='logging/testing_images'
        )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='logging/model_saves'
        )
    parser.add_argument(
        '--load_checkpoint',
        type=bool,
        default=True
        )
    parser.add_argument(
        '--g_weights',
        type=str,
        default='logging/model_saves/stylegan2_generator_weights_70_3.755.h5'
        )
    parser.add_argument(
        '--d_weights',
        type=str,
        default='logging/model_saves/stylegan2_discriminator_weights_70_0.553.h5'
        )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000
        )
    parser.add_argument(
        '--n_cpu',
        type=int,
        default=N_CPU
        )
    parser.add_argument(
        '--n_gpu',
        type=int,
        default=2
        )

    # model parameters
    parser.add_argument(
        '--img_dim_x',
        type=int,
        default=512
        )
    parser.add_argument(
        '--img_dim_y',
        type=int,
        default=512
        )
    parser.add_argument(
        '--img_depth',
        type=int,
        default=3
        )
    parser.add_argument(
        '--z_len',
        type=int,
        default=256
        )
    parser.add_argument(
        '--n_classes',
        type=int,
        default=5
        )
    parser.add_argument(
        '--g_lr',
        type=float,
        default=1e-4
        )
    parser.add_argument(
        '--d_lr',
        type=float,
        default=1e-4
        )
    parser.add_argument(
        '--save_freq',
        type=int,
        default=2
        )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8
        )

    return parser.parse_args()

def main():
    args = parse_args()
    #gan = MiniGAN(
    gan = StyleGAN2(
        img_dim_x=args.img_dim_x,
        img_dim_y=args.img_dim_y,
        img_depth=args.img_depth,
        z_len=args.z_len,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        save_freq=args.save_freq,
        training_dir=args.training_dir,
        validation_dir=args.validation_dir,
        checkpoint_dir=args.checkpoint_dir,
        testing_dir=args.testing_dir,
        batch_size=args.batch_size,
        n_classes=args.n_classes,
        n_cpu=args.n_cpu,
        n_gpu=args.n_gpu
        )

    if args.load_checkpoint:
            gan.discriminator.load_weights(args.d_weights, by_name=True)
            gan.generator.load_weights(args.g_weights, by_name=True)
            print('Success - Model Checkpoint Loaded...')

    if args.train:
        gan.build_model()
        gan.train(args.epochs)

    else:
        gan.generate_samples(
            savedir='logging/testing_images',
            n_samples=2048,
            batch_size=args.batch_size,
            z_var=0.5
            )

if __name__ == '__main__':
    main()
