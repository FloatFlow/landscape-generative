import argparse
import keras.backend as K
from model.vqvae2 import VQVAE2
import psutil
from tensorflow.python.client import device_lib

N_CPU = psutil.cpu_count()
N_GPU = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters for vae')

    # general parameters

    parser.add_argument(
        '--train',
        type=bool,
        default=True
        )
    parser.add_argument(
        '--train_phase',
        type=str,
        default='vae'
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
        default=False
        )
    parser.add_argument(
        '--g_weights',
        type=str,
        default='logging/model_saves/minivae_generator_weights_30_0.069.h5'
        )
    parser.add_argument(
        '--d_weights',
        type=str,
        default='logging/model_saves/minivae_discriminator_weights_30_0.901.h5'
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
        default=N_GPU
        )

    # model parameters
    parser.add_argument(
        '--img_width',
        type=int,
        default=512
        )
    parser.add_argument(
        '--img_height',
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
        '--lr',
        type=float,
        default=1e-4
        )
    parser.add_argument(
        '--save_freq',
        type=int,
        default=5
        )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
        )

    return parser.parse_args()

def main():
    args = parse_args()
    vae = VQVAE2(
        img_width=args.img_width,
        img_height=args.img_height,
        img_depth=args.img_depth,
        lr=args.lr,
        training_dir=args.training_dir,
        validation_dir=args.validation_dir,
        checkpoint_dir=args.checkpoint_dir,
        n_gpu=args.n_gpu
        )

    if args.load_checkpoint:
            vae.discriminator.load_weights(args.d_weights, by_name=True)
            vae.generator.load_weights(args.g_weights, by_name=True)
            print('Success - Model Checkpoint Loaded...')

    if args.train and args.train_phase == 'vae':
        vae.train(
            epochs=args.epochs,
            n_cpu=args.n_cpu,
            batch_size=args.batch_size
            )
    elif args.train and args.train_phase == 'pixelcnn':
        vae.train_pixelcnn(
            epochs=args.epochs,
            n_cpu=args.n_cpu,
            batch_size=args.batch_size
            )
    else:
        vae.predict_noise_testing(
            args.class_testing_labels,
            args.testing_dir
            )

if __name__ == '__main__':
    main()
