import os
import sys
import logging
import argparse
import configs
from wae import WAE
import improved_wae
from datahandler import DataHandler
import utils
import datetime
import fideval

import neptune

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument("--exp", default='mnist_small',
                    help='dataset [mnist/celebA/dsprites]')
parser.add_argument("--zdim",
                    help='dimensionality of the latent space',
                    type=int)
parser.add_argument("--lr",
                    help='ae learning rate',
                    type=float)
parser.add_argument("--w_aef",
                    help='weight of ae fixedpoint cost',
                    type=float)
parser.add_argument("--z_test",
                    help='method of choice for verifying Pz=Qz [mmd/gan]')
parser.add_argument("--pz",
                    help='Prior latent distribution [normal/sphere/uniform]')
parser.add_argument("--wae_lambda", help='WAE regularizer', type=float)
parser.add_argument("--work_dir")
parser.add_argument("--lambda_schedule",
                    help='constant or adaptive')
parser.add_argument("--enc_noise",
                    help="type of encoder noise:"\
                         " 'deterministic': no noise whatsoever,"\
                         " 'gaussian': gaussian encoder,"\
                         " 'implicit': implicit encoder,"\
                         " 'add_noise': add noise before feeding "\
                         "to deterministic encoder")
parser.add_argument("--mode", default='train',
                    help='train or test')
parser.add_argument("--checkpoint",
                    help='full path to the checkpoint file without extension')

parser.add_argument('--sinkhorn_epsilon', dest='sinkhorn_epsilon', type=float, default=0.01, help='The epsilon for entropy regularized Sinkhorn')
parser.add_argument('--sinkhorn_iters', dest='sinkhorn_iters', type=int, default=10, help='Sinkhorn rollout length')
parser.add_argument('--train_size', dest='train_size', type=int, default=None, help='Truncates train set to train_size')
parser.add_argument('--nat_size', dest='nat_size', type=int, default=None, help='NAT size')
parser.add_argument('--nat_resampling', dest='nat_resampling', type=str, default=None, help='NAT resampling mode, can be: epoch, batch or None')
parser.add_argument('--ot_lambda', dest='ot_lambda', type=float, default=1.0, help='Lambda for NAT OT loss')
parser.add_argument('--rec_lambda', dest='rec_lambda', type=float, default=1.0, help='Lambda for reconstruction loss')
parser.add_argument('--zxz_lambda', dest='zxz_lambda', type=float, default=0.0, help='Lambda for zxz loss')
parser.add_argument('--name', dest='name', type=str, default="experiment", help='Name of the experiment')
parser.add_argument('--epoch_num', dest='epoch_num', type=int, default=30, help='Number of epochs to train for')
parser.add_argument('--e_pretrain', dest='e_pretrain', type=str2bool, default=True, help='Pretrain or not.')
parser.add_argument('--tags', dest='tags', type=str, default="junk", help='Tags for the experiment (comma separated)')
parser.add_argument('--shuffle', dest='shuffle', type=str2bool, default=True, help='Shuffle train set when training')

FLAGS = parser.parse_args()

now = datetime.datetime.now()
FLAGS.name = FLAGS.name + now.strftime("%Y%m%d_%H%M%S%f")


def main():

    if FLAGS.exp == 'celebA':
        opts = configs.config_celebA
    elif FLAGS.exp == 'celebA_small':
        opts = configs.config_celebA_small
    elif FLAGS.exp == 'mnist':
        opts = configs.config_mnist
    elif FLAGS.exp == 'mnist_ord':
        opts = configs.config_mnist_ord
    elif FLAGS.exp == 'mnist_small':
        opts = configs.config_mnist_small
    elif FLAGS.exp == 'dsprites':
        opts = configs.config_dsprites
    elif FLAGS.exp == 'grassli':
        opts = configs.config_grassli
    elif FLAGS.exp == 'grassli_small':
        opts = configs.config_grassli_small
    else:
        assert False, 'Unknown experiment configuration'

    opts['mode'] = FLAGS.mode
    if opts['mode'] == 'test':
        assert FLAGS.checkpoint is not None, 'Checkpoint must be provided'
        opts['checkpoint'] = FLAGS.checkpoint

    if FLAGS.zdim is not None:
        opts['zdim'] = FLAGS.zdim
    if FLAGS.pz is not None:
        opts['pz'] = FLAGS.pz
    if FLAGS.lr is not None:
        opts['lr'] = FLAGS.lr
    if FLAGS.w_aef is not None:
        opts['w_aef'] = FLAGS.w_aef
    if FLAGS.z_test is not None:
        opts['z_test'] = FLAGS.z_test
    if FLAGS.lambda_schedule is not None:
        opts['lambda_schedule'] = FLAGS.lambda_schedule
    if FLAGS.work_dir is not None:
        opts['work_dir'] = FLAGS.work_dir
    if FLAGS.wae_lambda is not None:
        opts['lambda'] = FLAGS.wae_lambda
    if FLAGS.enc_noise is not None:
        opts['e_noise'] = FLAGS.enc_noise


    if FLAGS.ot_lambda is not None:
        opts['ot_lambda'] = FLAGS.ot_lambda
    if FLAGS.rec_lambda is not None:
        opts['rec_lambda'] = FLAGS.rec_lambda
    if FLAGS.zxz_lambda is not None:
        opts['zxz_lambda'] = FLAGS.zxz_lambda
    if FLAGS.train_size is not None:
        opts['train_size'] = FLAGS.train_size
    if FLAGS.nat_size is not None:
        opts['nat_size'] = FLAGS.nat_size
    else:
        opts['nat_size'] = FLAGS.train_size
    opts['nat_resampling'] = FLAGS.nat_resampling

    if FLAGS.sinkhorn_iters is not None:
        opts['sinkhorn_iters'] = FLAGS.sinkhorn_iters
    if FLAGS.sinkhorn_epsilon is not None:
        opts['sinkhorn_epsilon'] = FLAGS.sinkhorn_epsilon
    if FLAGS.name is not None:
        opts['name'] = FLAGS.name
    if FLAGS.tags is not None:
        opts['tags'] = FLAGS.tags
    if FLAGS.epoch_num is not None:
        opts['epoch_num'] = FLAGS.epoch_num
    if FLAGS.e_pretrain is not None:
        opts['e_pretrain'] = FLAGS.e_pretrain
    if FLAGS.shuffle is not None:
        opts['shuffle'] = FLAGS.shuffle

    if opts['verbose']:
        pass
        #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    utils.create_dir(opts['work_dir'])
    utils.create_dir(os.path.join(opts['work_dir'],
                     'checkpoints'))

    if opts['e_noise'] == 'gaussian' and opts['pz'] != 'normal':
        assert False, 'Gaussian encoders compatible only with Gaussian prior'
        return

    # Dumping all the configs to the text file
    with utils.o_gfile((opts['work_dir'], 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))

    # Loading the dataset
    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    if opts['train_size'] is not None:
        train_size = opts['train_size']
    else:
        train_size = data.num_points

    if "NEPTUNE_API_TOKEN" in os.environ:
        neptune.init(project_qualified_name="csadrian/global-sinkhorn")
        exp = neptune.create_experiment(params=opts, name=opts['name'])

        for tag in opts['tags'].split(','):
            neptune.append_tag(tag)


    if opts['mode'] == 'train':

        # Creating WAE model
        wae = WAE(opts, train_size)
        data.num_points = train_size
        
        # Training WAE
        wae.train(data)

    elif opts['mode'] == 'test':

        # Do something else
        improved_wae.improved_sampling(opts)

    elif opts['mode'] == 'generate':
        fideval.generate(opts)

    if "NEPTUNE_API_TOKEN" in os.environ:
        exp.stop()

main()
