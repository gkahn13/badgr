import argparse
import os

from badgr.file_manager import FileManager
from badgr.trainers.trainer import Trainer
from badgr.utils import tf_utils
from badgr.utils.python_utils import import_config

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--continue', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpu_frac', type=float, default=0.3)
args = parser.parse_args()

config_fname = os.path.abspath(args.config)
assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
params = import_config(config_fname)

tf_utils.enable_static_execution(gpu=args.gpu, gpu_frac=args.gpu_frac)

file_manager = FileManager(params.exp_name,
                           is_continue=getattr(args, 'continue'),
                           log_fname='log_train.txt',
                           config_fname=config_fname)
dataset_train = params.dataset.cls(params.dataset.kwargs_train)
dataset_holdout = params.dataset.cls(params.dataset.kwargs_holdout)
model = params.model.cls(params.model.kwargs_train)

trainer = Trainer(params.trainer, file_manager, model, dataset_train, dataset_holdout)
trainer.run()
