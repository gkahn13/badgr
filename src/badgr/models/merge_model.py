import numpy as np
import os
import tensorflow as tf

from badgr.file_manager import FileManager
from badgr.models.model import Model
from badgr.utils.python_utils import AttrDict, import_config
from badgr.utils import tf_utils


class MergeModel(Model):

    def __init__(self, params):
        tf.keras.Model.__init__(self)

        self._config_fnames = params.config_fnames
        self._ckptnums = params.ckptnums

        self._models = []
        self._ckpts_dirs = []
        for config_fname in self._config_fnames:
            assert os.path.exists(config_fname), '{0} does not exist'.format(config_fname)
            config_params = import_config(config_fname)

            config_file_manager = FileManager(config_params.exp_name, is_continue=True, add_logger=False)
            self._ckpts_dirs.append(config_file_manager.ckpts_dir)

            config_model = config_params.model.cls(config_params.model.kwargs_eval)
            self._models.append(config_model)

        self._obs_lowd_dims = None # NOTE(greg): will be filled in by get_obs_lowd

    @property
    def horizon(self):
        horizons = np.array([model.horizon for model in self._models])
        assert np.all(horizons == horizons[0])
        return horizons[0]

    ############
    ### Call ###
    ############

    def get_obs_lowd(self, inputs, training=False):
        obs_lowds = [model.get_obs_lowd(inputs, training=training) for model in self._models]
        if self._obs_lowd_dims is None:
            self._obs_lowd_dims = [obs_lowd_i.shape.as_list()[-1] for obs_lowd_i in obs_lowds]
        return tf.concat(obs_lowds, axis=-1)

    def call(self, inputs, obs_lowd=None, training=False):
        if obs_lowd is None:
            obs_lowd = [None] * len(self._models)
        else:
            obs_lowd = tf.split(obs_lowd, self._obs_lowd_dims, axis=-1)

        outputs = AttrDict()
        for model_i, obs_lowd_i in zip(self._models, obs_lowd):
            outputs_i = model_i(inputs, obs_lowd=obs_lowd_i, training=training)
            for key, value in outputs_i.get_leaf_items():
                outputs.add_recursive(key, value)

        return outputs

    ###############
    ### Restore ###
    ###############

    def restore(self, **kwargs):
        for model, ckpts_dir, ckptnum in zip(self._models, self._ckpts_dirs, self._ckptnums):
            tf_utils.restore_checkpoint(ckpts_dir, model, ckptnum=ckptnum)
