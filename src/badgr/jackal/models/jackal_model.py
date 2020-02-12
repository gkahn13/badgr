import tensorflow as tf

from badgr.models.model import Model


class JackalModel(Model):

    def __init__(self, params):
        self._use_both_images = params.get('use_both_images', True)

        super(JackalModel, self).__init__(params)

    def get_obs_lowd(self, inputs, training=False):
        obs_ims, obs_vecs = self._preprocess_observation_inputs(inputs)

        rgb_left = obs_ims.get_recursive('images/rgb_left')
        if self._use_both_images:
            rgb_right = obs_ims.get_recursive('images/rgb_right')
            obs_ims_concat = tf.concat([rgb_left, rgb_left - rgb_right], axis=-1)
        else:
            obs_ims_concat = rgb_left

        ### network

        # observations
        obs_im_lowd = self._obs_im_model(obs_ims_concat, training=training)
        obs_lowd = self._obs_lowd_model(obs_im_lowd, training=training)

        return obs_lowd

    def _get_outputs(self, preprocess_outputs, inputs):
        return Model._get_outputs(self, preprocess_outputs, inputs, denormalize=False)
