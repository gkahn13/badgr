import tensorflow as tf

from badgr.utils.python_utils import AttrDict
from badgr.utils import tf_utils


class Model(tf.keras.Model):

    def __init__(self, params):
        super(Model, self).__init__()

        self._horizon = params.horizon
        self._rnn_dim = params.rnn_dim

        self._env_spec = params.env_spec
        self._output_observations = tf.contrib.checkpoint.NoDependency(params.output_observations)

        self._output_dim = self._env_spec.dim(
            [output_observation.name for output_observation in self._output_observations])

        self._setup_layers()

    def _setup_layers(self):
        self._obs_im_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu', name='obs_im/conv0'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', name='obs_im/conv1'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', name='obs_im/conv2'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', name='obs_im/dense0'),
            tf.keras.layers.Dense(128, activation=None, name='obs_im/dense1'),
        ])

        self._obs_vec_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', name='obs_vec/dense0'),
            tf.keras.layers.Dense(32, activation=None, name='obs_vec/dense1'),
        ])

        self._obs_lowd_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', name='obs_lowd/dense0'),
            tf.keras.layers.Dense(2 * self._rnn_dim, activation=None, name='obs_lowd/dense1'),
        ])

        self._action_model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', name='action/dense0'),
            tf.keras.layers.Dense(16, activation=None, name='action/dense1'),
        ])

        self._rnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(self.horizon, self._rnn_dim)

        self._output_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', name='output/dense0'),
            tf.keras.layers.Dense(self._output_dim, activation=None, name='output/dense1'),
        ])

    @property
    def horizon(self):
        return self._horizon

    ############
    ### Call ###
    ############

    def _process_inputs(self, inputs):
        """
        Separate out observations/actions and normalize
        
        :param inputs (AttrDict): 
        :return: obs_ims (AttrDict), obs_vecs (AttrDict), actions (AttrDict)
        """
        ### separate out observations/actions

        # normalization
        inputs.modify_recursive(lambda arr: tf.cast(arr, tf.float32))
        normalized_inputs = self._env_spec.normalize(inputs)

        obs_ims = AttrDict()
        obs_vecs = AttrDict()
        for key in self._env_spec.observation_names:
            value = normalized_inputs.get_recursive(key)
            if len(value.shape) == 1:
                obs_vecs.add_recursive(key, value[:, tf.newaxis])
            elif len(value.shape) == 2:
                obs_vecs.add_recursive(key, value)
            elif len(value.shape) == 4:
                obs_ims.add_recursive(key, value)
            else:
                raise ValueError

        actions = AttrDict()
        for key in self._env_spec.action_names:
            value = normalized_inputs.get_recursive(key)
            if len(value.shape) == 2:
                value = value[..., tf.newaxis]
            elif len(value.shape) == 3:
                pass
            else:
                raise ValueError
            actions.add_recursive(key, value)

        return obs_ims, obs_vecs, actions

    def _preprocess_observation_inputs(self, inputs):
        ### separate out observations
        obs_inputs = inputs.filter_recursive(lambda key, value: key in self._env_spec.observation_names)

        # normalization
        obs_inputs.modify_recursive(lambda arr: tf.cast(arr, tf.float32))
        normalized_inputs = self._env_spec.normalize(obs_inputs)

        obs_ims = AttrDict()
        obs_vecs = AttrDict()
        for key, value in normalized_inputs.get_leaf_items():
            value = normalized_inputs.get_recursive(key)
            if len(value.shape) == 1:
                obs_vecs.add_recursive(key, value[:, tf.newaxis])
            elif len(value.shape) == 2:
                obs_vecs.add_recursive(key, value)
            elif len(value.shape) == 4:
                obs_ims.add_recursive(key, value)
            else:
                raise ValueError

        return obs_ims, obs_vecs

    def _preprocess_action_inputs(self, inputs):
        ### separate out actions
        action_inputs = inputs.filter_recursive(lambda key, value: key in self._env_spec.action_names)

        # normalization
        action_inputs.modify_recursive(lambda arr: tf.cast(arr, tf.float32))
        normalized_inputs = self._env_spec.normalize(action_inputs )

        actions = AttrDict()
        for key in self._env_spec.action_names:
            value = normalized_inputs.get_recursive(key)
            if len(value.shape) == 2:
                value = value[..., tf.newaxis]
            elif len(value.shape) == 3:
                pass
            else:
                raise ValueError
            actions.add_recursive(key, value)

        return actions

    def get_obs_lowd(self, inputs, training=False):
        obs_ims, obs_vecs = self._preprocess_observation_inputs(inputs)

        ### stack them
        obs_ims_concat= tf.concat(
            [obs_ims.get_recursive(key) for key in sorted(obs_ims.get_leaf_keys())],
            axis=-1
        )
        obs_vecs_concat = tf.concat(
            [obs_vecs.get_recursive(key) for key in sorted(obs_vecs.get_leaf_keys())],
            axis=-1
        )

        ### network

        # observations
        obs_im_lowd = self._obs_im_model(obs_ims_concat, training=training)
        obs_vec_lowd = self._obs_vec_model(obs_vecs_concat, training=training)
        obs_lowd = self._obs_lowd_model(tf.concat([obs_im_lowd, obs_vec_lowd], axis=1), training=training)

        return obs_lowd

    def _get_preprocess_outputs(self, obs_lowd, actions, training=False):
        ### actions
        actions_concat = tf.concat(
            [actions.get_recursive(key) for key in sorted(actions.get_leaf_keys())],
            axis=-1
        )
        actions_lowd = self._action_model(actions_concat, training=training)

        # rnn
        initial_state_c, initial_state_h = tf.split(obs_lowd, 2, axis=1)
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_state_c[tf.newaxis], initial_state_h[tf.newaxis])
        actions_lowd_time_major = tf.transpose(actions_lowd, (1, 0, 2))
        rnn_outputs_time_major, _ = self._rnn_cell(actions_lowd_time_major, initial_state=initial_state)
        rnn_outputs = tf.transpose(rnn_outputs_time_major, (1, 0, 2))

        # outputs
        outputs_concat = self._output_model(rnn_outputs, training=training)
        
        return outputs_concat

    def _get_outputs(self, preprocess_outputs, inputs, denormalize=True):
        """
        Split the outputs into each prediction component and denormalize
        
        :param preprocess_outputs (tensor): [batch_size, horizon, dim]
        :return: AttrDict
        """
        ### split and denormalize
        outputs_denormalized = AttrDict()
        start_idx = 0
        for output_observation in self._output_observations:
            name = output_observation.name

            shape = self._env_spec.names_to_shapes.get_recursive(name)

            assert len(shape) == 1, 'Can only predict vector quantities'
            dim = shape[0]

            outputs_slice_denormalized = preprocess_outputs[..., start_idx:start_idx+dim]
            outputs_denormalized.add_recursive(name, outputs_slice_denormalized)

            start_idx += dim

        outputs = self._env_spec.denormalize(outputs_denormalized) if denormalize else outputs_denormalized

        ### make relative
        for output_observation in self._output_observations:
            name = output_observation.name
            is_relative = output_observation.is_relative

            if is_relative:
                value = outputs.get_recursive(name)
                value += inputs.get_recursive(name)[:, tf.newaxis, :]
                outputs.add_recursive(name, value)

        return outputs

    def call(self, inputs, obs_lowd=None, training=False):
        """
        :param inputs (AttrDict):
        :param training (bool):
        :return: (AttrDict)
        """
        obs_lowd = obs_lowd if obs_lowd is not None else self.get_obs_lowd(inputs, training=training)
        actions = self._preprocess_action_inputs(inputs)
        preprocess_outputs = self._get_preprocess_outputs(obs_lowd, actions, training=training)
        outputs = self._get_outputs(preprocess_outputs, inputs)
        if training:
            outputs.kernels = tf_utils.get_kernels(self.layers)
        return outputs

    ###############
    ### Restore ###
    ###############

    def restore(self, ckpts_dir, ckptnum=None):
        tf_utils.restore_checkpoint(ckpts_dir, model=self, ckptnum=ckptnum)
