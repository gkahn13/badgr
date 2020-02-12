from loguru import logger
import numpy as np
import tensorflow as tf

from badgr.utils.python_utils import AttrDict


class MPPIplanner(object):

    def __init__(self, file_manager, params):
        self._file_manager = file_manager
        self._env_spec = params.env_spec
        self._action_selection_limits = params.action_selection_limits
        self._cost_fn = params.cost_fn

        lower, upper = [], []
        for name in self._env_spec.action_names:
            l, u = self._action_selection_limits.get_recursive(name)
            lower.append(np.ravel(l))
            upper.append(np.ravel(u))
        self._action_selection_lower_limits = np.ravel(list(zip(*lower))).astype(np.float32)
        self._action_selection_upper_limits = np.ravel(list(zip(*upper))).astype(np.float32)
        self._action_dim = self._env_spec.dim(self._env_spec.action_names)

        # MPPI params
        self._sigma = params.sigma
        self._N = params.N
        self._gamma = params.gamma
        self._beta = params.beta

        # static graph
        self._session = None
        self._obs_placeholders = None
        self._goal_placeholders = None
        self._mppi_mean_placeholder = None
        self._mppi_mean_np = None
        self._get_action_outputs = None

    def _split_action(self, action):
        """
        :param action (AttrDict):
        """
        d = AttrDict()

        idx = 0
        for name in self._env_spec.action_names:
            dim = np.sum(self._env_spec.names_to_shapes.get_recursive(name))
            value = action[..., idx:idx+dim]
            d.add_recursive(name, value)
            idx += dim

        return d

    def _setup_mppi_graph(self, model, goals):
        ### create placeholders
        obs_placeholders = AttrDict()
        for name in self._env_spec.observation_names:
            shape = list(self._env_spec.names_to_shapes.get_recursive(name))
            dtype = tf.as_dtype(self._env_spec.names_to_dtypes.get_recursive(name))
            ph = tf.placeholder(dtype, shape=shape, name=name)
            obs_placeholders.add_recursive(name, ph)

        goal_placeholders = AttrDict()
        for name, value in goals.get_leaf_items():
            goal_placeholders.add_recursive(name,
                                            tf.placeholder(tf.as_dtype(value.dtype),
                                                           shape=value.shape,
                                                           name=name))

        mppi_mean_placeholder = tf.placeholder(tf.float32,
                                               name='mppi_mean',
                                               shape=[model.horizon, self._action_dim])

        ### get obs lowd
        inputs = obs_placeholders.apply_recursive(lambda value: value[tf.newaxis])
        obs_lowd = model.get_obs_lowd(inputs)

        past_mean = mppi_mean_placeholder[0]
        shifted_mean = tf.concat([mppi_mean_placeholder[1:], mppi_mean_placeholder[-1:]], axis=0)

        # sample through time
        delta_limits = 0.5 * (self._action_selection_upper_limits - self._action_selection_lower_limits)
        eps = tf.random_normal(mean=0, stddev=self._sigma * delta_limits,
                               shape=(self._N, model.horizon, self._action_dim))
        actions = []
        for h in range(model.horizon):
            if h == 0:
                action_h = self._beta * (shifted_mean[h, :] + eps[:, h, :]) + (1. - self._beta) * past_mean
            else:
                action_h = self._beta * (shifted_mean[h, :] + eps[:, h, :]) + (1. - self._beta) * actions[-1]
            actions.append(action_h)
        actions = tf.stack(actions, axis=1)
        actions = tf.clip_by_value(
            actions,
            self._action_selection_lower_limits[np.newaxis, np.newaxis],
            self._action_selection_upper_limits[np.newaxis, np.newaxis]
        )

        # forward simulate
        actions_split = self._split_action(actions)
        inputs_tiled = inputs.apply_recursive(
            lambda v: tf.tile(v, [self._N] + [1] * (len(v.shape) - 1))
        )
        for k, v in actions_split.get_leaf_items():
            inputs_tiled.add_recursive(k, v)
        obs_lowd_tiled = tf.tile(obs_lowd, (self._N, 1))

        ### call model and evaluate cost
        model_outputs = model(inputs_tiled, obs_lowd=obs_lowd_tiled)
        model_outputs = model_outputs.filter_recursive(lambda key, value: key[0] != '_')
        costs_per_timestep = self._cost_fn(inputs_tiled, model_outputs, goal_placeholders, actions_split)
        costs = tf.reduce_mean(costs_per_timestep.total, axis=1)

        # MPPI update
        scores = -costs
        probs = tf.exp(self._gamma * (scores - tf.reduce_max(scores)))
        probs /= tf.reduce_sum(probs) + 1e-10
        new_mppi_mean = tf.reduce_sum(actions * probs[:, tf.newaxis, tf.newaxis], axis=0)

        best_idx = tf.argmin(costs)
        best_actions = self._split_action(new_mppi_mean)

        get_action_outputs = AttrDict(
            cost=costs[best_idx],
            cost_per_timestep=costs_per_timestep.apply_recursive(lambda v: v[best_idx]),
            action=best_actions.apply_recursive(lambda v: v[0]),
            action_sequence=best_actions,
            model_outputs=model_outputs.apply_recursive(lambda v: v[best_idx]),

            all_costs=costs,
            all_costs_per_timestep=costs_per_timestep,
            all_actions=actions_split,
            all_model_outputs=model_outputs,

            mppi_mean=new_mppi_mean,
        )

        for key, value in get_action_outputs.get_leaf_items():
            get_action_outputs.add_recursive(
                key,
                tf.identity(value, 'get_action_outputs/' + key)
            )

        return obs_placeholders, goal_placeholders, mppi_mean_placeholder, get_action_outputs

    def warm_start(self, model, inputs, goals):
        assert not tf.executing_eagerly()

        logger.debug('Setting up MPPI graph....')
        self._session = tf.get_default_session()
        assert self._session is not None

        self._obs_placeholders, self._goal_placeholders, self._mppi_mean_placeholder, self._get_action_outputs = \
            self._setup_mppi_graph(model, goals)
        self._mppi_mean_np = np.zeros([model.horizon, self._action_dim], dtype=np.float32)

        logger.debug('MPPI graph setup complete')

    def get_action(self, model, inputs, goals):
        assert self._session is not None

        feed_dict = {}
        for name, ph in self._obs_placeholders.get_leaf_items():
            value = np.array(inputs.get_recursive(name))
            if value.shape == tuple():
                value = value[np.newaxis]
            feed_dict[ph] = value
        for name, ph in self._goal_placeholders.get_leaf_items():
            feed_dict[ph] = np.array(goals.get_recursive(name))
        feed_dict[self._mppi_mean_placeholder] = self._mppi_mean_np

        get_action_tf = {}
        for name, tensor in self._get_action_outputs.get_leaf_items():
            get_action_tf[name] = tensor

        get_action_tf_output = self._session.run(get_action_tf, feed_dict=feed_dict)

        get_action = AttrDict()
        for name, value in get_action_tf_output.items():
            get_action.add_recursive(name, value)
        self._mppi_mean_np = get_action.mppi_mean

        return get_action
