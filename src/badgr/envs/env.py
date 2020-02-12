import numpy as np

from badgr.utils.np_utils import imresize
from badgr.utils.python_utils import AttrDict


class EnvSpec(object):

    def __init__(self, names_shapes_limits_dtypes):
        names_shapes_limits_dtypes = list(names_shapes_limits_dtypes)
        names_shapes_limits_dtypes += [('done', (1,), (0, 1), np.bool)]

        self._names_to_shapes = AttrDict()
        self._names_to_limits = AttrDict()
        self._names_to_dtypes = AttrDict()
        for name, shape, limit, dtype in names_shapes_limits_dtypes:
            self._names_to_shapes.add_recursive(name, shape)
            self._names_to_limits.add_recursive(name, limit)
            self._names_to_dtypes.add_recursive(name, dtype)

    @property
    def observation_names(self):
        raise NotImplementedError

    @property
    def output_observation_names(self):
        return self.observation_names

    @property
    def action_names(self):
        raise NotImplementedError

    @property
    def names(self):
        return self.observation_names + self.action_names

    @property
    def names_to_shapes(self):
        return self._names_to_shapes

    @property
    def names_to_limits(self):
        return self._names_to_limits

    @property
    def names_to_dtypes(self):
        return self._names_to_dtypes

    def dims(self, names):
        return np.array([np.sum(self.names_to_shapes.get_recursive(name)) for name in names])

    def dim(self, names):
        return np.sum(self.dims(names))

    def normalize(self, inputs):
        """
        :param inputs (AttrDict):
        :return: AttrDict
        """
        inputs_normalized = AttrDict()
        for key, value in inputs.get_leaf_items():
            lower, upper = self.names_to_limits.get_recursive(key)

            lower, upper = np.array(lower), np.array(upper)
            mean = 0.5 * (lower + upper)
            std = 0.5 * (upper - lower)

            value_normalized = (value - mean) / std

            inputs_normalized.add_recursive(key, value_normalized)

        return inputs_normalized

    def denormalize(self, inputs):
        """
        :param inputs (AttrDict):
        :return: AttrDict
        """
        inputs_denormalized = AttrDict()
        for key, value in inputs.get_leaf_items():
            lower, upper = self.names_to_limits.get_recursive(key)

            lower, upper = np.array(lower), np.array(upper)
            mean = 0.5 * (lower + upper)
            std = 0.5 * (upper - lower)

            value_denormalized = value * std + mean

            inputs_denormalized.add_recursive(key, value_denormalized)

        return inputs_denormalized

    def process_image(self, name, image):
        """
        Default behavior: resize the image
        """
        if len(image.shape) == 4:
            return np.array([self.process_image(name, im_i) for im_i in image])

        return imresize(image, self.names_to_shapes.get_recursive(name))



class Env(object):

    def __init__(self, env_spec, params):
        self.spec = env_spec

    def step(self, get_action):
        raise NotImplementedError
        return obs, goal, done

    def reset(self):
        raise NotImplementedError
        return obs, goal
