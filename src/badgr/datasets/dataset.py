class Dataset(object):

    def __init__(self, env_spec):
        self._env_spec = env_spec

    def get_batch(self, batch_size, horizon):
        raise NotImplementedError

    def get_batch_iterator(self, batch_size, horizon, randomize_order=False, is_tf=True):
        raise NotImplementedError
