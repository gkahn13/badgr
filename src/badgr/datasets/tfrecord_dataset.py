from loguru import logger
import numpy as np
import tensorflow as tf

from badgr.datasets.dataset import Dataset
from badgr.utils import file_utils
from badgr.utils.python_utils import AttrDict


class TfrecordDataset(Dataset):

    def __init__(self, params):
        assert not tf.executing_eagerly()
        super(TfrecordDataset, self).__init__(env_spec=params.env_spec)

        self._tfrecord_folders = params.tfrecord_folders

        self._horizon = params.horizon
        self._batch_size = params.batch_size

        self._is_shuffle_and_repeat = params.get('is_shuffle_and_repeat', True)
        self._num_parallel_calls = params.num_parallel_calls
        self._shuffle_buffer_size = params.shuffle_buffer_size
        self._prefetch_buffer_size_multiplier = params.prefetch_buffer_size_multiplier

        self._iterator = self._load_tfrecords()
        self._static_inputs_and_outputs = None

    #############
    ### Setup ###
    #############

    def _tfrecord_parse_fn(self, dataset):
        names = ['inputs/' + name for name in self._env_spec.observation_names + self._env_spec.action_names] + \
                ['outputs/' + name for name in self._env_spec.output_observation_names]

        dtypes = [
            tf.dtypes.as_dtype(self._env_spec.names_to_dtypes.get_recursive(
                name.replace('inputs/', '').replace('outputs/', '')))
            for name in names
        ]
        dtypes = [dtype if dtype != tf.bool else tf.uint8 for dtype in dtypes]

        shapes = []
        for name in names:
            name_suffix = name.replace('inputs/', '').replace('outputs/', '')
            shape = list(self._env_spec.names_to_shapes.get_recursive(name_suffix))
            if name.startswith('outputs/') or name_suffix in self._env_spec.action_names:
                shape = [self._horizon] + shape
            shapes.append(shape)

        names.append('outputs/done')
        dtypes.append(tf.uint8)
        shapes.append((self._horizon,))

        parsed = tf.parse_single_example(
            dataset,
            {name: tf.FixedLenFeature([], tf.string) for name in names}
        )
        decoded_parsed = {name: tf.decode_raw(parsed[name], dtype) for name, dtype in zip(names, dtypes)}

        reshaped_decoded_parsed = dict()
        for name, shape in zip(names, shapes):
            tensor = decoded_parsed[name]
            tensor.set_shape([np.prod(shape)])
            tensor = tf.reshape(tensor, shape)
            reshaped_decoded_parsed[name] = tensor

        reshaped_decoded_parsed['outputs/done'] = tf.cast(reshaped_decoded_parsed['outputs/done'], tf.bool)

        # randomize actions after done
        done_float = tf.cast(reshaped_decoded_parsed['outputs/done'], tf.float32)[:, tf.newaxis]
        for name in self._env_spec.action_names:
            lower, upper = self._env_spec.names_to_limits.get_recursive(name)
            shape = self._env_spec.names_to_shapes.get_recursive(name)
            action = reshaped_decoded_parsed['inputs/' + name]
            horizon = action.shape[0].value
            action = (1 - done_float) * action + done_float * \
                                                 tf.random.uniform(shape=[horizon] + list(shape),
                                                                   minval=lower, maxval=upper)
            reshaped_decoded_parsed['inputs/' + name] = action

        return reshaped_decoded_parsed

    def _filter_out_input_nans(self, dataset):
        return dataset.filter(
            lambda x: tf.reduce_all([tf.reduce_all(tf.is_finite(tensor)) for key, tensor in x.items()
                                     if key.startswith('inputs/') and tensor.dtype != tf.uint8]))

    def _load_tfrecords(self):
        logger.debug('Loading tfrecords...')
        for tfrecord_folder in self._tfrecord_folders:
            logger.debug(tfrecord_folder)
        tfrecord_fnames = file_utils.get_files_ending_with(self._tfrecord_folders, '.tfrecord')
        assert len(tfrecord_fnames) > 0
        if self._is_shuffle_and_repeat:
            np.random.shuffle(tfrecord_fnames)
        else:
            tfrecord_fnames = sorted(tfrecord_fnames)

        dataset = tf.data.TFRecordDataset(tfrecord_fnames)
        dataset = dataset.map(self._tfrecord_parse_fn)
        dataset = self._filter_out_input_nans(dataset)
        if self._is_shuffle_and_repeat:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=self._shuffle_buffer_size))
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(buffer_size=self._prefetch_buffer_size_multiplier * self._batch_size)

        iterator = dataset.make_one_shot_iterator()

        return iterator

    ################
    ### Get data ###
    ################

    def get_batch(self, batch_size, horizon):
        if self._static_inputs_and_outputs is None:
            self._static_inputs_and_outputs = self._iterator.get_next()
        inputs_and_outputs = self._static_inputs_and_outputs

        inputs = AttrDict()
        outputs = AttrDict()
        for key, value in inputs_and_outputs.items():
            if key.startswith('inputs/'):
                inputs.add_recursive(key.replace('inputs/', ''), value)
            else:
                outputs.add_recursive(key.replace('outputs/', ''), value)

        return inputs, outputs