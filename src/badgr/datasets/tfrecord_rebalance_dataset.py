from loguru import logger
import numpy as np
import tensorflow as tf

from badgr.datasets.tfrecord_dataset import TfrecordDataset
from badgr.utils import file_utils


class TfrecordRebalanceDataset(TfrecordDataset):

    def __init__(self, params):
        self._rebalance_key = params.rebalance_key
        self._rebalance_logical_not = params.get('rebalance_logical_not', False)

        super(TfrecordRebalanceDataset, self).__init__(params)

    #############
    ### Setup ###
    #############

    def _load_tfrecords(self):
        logger.debug('Loading tfrecords...')
        for tfrecord_folder in sorted(self._tfrecord_folders):
            logger.debug(tfrecord_folder)
        tfrecord_fnames = file_utils.get_files_ending_with(sorted(self._tfrecord_folders), '.tfrecord')
        assert len(tfrecord_fnames) > 0
        np.random.shuffle(tfrecord_fnames)

        dataset = tf.data.TFRecordDataset(tfrecord_fnames)
        dataset = dataset.map(self._tfrecord_parse_fn, num_parallel_calls=self._num_parallel_calls)
        dataset = self._filter_out_input_nans(dataset)

        dataset_true = dataset.filter(lambda x: tf.reduce_any(
            tf.cast(x[self._rebalance_key], tf.bool) if not self._rebalance_logical_not else
            tf.logical_not(tf.cast(x[self._rebalance_key], tf.bool))))
        dataset_false = dataset.filter(lambda x: tf.logical_not(tf.reduce_any(
            tf.cast(x[self._rebalance_key], tf.bool) if not self._rebalance_logical_not else
            tf.logical_not(tf.cast(x[self._rebalance_key], tf.bool)))))

        dataset_true = dataset_true.apply(tf.data.experimental.shuffle_and_repeat(
            buffer_size=self._shuffle_buffer_size // 2))
        dataset_false = dataset_false.apply(tf.data.experimental.shuffle_and_repeat(
            buffer_size=self._shuffle_buffer_size // 2))

        dataset = tf.data.experimental.sample_from_datasets([dataset_true, dataset_false])
        dataset = dataset.batch(self._batch_size)

        dataset = dataset.prefetch(buffer_size=self._prefetch_buffer_size_multiplier * self._batch_size)

        iterator = dataset.make_one_shot_iterator()

        return iterator
