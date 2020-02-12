from collections import defaultdict
import numpy as np
import tensorflow as tf
from loguru import logger

from badgr.datasets.tfrecord_dataset import TfrecordDataset
from badgr.utils.python_utils import timeit


class Trainer(object):

    def __init__(self, params, file_manager, model, dataset_train, dataset_holdout):
        assert isinstance(dataset_train, TfrecordDataset)
        assert isinstance(dataset_holdout, TfrecordDataset)

        self._file_manager = file_manager
        self._model = model
        self._dataset_train = dataset_train
        self._dataset_holdout= dataset_holdout

        # steps
        self._max_steps = int(params.max_steps)
        self._holdout_every_n_steps = int(params.holdout_every_n_steps)
        self._log_every_n_steps = int(params.log_every_n_steps)
        self._save_every_n_steps = int(params.save_every_n_steps)

        # dataset
        self._batch_size = params.batch_size

        # optimizer
        self._cost_fn = params.cost_fn
        self._optimizer_cls = params.optimizer_cls
        self._learning_rate = params.learning_rate

        # create optimizer and checkpoint
        self._optimizer = self._optimizer_cls(self._learning_rate)
        self._global_step = tf.train.get_or_create_global_step()
        self._checkpointer = tf.train.Checkpoint(optimizer=self._optimizer,
                                                 model=model,
                                                 optimizer_step=self._global_step)

        self._train_cost_dict_tensors, self._train_op = self._setup_model(
            self._dataset_train, is_train=True, summary_prefix='train_')
        self._holdout_cost_dict_tensors, _ = self._setup_model(
            self._dataset_holdout, is_train=False, summary_prefix='holdout_')

        # tensorboard logging
        self._tb_logger = defaultdict(list)
        self._summary_op = tf.summary.merge_all()
        self._summary_writer = tf.summary.FileWriter(file_manager.exp_dir, graph=tf.get_default_graph())

        # session and init
        self._session = tf.get_default_session()
        self._session.run(tf.global_variables_initializer())

    def _setup_model(self, dataset, is_train, summary_prefix=''):
        inputs, outputs = dataset.get_batch(self._batch_size, self._model.horizon)
        model_outputs = self._model.call(inputs, training=True)
        cost_dict_tensors = self._cost_fn(model_outputs, outputs)
        if is_train:
            train_op = self._optimizer.minimize(tf.reduce_mean(cost_dict_tensors.total),
                                                      global_step=self._global_step,
                                                      var_list=self._model.trainable_variables)
        else:
            train_op = None

        for name, cost_tensor in cost_dict_tensors.items():
            tf.summary.scalar(summary_prefix + 'cost_' + name, tf.reduce_mean(cost_tensor))

        return cost_dict_tensors, train_op

    def run(self):
        # restore checkpoint
        latest_ckpt_fname = tf.train.latest_checkpoint(self._file_manager.ckpts_dir)
        if latest_ckpt_fname:
            logger.info('Restoring ckpt {0}'.format(latest_ckpt_fname))
            self._checkpointer.restore(latest_ckpt_fname)
            logger.info('Starting training from step = {0}'.format(self._get_global_step_value()))

        for step in range(self._get_global_step_value(), self._max_steps + 1):
            with timeit('total'):
                self._train_step()

                if step > 0 and step % self._holdout_every_n_steps == 0:
                    self._holdout_step()

                # save
                if step > 0 and step % self._save_every_n_steps == 0:
                    with timeit('save'):
                        self._checkpointer.save(self._file_manager.ckpt_prefix)

            # log
            if step > 0 and step % self._log_every_n_steps == 0:
                self._log()

    def _get_global_step_value(self):
        return self._session.run(self._global_step)

    def _train_step(self):
        with timeit('train'):
            cost_dict, summary, global_step, _ = self._session.run([
                self._train_cost_dict_tensors, self._summary_op, self._global_step, self._train_op])
            self._summary_writer.add_summary(summary, global_step)

        for name, cost_tensor in cost_dict.items():
            self._tb_logger['train_cost_' + name] += cost_tensor.tolist()

    def _holdout_step(self):
        with timeit('holdout'):
            cost_dict, summary, global_step = self._session.run([
                self._holdout_cost_dict_tensors, self._summary_op, self._global_step])
            self._summary_writer.add_summary(summary, global_step)

        for name, cost_tensor in cost_dict.items():
            self._tb_logger['holdout_cost_' + name] += cost_tensor.tolist()

    def _log(self):
        logger.info('')
        logger.info('Step {0}'.format(self._get_global_step_value() - 1))
        for key, value in sorted(self._tb_logger.items(), key=lambda kv: kv[0]):
            logger.info('{0} {1:.6f}'.format(key, np.mean(value)))
        self._tb_logger.clear()

        for line in str(timeit).split('\n'):
            logger.debug(line)
        timeit.reset()
