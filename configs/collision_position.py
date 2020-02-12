import os
import tensorflow as tf

from badgr.file_manager import FileManager
from badgr.datasets.tfrecord_rebalance_dataset import TfrecordRebalanceDataset
from badgr.jackal.envs.jackal_env_specs import JackalPositionCollisionEnvSpec
from badgr.jackal.models.jackal_position_model import JackalPositionModel
from badgr.utils.python_utils import AttrDict as d



###############
### Dataset ###
###############

def get_dataset_params(env_spec, horizon, batch_size):
    all_tfrecord_folders = [
        os.path.join(FileManager.data_dir, 'tfrecords_collision/{0}-2019_horizon_{1}'.format(f, horizon)) for f in
        ['08-02', '08-06', '08-08', '08-13', '08-15', '08-18', '08-20', '08-27', '08-29', '09-09', '09-12', '09-17',
         '09-19', '10-20', '10-24', '10-31']
        ]
    train_tfrecord_folders = [fname for fname in all_tfrecord_folders if '09-12' not in fname],
    holdout_tfrecord_folders = [fname for fname in all_tfrecord_folders if '09-12' in fname],

    kwargs_train = d(
        rebalance_key='outputs/collision/close',

        env_spec=env_spec,
        tfrecord_folders=train_tfrecord_folders,

        horizon=horizon,
        batch_size=batch_size,

        num_parallel_calls=6,
        shuffle_buffer_size=1000,
        prefetch_buffer_size_multiplier=10,
    )

    kwargs_holdout = kwargs_train.copy()
    kwargs_holdout.tfrecord_folders = holdout_tfrecord_folders

    return d(
        cls=TfrecordRebalanceDataset,
        kwargs_train=kwargs_train,
        kwargs_holdout=kwargs_holdout

    )


#############
### Model ###
#############

def get_model_params(env_spec, horizon):
    kwargs_train = d(
            # jackal mode
            use_both_images=False,

            # RNN
            horizon=horizon,
            rnn_dim=64,

            # inputs/outputs
            env_spec=env_spec,
            output_observations=[
                d(
                    name='jackal/position',
                    is_relative=True
                ),
                d(
                    name='collision/close',
                    is_relative=False
                )
            ],

            is_output_gps=False,
        )

    kwargs_eval = kwargs_train.copy()
    kwargs_eval.is_output_gps = True

    return d(
        cls=JackalPositionModel,
        kwargs_train=kwargs_train,
        kwargs_eval=kwargs_eval
    )


################
### Training ###
################

def get_trainer_params():

    def cost_fn(model_outputs, outputs):
        batch_size = tf.shape(outputs.done)[0]
        batch_size_float = tf.cast(batch_size, tf.float32)

        done = tf.concat([tf.zeros([batch_size, 1], dtype=tf.bool), outputs.done[:, :-1]], axis=1)
        mask = tf.cast(tf.logical_not(done), tf.float32)
        tf.debugging.assert_positive(tf.reduce_sum(mask, axis=1))
        mask = batch_size_float * (mask / tf.reduce_sum(mask))
        mask = mask[..., tf.newaxis]

        ### position

        cost_position = tf.reduce_sum(
            mask * 0.5 * tf.square(model_outputs.jackal.position - outputs.jackal.position),
            axis=(1, 2)
        )

        ### collision

        model_output_collision = model_outputs.collision.close[..., 0]

        collision = tf.cast(outputs.collision.close, tf.bool)[..., 0]
        collision = tf.logical_and(collision, tf.logical_not(done)) # don't count collisions after done!
        collision = tf.cumsum(tf.cast(collision, tf.float32), axis=-1) > 0.5

        # collision mask should be same as normal mask, but turned on for dones with collision = true
        mask_collision = tf.cast(tf.logical_or(tf.logical_not(done), collision), tf.float32)
        mask_collision = batch_size_float * (mask_collision / tf.reduce_sum(mask_collision))

        cost_collision = 2.0 * tf.reduce_sum(
            mask_collision * tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(collision, tf.float32),
                                                                     logits=model_output_collision),
            axis=1
        )
        collision_accuracy = tf.reduce_mean(tf.cast(tf.equal(model_output_collision > 0,
                                                             tf.cast(collision, tf.bool)),
                                                    tf.float32),
                                            axis=1)
        collision_accuracy_random = tf.reduce_mean(1. - tf.cast(collision, tf.float32), axis=1)

        ### regularization

        cost_l2_reg = 1e-2 * \
                      tf.reduce_mean([0.5 * tf.reduce_mean(kernel * kernel) for kernel in model_outputs.kernels]) * \
                      tf.ones(batch_size)

        ### filter out nans

        costs_is_finite = tf.logical_and(tf.is_finite(cost_position), tf.is_finite(cost_collision))
        cost_position = tf.boolean_mask(cost_position, costs_is_finite)
        cost_collision = tf.boolean_mask(cost_collision, costs_is_finite)
        cost_l2_reg = tf.boolean_mask(cost_l2_reg, costs_is_finite)
        # assert cost_l2_reg.shape[0].value > 0.5 * batch_size

        ### total

        cost = cost_position + cost_collision + cost_l2_reg

        return d(
            total=cost,
            position=cost_position,
            collision=cost_collision,
            collision_accuracy=collision_accuracy,
            collision_accuracy_random=collision_accuracy_random,
            l2_reg=cost_l2_reg
        )


    return d(
        # steps
        max_steps=2e5,
        holdout_every_n_steps=50,
        log_every_n_steps=1e3,
        save_every_n_steps=1e4,

        # dataset
        batch_size=32,

        # optimizer
        cost_fn=cost_fn,
        optimizer_cls=tf.train.AdamOptimizer,
        learning_rate=1e-4,
    )

##################
### Get params ###
##################

def get_params():
    horizon = 8

    env_spec = JackalPositionCollisionEnvSpec(left_image_only=True)

    model_params = get_model_params(env_spec, horizon)
    trainer_params = get_trainer_params()
    dataset_params = get_dataset_params(env_spec, horizon, trainer_params.batch_size)

    return d(
        exp_name='collision_position',

        dataset=dataset_params,
        model=model_params,
        trainer=trainer_params,
    )

params = get_params()
