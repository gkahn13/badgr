from loguru import logger
import os
import tensorflow as tf


def config_gpu(gpu=0, gpu_frac=0.3):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
    return config


def enable_static_execution(gpu=0, gpu_frac=0.3):
    graph = tf.Graph()
    session = tf.Session(graph=graph, config=config_gpu(gpu=gpu, gpu_frac=gpu_frac))
    session.__enter__() # so get default session works


def yaw_rotmat(yaw):
    batch_size = tf.shape(yaw)[0]
    return tf.reshape(
        tf.stack([tf.cos(yaw), -tf.sin(yaw), tf.zeros(batch_size),
                  tf.sin(yaw), tf.cos(yaw), tf.zeros(batch_size),
                  tf.zeros(batch_size), tf.zeros(batch_size), tf.ones(batch_size)],
                 axis=1),
        (batch_size, 3, 3)
    )


def rotate_to_global(curr_position, curr_yaw, local_position):
    """
    :param curr_position (tensor): [batch, 3]
    :param curr_yaw (tensor): [batch]
    :param local_position (tensor): [batch, H, 3]
    :return: [batch, H, 3]
    """
    return tf.matmul(local_position, yaw_rotmat(-curr_yaw)) + curr_position[:, tf.newaxis]


def restore_checkpoint(ckpts_dir, model, ckptnum=None):
    if ckptnum is None:
        ckpt_fname = tf.train.latest_checkpoint(ckpts_dir)
    else:
        ckpt_fname = os.path.join(ckpts_dir, 'ckpt-{0:d}'.format(ckptnum))
    logger.debug('Restoring ckpt {0}'.format(ckpt_fname))
    assert tf.train.checkpoint_exists(ckpt_fname)
    checkpointer = tf.train.Checkpoint(model=model)
    status = checkpointer.restore(ckpt_fname)
    if not tf.executing_eagerly():
        status.initialize_or_restore(tf.get_default_session())


def get_kernels(layers):
    kernels = []
    for layer in layers:
        if hasattr(layer, 'layers'):
            kernels += get_kernels(layer.layers)
        elif hasattr(layer, 'kernel'):
            kernels.append(layer.kernel)
    return kernels
