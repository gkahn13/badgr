import numpy as np
import os
import tensorflow as tf

from badgr.file_manager import FileManager
from badgr.jackal.envs.jackal_env import JackalEnv
from badgr.jackal.envs.jackal_env_specs import JackalPositionCollisionEnvSpec
from badgr.models.merge_model import MergeModel
from badgr.planner.mppi_planner import MPPIplanner
from badgr.utils.python_utils import AttrDict as d


#############
### Model ###
#############

def get_model_params():
    kwargs_train = d(
            config_fnames=(
                os.path.join(FileManager.configs_dir, 'bumpy.py'),
                os.path.join(FileManager.configs_dir, 'collision_position.py'),
            ),
            ckptnums=(
                4,
                8,
            )
        )

    kwargs_eval = kwargs_train.copy()

    return d(
        cls=MergeModel,
        kwargs_train=kwargs_train,
        kwargs_eval=kwargs_eval
    )


###############
### Testing ###
###############

def get_planner_params(env_spec):
    def cost_fn(inputs, model_outputs, goals, actions):
        ### collision
        model_collision = tf.nn.sigmoid(model_outputs.collision.close[..., 0])
        clip_value = 0.02
        model_collision = tf.clip_by_value(model_collision, clip_value, 1. - clip_value)
        model_collision = (model_collision - clip_value) / (1. - 2 * clip_value)
        cost_collision = model_collision

        ### bumpy
        model_bumpy = tf.nn.sigmoid(model_outputs.bumpy[..., 0])
        clip_value = 0.1
        model_bumpy = tf.clip_by_value(model_bumpy, clip_value, 1. - clip_value)
        model_bumpy = (model_bumpy - clip_value) / (1. - 2 * clip_value)
        cost_bumpy = model_bumpy

        ### goal position cost
        a = model_outputs.jackal.position[..., :2]
        b = tf.cast(goals.position, tf.float32)[..., :2][tf.newaxis, tf.newaxis]
        dot_product = tf.reduce_sum(a * b, axis=2)
        a_norm = tf.linalg.norm(a, axis=2)
        b_norm = tf.linalg.norm(b, axis=2)
        cos_theta = dot_product / tf.maximum(a_norm * b_norm, 1e-4)
        theta = tf.acos(tf.clip_by_value(cos_theta, -1+1e-4, 1-1e-4))
        cost_position = (1. / np.pi) * tf.abs(theta)
        cost_position = tf.nn.sigmoid(goals.cost_weights.position_sigmoid_scale * (cost_position - goals.cost_weights.position_sigmoid_center)) # more expensive when outside XX degrees difference

        ### for all costs, you only get them if you don't collied
        # cost_position = (1. - model_collision) * cost_position + model_collision * 1.
        # cost_bumpy = (1. - model_collision) * cost_bumpy + model_collision * 1.

        ### magnitude action cost
        steer = actions.commands.angular_velocity[..., 0]
        batch_size = tf.shape(steer)[0]
        cost_action_magnitude = tf.square(steer)

        ### smooth action cost
        cost_action_smooth = tf.concat([tf.square(steer[:, 1:] - steer[:, :-1]), tf.zeros([batch_size, 1])], axis=1)

        total_cost = goals.cost_weights.collision * cost_collision + \
                     goals.cost_weights.bumpy * cost_bumpy + \
                     goals.cost_weights.position * cost_position + \
                     goals.cost_weights.action_magnitude * cost_action_magnitude + \
                     goals.cost_weights.action_smooth * cost_action_smooth

        return d(
            total=total_cost,
            collision=cost_collision,
            bumpy=cost_bumpy,
            position=cost_position,
            action_magnitude=cost_action_magnitude,
            action_smooth=cost_action_smooth,
        ) # [batch, horizon]

    return d(
        cls=MPPIplanner,
        kwargs=d(
            env_spec=env_spec,
            action_selection_limits=d(
                commands=d(
                    angular_velocity=(-1., 1.),
                    linear_velocity=(0.8, 0.8),
                )
            ),
            cost_fn=cost_fn,

            # MPPI params
            sigma=1.0,
            N=4096,
            gamma=50.,
            beta=0.6,
        ),
    )

def get_env_params(env_spec):
    return d(
        cls=JackalEnv,
        env_spec=env_spec,
        params=d(
            speed=0.8,

            debug=True,
            debug_normalize_cost_colors=False,
            debug_color_cost_key='collision',
        )
    )


##################
### Get params ###
##################

def get_params():
    env_spec = JackalPositionCollisionEnvSpec(left_image_only=True)

    model_params = get_model_params()

    return d(
        exp_name='bumpy_collision_position',

        model=model_params,
        planner=get_planner_params(env_spec),
        env=get_env_params(env_spec),
    )

params = get_params()
