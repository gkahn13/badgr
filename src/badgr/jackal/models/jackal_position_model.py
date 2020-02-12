import numpy as np
import tensorflow as tf

from badgr.jackal.models.jackal_model import JackalModel
from badgr.utils.python_utils import AttrDict
from badgr.utils.tf_utils import rotate_to_global


class JackalPositionModel(JackalModel):

    def __init__(self, params):
        super(JackalPositionModel, self).__init__(params)

        self._is_output_gps = params.is_output_gps

    def _get_position_outputs(self, preprocess_outputs, inputs):
        assert preprocess_outputs.shape.as_list()[-1] == 3

        if self._is_output_gps:
            batch_size = tf.shape(preprocess_outputs)[0]
            position = tf.zeros([batch_size, 3])
            yaw = -inputs.imu.compass_bearing[:, 0] + 0.5 * np.pi # so that east is 0 degrees
        else:
            position = inputs.jackal.position
            yaw = inputs.jackal.yaw[:, 0]

        output_positions = rotate_to_global(curr_position=position,
                                            curr_yaw=yaw,
                                            local_position=preprocess_outputs)

        outputs = AttrDict()
        outputs.add_recursive('jackal/position', output_positions)

        return outputs

    def _get_outputs(self, preprocess_outputs, inputs):
        # assumes Model slices in order of output_observations, so need jackal/position first
        assert self._output_observations[0].name == 'jackal/position'
        preprocess_output_position = preprocess_outputs[..., :3]

        position_model_outputs = self._get_position_outputs(preprocess_output_position, inputs)
        model_outputs = super(JackalPositionModel, self)._get_outputs(preprocess_outputs, inputs)

        outputs = model_outputs
        outputs.jackal.position = position_model_outputs.jackal.position
        return outputs
