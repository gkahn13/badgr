import cv2
import numpy as np

from badgr.envs.env import EnvSpec
from badgr.utils.np_utils import imrectify


class JackalEnvSpec(EnvSpec):

    def __init__(self):
        super(JackalEnvSpec, self).__init__(
            names_shapes_limits_dtypes=(
                ('images/rgb_left', (96, 128, 3), (0, 255), np.uint8),
                ('images/rgb_right', (96, 128, 3), (0, 255), np.uint8),
                ('images/thermal', (32, 32), (-1, 1), np.float32), # TODO: don't know good limits
                ('lidar', (360,), (0., 12.), np.float32),
                ('collision/close', (1,), (0, 1), np.bool),
                ('collision/flipped', (1,), (0, 1), np.bool),
                ('collision/stuck', (1,), (0, 1), np.bool),
                ('collision/any', (1,), (0, 1), np.bool),
                ('gps/is_fixed', (1,), (0, 1), np.float32),
                ('gps/latlong', (2,), (0, 1), np.float32),
                ('gps/utm', (2,), (0, 1), np.float32),
                ('imu/angular_velocity', (3,), (-1.0 * np.pi, 1.0 * np.pi), np.float32),
                ('imu/compass_bearing', (1,), (-np.pi, np.pi), np.float32),
                ('imu/linear_acceleration', (3,), ((-1., -1., 9.81-1.), (1., 1., 9.81+1.)), np.float32),
                ('jackal/angular_velocity', (1,), (-1.0 * np.pi, 1.0 * np.pi), np.float32),
                ('jackal/linear_velocity', (1,), (-1., 1.), np.float32),
                ('jackal/imu/angular_velocity', (3,), (-1.0 * np.pi, 1.0 * np.pi), np.float32),
                ('jackal/imu/linear_acceleration', (3,), ((-1., -1., 9.81-1.), (1., 1., 9.81+1.)), np.float32),
                ('jackal/position', (3,), (-0.5, 0.5), np.float32),
                ('jackal/yaw', (1,), (-np.pi, np.pi), np.float32),
                ('android/illuminance', (1,), (0., 200.), np.float32),
                ('bumpy', (1,), (0, 1), np.bool),

                ('commands/angular_velocity', (1,), (-1.0, 1.0), np.float32),
                ('commands/linear_velocity', (1,), (0.75, 1.25), np.float32)
            )
        )

        fx, fy, cx, cy = 272.547000, 266.358000, 320.000000, 240.000000
        self._dim = (640, 480)
        self._K = np.array([[fx, 0., cx],
                      [0., fy, cy],
                      [0., 0., 1.]])
        self._D = np.array([[-0.038483, -0.010456, 0.003930, -0.001007]]).T
        self._balance = 0.5

    @property
    def observation_names(self):
        return (
            'images/rgb_left',
            'images/rgb_right',
            'images/thermal',
            'collision/close',
            'collision/flipped',
            'collision/stuck',
            'collision/any',
            'gps/is_fixed',
            'gps/latlong',
            'imu/angular_velocity',
            'imu/compass_bearing',
            'imu/linear_acceleration',
            'jackal/angular_velocity',
            'jackal/linear_velocity',
            'jackal/imu/angular_velocity',
            'jackal/imu/linear_acceleration',
            'jackal/position',
            'jackal/yaw',
            'android/illuminance',
        )

    @property
    def output_observation_names(self):
        return (name for name in self.observation_names if 'rgb' not in name)

    @property
    def action_names(self):
        return (
            'commands/angular_velocity',
            'commands/linear_velocity'
        )

    def process_image(self, name, image):
        if len(image.shape) == 4:
            return np.array([self.process_image(name, im_i) for im_i in image])

        if name in ('images/rgb_left', 'images/rgb_right'):
            image = imrectify(image, self._K, self._D, balance=self._balance)

        return super(JackalEnvSpec, self).process_image(name, image)

    @property
    def image_intrinsics(self):
        return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self._K, self._D, self._dim, np.eye(3), balance=self._balance)

    @property
    def image_distortion(self):
        return self._D


class JackalPositionCollisionEnvSpec(JackalEnvSpec):

    def __init__(self, left_image_only=False):
        self._left_image_only = left_image_only
        super(JackalPositionCollisionEnvSpec, self).__init__()

    @property
    def observation_names(self):
        names = [
            'images/rgb_left',

            'jackal/position',
            'jackal/yaw',
            'jackal/angular_velocity',
            'jackal/linear_velocity',

            'jackal/imu/angular_velocity',
            'jackal/imu/linear_acceleration',
            'imu/angular_velocity',
            'imu/linear_acceleration',

            'imu/compass_bearing',
            'gps/latlong',

            'collision/close',
            'collision/stuck'
        ]
        if not self._left_image_only:
            names.append('images/rgb_right')
        return tuple(names)


class JackalBumpyEnvSpec(JackalEnvSpec):

    def __init__(self, left_image_only=False):
        self._left_image_only = left_image_only
        super(JackalBumpyEnvSpec, self).__init__()

    @property
    def observation_names(self):
        names = [
            'images/rgb_left',

            'jackal/imu/angular_velocity',
            'jackal/imu/linear_acceleration',
            'imu/angular_velocity',
            'imu/linear_acceleration',

            'bumpy',
        ]
        if not self._left_image_only:
            names.append('images/rgb_right')
        return tuple(names)
