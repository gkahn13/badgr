import cv2
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from geometry_msgs.msg import Pose2D
import rospy

from badgr.envs.env import Env
from badgr.jackal.utils import pyblit
from badgr.jackal.utils.gps import latlong_to_utm, GPSPlotter
from badgr.jackal.utils.jackal_subscriber import JackalSubscriber
from badgr.utils.np_utils import yaw_rotmat
from badgr.utils.python_utils import AttrDict, Rate


class JackalEnv(Env):

    def __init__(self, env_spec, params):
        super(JackalEnv, self).__init__(env_spec=env_spec, params=params)

        self._debug = params.debug
        self._debug_normalize_cost_colors = params.get('debug_normalize_cost_colors', True)
        self._debug_color_cost_key = params.get('debug_color_cost_key', 'total')
        self._debug_dt = params.get('debug_dt', 0.25)

        self._setup_ros()
        if self._debug:
            self._setup_debug()

    def _setup_debug(self):
        font = {'family': 'normal',
                'size': 16}
        matplotlib.rc('font', **font)
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"

        self._debug_fig, (ax_sat, ax_satzoom, ax_fpv, ax_egobirdseye, ax_steer, ax_speed) = \
            plt.subplots(1, 6, figsize=(35, 4))

        self._gps_plotter = GPSPlotter()

        self._debug_N_rollouts = 20

        self._pyblit_sat_imshow = pyblit.Imshow(ax_sat)
        self._pyblit_sat_batchline = pyblit.BatchLineCollection(ax_sat)
        self._pyblit_sat_goal = pyblit.Scatter(ax_sat)
        self._pyblit_sat_ax = pyblit.Axis(
            ax_sat,
            [self._pyblit_sat_imshow, self._pyblit_sat_goal, self._pyblit_sat_batchline]
        )

        self._pyblit_satzoom_imshow = pyblit.Imshow(ax_satzoom)
        self._pyblit_satzoom_batchline = pyblit.BatchLineCollection(ax_satzoom)
        self._pyblit_satzoom_ax = pyblit.Axis(ax_satzoom, [self._pyblit_satzoom_imshow, self._pyblit_satzoom_batchline])

        self._pyblit_fpv_imshow = pyblit.Imshow(ax_fpv)
        self._pyblit_fpv_batchline = pyblit.BatchLineCollection(ax_fpv)
        self._pyblit_fpv_ax = pyblit.Axis(ax_fpv, [self._pyblit_fpv_imshow, self._pyblit_fpv_batchline])

        self._pyblit_egobirdseye_batchline = pyblit.BatchLineCollection(ax_egobirdseye)
        self._pyblit_egobirdseye_ax = pyblit.Axis(ax_egobirdseye, [self._pyblit_egobirdseye_batchline])

        self._pyblit_steer_bar = pyblit.Barh(ax_steer)
        self._pyblit_steer_ax = pyblit.Axis(ax_steer, [self._pyblit_steer_bar])

        self._pyblit_speed_bar = pyblit.Bar(ax_speed)
        self._pyblit_speed_ax = pyblit.Axis(ax_speed, [self._pyblit_speed_bar])

        self._is_first_debug = True

    ###########
    ### ROS ###
    ###########

    def _setup_ros(self):
        rospy.init_node('jackal_position_only_env', anonymous=True)
        subscribe_names = set(self.spec.observation_names)
        subscribe_names.add('collision/any')
        subscribe_names.add('gps/utm')
        subscribe_names.add('joy')
        self._jackal_subscriber = JackalSubscriber(names=subscribe_names)

        rospy.Subscriber('/goal_intermediate_latlong', Pose2D, callback=self._goal_callback)
        self._goal_latlong = np.array([37.915021, -122.334439]) # default to next to the bathroom

    def _goal_callback(self, msg):
        self._goal_latlong = np.array([msg.x, msg.y])

    ##########################
    ### Observation / Goal ###
    ##########################

    def _get_observation(self):
        obs_names = set(self.spec.observation_names)
        obs_names.add('gps/utm')
        obs = AttrDict.from_dict(self._jackal_subscriber.get(names=obs_names))

        if 'images/rgb_left' in obs.get_leaf_keys():
            obs.images.rgb_left = self.spec.process_image('images/rgb_left', obs.images.rgb_left)
        if 'images/rgb_right' in obs.get_leaf_keys():
            obs.images.rgb_right = self.spec.process_image('images/rgb_right', obs.images.rgb_right)

        obs.modify_recursive(lambda v: np.asarray(v))

        return obs

    def _get_goal(self, obs):
        goal_utm = latlong_to_utm(self._goal_latlong)
        goal_utm -= obs.gps.utm
        goal_utm = np.append(goal_utm, 0.)

        cost_weights = rospy.get_param(
            '/cost_weights',
            {'collision': 1.0,
             'position': 0.0,
             'action_magnitude': 0.001,
             'action_smooth': 0.0,
             'bumpy': 0.8,
             'position_sigmoid_center': 0.6,
             'position_sigmoid_scale': 100.}
        )
        for key, weight in cost_weights.items():
            cost_weights[key] = np.ravel(weight).astype(np.float32)

        return AttrDict(
            position=goal_utm,
            cost_weights=AttrDict.from_dict(cost_weights)
        )

    def _get_done(self):
        names = [
            'collision/any',
            'gps/latlong',
            'joy'
        ]
        obs = AttrDict.from_dict(self._jackal_subscriber.get(names=names))
        is_collision = obs.collision.any
        is_close_to_goal = np.linalg.norm(latlong_to_utm(self._goal_latlong) - latlong_to_utm(obs.gps.latlong)) < 2.0
        return is_collision or is_close_to_goal

    ####################
    ### Step / Reset ###
    ####################

    def _plot_satellite(self, obs, positions, colors, goal, zoom_size=150):
        ### sat
        self._pyblit_sat_imshow.draw(np.flipud(self._gps_plotter.satellite_image), origin='lower')
        self._pyblit_satzoom_imshow.draw(np.flipud(self._gps_plotter.satellite_image), origin='lower')

        goal_coord = self._gps_plotter.utm_to_coordinate(goal.position[:2] + obs.gps.utm)
        self._pyblit_sat_goal.draw([goal_coord[0]], [goal_coord[1]],
                                      c='r', marker='x', s=50.)

        x_list, y_list = [], []
        for pos in positions:
            yaw = obs.imu.compass_bearing - 0.5 * np.pi  # so that east is 0 degrees
            R = yaw_rotmat(yaw)
            pos = np.hstack((pos, np.zeros([len(pos), 1])))

            positions_in_origin = (pos - pos[0]).dot(R)[:, :2]

            coords_i = self._gps_plotter.utm_to_coordinate(positions_in_origin + obs.gps.utm)

            x_list.append(coords_i[:, 0])
            y_list.append(coords_i[:, 1])
        self._pyblit_sat_batchline.draw(x_list, y_list, color=colors)
        self._pyblit_satzoom_batchline.draw(x_list, y_list, color=colors)

        self._pyblit_sat_ax.draw()

        ax = self._pyblit_satzoom_ax.ax
        curr_xcoord, curr_ycoord = self._gps_plotter.utm_to_coordinate(obs.gps.utm)
        ax.set_xlim((curr_xcoord - 0.5*zoom_size, curr_xcoord + 0.5*zoom_size))
        ax.set_ylim((curr_ycoord - 0.5*zoom_size, curr_ycoord + 0.5*zoom_size))
        self._pyblit_satzoom_ax.draw()

    def _plot_fpv(self, obs, positions, colors):
        def project_points(xy):
            """
            :param xy: [batch_size, horizon, 2]
            :return: [batch_size, horizon, 2]
            """
            batch_size, horizon, _ = xy.shape

            # camera is ~0.35m above ground
            xyz = np.concatenate([xy, -0.35 * np.ones(list(xy.shape[:-1]) + [1])], axis=-1) # 0.35
            rvec = tvec = (0, 0, 0)
            camera_matrix = self.spec.image_intrinsics.copy()
            k1, k2, p1, p2 = self.spec.image_distortion.ravel()
            k3 = k4 = k5 = k6 = 0.
            dist_coeffs = (k1, k2, p1, p2, k3, k4, k5, k6)

            # x = y
            # y = -z
            # z = x
            xyz[..., 0] += 0.15  # NOTE(greg): shift to be in front of image plane
            xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
            uv, _ = cv2.projectPoints(xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs)
            uv = uv.reshape(batch_size, horizon, 2)

            return uv

        pixels = project_points(positions)

        image = obs.images.rgb_left
        self._pyblit_fpv_imshow.draw(np.flipud(image), origin='lower')
        im_lims = obs.images.rgb_left.shape
        x_list, y_list = [], []
        for pix in pixels:
            pix_lims = (480., 640.)

            assert pix_lims[1] / pix_lims[0] == im_lims[1] / float(im_lims[0])
            resize = im_lims[0] / pix_lims[0]

            pix = resize * pix
            x_list.append(im_lims[1] - pix[:, 0])
            y_list.append(im_lims[0] - pix[:, 1])
        self._pyblit_fpv_batchline.draw(x_list, y_list, color=colors, linewidth=3.)
        ax = self._pyblit_fpv_ax.ax
        ax.set_xlim((0, im_lims[1]))
        ax.set_ylim((0, im_lims[0]))
        ax.set_aspect('equal')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('left', size='5%', pad=0.05)
        data = np.linspace(1., 0., 100).reshape(10, 10)
        im = ax.imshow(data, cmap=plt.cm.autumn_r)
        self._debug_fig.colorbar(im, cax=cax, orientation='vertical')
        cax.yaxis.set_ticks_position('left')
        cax.yaxis.set_label_position('left')

        self._pyblit_fpv_ax.draw()

    def _plot_ego_birdseye(self, positions, colors, xlim=(-1.3, 1.3), ylim=(-0.1, 2.0)):
        _, horizon, _ = positions.shape
        xlim = (horizon / 8.) * np.array(xlim)
        ylim = (horizon / 8.) * np.array(ylim)
        x_list, y_list = [], []
        for pos in positions:
            forward_is_up = np.array([[0., -1.], [1., 0.]])
            positions_in_origin = forward_is_up.dot(pos.T).T

            x_list.append(positions_in_origin[:, 0])
            y_list.append(positions_in_origin[:, 1])
        self._pyblit_egobirdseye_batchline.draw(x_list, y_list, color=colors)
        ax = self._pyblit_egobirdseye_ax.ax
        ax.set_xlim(1.0 * np.array(xlim))
        ax.set_ylim(1.0 * np.array(ylim))
        ax.set_aspect('equal')
        ax.set_xlabel('meters')
        ax.set_ylabel('meters')
        ax.set_title('candidate plans', fontweight='bold')
        self._pyblit_egobirdseye_ax.draw()

    def _plot_steer(self, get_action):
        steers = -get_action.action_sequence.commands.angular_velocity.ravel().copy()
        steers[0] = -get_action.action.commands.angular_velocity
        self._pyblit_steer_bar.draw(np.arange(len(steers)), steers)
        ax = self._pyblit_steer_ax.ax
        ax.set_xlim(1.1 * np.array(self.spec.names_to_limits.commands.angular_velocity))
        ax.set_xlabel('angular velocity (rad/s)')
        ax.set_ylabel('planning timestep')
        ax.set_title('planned steering', fontweight='bold')
        ax.set_yticks(np.arange(len(steers)))
        self._pyblit_steer_ax.draw()

    def _plot_speed(self, get_action):
        speeds = get_action.action_sequence.commands.linear_velocity.ravel()
        speeds[0] = get_action.action.commands.linear_velocity
        self._pyblit_speed_bar.draw(np.arange(len(speeds)), speeds)
        ax = self._pyblit_speed_ax.ax
        ax.set_ylim((0, 1.6))
        self._pyblit_speed_ax.draw()

    def _step_debug(self, get_action, obs, goal, use_per_timestep_costs=True):

        def commands_to_positions(linvel, angvel):
            N = len(linvel)
            all_angles = [np.zeros(N)]
            all_positions = [np.zeros((N, 2))]
            for linvel_i, angvel_i in zip(linvel.T, angvel.T):
                angle_i = all_angles[-1] + self._debug_dt * angvel_i
                position_i = all_positions[-1] + \
                             self._debug_dt * linvel_i[..., np.newaxis] * np.stack([np.cos(angle_i), np.sin(angle_i)], axis=1)

                all_angles.append(angle_i)
                all_positions.append(position_i)

            all_positions = np.stack(all_positions, axis=1)
            return all_positions

        all_costs = get_action.all_costs
        all_costs_per_timestep = get_action.all_costs_per_timestep
        all_positions = commands_to_positions(get_action.all_actions.commands.linear_velocity[..., 0],
                                              get_action.all_actions.commands.angular_velocity[..., 0])

        sorted_idxs = np.argsort(all_costs)[::-1]
        all_costs = all_costs[sorted_idxs]
        all_costs_per_timestep = all_costs_per_timestep[self._debug_color_cost_key][sorted_idxs]
        all_positions = all_positions[sorted_idxs]

        subsample_idxs = np.linspace(0, len(all_costs) - 1, self._debug_N_rollouts).astype(int)
        costs = all_costs[subsample_idxs]
        costs_per_timestep = all_costs_per_timestep[subsample_idxs]
        positions = all_positions[subsample_idxs]

        batch_size, horizon = costs_per_timestep.shape
        color_costs = costs_per_timestep if use_per_timestep_costs else np.tile(costs[..., np.newaxis], (1, horizon))
        color_costs = color_costs.ravel()
        if self._debug_normalize_cost_colors:
            colors = plt.cm.autumn_r((color_costs - color_costs.min()) / (color_costs.max() - color_costs.min()))
        else:
            colors = plt.cm.autumn_r(np.clip(color_costs, 0., 1.))
        colors = np.reshape(colors, (batch_size, horizon, -1))

        self._plot_satellite(obs, positions, colors, goal)
        self._plot_fpv(obs, positions, colors)
        self._plot_ego_birdseye(positions, colors)
        self._plot_steer(get_action)
        self._plot_speed(get_action)

        if self._is_first_debug:
            self._debug_fig.tight_layout(pad=1.5)
            plt.show(block=False)
            plt.pause(0.1)
        self._is_first_debug = False

    def step(self, get_action):
        action = get_action.action
        # would normally publish the action here

        obs = self._get_observation()
        goal = self._get_goal(obs)
        done = self._get_done()

        if self._debug:
            self._step_debug(get_action, obs, goal)

        return obs, goal, done

    def reset(self):
        obs = self._get_observation()
        goal = self._get_goal(obs)
        return obs, goal
