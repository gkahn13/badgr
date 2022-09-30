#!/bin/bash

roscore > /dev/null 2>&1 &

echo "roscore starting up..."
while true; do
    sleep 1
    rostopic list > /dev/null 2>&1
    [[ $? != 0 ]] || break
done

rosbag play -l data/rosbags/collision.bag > /dev/null 2>&1 &
rosparam set /cost_weights "{'collision': 1.0, 'position': 0.0, 'position_sigmoid_center': 0.4, 'position_sigmoid_scale': 100., 'action_magnitude': 0.01, 'action_smooth': 0.0, 'bumpy': 0.0}";
python3 scripts/eval.py configs/bumpy_collision_position.py;
bash
