
# BADGR: An Autonomous Self-Supervised Learning-Based Navigation System

Gregory Kahn, Pieter Abbeel, Sergey Levine

[Website link](https://sites.google.com/view/badgr)

<img src="./images/bair_logo.png" width="250">

## Installation

Make sure you have 90GB of space available, [anaconda](https://www.anaconda.com/distribution/) installed, and [ROS](https://www.ros.org/) installed. Our installation was on Ubuntu 16.04 with ROS Kinetic.

Clone the repository and go into the folder:

```bash
git clone https://github.com/gkahn13/badgr.git
cd badgr
```

From now on, we will assume you are in the badgr folder.

Download the training data tfrecords and sample rosbags from [here](https://drive.google.com/drive/folders/1zjtuqMIfgEbKTZ-H-uHqCKS8RPDoYiwc?usp=sharing), and extract them into the data folder:
```bash
mkdir data
cd data
mv </path/to/BADGR_collision_tfrecords.zip> .
mv </path/to/BADGR_bumpy_tfrecords.zip> .
mv </path/to/BADGR_rosbags.zip> .
unzip BADGR_collision_tfrecords.zip
rm BADGR_collision_tfrecords.zip
unzip BADGR_bumpy_tfrecords.zip
rm BADGR_bumpy_tfrecords.zip
unzip BADGR_rosbags.zip
rm BADGR_rosbags.zip
cd ..
```

Then setup the anaconda environment:
```bash
conda create -y --name badgr python==3.6.9
source activate badgr
pip install -r requirements.txt
sudo apt-get install ros-${ROS_DISTRO}-ros-numpy
```

Add the src directory to your PYTHONPATH:
```bash
echo 'export PYTHONPATH=</path/to/badgr/src>:$PYTHONPATH' >> ~/.bashrc
```

## Training

Open a new terminal and activate the badgr anaconda environment:
```bash
source activate badgr
```

We train two separate neural networks: one that predicts future collisions and positions, and one that predicts bumpy terrain:
```bash
python scripts/train.py configs/collision_position.py
python scripts/train.py configs/bumpy.py
```

These networks are trained separately so that the data can be rebalanced for either equal proportion collision or bumpy labels. However, at test time these models are combined into a single predictive model, which is possible because the models both have the same inputs.

Create the folder for the combined model:
```bash
mkdir data/bumpy_collision_position
```

## Evaluation

First, play the collision rosbag in a loop,
```bash
rosbag play -l data/rosbags/collision.bag
```

set the cost function weights to only account for collisions,
```bash
rosparam set /cost_weights "{'collision': 1.0, 'position': 0.0, 'position_sigmoid_center': 0.4, 'position_sigmoid_scale': 100., 'action_magnitude': 0.01, 'action_smooth': 0.0, 'bumpy': 0.0}"
```

and then start the policy
```bash
python scripts/eval.py configs/bumpy_collision_position.py
```

This will display a visualizer showing the candidate action sequences, predicted probabilities of collision, and the optimal action sequence for purely avoiding collisions.

![Evaluation Display](/images/eval_display.jpg)


If you wish to visualize the planner for avoiding bumpy terrain, start the bumpy rosbag in a loop,
```bash
rosbag play -l data/rosbags/bumpy.bag
```

change the cost function,
```bash
rosparam set /cost_weights "{'collision': 0.0, 'position': 0.0, 'position_sigmoid_center': 0.4, 'position_sigmoid_scale': 100., 'action_magnitude': 0.01, 'action_smooth': 0.0, 'bumpy': 1.0}"
```

change the visualizer to show bumpiness by modifying `configs/bumpy_collision_position.py` to have `debug_color_cost_key='bumpy'`, and restart the policy
```bash
python scripts/eval.py configs/bumpy_collision_position.py
```

## FAQ
1. If you are having issues with importing OpenCV (e.g., `ImportError: /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so: undefined symbol: PyCObject_Type`), try the following to have python look for the Python 3 OpenCV first:
```bash
export PYTHONPATH=<path/to/anaconda/envs>/badgr/lib/python3.6/site-packages/:$PYTHONPATH
```
