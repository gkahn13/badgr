FROM tensorflow/tensorflow:1.13.1-gpu

# fix nvidia public key
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub &&\
    apt update

# install pip and requirements
COPY req_docker.txt ./requirements.txt
RUN apt install -y python3-dev python3-tk &&\
    curl -sSL https://bootstrap.pypa.io/pip/3.5/get-pip.py -o get-pip.py &&\
    python3 get-pip.py "pip < 21.0" "setuptools < 50.0" "wheel < 1.0" &&\
    python3 -m pip install -r requirements.txt

# install ROS kinetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' &&\
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - &&\
    apt update &&\
    apt install -y ros-kinetic-ros-base ros-kinetic-ros-numpy git nano &&\
    echo "source /opt/ros/kinetic/setup.bash" >> /root/.bashrc

RUN echo 'export PYTHONPATH=/badgr/src:$PYTHONPATH' >> /root/.bashrc

WORKDIR /badgr
COPY bgr.sh .
