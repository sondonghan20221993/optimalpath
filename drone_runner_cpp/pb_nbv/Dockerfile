FROM osrf/ros:noetic-desktop-full

RUN apt-get update && apt-get install -y \
    build-essential cmake git unzip \
    libnlopt-dev libnlopt-cxx-dev \
    libjsoncpp-dev libcgal-dev liboctomap-dev \
    libgoogle-glog-dev libeigen3-dev \
    ros-noetic-behaviortree-cpp-v3 \
    python3-pip libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . ./pb_nbv

ENV WORK_DIR=/workspace/pb_nbv_minimal/

RUN cd /workspace/pb_nbv/3rdparty && \
    unzip -qo glog-0.6.0.zip && \
    cd glog-0.6.0 && rm -rf build && mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j4 && make install && \
    ldconfig

RUN cd /workspace && mkdir -p pb_nbv_minimal/src && \
    cp -r /workspace/pb_nbv/src/{pb_core,drone_runner,utils_msgs,see_common} pb_nbv_minimal/src/ && \
    cd pb_nbv_minimal && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20"

CMD ["/bin/bash"]
