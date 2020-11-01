
## SECTION 1 : Indoor Navigation 

### Monocular Indoor Navigation - Single Instance Demonstration

Following video provides the indoor navigation for monocular camera raw feed

[Raw Feed](https://www.youtube.com/watch?v=sht2fJMM70s&feature=youtu.be)

[RVIZ](https://youtu.be/VTLj2SsOUTY)

[VSLAM](https://youtu.be/RPuE93142bE)



Following video provides the indoor navigation for stereo camera raw feed

[Right Eye](https://youtu.be/paSayvZxmWQ)

[Left Eye](https://youtu.be/paSayvZxmWQ)

[RVIZ](https://youtu.be/qMxSWc_pv7g)

[VSLAM](https://youtu.be/1g-4wYyUqo0)

---
## Section 2: How to build & run a standalone application

## Prerequisite

```bash
Ubuntu 16.04
```

## Setup OpenVSLAM 

Install dependencies
```bash
# basic 
apt install -y build-essential pkg-config cmake git wget curl unzip
# g2o 
apt install -y libatlas-base-dev libsuitesparse-dev
# OpenCV 
apt install -y libgtk-3-dev
apt install -y ffmpeg
apt install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavresample-dev
# eigen 
apt install -y gfortran
# other 
apt install -y libyaml-cpp-dev libgoogle-glog-dev libgflags-dev
# SocketViewer
# Protobuf dependencies
apt install -y autogen autoconf libtool
# Node.js
curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
apt install -y nodejs
```

Install Eigen
```bash
cd /path/to/working/dir
wget -q https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2
tar xf eigen-3.3.7.tar.bz2
rm -rf eigen-3.3.7.tar.bz2
cd eigen-3.3.7
mkdir -p build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    ..
make -j4
make install
```

Install OpenCV
```bash
cd /path/to/working/dir
wget -q https://github.com/opencv/opencv/archive/3.4.0.zip
unzip -q 3.4.0.zip
rm -rf 3.4.0.zip
cd opencv-3.4.0
mkdir -p build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DENABLE_CXX11=ON \
    -DBUILD_DOCS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_JASPER=OFF \
    -DBUILD_OPENEXR=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DWITH_EIGEN=ON \
    -DWITH_FFMPEG=ON \
    -DWITH_OPENMP=ON \
    ..
make -j4
make install
```
Install DBoW2
```bash
cd /path/to/working/dir
git clone https://github.com/shinsumicco/DBoW2.git
cd DBoW2
mkdir build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    ..
make -j4
make install
```	

Install g2o
```bash
cd /path/to/working/dir
git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o
git checkout 9b41a4ea5ade8e1250b9c1b279f3a9c098811b5a
mkdir build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_CXX_FLAGS=-std=c++11 \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_UNITTESTS=OFF \
    -DBUILD_WITH_MARCH_NATIVE=ON \
    -DG2O_USE_CHOLMOD=OFF \
    -DG2O_USE_CSPARSE=ON \
    -DG2O_USE_OPENGL=OFF \
    -DG2O_USE_OPENMP=ON \
    ..
make -j4
make install
```	

Install SocketViewer
```bash
cd /path/to/working/dir
git clone https://github.com/shinsumicco/socket.io-client-cpp.git
cd socket.io-client-cpp
git submodule init
git submodule update
mkdir build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_UNIT_TESTS=OFF \
    ..
make -j4
make install
```	

Install Protobuf 
```bash
wget -q https://github.com/google/protobuf/archive/v3.6.1.tar.gz
tar xf v3.6.1.tar.gz
cd protobuf-3.6.1
./autogen.sh
./configure \
    --prefix=/usr/local \
    --enable-static=no
make -j4
make install
```	

Install OpenVSLAM
```bash
git clone https://github.com/xdspacelab/openvslam
cd /path/to/openvslam
mkdir build && cd build
cmake \
    -DBUILD_WITH_MARCH_NATIVE=ON \
    -DUSE_PANGOLIN_VIEWER=OFF \
    -DUSE_SOCKET_PUBLISHER=ON \
    -DUSE_STACK_TRACE_LOGGER=ON \
    -DBOW_FRAMEWORK=DBoW2 \
    -DBUILD_TESTS=ON \
    ..
make -j4
```	

Install Server for SockertViewer
```bash
cd /path/to/openvslam/viewer
npm install
```	

Export Environment Varibables
```bash
export Eigen3_DIR=/usr/local/share/eigen3/cmake
export OpenCV_DIR=/usr/local/share/OpenCV
export DBoW2_DIR=/usr/local/lib/cmake/DBoW2
export g2o_DIR=/usr/local/lib/cmake/g2o
export sioclient_DIR=/usr/local/lib/cmake/sioclient
```	

Test the built of OpenVSLAM

```bash
   ./run_kitti_slam -h
```	

## Run OpenVSLAM (standalone)
Download ORB vocabulary
```bash
FILE_ID="1wUPb328th8bUqhOk-i8xllt5mgRW4n84"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -sLb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o orb_vocab.zip
unzip orb_vocab.zip
```	

Download test data
```bash
FILE_ID="1TVf2D2QvMZPHsFoTb7HNxbXclPoFMGLX"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -sLb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o aist_living_lab_2.zip
unzip aist_living_lab_2.zip
```	

Start the SockertViewer Server
```bash
cd /path/to/openvslam/viewer
node app.js
open browser : http://localhost:3001/
```	

Start localization
```bash
./run_video_localization -v ./orb_vocab/orb_vocab.dbow2 -m ./aist_living_lab_2/video.mp4 -c ./aist_living_lab_2/config.yaml --frame-skip 3 --no-sleep --map-db map.msg
```	

## Setup ROS Kinetic
Install ros core
```bash
sudo apt-get install ros-kinetic-ros-base
```

Environment setup
```bash
echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
sudo rosdep init
rosdep update
```

Install image_transport
```bash
apt install ros-${ROS_DISTRO}-image-transport
```

Install cv_bridge
```bash
cd /path/to/openvslam/ros
git clone --branch ${ROS_DISTRO} --depth 1 https://github.com/ros-perception/vision_opencv.git
cp -r vision_opencv/cv_bridge src/
rm -rf vision_opencv
```

Build ROS
```bash
cd /path/to/openvslam/ros
catkin_make \
    -DBUILD_WITH_MARCH_NATIVE=ON \
    -DUSE_PANGOLIN_VIEWER=OFF \
    -DUSE_SOCKET_PUBLISHER=ON \
    -DUSE_STACK_TRACE_LOGGER=ON \
    -DBOW_FRAMEWORK=DBoW2
```

# Run localization on ROS Kinetic
```bash
# start the image-transport
source /path/to/openvslam/ros/devel/setup.bash
rosrun image_transport republish raw in:=/video/image_raw raw out:=/camera/image_raw

# start the socket viewer
cd /path/to/openvslam/viewer
node app.js
open browser : http://localhost:3001/

# start openvslam localization
rosrun openvslam run_localization \
    -v /path/to/orb_vocab.dbow2 \
    -c /path/to/config.yaml \
    --map-db /path/to/map.msg

# publish video to test and view in socketviewer
rosrun publisher video -m /path/to/video.mp4
```

# Setup Navigation Stack
Install rviz
```bash
sudo apt-get install rviz
```

Install OctoMap,Image Pipeline,Path Planner,Move base
```bash
git clone https://e0508624@dev.azure.com/e0508624/Navcon/_git/Navcon
```

Build ROS
```bash
cd /home/zy2020/openvslam/ros
catkin_make \
    -DBUILD_WITH_MARCH_NATIVE=ON \
    -DUSE_PANGOLIN_VIEWER=OFF \
    -DUSE_SOCKET_PUBLISHER=ON \
    -DUSE_STACK_TRACE_LOGGER=ON \
    -DBOW_FRAMEWORK=DBoW2
```

# Start Navigation
```bash
#Start ROS
roscore

#Start Rviz
rosrun rviz rviz

#[Alternative]Start Rviz in WSL(windows subsystem for linux)
export LIBGL_ALWAYS_INDIRECT=0
rosrun rviz rviz

#Start socket viewer
node app.js

#Start image_transport
rosrun image_transport republish raw in:=/video/right/image_raw raw out:=/stereo/right/image_raw & rosrun image_transport republish raw in:=/video/left/image_raw raw out:=/stereo/left/image_raw

#Start Image processor
ROS_NAMESPACE=stereo rosrun stereo_image_proc stereo_image_proc

#Start localization
rosrun openvslam run_localization -v /home/zy2020/openvslam/build/orb_vocab/orb_vocab.dbow2 -c /home/zy2020/openvslam/build/config.yaml --map-db /home/zy2020/openvslam/build/map.msg --mapping

#Launch Octomap suite
roslaunch octomap_server octomap_mapping.launch 

#Launch nevigation
roslaunch my_robot_configuration.launch
roslaunch move_base.launch

#[Alternative]Stereo publisher for testing
rosrun publisher_left video_left -m /home/zy2020/openvslam/ros/video/stereo/left/VID_20201011_143838_3.mp4 & rosrun publisher_right video_right -m /home/zy2020/openvslam/ros/video/stereo/right/IMG_8025_3.MOV
```

