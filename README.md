# SEMSCS
Installation:
1.Create catkin workspace:  
$ mkdir -p ~/gaitech_bci_ws  
$ cd gaitech_bci_ws  
$ mkdir src  
$ catkin_make  

2.Clone gaitech_bci repository in gaitech_bci_ws/src and catkin_make the workspace:  
$ cd ~/gaitech_bci_ws/src  
$ git clone https://github.com/chiwuan6/SEMSCS  
$ cd ../  
$ catkin_make  

3.Source the workspace environment:  
$ source ~/gaitech_bci_ws/devel/setup.bash

4.Run the ROS launch file:
$ roslaunch gaitech_bci_bringup start_driver.launch

5.Connecting to device:
$ rosservice call /gaitech_bci_device_1/connect


6.GAITECH_BCI_TOOLS

View_bci_data:
$ rosrun gaitech_bci_tools view_bci_data

View_bci_experiment:
$ rosrun gaitech_bci_tools view_bci_experiment

Make_experiment:
$ rosrun gaitech_bci_tools make_experiment –i filename.csv –o experiment_video

Video_experiment_builder:
$ rosrun gaitech_bci_tools video_experiment_builder

Rosbag_matlab:
$ rosrun gaitech_bci_tools rosbag_matlab –i input.bag –o output.mat

Rosbag_csv:
$ rosrun gaitech_bci_tools rosbag_csv –i input.bag –o output.

rosbag_mne:
$ rosrun gaitech_bci_tools rosbag_matlab –i input.bag –o 

View_psd:
$ rosrun gaitech_bci_tools view_psd bci_data:=/gaitech_bci_device_1/data_comref

View_image:
$ rosrun gaitech_bci_tools view_image _frequency:=10_background:=blue_foreground:=white _type:=2

Use following command to run this gaitech_bci_teleop：
$ rosrun gaitech_bci_teleop bci_teleop


Depending on your system you might need to install following Dependencies: 
pyqtgraph  
python-qt4  
numpy  
scipy  
gstreamer-1.0  

For details on the model usage and training strategies, please refer to the paper.



