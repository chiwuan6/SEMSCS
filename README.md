# SEMSCS
Installation:  
1. Create catkin workspace  
$ mkdir -p ~/gaitech_bci_ws  
$ cd gaitech_bci_ws  
$ mkdir src  
$ catkin_make  

2. Clone gaitech_bci repository in gaitech_bci_ws/src and catkin_make the workspace  
$ cd ~/gaitech_bci_ws/src  
$ git clone https://github.com/gaitech-robotics/gaitech-bci.git  
$ cd ../  
$ catkin_make  

3. source the workspace environment  
$ source ~/gaitech_bci_ws/devel/setup.bash  


Depending on your system you might need to install following Dependencies:  
python-avertuseegheadset (get it from Gaitech Robotics along with licence key if you already have H10C EEG headset)    
pyqtgraph  
python-qt4  
numpy  
scipy  
gstreamer-1.0  
