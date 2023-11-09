# TurtleBot3 Lane Detection README

This package provides a lane detection system for TurtleBot3 in a simulated environment. It allows the TurtleBot3 to autonomously follow lanes in a Gazebo simulation.

## Installation and Usage

Follow these steps to install and use the `turtlebot3_detect_lane` package:

1. Copy the `turtlebot3_detect_lane` package to your `catkin_ws/src` directory.
2. Build the package by navigating to your Catkin workspace and running `catkin_make`:

   ```
   cd ~/catkin_ws/catkin_make
   ```
3. Copy the Gazebo models from the `models` folder to your Gazebo models directory. This is usually located at `~/.gazebo/models`.
4. Now you can run the lane detection system in the Gazebo simulation by launching the `city.launch` file:

   ```
   roslaunch turtlebot3_detect_lane city.launch
   python3 get_image.py
   ```

   This launch file will start the simulation and the lane detection system for your TurtleBot3. The TurtleBot3 will follow the lanes in the simulated city environment.

## Additional Notes

* Make sure you have the necessary dependencies and TurtleBot3 packages installed in your workspace before building the package. You can refer to the TurtleBot3 documentation for more information on installation and dependencies.
* You can modify the launch file or the parameters in the package to customize the behavior of the lane detection system according to your requirements.
* Ensure that you have ROS and Gazebo properly set up in your environment for this package to work.
* If you encounter any issues or have questions, please refer to the package documentation or the TurtleBot3 community for support.

Enjoy using the TurtleBot3 Lane Detection package in your Gazebo simulation!
