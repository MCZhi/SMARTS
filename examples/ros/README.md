# ROS Driver for SMARTS

This is a catkin workspace for a ROS (v1) node that wraps/drives a SMARTS simulation.

## ROS Installation and Configuration

First, see [wiki.ros.org](http://wiki.ros.org) for instructions about installing and configuring a ROS environment.

### Python version issues

Note that SMARTS uses **python3**, whereas ROS verions `1.*` (kinetic, lunar, melodic, or noetic) were designed for **python2**.

The example node in `src/src/ros_wrapper.py` was created for the "_kinetic_" ROS distribution and may not work with a ROS `2.*` distribution.
Therefore, you may need to tweak your ROS and/or python environment(s) slightly to get things to work.

The exact tweaks/workarounds to get python3 code running correctly with ROS version `1.*` will depend upon your local setup.
But among other things, you may need to do the following (after the "normal" SMARTS and ROS installation and setups):
```bash
source .venv/bin/activate
pip3 install rospkg catkin_pkg
```

## Setup

Setup your environment:
```bash
cd examples/ros
catkin_make
source devel/setup.bash
```


## Running

From the main SMARTS repo folder:
```bash
rosrun smarts_ros ros_driver.py
```
Or if you prefer (or if required due to the python version issues desribed above):
```bash
python3 exmples/src/src/ros_wrapper.py
```

These may require you to explicitly start `rosmaster` node first
if you don't already have an instance running, like:
```bash
roscore &
```
which will run one in the background.

Alternatively, if you have parameters that you want to override on a regular basis,
create a [roslaunch](http://wiki.ros.org/roslaunch) file in your package folder like:
```xml
<launch>
  <node pkg="smarts_ros" name="SMARTS" type="ros_driver.py" output="screen">
    <param name="~buffer_size" value="3" />
    <param name="~target_freq" value="30" />
    <param name="~time_ratio" value="1" />
  </node>
</launch>
```
And then, if you called it `ros_driver.launch`:
```bash
roslaunch smarts_ros ros_driver.launch
```
(This approach will automatically start the `rosmaster` node.)


### Parameters and Arguments

The node started by `ros_driver.py` accepts several parameters.  
These can be specified as arguments on the command line when using `rosrun`
or added to a `.launch` file when using `roslaunch`, or set in the 
ROS parameter server.

In addition to the normal arguments that `roslaunch` supplies on
the command line (e.g., `__name`, `__ns`, `__master`, `__ip`, etc.)
the following other (optional) arguments will set the associated
ROS private node parameters if used via `rosrun`:

- `_buffer_size`:  The number of entity messages to buffer to use for smoothing/extrapolation.  Must be a positive integer.  Defaults to `3`.

- `_target_freq`:  The target frequencey in Hz.  If not specified, the node will publish as quickly as SMARTS permits.

- `_time_ratio`:  How many real seconds should a simulation second take.  Must be a positive float.  Default to `1.0`.

- `_headless`:  Controls whether SMARTS should also emit its state to an Envision server.  Defaults to `True`.

- `_seed`:  Seed to use when initializing SMARTS' random number generator(s).  Defaults to `42`.


To specify these via the command line, use syntax like:
```bash
rosrun smarts_ros ros_driver.py _target_freq:=20
```


### Scenarios

Then, when you want to initialize SMARTS on a scenario,
have one of the nodes on the ROS network publish an appropriate `SmartsControl` messsage on the `SMARTS/control` topic,
after which SMARTS will begin handling messages from the `SMARTS/entities_in` channel.

Or you could manually reset SMARTS from the command line with:
```bash
rostopic pub /SMARTS/control smarts_ros/SmartsControl '{ reset_with_scenario_path: /full/path/to/scenario }'
```
