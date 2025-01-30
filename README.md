# context_adaptation
Framework on the viking for detecting new environmental contexts and adapting parameters for downstream tasks

### ARL sim instructions
* Make sure you have the saich/ros2/salon/arl-sim branch for the following packages

```
physics_atv_visual_mapping
torch_coordinator
(this package) context_adaptation
```

* Build everything inside the phoenix-r2 stack. gpytorch is not in the phoenix docker. A temporary fix is to install that within the container.

```
pip install gpytorch
```

* Run the following commands:

```
# simulator
./sim.x86_64

# rviz
ros2 launch phoenix_launch rviz_launch.xml name:=warty

# launchers
ros2 launch phoenix_launch salon_launch.xml sim:=true terrainnet:=false vfm_voxels:=true
```

* Check if the cost is being published in /warty/map/planning/local
* Run the ARL MPPI controller through rviz following the sim documentation.
* Ask Matthew Sivaprakasam in case you have any questions.