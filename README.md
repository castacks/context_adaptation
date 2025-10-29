# context_adaptation
Framework on the viking for detecting new environmental contexts and adapting parameters for downstream tasks

### ATV instructions
* Make sure you have the following branches for the following packages

```
physics_atv_visual_mapping (branch: saich/ros2/humble/for-context-adaptation)
torch_coordinator (branch: saich/ros2/salon/release)
context_adaptation (branch: saich/ros2/humble/release)
```

* This package needs visual mapping and super odometry running online.
* Then launch

```
ros2 launch context_adaptation online_hdif.launch.py
```

* Check shortrange_costmap and shortrange_speedmap in rviz.
* Ask Matthew Sivaprakasam in case you have any questions.