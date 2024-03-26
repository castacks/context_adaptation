#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Imu, Joy
#from learned_cost_map.msg import FloatStamped
from nav_msgs.msg import Odometry
from racepak.msg import rp_controls, rp_shock_sensors, rp_wheel_encoders
from torch_mpc.msg import KBMParameters, MPPIStats, SteerSetpointKBMState

import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib
# import rasterio
# from rasterio.enums import Resampling
# from matplotlib.animation import FuncAnimation, ArtistAnimation


class SpeedMismatchNode(object):
    def __init__(self):

        # Set up subscribers
        # rospy.Subscriber('/lester/imu/data', Imu, self.handle_imu, queue_size=1)
        # rospy.Subscriber('/novatel/imu/data', Imu, self.handle_imu, queue_size=1)
        # rospy.Subscriber('/shock_pos', rp_shock_sensors, self.handle_shock, queue_size=1)
        # rospy.Subscriber('/wheel_rpm', rp_wheel_encoders, self.handle_wheel, queue_size=1)
        rospy.Subscriber('/mppi/stats', MPPIStats, self.handle_stats, queue_size=1)
        rospy.Subscriber('/odometry/filtered_odom', Odometry, self.handle_odom, queue_size=1)
        # rospy.Subscriber('/integrated_to_init', Odometry, self.handle_odom, queue_size=1)



        # Set up publishers
        self.desired = 0.
        self.velocity = 0.
        self.mismatch = 0.
        self.mismatch_publisher = rospy.Publisher('/speed_mismatch', Float32, queue_size=10)

        self.viz = False
        if self.viz:


            self.fig, self.ax = plt.subplots(1)
            self.ln, = self.ax.plot([], [], 'ro')
            self.x_data, self.y_data = [] , []
            self.viz_skip = 10
            self.skip_counter = 0
            self.colormap = plt.get_cmap('plasma')

            self.dat = rasterio.open(r"/home/physics_atv/physics_atv_ws/gascola.tif")
            self.scale = 1
            self.map = self.dat.read(out_shape=(self.dat.count,
                int(self.dat.height * self.scale),
                int(self.dat.width * self.scale)
            ),
            resampling=Resampling.bilinear)[:3]/255.0

            matplotlib.rcParams['figure.raise_window'] = False
            self.ax.imshow(np.transpose(self.map,(1,2,0)),zorder=1)
            # plt.show(block=False)



    def handle_odom(self, msg):
        self.velocity = np.linalg.norm([msg.twist.twist.linear.x,msg.twist.twist.linear.y])

        mismatch = np.abs(self.velocity - self.desired)
        out_msg = Float32()

        self.mismatch = self.mismatch + (mismatch - self.mismatch)*.5
        out_msg.data = self.mismatch

        self.mismatch_publisher.publish(out_msg)

        # self.skip_counter += 1
        # if self.viz and (self.skip_counter % self.viz_skip == 0):
        #     self.skip_counter = 0
        #     x = -msg.pose.pose.position.y
        #     y = msg.pose.pose.position.x
        #
        #     # self.vel_color = self.vel_colormap(self.velocity/5.5)
        #
        #     row,col = self.dat.index(x, y)
        #     row *= self.scale
        #     col *= self.scale
        #     # self.ax.plot(col,row,'.',c=self.color,zorder=2)
        #     self.ax.plot(col,row,'.',c=self.colormap(self.mismatch*5),markersize=2,zorder=2)


        print(self.desired,'    ', self.velocity)

    def handle_stats(self, msg):
        setpoint = msg.trajectory[0]
        speed = setpoint.v

        self.desired = speed

    def update_plot(self, frame):
        # self.ln.set_data(self.x_data, self.y_data)
        # return self.ln
        return





if __name__ == "__main__":
    rospy.init_node("speed_mismatch_publisher", log_level=rospy.INFO)
    # rospy.loginfo("Initialized traversability_cost_publisher node")
    # cost_stats_dir = rospy.get_param("~cost_stats_dir")
    # rp = rospkg.RosPack()
    # cost_stats_dir = os.path.join(rp.get_path("context_adaptation"), "assets","cost_configs") + '/' + cost_stats_dir
    # cost_stats_dir = './wanda_cost_statistics.yaml'
    # cost_stats_dir = './wanda_cost_statistics.yaml'
    node = SpeedMismatchNode()
    rate = rospy.Rate(100)

    # ani = FuncAnimation(node.fig, node.update_plot)
    # plt.show(block=True)

    while not rospy.is_shutdown():
        rate.sleep()
