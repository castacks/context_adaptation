#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Imu, Joy
#from learned_cost_map.msg import FloatStamped
from nav_msgs.msg import Odometry
# from racepak.msg import rp_controls, rp_shock_sensors, rp_wheel_encoders
from racepak_interfaces.msg import RpControls, RpShockSensors, RpWheelEncoders

import numpy as np

import scipy
import scipy.signal
from scipy.signal import welch
from scipy.integrate import simpson

import os
import yaml

# import pywt
import time

from ament_index_python.packages import get_package_share_directory

class Buffer:
    '''Maintains a scrolling buffer to maintain a window of data in memory

    Args:
        buffer_size:
            Int, number of data points to keep in buffer
    '''
    def __init__(self, buffer_size, padded=False, pad_val=None):
        self.buffer_size = buffer_size
        if not padded:
            self._data = []
            self.data = np.array(self._data)
        else:
            assert pad_val is not None, "For a padded array, pad_val cannot be None."
            self._data = [pad_val] * buffer_size
            self.data = np.array(self._data)

    def insert(self, data_point):
        self._data.append(data_point)
        if len(self._data) > self.buffer_size:
            self._data = self._data[1:]
        self.data = np.array(self._data)

    def get_data(self):
        return self.data

    def show(self):
        print(self.data)


def psd(x, fs):
    '''Return Poswer
    '''
    # f, Pxx = scipy.signal.periodogram(x, fs=fs)
    f, Pxx = scipy.signal.welch(x, fs)

    return f, Pxx

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Taken from: https://raphaelvallat.com/bandpower.html

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        # nperseg = (2 / low) * sf
        nperseg = None

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simpson(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simpson(psd, dx=freq_res)
    return bp

def cost_function(data, sensor_freq, cost_name, cost_stats, freq_range=None, num_bins=None):
    '''Average bandpower in bins of 10 Hz in z axis

    Args:
        - data:
            Input signal to be analyzed
        - sensor_freq:
            Frequency of the recorded signal
        - num_bins:
            Number of bins to split data into
    '''
    cost = 1000000

    # import pdb;pdb.set_trace()
    if "bins" in cost_name:
        assert num_bins is not None, "num_bins should not be None"
        freq_width = (sensor_freq//2)/num_bins
        bins_start = [i*freq_width + 1 for i in range(num_bins)]
        bins_end   = [(i+1)*freq_width + 1 for i in range(num_bins)]

        bps = []
        for i in range(num_bins):
            bp_z = bandpower(data, sensor_freq, [bins_start[i], bins_end[i]], window_sec=None, relative=False)
            total_bp = bp_z
            bps.append(total_bp)

        cost = np.mean(bps)

        # Normalize cost:
        cost = (cost-cost_stats["min"])/(cost_stats["max"]-cost_stats["min"])
        cost = max(min(cost, 1), 0)

    elif "band" in cost_name:
        assert freq_range is not None, "range should not be None"
        bp_z = bandpower(data, sensor_freq, freq_range, window_sec=None, relative=False)

        cost = bp_z

        # Normalize cost:
        # cost = (cost-cost_stats["min"])/(cost_stats["max"]-cost_stats["min"])
        cost = (cost-cost_stats["min"])/(cost_stats["max"]-cost_stats["min"])

        cost = max(min(cost, 3), 0)

    else:
        raise NotImplementedError("cost_name needs to include bins or band")

    return cost

class TraversabilityCostNode(Node):
    def __init__(self):

        super().__init__("cost_publisher")

        self.declare_parameter("cost_stats_dir", "")
        self.declare_parameter("imu_topic", "")

        cost_stats_dir = self.get_parameter('cost_stats_dir').get_parameter_value().string_value
        imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value

        self.assets_dir = os.path.join(get_package_share_directory('context_adaptation'), 'assets')

        cost_stats_dir = os.path.join(self.assets_dir, cost_stats_dir)

        # Set up subscribers
        self.create_subscription(Imu, imu_topic, self.handle_imu, 1)
        self.create_subscription(RpShockSensors, '/shock_pos', self.handle_shock, 1)
        self.create_subscription(Joy, '/mux/joy', self.handle_joy, 1)
        self.create_subscription(Float32, '/terrain_mismatch', self.handle_terrain, 3)
        self.create_subscription(Odometry, '/integrated_to_init', self.handle_odom, 1)

        # Set up publishers
        self.cost = 0
        self.cost_publisher = self.create_publisher(Float32, '/traversability_cost', 10)
        self.cost_array_publisher = self.create_publisher(Float32MultiArray, '/traversability_breakdown', 10)
        self.cost_publisher_baseline = self.create_publisher(Float32, '/traversability_cost_baseline', 10)
        self.speed_mismatch_publisher = self.create_publisher(Float32, '/speed_mismatch', 10)

        self.params = {'IMU_min_freq': {'z': 2, 'y': 9, 'x': 0}, 'IMU_max_freq': {'z': 30, 'y': 13, 'x': 22}, 'IMU_mult': {'z': 1.0, 'y': 0.7, 'x': 0.6}, 'shock_min_freq': 0, 'shock_max_freq': 46, 'mean_mult': 0.1, 'shock_mult': 0.5, 'cutoff_factor': 0.6, 'ALL_MAX': 0.09220535518703862, 'ALL_MIN': 0.002474401263335543, 'ALL_AVG': 0.019438000929697122}

        self.stats = {'IMU_MIN': [-0.5554102710448205, -0.6436653784476221, -0.5440625478513539, -16.969271264076234, -20.00854387164116, -4.090186840295792], 'IMU_MAX': [0.5882288794964552, 0.5955823929980397, 0.6560320965945721, 16.157508494853975, 37.19758415222168, 23.688631061911583], 'SHOCK_MIN': [4.366000175476074, 4.552999973297119], 'SHOCK_MAX': [6.920000076293945, 7.138999938964844]}

        for key in self.stats.keys():
            temp = self.stats[key]
            temp = np.array(temp)
            self.stats[key] = temp
        self.get_logger().info('___________________')
        # Set data buffer
        pad_val = Imu()
        pad_val.linear_acceleration.z = 9.81
        pad_val.linear_acceleration.z = (pad_val.linear_acceleration.z -self.stats['IMU_MIN'][5])/(self.stats['IMU_MAX'][5] - self.stats['IMU_MIN'][5])
        self.imu_freq = 100
        self.shock_freq = 50
        self.num_secs = 1
        self.buffer_size = int(self.num_secs*self.imu_freq)  # num_seconds*imu_freq
        self.shock_buff_size = int(self.num_secs*self.shock_freq)  # num_seconds*imu_freq
        self.shock_mult = self.params['shock_mult']
        self.shock_max = 1
        self.bufferZ = Buffer(self.buffer_size, padded=True, pad_val=pad_val.linear_acceleration.z)
        self.bufferX = Buffer(self.buffer_size, padded=True, pad_val=0)
        self.bufferY = Buffer(self.buffer_size, padded=True, pad_val=0)
        self.bufferRoll = Buffer(self.buffer_size, padded=True, pad_val=0)
        self.bufferPitch = Buffer(self.buffer_size, padded=True, pad_val=0)

        self.bufferL = Buffer(self.shock_buff_size, padded=True, pad_val=6)
        self.bufferR = Buffer(self.shock_buff_size, padded=True, pad_val=6)

        self.bufferCurv = Buffer(30, padded=True, pad_val=0)

        self.bufferJoy = Buffer(32, padded=True, pad_val=0)


        # Load stats for different cost functions:
        self.cost_stats_dir = cost_stats_dir

        with open(cost_stats_dir, 'r') as f:
            self.all_costs_stats = yaml.load(f, Loader=yaml.FullLoader)
        # Information about sensor and sensor frequency, Min and max frequencies set the band to be analyzed for the cost function.
        # self.cost_name = "freq_bins_5"
        self.cost_name = "freq_band_1_30"
        self.cost_stats = self.all_costs_stats[self.cost_name]
        self.sensor_name = "imu_z"
        self.sensor_freq = 100
        # self.min_freq = .5
        # self.max_freq = 40
        self.min_freq = 1.0
        self.max_freq = 30
        # self.cost_stats['min'] = self.min_freq
        # self.cost_stats['max'] = self.max_freq
        # self.num_bins = 5

        #this seemed to work decently
        # self.min_freq = 1
        # self.max_freq = 50

        self.wheel_diff = 0
        self.joy_cost = 0
        self.shock_cost = 0

        self.diff_cost = 0
        self.velocity = 0
        self.vel_mismatch = 0.0
        self.desired_vel = None

        self.cwt_cost = 0

        self.get_logger().info("cost_publisher node initialized");
        self.timer = self.create_timer(0.01, self.dummy_timer_callback) # 100hz

    def handle_terrain(self,msg):
        diff = msg.data

        #assume max vel of 8 and max diff of 1 for now
        mv = 8.0
        md = 1.0
        maxcost = mv*md

        diff_cost = self.velocity * diff

        self.diff_cost = diff_cost/maxcost

        # print('*\n*\n**\n*\n***terraincost', self.diff_cost)


    def handle_odom(self, msg):
        self.velocity = np.linalg.norm([msg.twist.twist.linear.x,msg.twist.twist.linear.y])

        if self.desired_vel is not None:
            mismatch = np.abs(self.velocity - self.desired_vel)
        else:
            mismatch = 0.0

        self.vel_mismatch = self.vel_mismatch + (mismatch - self.vel_mismatch)*.5
        out_msg = Float32()
        out_msg.data = self.vel_mismatch
        self.speed_mismatch_publisher.publish(out_msg)

    def handle_joy(self, msg):
        self.bufferJoy.insert(msg.axes[2])

        cost = cost_function(self.bufferJoy.data, 8, self.cost_name, self.cost_stats, freq_range=[.1, 10], num_bins=None)

        self.joy_cost = self.joy_cost + (cost - self.joy_cost)*.05

    def handle_stats(self, msg):

        setpoint = msg.trajectory[0]
        speed = setpoint.v

        self.desired_vel = speed

    def handle_imu(self, msg):
        self.get_logger().info("-----")
        self.get_logger().info("Received IMU message")

        imu_data = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z, msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

        imu_data = (imu_data - self.stats['IMU_MIN'])/(self.stats['IMU_MAX'] - self.stats['IMU_MIN'])
        imu_data[0] -= .5
        imu_data[1] -= .5

        self.bufferZ.insert(imu_data[5])
        self.bufferRoll.insert(imu_data[0])
        self.bufferPitch.insert(imu_data[1])
        self.bufferX.insert(imu_data[3])
        self.bufferY.insert(imu_data[4])

        bp = bandpower(self.bufferZ.data, self.imu_freq, band=[self.params['IMU_min_freq']['z'], self.params['IMU_max_freq']['z']], window_sec=self.num_secs)
        bp *= self.params['IMU_mult']['z']

        xbp = bandpower(self.bufferX.data, self.imu_freq, band=[self.params['IMU_min_freq']['x'], self.params['IMU_max_freq']['x']], window_sec=self.num_secs)
        xbp *= self.params['IMU_mult']['x']
        bp += xbp

        ybp = bandpower(self.bufferY.data, self.imu_freq, band=[self.params['IMU_min_freq']['y'], self.params['IMU_max_freq']['y']], window_sec=self.num_secs)
        ybp *= self.params['IMU_mult']['y']
        bp += ybp

        MEAN = np.mean(np.abs(self.bufferRoll.data)) #+ np.mean(np.abs(self.bufferPitch.data))
        bp += MEAN*self.params['mean_mult']

        sbp = self.shock_cost * self.params['shock_mult']
        bp += sbp

        bp = (bp - self.params['ALL_MIN'])/(self.params['ALL_MAX']*self.params['cutoff_factor'] - self.params['ALL_MIN'])
        bp = np.clip(bp,0,1)
        cost = bp

        jerk = np.mean(np.abs(np.diff(self.bufferL.data[-30:])))
        cost += jerk

        self.get_logger().info(f"Publishing cost: {cost}, {self.diff_cost}")
        cost_msg = Float32()

        cost_msg.data = cost
        self.cost_publisher.publish(cost_msg)
        self.get_logger().info("Published cost!")


    def handle_shock(self, msg):
        shock_data = np.array([msg.rear_left, msg.rear_right])
        shock_data = (shock_data - self.stats['SHOCK_MIN'])/(self.stats['SHOCK_MAX']- self.stats['SHOCK_MIN'])

        self.bufferL.insert(shock_data[0])

        costL = bandpower(self.bufferL.data, self.shock_freq, band=[self.params['shock_min_freq'], self.params['shock_max_freq']], window_sec=self.num_secs)

        self.shock_cost = costL

    def dummy_timer_callback(self) :
        pass


def main(args=None):
    rclpy.init(args=args)

    node = TraversabilityCostNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()