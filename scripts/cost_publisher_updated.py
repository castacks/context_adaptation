#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Imu, Joy
#from learned_cost_map.msg import FloatStamped
from racepak.msg import rp_controls, rp_shock_sensors, rp_wheel_encoders
from torch_mpc.msg import KBMParameters, MPPIStats, SteerSetpointKBMState

import numpy as np

import scipy
import scipy.signal
from scipy.signal import welch
from scipy.integrate import simps

import os
import yaml

import rospkg

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
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
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

class TraversabilityCostNode(object):
    def __init__(self, cost_stats_dir):

        # Set up subscribers
        # rospy.Subscriber('/lester/imu/data', Imu, self.handle_imu, queue_size=1)
        rospy.Subscriber('/novatel/imu/data', Imu, self.handle_imu, queue_size=1)
        rospy.Subscriber('/shock_pos', rp_shock_sensors, self.handle_shock, queue_size=1)
        # rospy.Subscriber('/wheel_rpm', rp_wheel_encoders, self.handle_wheel, queue_size=1)
        rospy.Subscriber('/mppi/stats', MPPIStats, self.handle_stats, queue_size=1)
        rospy.Subscriber('/mux/joy', Joy, self.handle_joy, queue_size=1)

        # Set up publishers
        self.cost = 0
        self.cost_publisher = rospy.Publisher('/traversability_cost_desktop', Float32, queue_size=10)
        self.cost_array_publisher = rospy.Publisher('/traversability_breakdown', Float32MultiArray, queue_size=10)
        self.cost_publisher_baseline = rospy.Publisher('/traversability_cost_baseline', Float32, queue_size=10)
        # self.cost_wheel = rospy.Publisher('/wheel_cost', Float32, queue_size=10)
        # self.cost_curv = rospy.Publisher('/curv_cost', Float32, queue_size=10)



        # Set data buffer
        pad_val = Imu()
        pad_val.linear_acceleration.z = 9.81
        self.imu_freq = 100
        self.shock_freq = 50
        self.num_secs = 2
        self.buffer_size = int(self.num_secs*self.imu_freq)  # num_seconds*imu_freq
        self.shock_buff_size = int(self.num_secs*self.shock_freq)  # num_seconds*imu_freq
        self.shock_mult = 350
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
            self.all_costs_stats = yaml.safe_load(f)
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


    def handle_joy(self, msg):
        print('axes', msg.axes[2])
        self.bufferJoy.insert(msg.axes[2])

        # print(self.bufferJoy.data)
        cost = cost_function(self.bufferJoy.data, 8, self.cost_name, self.cost_stats, freq_range=[.1, 10], num_bins=None)
        # cost = np.var(self.bufferJoy.data)
        print('cost', cost)


        self.joy_cost = self.joy_cost + (cost - self.joy_cost)*.05

        # joy_msg = Float32()
        # joy_msg.data = cost
        # print('joy: ', cost)
        # self.cost_curv.publish(joy_msg)

    def handle_stats(self, msg):
        self.bufferCurv.insert(msg.average_curvature)

        curv_msg = Float32()
        #cost_msg.header = msg.header
        # curv_msg.data = np.var(self.bufferCurv.data)
        # cost = cost_function(self.bufferCurv.data, 10, self.cost_name, self.cost_stats, freq_range=[self.min_freq, self.max_freq], num_bins=None)
        # curv_msg.data = msg.average_curvature
        # curv_msg.data = cost*10

        # cost = msg.cost
        #
        # if cost > 1e5:
        #     cost = 1
        # else:
        #     cost = 0
        #
        # curv_msg.data = cost
        # self.cost_curv.publish(curv_msg)

    # def handle_wheel(self, msg):
    #     # wheel_diff = np.abs((msg.front_left + msg.front_right)/2 - (msg.rear_left + msg.rear_right)/2)
    #     # wheel_diff = np.abs(msg.front_left  - msg.rear_left)
    #     wheel_diff = np.abs(msg.rear_right  - msg.rear_left)
    #     self.wheel_diff = self.wheel_diff + (wheel_diff - self.wheel_diff) * .25
    #     wheel_diff = self.wheel_diff
    #     # wheel_diff = np.minimum(wheel_diff, 15)
    #     # if wheel_diff > 10:
    #     #     wheel_diff = 0
    #
    #     # if wheel_diff > 8:
    #     #     wheel_diff = 2
    #     # else:
    #     #     wheel_diff = 0
    #
    #     wheel_msg = Float32()
    #     #cost_msg.header = msg.header
    #     wheel_msg.data = wheel_diff
    #     self.cost_wheel.publish(wheel_msg)


    def handle_imu(self, msg):
        print("-----")
        print("Received IMU message")
        # self.buffer.insert(msg.linear_acceleration.z + np.abs(msg.angular_velocity.x))
        self.bufferZ.insert(msg.linear_acceleration.z)
        self.bufferRoll.insert(msg.angular_velocity.x)
        self.bufferPitch.insert(msg.angular_velocity.y)
        self.bufferX.insert(msg.linear_acceleration.x)
        self.bufferY.insert(msg.linear_acceleration.y)


        # print('here')
        # cost = cost_function(self.buffer.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=None, num_bins=self.num_bins)
        costZ = cost_function(self.bufferZ.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=[5, self.max_freq], num_bins=None)
        costRoll = cost_function(self.bufferRoll.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=[self.min_freq, self.max_freq], num_bins=None)
        costPitch = cost_function(self.bufferPitch.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=[4, self.max_freq], num_bins=None)
        costX = cost_function(self.bufferX.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=[self.min_freq, self.max_freq], num_bins=None)
        costY = cost_function(self.bufferY.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=[self.min_freq, self.max_freq], num_bins=None)
        # cost = costZ*.4 + costRoll*100 + costPitch*600
        # cost = costX*.7
        # cost = costY
        # cost = (costZ*.3 + costRoll*100 + costPitch*400 + costX*.7 + costY*.5)*.2
        # cost = (costZ*.8 + costRoll*700 + costPitch*400 + costX*.8 + costY*.8 + self.joy_cost*1000)*.1
        # cost = (costZ*.8 + costRoll*700 + costPitch*400 + costX*.8 + costY*.8 + self.joy_cost*1000 + self.shock_cost*350)*.4

        # cost = (costZ*.8 + costRoll*1700 + costPitch*800 + costX*0 + costY*.0 + self.joy_cost*0 + self.shock_cost*0)*.4

        cost = (costZ*.8 + costRoll*1700 + costPitch*800 + costX*.0 + costY*.0 + self.joy_cost*10 + self.shock_cost*350)*.075

        # cost = (costZ*.8 + costRoll*0 + costPitch*0 + costX*.0 + costY*.0 + self.joy_cost*0 + self.shock_cost*0)*1.8



        # print(costX)
        print(f"Publishing cost: {cost}")
        cost_msg = Float32()
        #cost_msg.header = msg.header

        # if cost > .25:
        #     cost = 1
        # else:
        #     cost = 0

        cost_msg.data = cost
        self.cost_publisher.publish(cost_msg)
        print("Published cost!")

        array = [costZ*.8 , costRoll*1700 , costPitch*800 , costX*.0 , costY*.0 , self.joy_cost*10, self.shock_cost*350]
        # print(array)
        arr_msg = Float32MultiArray()
        arr_msg.data = array
        self.cost_array_publisher.publish(arr_msg)

        # cost = (costZ*.8 + costRoll*0 + costPitch*0 + costX*.0 + costY*.0 + self.joy_cost*0 + self.shock_cost*0)*1.0
        #
        # # if cost > .25:
        # #     cost = 1
        # # else:
        # #     cost = 0
        #
        # cost_msg.data = cost
        # self.cost_publisher_baseline.publish(cost_msg)
        # print("Published cost!")

        #self.min_freq
        # cost_msg.data = (costZ*.3 + costRoll*100 + costPitch*400 + costX*.7 + costY*.5)*.3
        # self.cost_publisher_baseline.publish(cost_msg)

    def handle_shock(self, msg):
        print("-----")
        print("Received Shock message")
        # self.buffer.insert(msg.linear_acceleration.z + np.abs(msg.angular_velocity.x))
        # print(msg.front_left)
        self.bufferL.insert(msg.rear_left)
        self.bufferR.insert(msg.rear_right)

        # print('here')
        # cost = cost_function(self.buffer.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=None, num_bins=self.num_bins)
        costL = cost_function(self.bufferL.data, self.shock_freq, self.cost_name, self.cost_stats, freq_range=[2, self.max_freq], num_bins=None)
        costR = cost_function(self.bufferR.data, self.shock_freq, self.cost_name, self.cost_stats, freq_range=[2, self.max_freq], num_bins=None)
        cost = costL + costR
        #
        # cost *= self.shock_mult
        # cost = min(cost,self.shock_max)

        self.shock_cost = self.shock_cost + (cost - self.shock_cost)*.3



if __name__ == "__main__":
    rospy.init_node("traversability_cost_publisher", log_level=rospy.INFO)
    rospy.loginfo("Initialized traversability_cost_publisher node")
    cost_stats_dir = rospy.get_param("~cost_stats_dir")
    rp = rospkg.RosPack()
    cost_stats_dir = os.path.join(rp.get_path("context_adaptation"), "assets","cost_configs") + '/' + cost_stats_dir
    # cost_stats_dir = './wanda_cost_statistics.yaml'
    # cost_stats_dir = './wanda_cost_statistics.yaml'
    node = TraversabilityCostNode(cost_stats_dir)
    rate = rospy.Rate(100)

    while not rospy.is_shutdown():

        rate.sleep()
