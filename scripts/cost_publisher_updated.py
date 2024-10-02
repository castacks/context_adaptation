#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Imu, Joy
#from learned_cost_map.msg import FloatStamped
from nav_msgs.msg import Odometry
from racepak.msg import rp_controls, rp_shock_sensors, rp_wheel_encoders

import numpy as np

import scipy
import scipy.signal
from scipy.signal import welch
from scipy.integrate import simps

import os
import yaml

import rospkg

# import pywt
import time

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
    def __init__(self, cost_stats_dir, imu_topic):

        # Set up subscribers
        # rospy.Subscriber('/lester/imu/data', Imu, self.handle_imu, queue_size=1)
        rospy.Subscriber(imu_topic, Imu, self.handle_imu, queue_size=1)
        rospy.Subscriber('/shock_pos', rp_shock_sensors, self.handle_shock, queue_size=1)
        # rospy.Subscriber('/wheel_rpm', rp_wheel_encoders, self.handle_wheel, queue_size=1)
        rospy.Subscriber('/mux/joy', Joy, self.handle_joy, queue_size=1)
        rospy.Subscriber('/terrain_mismatch', Float32, self.handle_terrain, queue_size=3)
        # rospy.Subscriber('/odometry/filtered_odom', Odometry, self.handle_odom, queue_size=1)
        rospy.Subscriber('/integrated_to_init', Odometry, self.handle_odom, queue_size=1)


        # Set up publishers
        self.cost = 0
        self.cost_publisher = rospy.Publisher('/traversability_cost', Float32, queue_size=10)
        self.cost_array_publisher = rospy.Publisher('/traversability_breakdown', Float32MultiArray, queue_size=10)
        self.cost_publisher_baseline = rospy.Publisher('/traversability_cost_baseline', Float32, queue_size=10)
        self.speed_mismatch_publisher = rospy.Publisher('/speed_mismatch', Float32, queue_size=10)
        # self.cost_wheel = rospy.Publisher('/wheel_cost', Float32, queue_size=10)
        # self.cost_curv = rospy.Publisher('/curv_cost', Float32, queue_size=10)

        # self.params = {'IMU_min_freq': 1, 'IMU_max_freq': 10, 'shock_min_freq': 4, 'shock_max_freq': 15, 'mean_mult': 0, 'shock_mult': 0, 'cutoff_factor': 0.3}
        # self.params = {'IMU_min_freq': 13, 'IMU_max_freq': 21, 'shock_min_freq': 0, 'shock_max_freq': 43, 'mean_mult': 7, 'shock_mult': 30, 'cutoff_factor': 0.5}
        # self.params = {'IMU_min_freq': {'z': 2, 'y': 12, 'x': 7}, 'IMU_max_freq': {'z': 46, 'y': 36, 'x': 13}, 'IMU_mult': {'z': 1.0, 'y': 0.3, 'x': 0.3}, 'shock_min_freq': 0, 'shock_max_freq': 24, 'mean_mult': 0.05, 'shock_mult': 0.8, 'cutoff_factor': 0.6, 'ALL_MAX': 0.08281747515071684, 'ALL_MIN': 0.0031987638810578706, 'ALL_AVG': 0.016374639824156868}

        self.params = {'IMU_min_freq': {'z': 2, 'y': 9, 'x': 0}, 'IMU_max_freq': {'z': 30, 'y': 13, 'x': 22}, 'IMU_mult': {'z': 1.0, 'y': 0.7, 'x': 0.6}, 'shock_min_freq': 0, 'shock_max_freq': 46, 'mean_mult': 0.1, 'shock_mult': 0.5, 'cutoff_factor': 0.6, 'ALL_MAX': 0.09220535518703862, 'ALL_MIN': 0.002474401263335543, 'ALL_AVG': 0.019438000929697122}

        self.stats = {'IMU_MIN': [-0.5554102710448205, -0.6436653784476221, -0.5440625478513539, -16.969271264076234, -20.00854387164116, -4.090186840295792], 'IMU_MAX': [0.5882288794964552, 0.5955823929980397, 0.6560320965945721, 16.157508494853975, 37.19758415222168, 23.688631061911583], 'SHOCK_MIN': [4.366000175476074, 4.552999973297119], 'SHOCK_MAX': [6.920000076293945, 7.138999938964844]}

        for key in self.stats.keys():
            temp = self.stats[key]
            temp = np.array(temp)
            self.stats[key] = temp
        print('___________________-')
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

        self.diff_cost = 0
        self.velocity = 0
        self.vel_mismatch = 0.0
        self.desired_vel = None

        self.cwt_cost = 0

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

        # self.mismatch_publisher.publish(out_msg)

    def handle_joy(self, msg):
        # print('axes', msg.axes[2])
        self.bufferJoy.insert(msg.axes[2])

        # print(self.bufferJoy.data)
        cost = cost_function(self.bufferJoy.data, 8, self.cost_name, self.cost_stats, freq_range=[.1, 10], num_bins=None)
        # cost = np.var(self.bufferJoy.data)
        # print('cost', cost)


        self.joy_cost = self.joy_cost + (cost - self.joy_cost)*.05

        # joy_msg = Float32()
        # joy_msg.data = cost
        # print('joy: ', cost)
        # self.cost_curv.publish(joy_msg)

    def handle_stats(self, msg):

        setpoint = msg.trajectory[0]
        speed = setpoint.v

        self.desired_vel = speed

        # self.bufferCurv.insert(msg.average_curvature)

        # curv_msg = Float32()
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

        imu_data = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z, msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

        imu_data = (imu_data - self.stats['IMU_MIN'])/(self.stats['IMU_MAX'] - self.stats['IMU_MIN'])
        imu_data[0] -= .5
        imu_data[1] -= .5

        self.bufferZ.insert(imu_data[5])
        self.bufferRoll.insert(imu_data[0])
        self.bufferPitch.insert(imu_data[1])
        self.bufferX.insert(imu_data[3])
        self.bufferY.insert(imu_data[4])


        # print('here')
        # cost = cost_function(self.buffer.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=None, num_bins=self.num_bins)
        # costZ = cost_function(self.bufferZ.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=[5, self.max_freq], num_bins=None)
        # costRoll = cost_function(self.bufferRoll.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=[self.min_freq, self.max_freq], num_bins=None)
        # costPitch = cost_function(self.bufferPitch.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=[4, self.max_freq], num_bins=None)
        # costX = cost_function(self.bufferX.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=[self.min_freq, self.max_freq], num_bins=None)
        # costY = cost_function(self.bufferY.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=[self.min_freq, self.max_freq], num_bins=None)
        # # cost = costZ*.4 + costRoll*100 + costPitch*600
        # # cost = costX*.7
        # # cost = costY
        # # cost = (costZ*.3 + costRoll*100 + costPitch*400 + costX*.7 + costY*.5)*.2
        # # cost = (costZ*.8 + costRoll*700 + costPitch*400 + costX*.8 + costY*.8 + self.joy_cost*1000)*.1
        # # cost = (costZ*.8 + costRoll*700 + costPitch*400 + costX*.8 + costY*.8 + self.joy_cost*1000 + self.shock_cost*350)*.4
        #
        # # cost = (costZ*.8 + costRoll*1700 + costPitch*800 + costX*0 + costY*.0 + self.joy_cost*0 + self.shock_cost*0)*.4
        #
        # cost = (costZ*.8 + costRoll*1700 + costPitch*800 + costX*.0 + costY*.0 + self.joy_cost*10 + self.shock_cost*350)*.075 + self.diff_cost*8
        # # cost = (self.diff_cost)*.075
        # # print(self.joy_cost)
        # baseline_cost = cost

        bp = bandpower(self.bufferZ.data, self.imu_freq, band=[self.params['IMU_min_freq']['z'], self.params['IMU_max_freq']['z']], window_sec=self.num_secs)
        bp *= self.params['IMU_mult']['z']

        xbp = bandpower(self.bufferX.data, self.imu_freq, band=[self.params['IMU_min_freq']['x'], self.params['IMU_max_freq']['x']], window_sec=self.num_secs)
        xbp *= self.params['IMU_mult']['x']
        bp += xbp

        ybp = bandpower(self.bufferY.data, self.imu_freq, band=[self.params['IMU_min_freq']['y'], self.params['IMU_max_freq']['y']], window_sec=self.num_secs)
        ybp *= self.params['IMU_mult']['y']
        bp += ybp

        MEAN = np.mean(np.abs(self.bufferRoll.data)) #+ np.mean(np.abs(self.bufferPitch.data))
        # MEAN = np.mean(np.abs(np.diff(self.bufferRoll.data))) + np.mean(np.abs(np.diff(self.bufferPitch.data)))
        # print("MEAN - ", MEAN,  MEAN*self.params['mean_mult'])
        bp += MEAN*self.params['mean_mult']

        sbp = self.shock_cost * self.params['shock_mult']
        # print(bp, sbp)
        bp += sbp

        # bp = (bp - .0284)/8.667
        bp = (bp - self.params['ALL_MIN'])/(self.params['ALL_MAX']*self.params['cutoff_factor'] - self.params['ALL_MIN'])
        bp = np.clip(bp,0,1)
        cost = bp
        # cost = baseline_cost

        jerk = np.mean(np.abs(np.diff(self.bufferL.data[-30:])))
        # print("JERK -- ", jerk)
        cost += jerk

        # cost = .7*cost + self.vel_mismatch + self.diff_cost*.2
        # cost = cost + 0*self.vel_mismatch + 0*self.diff_cost*.2
        # cost += jerk*10

        # cost = (costZ*.8 + costRoll*0 + costPitch*0 + costX*.0 + costY*.0 + self.joy_cost*0 + self.shock_cost*0)*1.8
        # now = time.perf_counter()
        # mor_freqs = 0.16 * 2**np.arange(6)
        # frequencies = np.array([100, 50, 33.33333333, 25,15,10,5,1,.8]) / self.imu_freq # normalize
        # frequencies = np.array([8,7,6]) / self.imu_freq # normalize
        #
        # scales = pywt.frequency2scale('morl', frequencies)
        # # scales = pywt.frequency2scale('morl', mor_freqs)
        # # print(scales)
        # # scales = [10,50,100]
        # cwt_coeffs,_ = pywt.cwt(self.bufferZ.data, scales, 'morl')
        # # print(cwt_coeffs.shape, self.bufferZ.data.shape)
        # cwt_cost = np.sum(cwt_coeffs[:,-1]**2/scales)
        # cwt_cost = np.mean(np.abs(self.bufferPitch.data[-50:])) + np.mean(np.abs(self.bufferRoll.data[-50:])) + np.mean(np.abs(self.bufferZ.data[-50:]))
        # self.cwt_cost = self.cwt_cost + (cwt_cost - self.cwt_cost) * .5

        # cwt_cost = np.sum(cwt_coeffs[:,-1])
        # print(now - time.perf_counter(), '. t')
        # print((cwt_coeffs[:,-1]**2/scales).shape)
        # cwt_cost = np.sum(np.mean(cwt_coeffs,axis=1)**2/scales)
        # print("CWT: ", cwt_cost)


        # print(costX)
        print(f"Publishing cost: {cost}", self.diff_cost)
        cost_msg = Float32()
        #cost_msg.header = msg.header

        # if cost > .25:
        #     cost = 1
        # else:
        #     cost = 0

        cost_msg.data = cost
        self.cost_publisher.publish(cost_msg)
        print("Published cost!")

        # array = [costZ, costRoll, costPitch, costX, costY, self.joy_cost, self.shock_cost, self.diff_cost]
        # print(array)
        # arr_msg = Float32MultiArray()
        # arr_msg.data = array
        # self.cost_array_publisher.publish(arr_msg)

        # cost = (costZ*.8 + costRoll*0 + costPitch*0 + costX*.0 + costY*.0 + self.joy_cost*0 + self.shock_cost*0)*1.0
        #
        # # if cost > .25:
        # #     cost = 1
        # # else:
        # #     cost = 0
        #
        # cost_msg.data = self.cwt_cost/10
        # cost_msg.data = baseline_cost
        # self.cost_publisher_baseline.publish(cost_msg)
        # print("Published cost!")

        #self.min_freq
        # cost_msg.data = (costZ*.3 + costRoll*100 + costPitch*400 + costX*.7 + costY*.5)*.3
        # self.cost_publisher_baseline.publish(cost_msg)

    def handle_shock(self, msg):
        # print("-----")
        # print("Received Shock message")
        # self.buffer.insert(msg.linear_acceleration.z + np.abs(msg.angular_velocity.x))
        # print(msg.front_left)
        shock_data = np.array([msg.rear_left, msg.rear_right])
        shock_data = (shock_data - self.stats['SHOCK_MIN'])/(self.stats['SHOCK_MAX']- self.stats['SHOCK_MIN'])

        self.bufferL.insert(shock_data[0])
        # self.bufferR.insert(msg.rear_right)

        # print('here')
        # cost = cost_function(self.buffer.data, self.imu_freq, self.cost_name, self.cost_stats, freq_range=None, num_bins=self.num_bins)
        # costL = cost_function(self.bufferL.data, self.shock_freq, self.cost_name, self.cost_stats, freq_range=[5, self.max_freq], num_bins=None)
        # costR = cost_function(self.bufferR.data, self.shock_freq, self.cost_name, self.cost_stats, freq_range=[5, self.max_freq], num_bins=None)
        # cost = costL + costR

        # costL = cost_function(self.bufferL.data, self.shock_freq, self.cost_name, self.cost_stats, freq_range=[5, self.max_freq], num_bins=None)
        costL = bandpower(self.bufferL.data, self.shock_freq, band=[self.params['shock_min_freq'], self.params['shock_max_freq']], window_sec=self.num_secs)

        #
        # cost *= self.shock_mult
        # cost = min(cost,self.shock_max)

        # self.shock_cost = self.shock_cost + (cost - self.shock_cost)*.3
        self.shock_cost = costL



if __name__ == "__main__":
    rospy.init_node("traversability_cost_publisher", log_level=rospy.INFO)
    rospy.loginfo("Initialized traversability_cost_publisher node")
    cost_stats_dir = rospy.get_param("~cost_stats_dir")
    imu_topic = rospy.get_param("~imu_topic")
    rp = rospkg.RosPack()
    cost_stats_dir = os.path.join(rp.get_path("context_adaptation"), "assets","cost_configs") + '/' + cost_stats_dir
    # cost_stats_dir = './wanda_cost_statistics.yaml'
    # cost_stats_dir = './wanda_cost_statistics.yaml'
    node = TraversabilityCostNode(cost_stats_dir, imu_topic)
    rate = rospy.Rate(100)

    while not rospy.is_shutdown():

        rate.sleep()
