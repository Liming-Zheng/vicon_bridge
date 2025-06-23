#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, Vector3
import numpy as np
import tf
from collections import deque
from scipy.signal import savgol_filter

class AeroAngleNode:
    def __init__(self):
        rospy.init_node('aero_angle_node')

        # 发布器
        self.pub_euler = rospy.Publisher('/euler_angles', Vector3, queue_size=10)
        self.pub_flight_path = rospy.Publisher('/flight_path', Vector3, queue_size=10)
        self.pub_aero = rospy.Publisher('/aero_angles', Vector3, queue_size=10)

        # 订阅 pose
        self.sub = rospy.Subscriber('/mocap/liming/pose', PoseStamped, self.pose_callback)

        # 滤波参数
        self.buffer_size = rospy.get_param('~buffer_size', 15)
        self.poly_order = rospy.get_param('~poly_order', 3)
        self.dt_nominal = rospy.get_param('~dt', 0.01)
        self.alpha_lpf_alpha = rospy.get_param("~alpha_lpf_alpha", 0.1)
        self.gamma_v = rospy.get_param("~gamma_v", 0.05)

        # 状态
        self.pos_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.quat_latest = None
        self.alpha_filtered = 0.0

    def lpf(self, prev, current):
        a = self.alpha_lpf_alpha
        return a * current + (1 - a) * prev

    def pose_callback(self, msg):
        t = msg.header.stamp.to_sec()
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        quat = [msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w]

        self.pos_buffer.append(pos)
        self.time_buffer.append(t)
        self.quat_latest = quat

        if len(self.pos_buffer) < self.buffer_size:
            rospy.logdebug("Buffer not full yet.")
            return

        # Step 1: 解算欧拉角
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(self.quat_latest)
        roll_deg = np.degrees(roll)
        theta_deg = -np.degrees(pitch)  # z朝下
        yaw_deg = -np.degrees(yaw)      # y朝右
        self.pub_euler.publish(Vector3(roll_deg, theta_deg, yaw_deg))

        # Step 2: 平滑位置 & 速度估计
        pos_array = np.array(self.pos_buffer)
        pos_smooth = savgol_filter(pos_array, self.buffer_size, self.poly_order, axis=0)
        
        # 手动差分计算速度，基于真实时间
        pos_np = pos_smooth  # shape: (N, 3)
        time_np = np.array(self.time_buffer)  # shape: (N,)

        dt = np.diff(time_np)  # shape: (N-1,)
        dp = np.diff(pos_np, axis=0)  # shape: (N-1, 3)

        # 防止除以0
        dt[dt < 1e-6] = 1e-6

        vel_array = dp / dt[:, np.newaxis]  # shape: (N-1, 3)

        # 中值滤波速度
        velocity = np.median(vel_array[-5:], axis=0)


        v_norm = np.linalg.norm(velocity)

        if v_norm < self.gamma_v:
            velocity[:] = 0.0
            gamma_deg = 0.0
            chi_deg = 0.0
        else:
            gamma_rad = np.arcsin(np.clip(velocity[2] / v_norm, -1.0, 1.0))
            gamma_deg = np.degrees(gamma_rad)

            vel_xy = np.array([velocity[0], velocity[1]])
            chi_rad = np.arctan2(vel_xy[1], vel_xy[0])
            chi_deg = np.degrees(chi_rad)

        self.pub_flight_path.publish(Vector3(gamma_deg, chi_deg, 0.0))

        # Step 3: alpha = theta - gamma, beta = chi - yaw
        if v_norm < self.gamma_v:
            alpha_deg = 0.0
            beta_deg = 0.0
        else:
            alpha_deg = theta_deg - gamma_deg
            beta_deg = chi_deg - yaw_deg

        # 一阶低通滤波 alpha
        self.alpha_filtered = self.lpf(self.alpha_filtered, alpha_deg)
        self.pub_aero.publish(Vector3(self.alpha_filtered, beta_deg, 0.0))

        # Debug 输出
        rospy.loginfo_throttle(0.5, 
            f"θ={theta_deg:.2f}, γ={gamma_deg:.2f}, α={self.alpha_filtered:.2f} | "+
            f"yaw={yaw_deg:.2f}, χ={chi_deg:.2f}, β={beta_deg:.2f} | | v=({velocity[0]:.4f}, {velocity[1]:.4f}, {velocity[2]:.4f})")

if __name__ == '__main__':
    try:
        AeroAngleNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
