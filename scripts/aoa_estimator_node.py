#!/usr/bin/env python
# coding: utf-8

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
import numpy as np
import tf

class AOAEstimator:
    def __init__(self):
        rospy.init_node('aoa_estimator_node')
        self.sub = rospy.Subscriber('/mocap/liming/pose', PoseStamped, self.pose_callback)
        self.pub = rospy.Publisher('/aoa_deg', Float32, queue_size=10)

        self.prev_time = None
        self.prev_pos = None
        self.filtered_pos = None

        self.alpha = rospy.get_param('~lpf_alpha', 0.1)  # 低通滤波器权重

    def lpf(self, prev, curr):
        return self.alpha * np.array(curr) + (1 - self.alpha) * np.array(prev)

    def pose_callback(self, msg):
        t = msg.header.stamp.to_sec()
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        quat = [msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w]

        if self.prev_time is None:
            self.prev_time = t
            self.prev_pos = pos
            self.filtered_pos = pos
            return

        dt = t - self.prev_time
        if dt < 1e-4:
            return

        # 滤波位置
        self.filtered_pos = self.lpf(self.filtered_pos, pos)

        # 速度向量（世界坐标）
        velocity = (np.array(self.filtered_pos) - np.array(self.prev_pos)) / dt

        # 当前姿态四元数 -> body x轴方向（在世界坐标中）
        rot_matrix = tf.transformations.quaternion_matrix(quat)[:3, :3]
        x_body_world = rot_matrix[:, 0]  # body x-axis in world

        # 在 x-z 平面上投影
        vel_xz = np.array([velocity[0], velocity[2]])
        x_body_xz = np.array([x_body_world[0], x_body_world[2]])

        if np.linalg.norm(vel_xz) < 1e-3 or np.linalg.norm(x_body_xz) < 1e-3:
            return

        # 计算攻角
        unit_vel = vel_xz / np.linalg.norm(vel_xz)
        unit_body = x_body_xz / np.linalg.norm(x_body_xz)

        dot = np.clip(np.dot(unit_vel, unit_body), -1.0, 1.0)
        aoa_rad = np.arccos(dot)

        # 用叉乘确定符号：速度在鼻子上方则为正攻角
        cross = np.cross(unit_vel, unit_body)
        sign = -1 if cross > 0 else 1
        aoa_deg = sign * np.degrees(aoa_rad)

        # 发布
        self.pub.publish(Float32(data=aoa_deg))
        rospy.loginfo_throttle(0.5, f"AoA: {aoa_deg:.2f} deg")

        # 更新
        self.prev_time = t
        self.prev_pos = self.filtered_pos

if __name__ == '__main__':
    try:
        estimator = AOAEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
