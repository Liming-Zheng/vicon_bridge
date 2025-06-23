#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float32.h>
#include <tf/transform_datatypes.h>
#include <cmath>

class AoaEstimator {
public:
  AoaEstimator() {
    sub_ = nh_.subscribe("/mocap/liming/pose", 10, &AoaEstimator::poseCallback, this);
    pub_ = nh_.advertise<std_msgs::Float32>("/aoa_deg", 10);
    has_prev_ = false;
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;

  geometry_msgs::Point prev_pos_;
  ros::Time prev_time_;
  bool has_prev_;

  void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    const geometry_msgs::Point& pos = msg->pose.position;
    const geometry_msgs::Quaternion& q = msg->pose.orientation;

    if (!has_prev_) {
      prev_pos_ = pos;
      prev_time_ = msg->header.stamp;
      has_prev_ = true;
      return;
    }

    double dt = (msg->header.stamp - prev_time_).toSec();
    if (dt < 1e-4) return;

    // Compute velocity vector in world frame
    tf::Vector3 vel((pos.x - prev_pos_.x) / dt,
                    (pos.y - prev_pos_.y) / dt,
                    (pos.z - prev_pos_.z) / dt);

    // Convert quaternion to rotation matrix
    tf::Quaternion quat(q.x, q.y, q.z, q.w);
    tf::Matrix3x3 rot(quat);

    // x-axis of body in world frame
    tf::Vector3 x_body_world = rot * tf::Vector3(1, 0, 0);

    // Project velocity and body-x onto x–z plane
    tf::Vector3 vel_xz(vel.x(), 0, vel.z());
    tf::Vector3 x_body_xz(x_body_world.x(), 0, x_body_world.z());

    if (vel_xz.length() < 1e-3 || x_body_xz.length() < 1e-3) return;

    // Compute AoA as angle between two vectors
    double dot = vel_xz.normalized().dot(x_body_xz.normalized());
    double aoa_rad = std::acos(std::max(-1.0, std::min(1.0, dot)));  // clamp for safety

    // Sign determination using cross product (positive when velocity above nose)
    double sign = (vel_xz.x() * x_body_xz.z() - vel_xz.z() * x_body_xz.x()) > 0 ? -1 : 1;
    double aoa_deg = sign * aoa_rad * 180.0 / M_PI;

    // Publish
    std_msgs::Float32 aoa_msg;
    aoa_msg.data = aoa_deg;
    pub_.publish(aoa_msg);

    // Update previous
    prev_pos_ = pos;
    prev_time_ = msg->header.stamp;
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "aoa_estimator_node");
  AoaEstimator estimator;
  ros::spin();
  return 0;
}
