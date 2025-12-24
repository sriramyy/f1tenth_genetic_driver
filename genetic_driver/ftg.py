#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import String
import json
from .parameters import GeneticParameters

class GeneticFollowGap(Node):
    def __init__(self):
        super().__init__("genetic_follow_gap")

        # load the params
        self.params = GeneticParameters()

        # ros2 setup
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        # lets supervisor send new genes to car
        self.param_sub = self.create_subscription(String, "/genetic_genes", self.update_params_callback, 10)

        # state
        self.ego_speed = 0.0
        self.prev_steer = 0.0
        self.radians_per_elem = None
        self.proc_latest = None

    # FTG implementation (overtake3 without overtake states)
    def odom_callback(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.ego_speed = float(np.hypot(vx, vy))

    def lidar_callback(self, scan):
        # 1. Preprocess
        ranges = np.array(scan.ranges, dtype=np.float32)
        proc = self.preprocess_lidar(ranges)
        self.proc_latest = proc.copy()

        if proc.size == 0: return

        # 2. Find Closest & Apply Safety Bubble
        closest_idx = int(np.argmin(proc))
        proc = self.mask_bubble(proc, closest_idx, self.params.BUBBLE_RADIUS)

        # 3. Find Max Gap & Goal Point
        gap_start, gap_end = self.find_max_gap(proc)
        best_idx = self.find_best_point(gap_start, gap_end, proc)

        # 4. Apply Handling Heuristics
        best_idx = self.apply_edge_guard(best_idx, proc.size)
        best_idx = self.apply_center_bias(best_idx, proc.size, self.params.CENTER_BIAS_ALPHA)

        # 5. Steering & Smoothing
        steer_raw = self.index_to_steer(best_idx, proc.size)
        steer_cmd = self.smooth_and_limit_steer(steer_raw)

        # 6. Speed Policy
        speed_cmd = self.speed_policy(steer_cmd)

        self.publish_drive(speed_cmd, steer_cmd)

    # --- Core Logic Methods ---

    def preprocess_lidar(self, ranges):
        n = len(ranges)
        self.radians_per_elem = (2.0 * np.pi) / n
        
        # Crop to front 180 degrees (approx)
        proc = ranges[135:-135].copy() if n > 270 else ranges.copy()

        if self.params.PREPROCESS_CONV_SIZE > 1:
            k = np.ones(self.params.PREPROCESS_CONV_SIZE) / self.params.PREPROCESS_CONV_SIZE
            proc = np.convolve(proc, k, mode="same")

        return np.clip(proc, 0.0, self.params.MAX_LIDAR_DIST)

    def mask_bubble(self, arr, center, radius):
        a = arr.copy()
        lo = max(0, center - radius)
        hi = min(a.size, center + radius + 1)
        a[lo:hi] = 0.0
        return a

    def find_max_gap(self, arr):
        masked = np.ma.masked_where(arr == 0.0, arr)
        spans = np.ma.notmasked_contiguous(masked)
        if not spans: return 0, arr.size
        best = max(spans, key=lambda sl: (sl.stop - sl.start))
        return best.start, best.stop

    def find_best_point(self, start, stop, arr):
        if stop <= start + 1: return start
        seg = arr[start:stop]
        if self.params.BEST_POINT_CONV_SIZE > 1:
            k = np.ones(self.params.BEST_POINT_CONV_SIZE) / self.params.BEST_POINT_CONV_SIZE
            seg = np.convolve(seg, k, mode="same")
        return int(np.argmax(seg)) + start

    def apply_edge_guard(self, idx, length):
        guard = int(round(np.deg2rad(self.params.EDGE_GUARD_DEG) / self.radians_per_elem))
        return int(np.clip(idx, guard, length - guard - 1))

    def apply_center_bias(self, idx, length, alpha):
        center = (length - 1) / 2.0
        return int(np.clip((1 - alpha) * idx + alpha * center, 0, length - 1))

    def index_to_steer(self, idx, length):
        angle = (idx - (length / 2.0)) * self.radians_per_elem
        return float(np.clip(angle / 2, -self.params.MAX_STEER_ABS, self.params.MAX_STEER_ABS))

    def smooth_and_limit_steer(self, steer):
        s = (1.0 - self.params.STEER_SMOOTH_ALPHA) * steer + self.params.STEER_SMOOTH_ALPHA * self.prev_steer
        delta = np.clip(s - self.prev_steer, -self.params.STEER_RATE_LIMIT, self.params.STEER_RATE_LIMIT)
        self.prev_steer += delta
        return self.prev_steer

    def forward_clearance(self):
        if self.proc_latest is None: return self.params.MAX_LIDAR_DIST
        center = self.proc_latest.size // 2
        half = int(round(np.deg2rad(self.params.FWD_WEDGE_DEG) / self.radians_per_elem))
        return float(np.min(self.proc_latest[center-half : center+half+1]))

    def speed_policy(self, steer):
        base = self.params.CORNER_SPEED if abs(steer) > np.deg2rad(10) else self.params.STRAIGHT_SPEED
        
        # TTC Braking
        fwd = self.forward_clearance()
        ttc = fwd / max(self.ego_speed, 0.05)
        
        if ttc < self.params.TTC_HARD_BRAKE:
            base = 0.0
        elif ttc < self.params.TTC_SOFT_BRAKE:
            scale = (ttc - self.params.TTC_HARD_BRAKE) / (self.params.TTC_SOFT_BRAKE - self.params.TTC_HARD_BRAKE)
            base *= scale

        return min(base, self.params.SPEED_MAX)

    def publish_drive(self, speed, steer):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def update_params_callback(self, msg):
        """Receive new genes from the Supervisor"""
        try:
            # expect a JSON string ex) {"BUBBLE_RADIUS": 120, "MAX_SPEED": 6.0}
            data = json.loads(msg.data)
            self.params.update_from_dict(data)
            # wipe prev memory
            self.prev_steer = 0.0
            self.ego_speed = 0.0
            
            # debug print
            # This proves the genes are actually changing inside the car
            print(f"\n🧬 DRIVER UPDATE: Bubble={self.params.BUBBLE_RADIUS}, Speed={self.params.SPEED_MAX}, SteerBias={self.params.CENTER_BIAS_ALPHA}")
        except Exception as e:
            self.get_logger().error(f"Failed to update params: {e}")

def main():
    rclpy.init()
    node = GeneticFollowGap()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # --- SHUTDOWN ---
        print("\n SHUTTING DOWN: Stopping the car...")
        
        # create a zero-speed message
        stop_msg = AckermannDriveStamped()
        stop_msg.header.stamp = node.get_clock().now().to_msg()
        stop_msg.drive.speed = 0.0
        stop_msg.drive.steering_angle = 0.0
        
        # Publish multiple times to ensure the Simulator hears it
        for _ in range(10):
            node.drive_pub.publish(stop_msg)
            
        # Clean up
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()