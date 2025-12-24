#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import json
import time
import numpy as np
import threading
import sys

from .genetic_ml import GeneticML

class GeneticSupervisor(Node):
    def __init__(self):
        super().__init__("genetic_supervisor")
        
        # init Genetic Manager
        self.ga = GeneticML(population_size=20, generations=50)
        
        # Publishers
        self.gene_pub = self.create_publisher(String, "/genetic_genes", 10)
        self.reset_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        
        # Subscribers (To detect crashes/lap completion)
        self.odom_sub = self.create_subscription(Odometry, "/ego_racecar/odom", self.odom_callback, 10)
        # Listen for crashes
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

        # State Variables
        self.start_time = time.time()
        self.is_running = False
        self.max_lap_time = 40.0 # Seconds before we kill a slow run
        
        self.get_logger().info("Genetic Supervisor Initiated. Starting...")
        print("NOTE: Press 'f' + Enter to view Fastest/Best Genome Stats mid-simulation.")
        print("NOTE: press 'r' + Enter for a manual reset")

        # interactive kb to be able to see stats mid run
        self.input_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.input_thread.start()
        
        # Give ROS a moment to connect, then start
        self.create_timer(2.0, self.start_first_run_callback)
        self.timer_started = False


    def keyboard_listener(self):
        """runs in background: waits for user to press 'f' + enter or 'r' + enter"""
        while True:
            try:
                cmd = sys.stdin.readline().strip()
                if cmd.lower() == 'f':
                    self.print_best_record()
                
                if cmd.lower() == 'r':
                    # Calculate time driven
                    run_time = time.time() - self.start_time
                    
                    # Report failure
                    self.finish_run(run_time, crash_count=1)
                    
                    print(f"\n 💥 MANUAL RESET (CRASH)")
            
            except Exception:
                break
    

    def print_best_record(self):
        """prints the best record on request (f in console)"""
        best = self.ga.overall_best_genome

        if best is None:
            print(f"\n 🏆 CURRENT CHAMPION: None")
        else:
            lap_time = best.laps[-1].time if best.laps else 999.9
            gene_list = []

            for key, value in vars(best.params).items():
            # if float then round to 3 decimals
                if isinstance(value, float): 
                    gene_list.append(round(value, 3))
                else:
                    gene_list.append(value)

            print(f"\n 🏆 CURRENT CHAMPION (As of Generation {self.ga.current_generation})")
            print(f"     GenomeID : {best.id}")
            print(f"     Time     : {lap_time}")
            print(f"     Score    : {best.fitness_score}")
            print(f"     Genes    : {gene_list} \n")


    def start_first_run_callback(self):
        """simple oneshot timer to start loop"""
        if not self.timer_started:
            self.start_next_run()
            self.timer_started = True


    def start_next_run(self):
        """Loads the next genome and begins tracking"""
        if self.ga.is_generation_complete():
            self.ga.evolve_next_generation()
            
        # Reset Lap State
        self.has_left_start = False
            
        # Get params for next car
        params = self.ga.get_next_genome_params()
        
        # Publish Genes (new params)
        param_dict = vars(params) 
        msg = String()
        msg.data = json.dumps(param_dict)
        self.gene_pub.publish(msg)
        
        # Reset sim & clock (resets location)
        self.reset_simulation()
        self.start_time = time.time()
        self.is_running = True

        idx = self.ga.current_genome_index
        gen = self.ga.current_generation
        
        print(f"\n 🚀 STARTING: Genome {idx} (Gen {gen})")

        # also print the genomes for reference
        gene_list = []
        for key, value in param_dict.items():
            # if float then round to 3 decimals
            if isinstance(value, float): 
                gene_list.append(round(value, 3))
            else:
                gene_list.append(value)
        print(f"\n 🧬 Genes: {gene_list}")


    def reset_simulation(self):
        """teleports car back to (0,0) in rviz"""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        # --- START LINE COORDINATES ---
        # ADJUST IF MAP START ISNT (0,0)
        msg.pose.pose.position.x = 0.0 
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.0
        
        # Orientation: (0,0,0,1) = Facing East (Positive X)
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        self.reset_pub.publish(msg)
        time.sleep(0.1)


    def scan_callback(self, msg):
        """Detects crashes via LiDAR data."""
        if not self.is_running: return

        # Convert to numpy
        ranges = np.array(msg.ranges)
        
        # Filter out invalid values (inf/nan)
        valid_ranges = ranges[np.isfinite(ranges)]
        
        if valid_ranges.size == 0: return

        min_dist = np.min(valid_ranges)
        
        # CRASH THRESHOLD: 0.15 meters (15cm)
        if min_dist < 0.15: 
            print(f"\n 💥 CRASH, Min Dist: {min_dist:.3f}m")
            
            # Calculate time driven
            run_time = time.time() - self.start_time
            
            # Report failure
            self.finish_run(run_time, crash_count=1)


    def odom_callback(self, msg):
        """Checks for timeouts and LAP COMPLETION."""
        if not self.is_running: return

        current_time = time.time() - self.start_time
        
        # --- 1. TIMEOUT CHECK ---
        if current_time > self.max_lap_time:
            print(f"❌ TIMEOUT! Car stuck or too slow ({current_time:.1f}s)")
            self.finish_run(current_time, crash_count=0) # Penalty for slow time
            return

        # --- 2. LAP COMPLETION CHECK ---
        # Calculate distance from (0,0)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        dist_from_start = np.sqrt(x**2 + y**2)


        # check for wrong way
        if dist_from_start < 3.0:
            orientation_z = msg.pose.pose.orientation.z

            # If z > 0.9, we are facing backwards
            if abs(orientation_z) > 0.9:
                print(f"🔄 CRASH: Facing Wrong Way (Z={orientation_z:.2f})")
                self.finish_run(current_time, crash_count=1)
                return

        # DEBUG for checking distance
        # print(f"DEBUG: Dist: {dist_from_start:.2f} | LeftStart: {getattr(self, 'has_left_start', False)}")

        # Initialize state if missing
        if not hasattr(self, 'has_left_start'):
            self.has_left_start = False

        # Phase A: Car must leave the start circle (> 2.0 meters)
        if not self.has_left_start:
            if dist_from_start > 2.0:
                self.has_left_start = True
                # print("DEBUG: Left Start Zone")

        # Phase B: Car returns to start circle (< 1.5 meters)
        elif self.has_left_start:
            if dist_from_start < 1.5:
                # SUCCESS!
                print(f"🏁 LAP COMPLETE! Time: {current_time:.3f}s 🏁")
                self.finish_run(current_time, crash_count=0)


    def finish_run(self, lap_time, crash_count):
        """finalizes the current car's attempt."""
        self.is_running = False
        
        # Send data to Manager
        self.ga.report_lap_result(lap_time, crash_count)

        # genetic_ML file deals with the printing to console for new record lap attempts
        
        # Loop immediately to next
        self.start_next_run()
        


def main():
    rclpy.init()
    node = GeneticSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()