import os
import sys
import multiprocessing
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crc_module import get_crc

import numpy as np
import torch
import faulthandler
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from unitree_go.msg import (
    WirelessController,
    LowState,
    MotorState,
    IMUState,
    LowCmd,
    MotorCmd,
)
import time
from web_hand import *
import atexit
import socket
from dynamixel_client import DynamixelClient

POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0
HW_DOF = 20

WALK_STRAIGHT = False
LOG_DATA = False
USE_GRIPPPER = False
NO_MOTOR = False

def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

def euler_from_quat(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]
    y = quat_angle[:,1]
    z = quat_angle[:,2]
    w = quat_angle[:,3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z

def isWeak(motor_index):
    return motor_index == 10 or motor_index == 11 or motor_index == 12 or motor_index == 13 or \
        motor_index == 14 or motor_index == 15 or motor_index == 16 or motor_index == 17 or \
        motor_index == 18 or motor_index == 19

def cleanup(ser_l,ser_r):
    print("closing ports")
    ser_r.flush()
    ser_r.reset_input_buffer()
    ser_r.reset_output_buffer()
    ser_l.flush()
    ser_l.reset_input_buffer()
    ser_l.reset_output_buffer()
    ser_r.close()
    ser_l.close()

class H1():
    def __init__(self,task='stand'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task = task

        self.num_envs = 1 
        if self.task=='stand':
            self.num_observations = 62 + 11 + 5
            self.num_actions = 11
        elif self.task=='stand_w_waist':
            self.num_observations = 62 + 11 + 5
            self.num_actions = 11
        elif self.task=='wb' or self.task=='squat':
            self.num_observations = 62 + 19 + 5
            self.num_actions = 19
        self.num_privileged_obs = None
        self.obs_context_len=8

        self.scale_lin_vel = 1.0
        self.scale_ang_vel = 1.0
        self.scale_orn = 1.0
        self.scale_dof_pos = 1.0
        self.scale_dof_vel = 1.0
        self.scale_action = 1.0

        # prepare action deployment joint positions offsets and PD gains
        if self.task=='stand':
            arm_pgain = 40
        elif self.task=='stand_w_waist' or self.task=='squat' or self.task=='wb':
            arm_pgain = 30

        
        arm_dgain = 1
        leg_pgain = 200
        leg_dgain = 5
        knee_pgain = 200
        knee_dgain = 5

        if self.task=='stand' or self.task=='stand_w_waist':
            ankle_pgain = 80
            ankle_dgain = 2
        elif self.task=='wb' or self.task=='squat':
            ankle_pgain = 50
            ankle_dgain = 1
        self.p_gains = np.array([leg_pgain,leg_pgain,knee_pgain,leg_pgain,leg_pgain,knee_pgain,leg_pgain,leg_pgain,leg_pgain,0,ankle_pgain,ankle_pgain,arm_pgain,arm_pgain,arm_pgain,arm_pgain,arm_pgain,arm_pgain,arm_pgain,arm_pgain],dtype=np.float64)
        self.d_gains = np.array([leg_dgain,leg_dgain,knee_dgain,leg_dgain,leg_dgain,knee_dgain,leg_dgain,leg_dgain,leg_dgain,0,ankle_dgain,ankle_dgain,arm_dgain,arm_dgain,arm_dgain,arm_dgain,arm_dgain,arm_dgain,arm_dgain,arm_dgain],dtype=np.float64)
        self.joint_limit_lo = [-0.43,-1.57,-0.26,-0.43,-1.57,-0.26,-2.35,-0.43,-0.43,0,-0.87,-0.87,-2.87,-3.11,-4.45,-1.25,-2.87,-0.34,-1.3,-1.25]
        self.joint_limit_hi = [0.43,1.57,2.05,0.43,1.57,2.05,2.35,0.43,0.43,0,0.52,0.52,2.87,0.34,1.3,2.61,2.87,3.11,4.45,2.61]
        self.default_dof_pos_np = np.array([0.0,-10/180*np.pi,20/180*np.pi,0.0,-10/180*np.pi,\
                                            20/180*np.pi,0.0,0.42,-0.42,0.0,\
                                                -10/180*np.pi, -10/180*np.pi,0.0,0.0,0.0,\
                                                    0.0,0.0,0.0,0.0,0.0])
        default_dof_pos = torch.tensor(self.default_dof_pos_np, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = default_dof_pos.unsqueeze(0)


        # prepare osbervations buffer
        self.obs_buf = torch.zeros(1, self.num_observations, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_history_buf = torch.zeros(1, self.obs_context_len, self.num_observations, dtype=torch.float, device=self.device, requires_grad=False)


class DeployNode(Node):
    def __init__(self,task='stand'):
        super().__init__("deploy_node")  # type: ignore
        self.task = task
        if self.task not in ['stand','stand_w_waist','wb','squat']:
            self.get_logger().info("Invalid task")
            raise SystemExit
        
        # init subcribers & publishers
        self.joy_stick_sub = self.create_subscription(WirelessController, "wirelesscontroller", self.joy_stick_cb, 1)
        self.joy_stick_sub  # prevent unused variable warning
        self.lowlevel_state_sub = self.create_subscription(LowState, "lowstate", self.lowlevel_state_cb, 1)  # "/lowcmd" or  "lf/lowstate" (low frequencies)
        self.lowlevel_state_sub  # prevent unused variable warning

        self.low_state = LowState()
        self.joint_pos = np.zeros(HW_DOF)
        self.joint_vel = np.zeros(HW_DOF)

        self.motor_pub = self.create_publisher(LowCmd, "lowcmd", 1)
        self.motor_pub_freq = 50
        self.cmd_msg = LowCmd()

        # init motor command
        self.motor_cmd = []
        for id in range(HW_DOF):
            if isWeak(id):
                mode = 0x01
            else:
                mode = 0x0A
            cmd=MotorCmd(q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, mode=mode, reserve=[0,0,0])
            self.motor_cmd.append(cmd)
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()

        # init policy
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_policy()
        self.prev_action = np.zeros(self.env.num_actions)
        self.start_policy = False

        # init multiprocess values
        self.wrist_l = multiprocessing.Value('f', 0.)
        self.wrist_r = multiprocessing.Value('f', 0.)
        self.body_ref = multiprocessing.Array('f', [0]*24) #[0]*12+[0.3]+[0]*3+[-0.3]+[0]*7)
        self.hand_ref = multiprocessing.Array('f', [160]*6+[0]+[160]*7+[0]+[160])

        # standing up
        self.get_logger().info("Standing up")
        self.stand_up = False
        self.stand_up = True

        # start
        self.start_time = time.monotonic()
        self.get_logger().info("Press L2 to start policy")
        self.get_logger().info("Press L1 for emergent stop")
        self.init_buffer = 0
        self.foot_contact_buffer = []
        self.time_hist = []
        self.obs_time_hist = []
        self.angle_hist = []
        self.action_hist = []
        self.dof_pos_hist = []
        self.dof_vel_hist = []
        self.imu_hist = []
        self.ang_vel_hist = []
        self.foot_contact_hist = []
        self.tau_hist = []
        self.obs_hist = []

    def reindex_urdf2hw(self, vec):
        vec = np.array(vec)
        assert len(vec)==19, "wrong dim for reindex"
        hw_vec = vec[[6, 7, 8, 1, 2, 3, 10, 0, 5, 0, 4, 9, 15, 16, 17, 18, 11, 12, 13, 14]]
        hw_vec[9] = 0
        return hw_vec

    def reindex_hw2urdf(self, vec):
        vec = np.array(vec)
        assert len(vec)==20, "wrong dim for reindex"
        return vec[[7, 3, 4, 5, 10, 8, 0, 1, 2, 11, 6, 16, 17, 18, 19, 12, 13, 14, 15]]
        
    ##############################
    # subscriber callbacks
    ##############################

    def joy_stick_cb(self, msg):
        if msg.keys == 2:  # L1: emergency stop
            self.get_logger().info("Emergency stop")
            self.set_gains(np.array([0.0]*HW_DOF),self.env.d_gains)
            self.set_motor_position(q=self.env.default_dof_pos_np)
            if LOG_DATA:
                print("Saving data")
                np.savez('captured_data.npz', action=np.array(self.action_hist), dof_pos=np.array(self.dof_pos_hist),
                        dof_vel=np.array(self.dof_vel_hist),imu=np.array(self.imu_hist),ang_vel=np.array(self.ang_vel_hist),
                        tau=np.array(self.tau_hist), obs=np.array(self.obs_hist))
            raise SystemExit
        if msg.keys == 32:  # L2: start policy
            if self.stand_up:
                self.get_logger().info("Start policy")
                self.start_policy = True
                self.policy_start_time = time.monotonic()
            else:
                self.get_logger().info("Wait for standing up first")

    def lowlevel_state_cb(self, msg: LowState):
        # imu data
        imu_data = msg.imu_state
        self.msg_tick = msg.tick/1000
        self.roll, self.pitch, self.yaw = imu_data.rpy
        self.obs_ang_vel = np.array(imu_data.gyroscope)*self.env.scale_ang_vel
        self.obs_imu = np.array([self.roll, self.pitch])*self.env.scale_orn

        # termination condition
        r_threshold = abs(self.roll) > 0.5
        p_threshold = abs(self.pitch) > 0.5
        if r_threshold or p_threshold:
            self.get_logger().warning("Roll or pitch threshold reached")

        # motor data
        self.joint_tau = [msg.motor_state[i].tau_est for i in range(HW_DOF)]
        self.joint_pos = [msg.motor_state[i].q for i in range(HW_DOF)]
        self.obs_joint_pos = (np.array(self.joint_pos) - self.env.default_dof_pos_np) * self.env.scale_dof_pos
        joint_vel = [msg.motor_state[i].dq for i in range(HW_DOF)]
        self.obs_joint_vel = np.array(joint_vel) * self.env.scale_dof_vel

        
    ##############################
    # motor commands
    ##############################

    def set_gains(self, kp: np.ndarray, kd: np.ndarray):
        self.kp = kp
        self.kd = kd
        for i in range(HW_DOF):
            self.motor_cmd[i].kp = kp[i]  #*0.5
            self.motor_cmd[i].kd = kd[i]  #*3

    def set_motor_position(
        self,
        q: np.ndarray,
    ):
        for i in range(HW_DOF):
            self.motor_cmd[i].q = q[i]
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        self.cmd_msg.crc = get_crc(self.cmd_msg)

    ##############################
    # deploy policy
    ##############################
    def init_policy(self):
        self.get_logger().info("Preparing policy")
        faulthandler.enable()

        # prepare environment
        self.env = H1(task=self.task)

        # load policy
        file_pth = os.path.dirname(os.path.realpath(__file__))
        self.policy = torch.jit.load(os.path.join(file_pth, "./ckpt/policy.pt"), map_location=self.env.device)  #0253 396
        self.policy.to(self.env.device)
        actions = self.policy(self.env.obs_history_buf.detach())  # first inference takes longer time

        # init p_gains, d_gains, torque_limits
        for i in range(HW_DOF):
            self.motor_cmd[i].q = self.env.default_dof_pos[0][i].item()
            self.motor_cmd[i].dq = 0.0
            self.motor_cmd[i].tau = 0.0
            self.motor_cmd[i].kp = 0.0  # self.env.p_gains[i]  # 30
            self.motor_cmd[i].kd = 0.0  # float(self.env.d_gains[i])  # 0.6
        self.cmd_msg.motor_cmd = self.motor_cmd.copy()
        self.angles = self.env.default_dof_pos

    def get_retarget(self):
        target_jt = self.body_ref[:19]
        if self.task == "stand":
            reference_pose = self.reindex_urdf2hw(target_jt) + self.env.default_dof_pos_np
            reference_pose_clip = np.clip(reference_pose, self.env.joint_limit_lo, self.env.joint_limit_hi)
            reference_pose_clip[:12] = self.env.default_dof_pos_np[:12]
            return reference_pose_clip
        elif self.task == "stand_w_waist":
            target_jt[10]*=2
            pose = np.array(self.body_ref[19:])
            reference_pose = self.reindex_urdf2hw(target_jt) + self.env.default_dof_pos_np
            reference_pose[:6] = self.env.default_dof_pos_np[:6]
            reference_pose[7:12] = self.env.default_dof_pos_np[7:12]
            return reference_pose, pose  # no clip for the target
        elif self.task == "wb" or self.task == "squat":
            pose = np.array(self.body_ref[19:])
            reference_pose = self.reindex_urdf2hw(target_jt) + self.env.default_dof_pos_np
        return reference_pose, pose  # no clip for the target

    @torch.no_grad()
    def main_loop(self):
        # keep stand up pose first
        standup_id = 0
        while self.stand_up and not self.start_policy:
            if standup_id==999:
                print("---Initialized---")
            time_ratio = min(1,standup_id / 1000)
            self.set_gains(kp=time_ratio * self.env.p_gains, kd=time_ratio * self.env.d_gains)
            self.set_motor_position(q=self.env.default_dof_pos_np)
            actions = self.policy(self.env.obs_history_buf.detach())  # first inference takes longer time
            if not NO_MOTOR:
                self.motor_pub.publish(self.cmd_msg)
            rclpy.spin_once(self)
            standup_id+=1

        cnt = 0
        fps_ckt = time.monotonic()
        self.obs_tick = self.msg_tick
        target_jt_hw = self.env.default_dof_pos_np
        pose = np.array([0]*5)
        # udp sending obs
        obs_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        obs_receiver_address = ('192.168.123.164', 5704)
        self.get_logger().info("start main loop")
        
        while rclpy.ok():
            loop_start_time = time.monotonic()

            # spin stuff
            rclpy.spin_once(self,timeout_sec=0.005)
            if self.msg_tick == self.obs_tick:
                rclpy.spin_once(self,timeout_sec=0.005)
            self.obs_tick = self.msg_tick

            if self.start_policy:
                # policy inference
                if cnt % 2 == 0:
                    if self.task == "stand":
                        target_jt_hw = self.get_retarget()
                    elif self.task == "stand_w_waist" or self.task=='wb' or self.task=='squat':
                        target_jt_hw,pose = self.get_retarget()

                # fetch proprioceptive data
                self.obs_joint_vel_ = self.reindex_hw2urdf(self.obs_joint_vel)
                self.obs_joint_pos_ = self.reindex_hw2urdf(self.obs_joint_pos)
                if self.task == "stand" or self.task == "stand_w_waist" or self.task=='squat':
                    self.obs_buf_np = np.concatenate((self.obs_imu, self.obs_ang_vel, self.obs_joint_pos_, self.obs_joint_vel_, self.prev_action, self.reindex_hw2urdf(target_jt_hw)*self.env.scale_dof_pos, np.zeros(5)))  # add reference input TODO
                elif self.task=='wb':
                    self.obs_buf_np = np.concatenate((self.obs_imu, self.obs_ang_vel, self.obs_joint_pos_, self.obs_joint_vel_, self.prev_action, self.reindex_hw2urdf(target_jt_hw)*self.env.scale_dof_pos, pose))  # add reference input TODO
                obs_buf = torch.tensor(self.obs_buf_np,dtype=torch.float, device=self.device).unsqueeze(0)
                self.env.obs_history_buf = torch.cat([
                    self.env.obs_history_buf[:, 1:],
                    obs_buf.unsqueeze(1)
                ], dim=1)

                # send obs through udp
                # qpos 19, qvel 19, orn 3, ang_vel 3, wrist 2
                obs_imitation = np.concatenate((self.obs_joint_pos_, self.obs_joint_vel_,[self.roll, self.pitch, self.yaw], self.obs_ang_vel, [self.wrist_l.value, self.wrist_r.value]))
                obs_imitation = np.array(obs_imitation).astype(np.float32)
                obs_bytes = obs_imitation.tobytes()
                obs_sock.sendto(obs_bytes, obs_receiver_address)

                if self.init_buffer < 0:
                    self.init_buffer += 1
                    raw_actions = self.policy(self.env.obs_history_buf.detach())
                    self.set_motor_position(q=self.env.default_dof_pos_np)
                    if not NO_MOTOR:
                        self.motor_pub.publish(self.cmd_msg)
                else:
                    if LOG_DATA:
                        self.dof_pos_hist.append(self.obs_joint_pos_)
                        self.dof_vel_hist.append(self.obs_joint_vel_)
                        self.imu_hist.append(self.obs_imu)
                        self.ang_vel_hist.append(self.obs_ang_vel)
                        self.tau_hist.append(self.joint_tau)
                        self.obs_hist.append(self.obs_buf_np)

                    raw_actions = self.policy(self.env.obs_history_buf.detach())
                    if torch.any(torch.isnan(raw_actions)):
                        self.get_logger().info("Emergency stop due to NaN")
                        self.set_gains(np.array([0.0]*HW_DOF),self.env.d_gains)
                        self.set_motor_position(q=self.env.default_dof_pos_np)
                        raise SystemExit
                    self.prev_action = raw_actions.clone().detach().cpu().numpy().squeeze(0)
                    
                    if self.task=="stand" or self.task == "stand_w_waist":
                        whole_body_action = np.concatenate((raw_actions.clone().detach().cpu().numpy().squeeze(0),[0]*8))
                        angles = self.reindex_urdf2hw(whole_body_action) * self.env.scale_action + self.env.default_dof_pos_np
                        angles[-8:] = target_jt_hw[-8:]
                        self.angles = np.clip(angles, self.env.joint_limit_lo, self.env.joint_limit_hi)
                        inference_time=time.monotonic()-loop_start_time
                        while 0.009-time.monotonic()+loop_start_time > 0:
                            pass
                    elif self.task=='wb' or self.task=='squat':
                        angles = self.reindex_urdf2hw(self.prev_action) * self.env.scale_action + self.env.default_dof_pos_np
                        self.angles = np.clip(angles, self.env.joint_limit_lo, self.env.joint_limit_hi)
                        inference_time=time.monotonic()-loop_start_time
                        self.get_logger().info(f"inference time: {inference_time}")
                        while 0.01-time.monotonic()+loop_start_time > 0:
                            pass
                    if LOG_DATA:
                        self.action_hist.append(self.prev_action)
                    if np.any(np.isnan(target_jt_hw)):
                        self.get_logger().info("Emergency stop due to NaN")
                        self.set_gains(np.array([0.0]*HW_DOF),self.env.d_gains)
                        self.set_motor_position(q=self.env.default_dof_pos_np)
                        raise SystemExit
                    self.set_motor_position(self.angles)
                    if not NO_MOTOR:
                        self.motor_pub.publish(self.cmd_msg)
                
            while 0.019969-time.monotonic()+loop_start_time>0:  #0.012473  0.019963
                pass
            cnt+=1
            if cnt == 500:
                dt = (time.monotonic()-fps_ckt)/cnt
                cnt = 0
                fps_ckt = time.monotonic()
                print(f"FPS: current dt ={dt}")



##############################
# hand controller process
##############################
def hand_process(wrist_l,wrist_r,hand_ref):
    test_start_time = time.time()
    print('open port')
    ser_r = openSerial('/dev/HandR', 115200)
    ser_l = openSerial('/dev/HandL', 57600)
    atexit.register(cleanup, ser_l,ser_r)

    print("init dynamixel")
    dxl_motor_ids = [1,2] #left:1, right:2
    dxl_port = '/dev/WRIST'
    dxl_client = DynamixelClient(dxl_motor_ids, port=dxl_port, lazy_connect=True)
    if not NO_MOTOR:
        dxl_client.set_torque_enabled(dxl_motor_ids, True)
    dxl_client.write_desired_pgain(dxl_motor_ids, np.array([1600]*2))
    dxl_client.write_desired_dgain(dxl_motor_ids, np.array([0]*2))

    if not NO_MOTOR:
        write6(ser_r, 2, 'speedSet', [1000, 1000, 1000, 1000, 1000, 1000])
        write6(ser_r, 2, 'forceSet', [500, 500, 500, 500, 500, 500])
        write6(ser_r, 2, 'angleSet', [1000, 1000, 1000, 1000, 1000, 1000])
        write6(ser_l, 1, 'speedSet', [1000, 1000, 1000, 1000, 1000, 1000])
        write6(ser_l, 1, 'forceSet', [500, 500, 500, 500, 500, 500])
        write6(ser_l, 1, 'angleSet', [1000, 1000, 1000, 1000, 1000, 1000])
    time.sleep(0.5)
    finger_cmd_r = np.array([-1]*6)
    finger_cmd_l = np.array([-1]*6)
    finger_angle_l_buf = []
    finger_angle_r_buf = []
    buffer_len = 12

    try:
        while rclpy.ok():
            start_time = time.time()

            requested_l = hand_ref[:8]
            requested_r = hand_ref[8:]
            finger_angle_r_buf.append(requested_r)
            if len(finger_angle_r_buf) > buffer_len:
                finger_angle_r_buf.pop(0)
            finger_angle_r = np.mean(finger_angle_r_buf,axis=0)

            dist_r = (finger_angle_r[7] - 0.03)/(0.1-0.03)
            finger_cmd_r[:3] = (np.clip((finger_angle_r[:3]-30)/(160-30)*1000,0,1000)).astype(int)  # range 30-160 30-close-cmd:0
            finger_cmd_r[3] = (np.clip(dist_r*(1000-490)+490,490,1000)).astype(int)
            finger_cmd_r[4] = (np.clip(dist_r*(1000-630)+630,630,1000)).astype(int)
            lat_r = (np.clip((finger_angle_r[5]-65)/(85-65)*1000,0,1000)).astype(int)
            if lat_r < 150:
                finger_cmd_r[5] = 50 #(np.clip((finger_angle_r[5]-65)/(85-65)*1000,0,1000)).astype(int) 
            else:
                finger_cmd_r[5] = lat_r
            wrist_ang_r = np.clip(-(finger_angle_r[6] + 0*30)*np.pi/180,-np.pi,np.pi)  #  30 -90
            
            finger_angle_l_buf.append(requested_l)
            if len(finger_angle_l_buf) > buffer_len:
                finger_angle_l_buf.pop(0)
            finger_angle_l = np.mean(finger_angle_l_buf,axis=0)

            dist_l = (finger_angle_l[7] - 0.03)/(0.1-0.03)
            finger_cmd_l[:3] = (np.clip((finger_angle_l[:3]-30)/(160-30)*1000,0,1000)).astype(int)  # range 30-160 30-close-cmd:0
            finger_cmd_l[3] = (np.clip(dist_l*(1000-490)+490,490,1000)).astype(int)
            finger_cmd_l[4] = (np.clip(dist_l*(1000-535)+535,535,1000)).astype(int)
            lat_l = 1000-(np.clip((finger_angle_l[5]-85)/(110-85)*1000,0,1000)).astype(int) 
            if lat_l < 300:
                finger_cmd_l[5] = 150 #1000-(np.clip((finger_angle_l[5]-85)/(110-85)*1000,0,1000)).astype(int)
            else:
                finger_cmd_l[5] = lat_l
            wrist_ang_l = np.clip((finger_angle_l[6] + 0*30)*np.pi/180,-np.pi,np.pi)  #  30 -90

            dxl_pos, _, _ = dxl_client.read_pos_vel_cur()
            if not NO_MOTOR:
                dxl_client.write_desired_pos(dxl_motor_ids, np.array([wrist_ang_l+np.pi,wrist_ang_r+np.pi]))

            wrist_l.value = dxl_pos[0] % (2*np.pi) - np.pi  # -pi~pi
            wrist_r.value = dxl_pos[1] % (2*np.pi)- np.pi

            if not NO_MOTOR:
                write6(ser_r, 2, 'angleSet', finger_cmd_r)
                write6(ser_l, 1, 'angleSet', finger_cmd_l)
            time.sleep(max(0, 0.033 - (time.time() - start_time)))

    finally:
        print("closing port")
        dxl_client.set_torque_enabled(dxl_motor_ids, False)

        ser_r.flush()
        ser_r.reset_input_buffer()
        ser_r.reset_output_buffer()
        ser_l.flush()
        ser_l.reset_input_buffer()
        ser_l.reset_output_buffer()
        ser_r.close()
        ser_l.close()

##############################
# udp process
##############################
def udp_body_recv(body_ref):
    sock_body = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    body_server_address = ('192.168.123.163', 5701) 
    sock_body.bind(body_server_address)

    while True:
        # get hand data: udp
        body_data, address = sock_body.recvfrom(300)
        target_jt = np.frombuffer(body_data, dtype=np.float32)
        body_ref[:] = target_jt[:]
        
def udp_hand_recv(hand_ref):
    sock_hand = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hand_server_address = ('192.168.123.163', 5702) 
    sock_hand.bind(hand_server_address)

    while True:
        # get hand data: udp
        hand_data, address = sock_hand.recvfrom(300)
        hand_arr = np.frombuffer(hand_data, dtype=np.float32)
        hand_ref[:] = hand_arr[:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name: stand, stand_w_waist, wb, squat', required=False, default='stand')
    args = parser.parse_args()
    
    rclpy.init(args=None)
    dp_node = DeployNode(args.task_name)
    dp_node.get_logger().info("Deploy node started")
    body_udp_process = multiprocessing.Process(target=udp_body_recv,args=(dp_node.body_ref,))
    hand_udp_process = multiprocessing.Process(target=udp_hand_recv,args=(dp_node.hand_ref,))
    serial_process = multiprocessing.Process(target=hand_process,args=(dp_node.wrist_l,dp_node.wrist_r,dp_node.hand_ref,))
    body_udp_process.daemon = True
    hand_udp_process.daemon = True
    serial_process.daemon = True 
    body_udp_process.start()
    hand_udp_process.start()
    serial_process.start()

    dp_node.main_loop()
    rclpy.shutdown()
    body_udp_process.join()
    hand_udp_process.join()
    serial_process.join()