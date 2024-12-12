import torch
import serial

from matrix import *

class Arm_Controller:
    def __init__(self, port="COM7", speed=0, acceleration=1, arm_lengths=[[0.23682, 0.03], 0.28015], default_joint_angles=[0, 0, 90]) -> None:
        self.joint_angles = [0, 0, 0] # degrees
        self.speed = speed
        self.acceleration = acceleration
        self.default_joint_angles = default_joint_angles # degrees
        self.arm_lengths = arm_lengths # meters

        self.ser = serial.Serial(port, baudrate=115200, dsrdtr=None)
        self.ser.setRTS(False)
        self.ser.setDTR(False)

        self.reset_all_joints()

        self.ang_a = torch.tensor(0.0, requires_grad=True)
        self.ang_b = torch.tensor(0.0, requires_grad=True)
        self.ang_c = torch.tensor(0.0, requires_grad=True)

        self.learning_rate = 1
        self.optimizer = torch.optim.SGD([self.ang_a, self.ang_b, self.ang_c], lr=self.learning_rate)

    # Degrees
    def rotate_joint(self, joint, angle):
        command = '{"T":121,"joint":' + str(int(joint)) + ',"angle":' + str(int(angle)) + ',"spd":' + str(self.speed) + ',"acc":' + str(self.acceleration) + '}'
        try:
            self.ser.write(command.encode() + b'\n')
            self.joint_angles[joint - 1] = angle
        except:
            print("ERROR: FAILED TO SEND DIRECTIONS TO ARM")

    # Value from 1.08 (Closed) to 3.14 (Open)
    # Function does normalize the value to 0 - 1
    def set_clamp_value(self, value):
        value = value*(3.14 - 1.08) + 1.08

        command = '{"T":106,"cmd":' + str(value) + ',"spd":0,"acc":0}'
        try:
            self.ser.write(command.encode() + b'\n')
        except:
            print("ERROR: FAILED TO SEND DIRECTIONS TO ARM")

    # Degrees
    def rotate_all_joints(self, base_angle, shoulder_angle, elbow_angle):
        command = '{"T":122,"b":' + str(int(base_angle)) + ',"s":' + str(int(shoulder_angle)) + ',"e":' + str(int(elbow_angle)) + ',"h":' + str(180) + ',"spd":' + str(self.speed) + ',"acc":' + str(self.speed) + '}'
        try:
            self.ser.write(command.encode() + b'\n')
            self.joint_angles = [base_angle, shoulder_angle, elbow_angle]
        except:
            print("ERROR: FAILED TO SEND DIRECTIONS TO ARM")

    def reset_all_joints(self):
        self.rotate_all_joints(self.default_joint_angles[0], self.default_joint_angles[1], self.default_joint_angles[2])
        self.joint_angles = self.default_joint_angles

    def get_arm_coords(self):
        arm_a_vec = Vector(0, self.arm_lengths[0][0], -self.arm_lengths[0][1])
        arm_a_vec = Matrix.get_rot_mat(0, -self.joint_angles[0], 0)*Matrix.get_rot_mat(-self.joint_angles[1], 0, 0)*arm_a_vec

        arm_b_vec = Vector(0, self.arm_lengths[1], 0)
        arm_b_vec = Matrix.get_rot_mat(0, -self.joint_angles[0], 0)*Matrix.get_rot_mat(-self.joint_angles[1] - self.joint_angles[2], 0, 0)*arm_b_vec

        return arm_a_vec+arm_b_vec

    def move_arm_to_pos(self, position, loop_iterations=100):
        target_position = torch.Tensor(position)

        for i in range(loop_iterations):
            self.optimizer.zero_grad()

            arm_a_vec = torch.tensor([0.0, self.arm_lengths[0][0], -self.arm_lengths[0][1]], requires_grad=True)
            arm_b_vec = torch.tensor([0.0, self.arm_lengths[1], 0.0], requires_grad=True)

            yaw_rot_mat = torch.stack([torch.cos(self.ang_a), torch.tensor(0.0), -torch.sin(self.ang_a),
                               torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0),
                               torch.sin(self.ang_a), torch.tensor(0.0), torch.cos(self.ang_a)]).reshape(3, 3)
            arm_a_x_rot_mat = torch.stack([torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0),
                                        torch.tensor(0.0), torch.cos(self.ang_b), torch.sin(self.ang_b),
                                        torch.tensor(0.0), -torch.sin(self.ang_b), torch.cos(self.ang_b)]).reshape(3, 3)
            arm_b_x_rot_mat = torch.stack([torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0),
                                        torch.tensor(0.0), torch.cos(self.ang_b + self.ang_c), torch.sin(self.ang_b + self.ang_c),
                                        torch.tensor(0.0), -torch.sin(self.ang_b + self.ang_c), torch.cos(self.ang_b + self.ang_c)]).reshape(3, 3)
            
            arm_a_vec = torch.matmul(yaw_rot_mat, torch.matmul(arm_a_x_rot_mat, arm_a_vec))
            arm_b_vec = torch.matmul(yaw_rot_mat, torch.matmul(arm_b_x_rot_mat, arm_b_vec))

            arm_coords = arm_a_vec + arm_b_vec

            # x_pos = (-torch.sin(self.ang_a)*(-torch.sin(self.ang_b)*self.arm_lengths[0][0] + torch.cos(self.ang_b)*(-self.arm_lengths[0][1]))) + (-torch.sin(self.ang_a)*(-torch.sin(self.ang_b + self.ang_c)*self.arm_lengths[1]))
            # y_pos = (torch.cos(self.ang_b)*self.arm_lengths[0][0] + torch.sin(self.ang_b)*(-self.arm_lengths[0][1])) + (torch.cos(-self.ang_b - self.ang_c)*self.arm_lengths[1])
            # z_pos = (torch.cos(self.ang_a)*(-torch.sin(self.ang_b)*self.arm_lengths[0][0] + torch.cos(self.ang_b)*(-self.arm_lengths[0][1]))) + (torch.cos(self.ang_a)*(-torch.sin(self.ang_b + self.ang_c)*self.arm_lengths[1]))

            #print(target_position, x_pos, y_pos, z_pos)

            loss = (target_position[0] - arm_coords[0])**2 + (target_position[1] - arm_coords[1])**2 + (target_position[2] - arm_coords[2])**2
            loss.backward()
            self.optimizer.step()

        self.rotate_all_joints(self.ang_a.item()*180/pi, self.ang_b.item()*180/pi, self.ang_c.item()*180/pi)
        self.joint_angles = [self.ang_a.item()*180/pi, self.ang_b.item()*180/pi, self.ang_c.item()*180/pi]
        #print(self.joint_angles)
       # print(self.joint_angles)
        #print(ang_a, ang_b, ang_c)

    def iterate_inverse_kinematics(self, position):
        target_position = torch.Tensor(position)

        self.optimizer.zero_grad()

        arm_a_vec = torch.tensor([0.0, self.arm_lengths[0][0], -self.arm_lengths[0][1]], requires_grad=True)
        arm_b_vec = torch.tensor([0.0, self.arm_lengths[1], 0.0], requires_grad=True)

        yaw_rot_mat = torch.stack([torch.cos(self.ang_a), torch.tensor(0.0), -torch.sin(self.ang_a),
                            torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0),
                            torch.sin(self.ang_a), torch.tensor(0.0), torch.cos(self.ang_a)]).reshape(3, 3)
        arm_a_x_rot_mat = torch.stack([torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0),
                                    torch.tensor(0.0), torch.cos(self.ang_b), torch.sin(self.ang_b),
                                    torch.tensor(0.0), -torch.sin(self.ang_b), torch.cos(self.ang_b)]).reshape(3, 3)
        arm_b_x_rot_mat = torch.stack([torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0),
                                    torch.tensor(0.0), torch.cos(self.ang_b + self.ang_c), torch.sin(self.ang_b + self.ang_c),
                                    torch.tensor(0.0), -torch.sin(self.ang_b + self.ang_c), torch.cos(self.ang_b + self.ang_c)]).reshape(3, 3)
        
        arm_a_vec = torch.matmul(yaw_rot_mat, torch.matmul(arm_a_x_rot_mat, arm_a_vec))
        arm_b_vec = torch.matmul(yaw_rot_mat, torch.matmul(arm_b_x_rot_mat, arm_b_vec))

        arm_coords = arm_a_vec + arm_b_vec

        # x_pos = (-torch.sin(self.ang_a)*(-torch.sin(self.ang_b)*self.arm_lengths[0][0] + torch.cos(self.ang_b)*(-self.arm_lengths[0][1]))) + (-torch.sin(self.ang_a)*(-torch.sin(self.ang_b + self.ang_c)*self.arm_lengths[1]))
        # y_pos = (torch.cos(self.ang_b)*self.arm_lengths[0][0] + torch.sin(self.ang_b)*(-self.arm_lengths[0][1])) + (torch.cos(-self.ang_b - self.ang_c)*self.arm_lengths[1])
        # z_pos = (torch.cos(self.ang_a)*(-torch.sin(self.ang_b)*self.arm_lengths[0][0] + torch.cos(self.ang_b)*(-self.arm_lengths[0][1]))) + (torch.cos(self.ang_a)*(-torch.sin(self.ang_b + self.ang_c)*self.arm_lengths[1]))

        #print(target_position, x_pos, y_pos, z_pos)

        loss = (target_position[0] - arm_coords[0])**2 + (target_position[1] - arm_coords[1])**2 + (target_position[2] - arm_coords[2])**2
        loss.backward()
        self.optimizer.step()

        self.rotate_all_joints(self.ang_a.item()*180/pi, self.ang_b.item()*180/pi, self.ang_c.item()*180/pi)
        self.joint_angles = [self.ang_a.item()*180/pi, self.ang_b.item()*180/pi, self.ang_c.item()*180/pi]

    def close_serial_port(self):
        self.ser.close()