from arm_controls import *

arm = Arm_Controller()
arm.reset_all_joints()

arm.set_clamp_value(1)

#target_position = [0.0, 0.23682, -0.4]
#target_position = [0.0, 1.0, -0.5]

# while True:
#     arm.move_arm_to_pos(target_position)