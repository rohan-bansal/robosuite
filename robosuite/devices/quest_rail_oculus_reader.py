import time
import threading
import numpy as np
import transforms3d as t3d

from oculus_reader import OculusReader


class QuestAgentRAIL:
    def __init__(
            self,
            robot_interface=None, # TODO can we do without it?
            robot_type="franka",
            bimanual=False,
            debug=False,
        ):

        self.robot_interface = robot_interface
        self.robot_type = robot_type
        self.bimanual = bimanual

        self.oculus_reader = OculusReader()

        self.controller_state_lock = threading.Lock() # lock to ensure safe access
        self.controller_state = None

        self._controller_names = ["l", "r"] # NOTE if there is one robot, it will use the "l" controller
        self._button_names = {
            "l": {
                "trigger_val": "leftTrig",
                "trigger_bool": "LTr",
                "grip_val": "leftGrip",
                "grip_bool": "LG",
            },
            "r": {
                "trigger_val": "rightTrig",
                "trigger_bool": "RTr",
                "grip_val": "rightGrip",
                "grip_bool": "RG",
            },
        }

        self.grip_pressed = False  # this can be used to control engage
        self.trigger_pressed = dict([(name, False) for name in self._controller_names])
        self._reset_pressed = False # whether we should reset the transform
        self.engaged = False # TODO(VS) why? remove, not useful
        # TODO above variables can perhaps be made local variables
        self.initialize_pose = True
        # self.pos_delta = [0, 0, 0]
        # self.last_pos = None

        # controller initial poses
        self.controller_init_pos = dict([(name, np.zeros(3)) for name in self._controller_names])
        self.controller_init_rot = dict([(name, np.array([1, 0, 0, 0])) for name in self._controller_names])
        self.ee_init_pos = dict([(name, np.zeros(3)) for name in self._controller_names])
        self.ee_init_rot = dict([(name, np.array([1, 0, 0, 0])) for name in self._controller_names])

        # # command poses
        # self.target_ori = self.ee_init_rot
        # self.target_pos = self.ee_init_pos
        # self.target_pose = (self.target_pos.tolist(), self.target_ori.tolist())

        # Golden offset for Grey Robot with Quest 3 # TODO(VS) why?
        self.controller_offset = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        self.debug = debug

        # Set controller offset from robot base frame.
        self.approx_controller_offset()

        if self.robot_interface is not None or self.debug:
            self.set_robot_transform()



### Example output from OculusReader.get_transformations_and_buttons()
# (
#     {
#         'l': array([[-0.15662  , -0.0793369,  0.177977 ,  0.012368 ],
#         [ 0.0902732, -0.231901 , -0.0239341, -0.0251249],
#         [ 0.172688 ,  0.0492721,  0.173929 , -0.0560221],
#         [ 0.       ,  0.       ,  0.       ,  1.       ]]),
#         'r': array([[ 0.923063 , -0.0438108, -0.382145 ,  0.183798 ],
#         [-0.351833 ,  0.305362 , -0.884854 , -0.303325 ],
#         [ 0.155459 ,  0.951228 ,  0.266455 , -0.102936 ],
#         [ 0.       ,  0.       ,  0.       ,  1.       ]])
#     },
#     {'A': False, 'B': False, 'RThU': True, 'RJ': False, 'RG': True, 'RTr': False, 'X': False, 'Y': False, 'LThU': True, 'LJ': False, 'LG': False, 'LTr': False, 'leftJS': (0.0, 0.0), 'leftTrig': (0.0,), 'leftGrip': (0.0,), 'rightJS': (0.0, 0.0), 'rightTrig': (0.0,), 'rightGrip': (1.0,)}
# )
## (VS) don't know what RThU, RJ are for

    def get_controller_state(self):
        with self.controller_state_lock:
            new_controller_state = {}

            # Get controller(s) data.
            controller_data = self.oculus_reader.get_transformations_and_buttons()
            # TODO check why need to handle empty controller_data ({}, {})
            while not controller_data or controller_data[0] == {}:  # busy wait for the headset to wake up
                time.sleep(0.001)
                controller_data = self.oculus_reader.get_transformations_and_buttons()

            transforms_data, buttons_data = controller_data

            # Parse button data.
            # Primary ('A') button, used to decide whether to save a demo
            if buttons_data["A"]:
                save_demo = True
            else:
                save_demo = False
            # Secondary ('B') button, use to delete the currently recording demo
            if buttons_data["B"]:
                delete_demo = True
            else:
                delete_demo = False
            if save_demo and delete_demo:
                # If both save and delete buttons are pressed, choose to save
                delete_demo = False

            # if buttons_data["A"] and not self._reset_pressed:
            #     self._reset_pressed = True
            #     print("Set controller offset orientation")
            #     print(f"original offset: {self.controller_offset}")
            #     self.approx_controller_offset() # TODO remove hardcode
            #     print(f"updated offset: {self.controller_offset}")
            # else:
            #     self._reset_pressed = buttons_data["A"]

            new_controller_state["save_demo"] = save_demo
            new_controller_state["delete_demo"] = delete_demo

            # Parse data per controller.
            for controller in transforms_data:
                new_controller_state[controller] = {}

                controller_state = transforms_data[controller]

                # Trigger
                if buttons_data[self._button_names[controller]["trigger_bool"]]:
                    if not self.trigger_pressed[controller]:
                        # trigger was just pressed, (re)initialize pose
                        self.initialize_pose = True
                    self.trigger_pressed[controller] = True
                else:
                    self.trigger_pressed[controller] = False
                    # self.pos_delta = [0, 0, 0]

                # Grip, used for gripper control
                if buttons_data[self._button_names[controller]["grip_bool"]]:
                    self.grip_pressed = True
                else:
                    self.grip_pressed = False

                if self.trigger_pressed[controller]: # Teleop only works if the trigger is pressed
                    # pos = controller_state["Position"]
                    # ori = controller_state["Rotation"]
                    pos_ori_mat = controller_state
                    controller_curr_pos = pos_ori_mat[:3, -1] # {"x": pos_ori_mat[0,-1], "y": pos_ori_mat[1,-1], "z": pos_ori_mat[2,-1]}
                    # ori = t3d.quaternions.mat2quat(pos_ori_mat[:3, :3]) # (w,x,y,z)
                    ori = pos_ori_mat[:3, :3]

                    if self.initialize_pose:
                        # once trigger is (re)pressed, rebase controller's initial pose to current pose, and compute deltas using it
                        self.controller_init_pos[controller] = controller_curr_pos
                        self.controller_init_rot[controller] = self.controller_offset @ ori
                        # reset self.ee_init_pos in case robot has moved;
                        # this will be added to dpos to compute absolute pose command for the robot
                        self.set_robot_transform()
                        if self.debug:
                            print("initialized pose")
                        self.initialize_pose = False

                    # Computing absolute action for the robot (in the robot's frame).
                    dpos = self.controller_offset @ (controller_curr_pos - self.controller_init_pos[controller]) # delta command pos
                    target_pos = self.ee_init_pos[controller] + dpos # abs command pos
                    if self.debug:
                        print(f"DEBUG get_controller_state(): target_pos: {target_pos}")
                    controller_curr_rot = self.controller_offset @ ori
                    rot_controller_delta = controller_curr_rot @ np.linalg.inv(self.controller_init_rot[controller])
                    dori = t3d.quaternions.mat2quat(rot_controller_delta)
                    target_ori = t3d.quaternions.mat2quat(rot_controller_delta @ t3d.quaternions.quat2mat(self.ee_init_rot[controller])).tolist() # (w, x, y, z) # abs command ori
                    target_pose = target_pos.tolist() + target_ori # abs command pose

                    # Computing delta action for the robot.
                    ee_curr_pose = self.robot_interface.last_eef_pose[{"l": 0, "r": 1}[controller]] # TODO save robot mapping in __init__
                    ee_curr_pos = ee_curr_pose[:3, 3]
                    ee_curr_rot = ee_curr_pose[:3, :3]
                    delta_pos = target_pos - ee_curr_pos # actual delta action
                    delta_rot = np.dot(t3d.quaternions.quat2mat(target_ori), np.linalg.inv(ee_curr_rot)) # (w, x, y, z)
                    delta_rot = t3d.quaternions.mat2quat(delta_rot).tolist() # (w, x, y, z)

                    # Gripper action.
                    if self.grip_pressed:
                        gripper_act = [1]
                    else:
                        gripper_act = [-1]

                    self.engaged = True # TODO maybe remove, not useful
                    new_controller_state[controller] = dict(
                        target_pose=target_pose,
                        target_pos=target_pos,
                        target_ori=target_ori,
                        dpos=dpos,
                        dori=dori,
                        delta_pos=delta_pos,
                        delta_ori=delta_rot,
                        gripper_act=gripper_act,
                        engaged=self.engaged, # TODO maybe remove, not useful
                    )
                    # return controller_state
                else:
                    self.engaged = False # TODO maybe remove, not useful
                    # new_controller_state[controller]['engaged'] = self.engaged # TODO maybe remove, not useful
                    # zero-ing delta actions; creates a minor gap b/w absolute and delta control
                    new_controller_state[controller] = dict(
                        dpos=[0, 0, 0],
                        dori=[1, 0, 0, 0], # (w, x, y, z)
                        delta_pos=[0, 0, 0],
                        delta_ori=[1, 0, 0, 0,], # (w, x, y, z)
                        engaged=self.engaged
                    )

            self.controller_state = self._nested_dict_update(self.controller_state, new_controller_state)

            return self.controller_state

    def _nested_dict_update(self, curr_dict, update_dict):
        for k, v in update_dict.items():
            if k in curr_dict and isinstance(v, dict) and isinstance(curr_dict[k], dict):
                curr_dict[k] = self._nested_dict_update(curr_dict[k], v)
            else:
                curr_dict[k] = v
        return curr_dict

    # def _get_robosuite_arm_pose(self, robot):
    #     # TODO goes to robosuite wrapper
    #     ee_pos = np.array(robot.sim.data.site_xpos[robot.sim.model.site_name2id(robot.controller.eef_name)])
    #     ee_ori_mat = np.array(
    #         robot.sim.data.site_xmat[robot.sim.model.site_name2id(robot.controller.eef_name)].reshape([3, 3])
    #     )
    #     pos_ori_mat = np.eye(4)
    #     pos_ori_mat[:3, :3] = ee_ori_mat
    #     pos_ori_mat[:3, -1] = ee_pos
    #     return pos_ori_mat

    def set_robot_transform(self):
        """
        Uses robot current pose to initialize controller_state
        """

        for i_robot in range(len(self.robot_interface._robots)):
            # Get robot's current pose.
            current_pose = self.robot_interface.last_eef_pose[i_robot]
            current_pos = current_pose[:3, 3]
            current_rot = current_pose[:3, :3]
            current_quat = t3d.quaternions.mat2quat(current_rot) # (w, x, y, z)
            print(f"robot {i_robot} current_pos: {current_pos}")

            # Set eef pose to robot's current pose.
            controller_name = self._controller_names[i_robot]
            self.ee_init_pos[controller_name] = np.array([current_pos[0], current_pos[1], current_pos[2]])
            self.ee_init_rot[controller_name] = current_quat

            # When called in __init__(), set controller state to robot's current pose.
            if self.controller_state is None:
                self.controller_state = dict([(name, None) for name in self._controller_names])
            if self.controller_state[controller_name] is None:
                print("resetting controller states to robot pose.")
                target_pose = current_pos.tolist() + current_quat.tolist()
                self.controller_state[controller_name] = dict(target_pose=target_pose, target_pos=current_pos.tolist(),
                                        target_ori=current_quat.tolist(), dpos=[0,0,0],
                                        dori=[1,0,0,0], gripper_act=[-1])
                print(self.controller_state)

    def TODO_approx_controller_offset(self):
        """
        Approximate a rotation matrix to its closest binary rotation matrix
        for rotations around the Z-axis.

        Parameters:
        R (numpy.ndarray): A 3x3 rotation matrix.

        Returns:
        numpy.ndarray: The closest binary rotation matrix for Z-axis rotation.
        """
        R = self.controller_offset
        # Define the binary rotation matrices around the Z-axis (90-degree increments)
        binary_matrices = [
            np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]),  # 0-degree rotation (Identity)
            np.array([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 1]]),  # 90-degree rotation ## NOTE(VS) looks like a -90 rotation around Z
            np.array([[-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 1]]),  # 180-degree rotation
            np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]]),  # 270-degree rotation ## NOTE(VS) looks like a -90 rotation around Z
        ]

        # Calculate the Frobenius norm distance to find the closest binary matrix
        closest_matrix = None
        min_distance = float('inf')

        for binary_matrix in binary_matrices:
            distance = np.linalg.norm(R - binary_matrix, 'fro')
            if distance < min_distance:
                min_distance = distance
                closest_matrix = binary_matrix

        self.controller_offset = closest_matrix

    def approx_controller_offset(self):
        # hard-coding the headset offset for now
        self.controller_offset = np.array([ # robot_T_headset
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0]
        ])
        # self.controller_offset = np.array([ # upside down headset
        #     [0, 0, -1],
        #     [1, 0, 0],
        #     [0, -1, 0]
        # ])

    def reset_internal_state(self):
        # self.trackpad_val = None

        self.trigger_pressed = dict([(name, False) for name in self._controller_names])
        self.initialize_pose = True

        self.controller_state = None
        self.set_robot_transform() # intializes robot pose and controller_state
        self.engaged = False # TODO maybe remove, not useful

if __name__ == "__main__":
    quest_controller = QuestAgent(debug=True)
    while True:
        state = quest_controller.get_controller_state()
        print(state)
        time.sleep(0.05)