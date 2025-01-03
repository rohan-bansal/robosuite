import time
import threading

import numpy as np
import quaternion

from oculus_reader import OculusReader

from robosuite.devices import Device
import robosuite.utils.transform_utils as transform_utils

class Quest(Device):
    def __init__(self, robots=None, debug=False):
        """
        # TODO maybe find a way to not need the robot interface passed into client

        Args:
            robots: the robots in the robosuite environment
            debug:
        """
        self.robots = robots
        self.debug = debug

        # Define reference frames and transforms


        # Default offset
        controller_offset = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])


        self.controller_offset = quaternion.from_rotation_matrix(controller_offset)
        self.controller_offset_reset = False  # whether we should reset the transform

        self.oculus_reader = OculusReader()

        # (pos, ori)
        self.grip_pressed = False  # this can be used to control engage

        self.pos_delta = [0, 0, 0]
        self.last_pos = None

        # TODO these variables below can perhaps be made local variables and not class variables
        self.trigger_pressed = False
        self.initialize_pose = True

        # state variable used to support user-commanded reset of environment
        self.reset_state = 0

        # task completion acknowledgment, we will toggle this
        self.task_completion_ack = 0
        self.task_timeout_ack = 0

        self.engaged = False

        # Locks to ensure safe access
        self.controller_state_lock = threading.Lock()

        self.trackpad_val = None

        # We have a variable to track when the reset button was last pressed to prevent multiple reads of the same click
        # TODO figure out how to track the reset state in a better way
        self.reset_time = time.time()
        self.reset_threshold = 2.0

        # set initial poses
        self.controller_init_pos = np.zeros(3)
        self.controller_init_rot = np.quaternion(1, 0, 0, 0)
        self.controller_curr_pos = np.zeros(3)
        self.controller_curr_rot = np.quaternion(1, 0, 0, 0)
        self.ee_init_pos = np.zeros(3)
        self.ee_init_rot = np.quaternion(1, 0, 0, 0)
        # self.get_ee_transform_from_tf()

        self.target_ori = quaternion.as_float_array(self.ee_init_rot)
        self.target_pos = self.ee_init_pos
        # This is the cartesian pose the robot EE needs to go to (in the robot base frame)
        self.target_pose = (self.target_pos.tolist(),
                            self.target_ori.tolist())

        # Init some variables
        self.robot_origin = None
        self.vr_origin = None
        self.vr_state = None
        self.prev_teleop_info = None

        if self.robots is not None:
            self.set_robot_transform()

        self._display_controls()


    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self.reset_internal_state()
        self.reset_state = 0
        self.engaged = True


    def set_robot_transform(self):

        #TODO this is hardcoded to use only the first robot in the environment. Change this
        current_pose = self.robots[0].last_eef_pose
        current_pos = current_pose[:3, 3]
        current_rot = current_pose[:3, :3]
        current_quat = transform_utils.mat2quat(current_rot)

        self.ee_init_pos = np.array([current_pos[0], current_pos[1], current_pos[2]])
        self.ee_init_rot = np.quaternion(current_quat[3], current_quat[0], current_quat[1], current_quat[2])

        if self.prev_teleop_info is None:
            target_pose = current_pos.tolist() + current_quat.tolist()
            self.prev_teleop_info = dict(target_pose=target_pose, target_pos=current_pos.tolist(),
                                    target_ori=current_quat.tolist(), dpos=[0,0,0], gripper_act=[-1], grasp = -1,
                                              reset=self.reset_state)


    def reset_internal_state(self):
        self.trackpad_val = None

        self.trigger_pressed = False
        self.initialize_pose = True

        self.prev_teleop_info = None

        self.set_robot_transform()

        self.engaged = False

    def approx_controller_offset(self):
        """
        Approximate a rotation matrix to its closest binary rotation matrix
        for rotations around the Z-axis.

        Parameters:
        R (numpy.ndarray): A 3x3 rotation matrix.

        Returns:
        numpy.ndarray: The closest binary rotation matrix for Z-axis rotation.
        """
        R = quaternion.as_rotation_matrix(self.controller_offset)
        # Define the binary rotation matrices around the Z-axis (90-degree increments)
        binary_matrices = [
            np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]]),  # 0-degree rotation (Identity)
            np.array([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 1]]),  # 90-degree rotation
            np.array([[-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 1]]),  # 180-degree rotation
            np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]]),  # 270-degree rotation
        ]

        # Calculate the Frobenius norm distance to find the closest binary matrix
        closest_matrix = None
        min_distance = float('inf')

        for binary_matrix in binary_matrices:
            distance = np.linalg.norm(R - binary_matrix, 'fro')
            if distance < min_distance:
                min_distance = distance
                closest_matrix = binary_matrix

        self.controller_offset = quaternion.from_rotation_matrix(closest_matrix)

    def get_controller_state(self):
        """
        Main function that updates the controller state and returns the teleop command
        Returns:

        """
        with self.controller_state_lock:
            controllers = self.oculus_reader.get_controller_inputs()

            while not controllers:  # busy wait for the headset to wake up
                time.sleep(0.001)
                controllers = self.oculus_reader.get_controller_inputs()

            for controller in controllers:
                # TODO: for now, we ignore the left controller, however this needs to be changed in the future,
                #  for example, for bimanual teleop
                if controller == "LeftController":
                    continue
                controller_state = controllers[controller]

                ## States below don't need the controller to be tracked

                # Grip, used for gripper control
                if controller_state["Grip"]:
                    self.grip_pressed = True
                    gripper_act = controller_state["GripValue"]
                    # TODO (NR): maybe the threshold for grasp should be set in config or somewhere else
                    if gripper_act > 0.3:
                        grasp = 1
                    else:
                        grasp = -1
                else:
                    self.grip_pressed = False
                    gripper_act = 0.0
                    grasp = -1

                # Primary ('A') button, used to decide whether to save a demo
                save_demo = True if controller_state["PrimaryButton"] else False

                # Secondary ('B') button, use to delete the currently recording demo
                delete_demo = True if controller_state["SecondaryButton"] else False

                # If both save and delete buttons are pressed, choose to save
                if save_demo and delete_demo:
                    delete_demo = False

                # If we are saving or deleting, reset state
                reset = True if (save_demo or delete_demo) else False

                # Set these values first
                teleop_info = dict(grasp = grasp, gripper_act=gripper_act,
                                        save_demo=save_demo, delete_demo=delete_demo, reset=reset
                )

                if controller_state["IsTracked"] == 1: # Can only control pose if the controller is tracked
                    # Trigger
                    if controller_state["Trigger"]:
                        if not self.trigger_pressed:
                            self.initialize_pose = True
                        self.trigger_pressed = True
                    else:
                        self.trigger_pressed = False
                        self.pos_delta = [0, 0, 0] # TODO (NR): unused, maybe delete or add code for delta control

                    # Joystick press, used for resetting controller transform, only works when controller is tracked
                    if (controller_state["AxisClicked"] and controller_state[
                        "AxisClicked"] != self.controller_offset_reset):
                        print("Set controller offset orientation")
                        print(f"original offset: {quaternion.as_rotation_matrix(self.controller_offset)}")
                        ori = controller_state["Rotation"]
                        self.controller_offset = np.quaternion(ori["w"], ori["x"], ori["y"], ori["z"])
                        self.approx_controller_offset()
                        print(f"updated offset: {quaternion.as_rotation_matrix(self.controller_offset)}")
                        self.controller_offset_reset = controller_state["AxisClicked"]
                    elif (controller_state["AxisClicked"] != self.controller_offset_reset):
                        self.controller_offset_reset = controller_state["AxisClicked"]

                    if self.trigger_pressed:  # Teleop only works if the trigger is pressed
                        pos = controller_state["Position"]
                        ori = controller_state["Rotation"]

                        if self.initialize_pose:

                            self.controller_init_pos = np.array([pos["x"], pos["y"], pos["z"]])

                            self.controller_init_rot = self.controller_offset * \
                                                       np.quaternion(ori["w"], ori["x"], ori["y"], ori["z"])

                            self.set_robot_transform()

                            if self.debug:
                                print("initialized pose")
                            self.initialize_pose = False

                        controller_curr_pos = np.array([pos["x"], pos["y"], pos["z"]])

                        target_pos = self.ee_init_pos + \
                                     quaternion.as_rotation_matrix(self.controller_offset) @ (
                                             controller_curr_pos - self.controller_init_pos)

                        self.controller_curr_rot = self.controller_offset * np.quaternion(ori["w"], ori["x"], ori["y"],
                                                                                          ori["z"])

                        rot_controller_delta = self.controller_curr_rot * self.controller_init_rot.inverse()

                        target_ori = quaternion.as_float_array(
                            rot_controller_delta * self.ee_init_rot).tolist()  # This is (w,x,y,z)

                        # Changing to (x,y,z,w)
                        target_ori = [target_ori[1], target_ori[2], target_ori[3], target_ori[0]]

                        target_pose = target_pos.tolist() + target_ori
                        dpos = controller_curr_pos - self.controller_init_pos

                        # Update the teleop info with the new target pose
                        teleop_pose_info = dict(target_pose=target_pose, target_pos=target_pos,
                                                target_ori=target_ori, dpos=dpos,
                                                engaged=self.engaged,
                                                )
                        teleop_info = self._nested_dict_update(teleop_info, teleop_pose_info)

                ## Final steps

                if self.trigger_pressed or self.grip_pressed: # are we controlling pose or gripper?
                    self.engaged = True
                else:
                    self.engaged = False

                teleop_info['engaged'] = self.engaged

                self.prev_teleop_info = self._nested_dict_update(self.prev_teleop_info, teleop_info)
                print(self.prev_teleop_info)
                return self.prev_teleop_info


    def _nested_dict_update(self, curr_dict, update_dict):
        for k, v in update_dict.items():
            if k in curr_dict and isinstance(v, dict) and isinstance(curr_dict[k], dict):
                curr_dict[k] = self._nested_dict_update(curr_dict[k], v)
            else:
                curr_dict[k] = v
        return curr_dict

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Front Grip", "control pose")
        print_command("Side Grip", "control gripper")
        print_command("A", "save demo")
        print_command("B", "delete demo")
        print_command("Press Joystick (Click)", "set controller transform")
        print("")

if __name__ == "__main__":
    quest_controller = Quest(debug=True)
    while True:
        state = quest_controller.get_dummy_controller_state()
        print(state)
        time.sleep(0.05)


