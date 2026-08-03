"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/lift/
"""

import argparse
import json
import os
import random

import h5py
import imageio
import numpy as np

import robosuite
import mimicgen  # noqa: F401  # Registers MimicGen environments with robosuite.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Path to save the playback video. Defaults to <folder>/playback.mp4.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Camera name to render from.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Rendered video height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Rendered video width.",
    )
    parser.add_argument(
        "--video-skip",
        type=int,
        default=1,
        help="Write every Nth frame.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of random episodes to write into the video.",
    )
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    video_path = args.video_path or os.path.join(demo_path, "playback.mp4")
    f = h5py.File(hdf5_path, "r")
    # Print all attributes and their values in the HDF5 'data' group
    for k, v in f["data"].attrs.items():
        print(f"{k}: {v}")
    # env_info = json.loads(f["data"].attrs["env_info"])

    # Extract the env_args attribute as a bytes object, decode as UTF-8 string, then parse as JSON
    # env_args is stored as a single JSON string in the attributes, not as a nested dict
    env_args_str = f["data"].attrs["env_args"]
    if isinstance(env_args_str, bytes):
        env_args_str = env_args_str.decode('utf-8')
    env_args = json.loads(env_args_str)

    # The env_name and env_kwargs are in the env_args dict directly
    env_name = env_args["env_name"]
    env_kwargs = dict(env_args.get("env_kwargs", {}))

    # MimicGen datasets are usually saved with headless/offscreen rendering.
    # Keep playback cluster-friendly and render frames directly from the sim.
    env_kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
    )

    env = robosuite.make(
        env_name,
        **env_kwargs,
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    video_writer = imageio.get_writer(video_path, fps=20)

    for i in range(args.n):
        print("Writing random episode {} / {} to {}".format(i + 1, args.n, video_path))

        # select an episode randomly
        ep = random.choice(demos)

        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]

        if args.use_actions:

            # load the initial state
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]

            for j, action in enumerate(actions):
                env.step(action)
                if j % args.video_skip == 0:
                    video_img = env.sim.render(height=args.height, width=args.width, camera_name=args.camera)[::-1]
                    video_writer.append_data(video_img)

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                    if not np.all(np.equal(states[j + 1], state_playback)):
                        err = np.linalg.norm(states[j + 1] - state_playback)
                        print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")

        else:

            # force the sequence of internal mujoco states one by one
            for j, state in enumerate(states):
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                if j % args.video_skip == 0:
                    video_img = env.sim.render(height=args.height, width=args.width, camera_name=args.camera)[::-1]
                    video_writer.append_data(video_img)

    video_writer.close()
    print("Saved playback video to {}".format(video_path))
    f.close()
