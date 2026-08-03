"""
Playback demonstrations from a MimicGen/robosuite hdf5 file while extracting and visualizing
end-effector contact forces (grasp + push).

This is a copy of ``playback_demonstrations_from_hdf5.py`` augmented to demonstrate the new contact
force API in ``robosuite.utils.sim_utils``:

    - ``get_contact_forces(sim, geoms_1, geoms_2)``  -> per-contact world-frame normal + force vector
    - ``get_total_contact_force(sim, geoms_1, geoms_2)`` -> summed resultant force + application point
    - ``get_sensor_measurement(sim, sensor_name)``   -> wrist force/torque sensor (net wrench)

For every playback frame we filter contacts to those involving the robot's gripper geoms (this
captures both fingerpad grasps and fingertip/body pushes), then project each contact's force vector
onto the rendered camera image and draw it as an arrow. Fingerpad (grasp) contacts are drawn in one
color and other gripper (push) contacts in another.

Arguments:
    --folder (str): Path to demonstrations (expects demo.hdf5 inside)
    --use-actions (optional): replay actions through the simulator instead of setting states
    --camera / --height / --width / --video-skip / --n: rendering options (as in the original script)
    --force-scale (float): meters of arrow length per Newton of force (visualization only)
    --max-force (float): forces above this (N) are clipped for arrow length (visualization only)
    --grasp-only: only consider fingerpad contacts (ignore other gripper geoms)

Example:
    $ python playback_with_contact_force_vectors.py --folder path/to/demo_dir --use-actions
"""

import argparse
import json
import os
import random
from collections import deque

import cv2
import h5py
import imageio
import numpy as np

import robosuite
from robosuite.utils import camera_utils
from robosuite.utils.sim_utils import aggregate_contact_forces, get_contact_forces, get_sensor_measurement

import mimicgen  # noqa: F401  # Registers MimicGen environments with robosuite.


# BGR-ish colors are not used (imageio writes RGB); these are RGB tuples.
GRASP_COLOR = (255, 60, 60)  # red-ish for fingerpad (grasp) contacts
PUSH_COLOR = (60, 160, 255)  # blue-ish for other gripper (push) contacts
NET_COLOR = (60, 255, 60)  # green for the summed resultant force


class ForceSmoother:
    """
    Per-group temporal denoiser for contact force vectors (and their application points) across playback
    frames. State-replay sets each frame independently and the solver re-solves from scratch, so raw
    contact forces are jittery frame-to-frame. Keyed on each aggregated group's stable ``key`` (the
    contacted body, or a (contacted body, actor finger body) pair when finger forces are kept separate),
    each force is denoised in two steps:

      1. Causal median over the last @median_window frames -- rejects isolated spikes/dropouts.
      2. EMA: ``smoothed = alpha * median + (1 - alpha) * previous`` -- low-pass for residual jitter.

    Measured on a grasp's sustained-contact frames, raw ~17 N / 8 deg frame-to-frame jitter drops to
    ~7 N / <2 deg with median(5) + EMA(0.4). Set median_window=1 and alpha=1.0 to disable smoothing.
    A group that loses contact for more than @forget frames is dropped so stale arrows do not linger.
    """

    def __init__(self, alpha=0.4, median_window=5, forget=2):
        self.alpha = alpha
        self.median_window = max(1, int(median_window))
        self.forget = forget
        self._state = {}  # key -> {"hist_f", "hist_p" (deques), "force", "pos", "missing", "meta"}

    def update(self, groups):
        """Smooths the current frame's aggregated groups; returns smoothed copies to draw."""
        seen = set()
        for g in groups:
            key = g["key"]
            seen.add(key)
            s = self._state.get(key)
            if s is None:
                s = {
                    "hist_f": deque(maxlen=self.median_window),
                    "hist_p": deque(maxlen=self.median_window),
                    "force": None,
                    "pos": None,
                    "missing": 0,
                }
                self._state[key] = s
            s["hist_f"].append(np.array(g["force"]))
            s["hist_p"].append(np.array(g["pos"]))
            # Step 1: causal elementwise median over the recent window.
            med_f = np.median(np.stack(s["hist_f"]), axis=0)
            med_p = np.median(np.stack(s["hist_p"]), axis=0)
            # Step 2: EMA toward the median.
            if s["force"] is None or self.alpha >= 1.0:
                s["force"], s["pos"] = med_f, med_p
            else:
                a = self.alpha
                s["force"] = a * med_f + (1 - a) * s["force"]
                s["pos"] = a * med_p + (1 - a) * s["pos"]
            s["missing"] = 0
            s["meta"] = g
        # Age out groups not in contact this frame.
        for key in list(self._state):
            if key not in seen:
                self._state[key]["missing"] += 1
                if self._state[key]["missing"] > self.forget:
                    del self._state[key]
        out = []
        for key, s in self._state.items():
            if s["missing"] == 0:
                out.append({**s["meta"], "force": s["force"], "pos": s["pos"],
                            "force_mag": float(np.linalg.norm(s["force"]))})
        return out

    def reset(self):
        self._state = {}


def _gripper_geom_sets(env):
    """
    Returns (all_gripper_geoms, fingerpad_geoms) name lists across all of the robot's grippers,
    handling both single-arm (GripperModel) and bimanual (dict of arm -> GripperModel) layouts.
    """
    all_geoms, pad_geoms = [], []
    for robot in env.robots:
        gripper = robot.gripper
        grippers = gripper.values() if isinstance(gripper, dict) else [gripper]
        for g in grippers:
            if g is None:
                continue
            all_geoms += list(g.contact_geoms)
            imp = g.important_geoms
            pad_geoms += list(imp.get("left_fingerpad", [])) + list(imp.get("right_fingerpad", []))
    return all_geoms, pad_geoms


def _wrist_sensor_names(env):
    """Returns the list of wrist force/torque sensor names exposed by the robots' grippers."""
    names = []
    for robot in env.robots:
        gripper = robot.gripper
        grippers = gripper.values() if isinstance(gripper, dict) else [gripper]
        for g in grippers:
            if g is None:
                continue
            for sensor_name in g.important_sensors.values():
                names.append(sensor_name)
    return names


def _draw_arrow(img, start_px, end_px, color, thickness=2):
    """Draws an arrow on @img (RGB, HxWx3 uint8). Pixels are (row, col); cv2 wants (x=col, y=row)."""
    p0 = (int(start_px[1]), int(start_px[0]))
    p1 = (int(end_px[1]), int(end_px[0]))
    cv2.arrowedLine(img, p0, p1, color, thickness, line_type=cv2.LINE_AA, tipLength=0.25)


def draw_contact_forces(
    env, img, camera_name, height, width, gripper_geoms, pad_geoms, self_geoms, force_scale, max_force,
    force_threshold, smoother,
):
    """
    Projects gripper contact forces onto @img (already vertically flipped to standard orientation)
    and draws them in place. Returns the smoothed per-body groups that were drawn.

    Denoising is two-stage: redundant contacts are summed per (object, finger) group (spatial), then
    EMA'd across frames by @smoother (temporal). Forces are oriented (via actor_geoms=gripper) as the
    force the gripper exerts ON whatever it touches, so arrows point away from the gripper into the
    object. Gripper self-contacts (both geoms in @self_geoms, e.g. the two fingerpads touching when the
    gripper closes on nothing) are dropped so they don't render as phantom forces.
    """
    contacts = get_contact_forces(env.sim, geoms_1=gripper_geoms, geoms_2=None, actor_geoms=gripper_geoms)
    # Drop gripper-internal self-contacts -- we only want forces on external objects.
    contacts = [c for c in contacts if not (c["geoms"][0] in self_geoms and c["geoms"][1] in self_geoms)]
    # Spatial denoising: sum per (object, finger) group, dropping sub-threshold jitter. Keeping the
    # finger in the key prevents a grasp's two opposing finger forces from cancelling to ~0 N.
    groups = aggregate_contact_forces(env.sim, contacts, actor_geoms=gripper_geoms, min_force=force_threshold)
    # Temporal denoising: EMA per body across frames.
    groups = smoother.update(groups)
    if len(groups) == 0:
        return groups

    world_to_pix = camera_utils.get_camera_transform_matrix(
        sim=env.sim, camera_name=camera_name, camera_height=height, camera_width=width
    )
    pad_set = set(pad_geoms)

    total_mag = 0.0
    for g in groups:
        mag = g["force_mag"]
        if mag < 1e-6:
            continue
        total_mag += mag
        # Clip the *length* (not the underlying value) so a huge force does not produce a giant arrow.
        draw_mag = min(mag, max_force)
        direction = g["force"] / mag
        start = g["pos"]
        end = g["pos"] + direction * draw_mag * force_scale

        pts = np.stack([start, end], axis=0)
        px = camera_utils.project_points_from_world_to_camera(pts, world_to_pix, height, width)

        # Grasp vs push is decided by the *actor* geom: a fingerpad doing the contact is a grasp,
        # any other gripper geom (hand/finger body) pushing is a push.
        is_grasp = len(g.get("actor_geoms", set()) & pad_set) > 0
        color = GRASP_COLOR if is_grasp else PUSH_COLOR
        _draw_arrow(img, px[0], px[1], color, thickness=3)
        cv2.circle(img, (int(px[0][1]), int(px[0][0])), 2, color, -1)

    cv2.putText(
        img,
        "sum |F|={:.1f}N".format(total_mag),
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        NET_COLOR,
        1,
        cv2.LINE_AA,
    )
    return groups


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
        help="Path to save the playback video. Defaults to <folder>/playback_contact_forces.mp4.",
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
    parser.add_argument(
        "--force-scale",
        type=float,
        default=0.01,
        help="Arrow length in meters per Newton of contact force (visualization only).",
    )
    parser.add_argument(
        "--max-force",
        type=float,
        default=50.0,
        help="Clip arrow length to this many Newtons (visualization only).",
    )
    parser.add_argument(
        "--grasp-only",
        action="store_true",
        help="Only consider fingerpad (grasp) contacts; ignore other gripper geoms.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.4,
        help="EMA factor for temporal force smoothing in (0, 1]. Lower = smoother/laggier; "
        "1.0 disables the EMA stage.",
    )
    parser.add_argument(
        "--median-window",
        type=int,
        default=5,
        help="Causal median-filter window (frames) applied before the EMA to reject spikes; "
        "1 disables the median stage.",
    )
    parser.add_argument(
        "--force-threshold",
        type=float,
        default=1.0,
        help="Deadband: drop per-body contact forces below this magnitude (N) before drawing.",
    )
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    video_path = args.video_path or os.path.join(demo_path, "playback_contact_forces.mp4")
    f = h5py.File(hdf5_path, "r")
    # Print all attributes and their values in the HDF5 'data' group
    for k, v in f["data"].attrs.items():
        print(f"{k}: {v}")

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

    # Resolve the gripper geom groups once (used to filter contacts to the end-effector).
    gripper_geoms, pad_geoms = _gripper_geom_sets(env)
    filter_geoms = pad_geoms if args.grasp_only else gripper_geoms
    if len(filter_geoms) == 0:
        print("[warning] no gripper geoms found; contact forces will not be filtered to the end-effector.")
        filter_geoms = None
    wrist_sensors = _wrist_sensor_names(env)
    print("Tracking {} gripper geoms ({} fingerpad). Wrist sensors: {}".format(
        len(gripper_geoms), len(pad_geoms), wrist_sensors))

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    video_writer = imageio.get_writer(video_path, fps=20)

    # One smoother per playback; reset between episodes so forces don't carry across resets.
    smoother = ForceSmoother(alpha=args.smooth, median_window=args.median_window)

    def render_and_annotate():
        """Render the current sim state and overlay contact-force arrows. Returns an RGB frame."""
        img = env.sim.render(height=args.height, width=args.width, camera_name=args.camera)[::-1]
        img = np.ascontiguousarray(img)
        draw_contact_forces(
            env,
            img,
            camera_name=args.camera,
            height=args.height,
            width=args.width,
            gripper_geoms=filter_geoms,
            pad_geoms=pad_geoms,
            self_geoms=set(gripper_geoms),
            force_scale=args.force_scale,
            max_force=args.max_force,
            force_threshold=args.force_threshold,
            smoother=smoother,
        )
        return img

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

        # re-resolve gripper geoms after the model was rebuilt from xml (geom prefixes are stable,
        # but the env objects are recreated on reset_from_xml_string)
        gripper_geoms, pad_geoms = _gripper_geom_sets(env)
        filter_geoms = pad_geoms if args.grasp_only else gripper_geoms

        # clear temporal smoothing state so forces don't bleed across episodes
        smoother.reset()

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
                    video_writer.append_data(render_and_annotate())

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
                # mj_forward (via sim.forward) runs collision + constraint solver, so contact
                # forces are populated for the current state.
                env.sim.forward()
                if j % args.video_skip == 0:
                    video_writer.append_data(render_and_annotate())

    video_writer.close()
    print("Saved playback video to {}".format(video_path))
    f.close()
