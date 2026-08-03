"""
Build a fused, colored, segmented 3D point cloud from a single frame of a robosuite/MimicGen demo and
write it to an interactive Plotly HTML you can open in a browser (rotate / zoom / toggle layers).

For the chosen demo episode the script:
  1. rebuilds the scene from the episode's stored model xml,
  2. loads the demo's recorded initial state (states[0]) and forwards the sim,
  3. renders each requested camera's RGB + depth (+ segmentation), back-projects depth to a world-frame
     point cloud via ``robosuite.utils.camera_utils.get_camera_pointcloud``, and fuses the cameras,
  4. writes a Plotly HTML with two views toggled by buttons:
       - "RGB": points colored by their rendered color,
       - "Segmentation": one legend-toggleable trace per body (table / robot / gripper / object / ...).

This is intentionally separate from the contact-force playback script.

Example:
    $ python pointcloud_from_demo_to_html.py --folder .../datasets/mug_cleanup_d0 --demo 0
"""

import argparse
import json
import os

import h5py
import numpy as np
import plotly.graph_objects as go

import robosuite
from robosuite.utils.camera_utils import get_camera_pointcloud

import mimicgen  # noqa: F401  # Registers MimicGen environments with robosuite.


# A qualitative palette for per-body segmentation coloring (cycled if there are more bodies).
_SEG_PALETTE = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
    "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def fuse_pointclouds(sim, camera_names, height, width, depth_max=None):
    """Renders each camera, back-projects, and concatenates into one fused cloud (points/rgb/body ids)."""
    points, rgb, body_ids = [], [], []
    for cam in camera_names:
        pc = get_camera_pointcloud(
            sim, cam, height, width, return_rgb=True, return_segmentation=True, depth_max=depth_max
        )
        points.append(pc["points"])
        rgb.append(pc["rgb"])
        body_ids.append(pc["body_ids"])
    return np.concatenate(points), np.concatenate(rgb), np.concatenate(body_ids)


def subsample(n, max_points):
    """Returns an index array that randomly subsamples n points down to at most max_points."""
    if max_points is None or n <= max_points:
        return np.arange(n)
    # np.random is fine here (visualization only); fixed seed for reproducible HTML.
    rng = np.random.default_rng(0)
    return rng.choice(n, size=max_points, replace=False)


def build_figure(points, rgb, body_ids, sim, marker_size, title):
    """Builds a Plotly figure: trace 0 = RGB cloud, traces 1..K = per-body segmentation clouds."""
    fig = go.Figure()

    # --- RGB trace (single Scatter3d, per-point color) ---
    rgb_strings = ["rgb({},{},{})".format(int(r), int(g), int(b)) for r, g, b in rgb]
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode="markers",
            marker=dict(size=marker_size, color=rgb_strings),
            name="RGB",
            visible=True,
            showlegend=False,
        )
    )

    # --- Segmentation traces (one per body so the legend can isolate each object) ---
    seg_trace_idx = []
    unique_bodies = sorted(set(int(b) for b in body_ids))
    for i, bid in enumerate(unique_bodies):
        mask = body_ids == bid
        name = sim.model.body_id2name(bid) if bid >= 0 else "background"
        color = _SEG_PALETTE[i % len(_SEG_PALETTE)]
        fig.add_trace(
            go.Scatter3d(
                x=points[mask, 0], y=points[mask, 1], z=points[mask, 2],
                mode="markers",
                marker=dict(size=marker_size, color=color),
                name=name,
                visible=False,
                showlegend=True,
            )
        )
        seg_trace_idx.append(len(fig.data) - 1)

    n_traces = len(fig.data)
    rgb_visible = [i == 0 for i in range(n_traces)]
    seg_visible = [i in seg_trace_idx for i in range(n_traces)]

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
            aspectmode="data",  # equal aspect so geometry is not distorted
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0, y=1.08, xanchor="left",
                buttons=[
                    dict(label="RGB", method="update", args=[{"visible": rgb_visible}]),
                    dict(label="Segmentation", method="update", args=[{"visible": seg_visible}]),
                ],
            )
        ],
        legend=dict(itemsizing="constant"),
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to the demonstration folder containing demo.hdf5.",
    )
    parser.add_argument(
        "--demo",
        type=str,
        default="0",
        help="Which episode: an integer index (e.g. 0) or a demo key (e.g. demo_0). Defaults to the first.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="agentview,robot0_eye_in_hand",
        help="Comma-separated camera names to fuse (defaults to the dataset's agentview + wrist cam).",
    )
    parser.add_argument("--height", type=int, default=256, help="Render height in pixels.")
    parser.add_argument("--width", type=int, default=256, help="Render width in pixels.")
    parser.add_argument(
        "--depth-max",
        type=float,
        default=None,
        help="Drop points farther than this many meters (discard far background). Default: keep all.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=150000,
        help="Randomly subsample the fused cloud to at most this many points (keeps the HTML responsive).",
    )
    parser.add_argument("--marker-size", type=float, default=1.5, help="Plotly marker size for points.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML path. Defaults to <folder>/pointcloud_<ep>.html.",
    )
    args = parser.parse_args()

    hdf5_path = os.path.join(args.folder, "demo.hdf5")
    f = h5py.File(hdf5_path, "r")

    env_args_str = f["data"].attrs["env_args"]
    if isinstance(env_args_str, bytes):
        env_args_str = env_args_str.decode("utf-8")
    env_args = json.loads(env_args_str)
    env_kwargs = dict(env_args.get("env_kwargs", {}))
    env_kwargs.update(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False)

    env = robosuite.make(env_args["env_name"], **env_kwargs)

    demos = list(f["data"].keys())
    # Resolve the requested episode (integer index or explicit key).
    if args.demo.isdigit():
        ep = demos[int(args.demo)]
    else:
        ep = args.demo
    print("Using episode '{}' from {}".format(ep, hdf5_path))

    # Rebuild the scene and load the demo's recorded initial state (states[0]).
    env.reset()
    xml = env.edit_model_xml(f["data/{}".format(ep)].attrs["model_file"])
    env.reset_from_xml_string(xml)
    env.sim.reset()
    states = f["data/{}/states".format(ep)][()]
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()

    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]
    points, rgb, body_ids = fuse_pointclouds(env.sim, camera_names, args.height, args.width, args.depth_max)
    print("Fused {} points from cameras: {}".format(len(points), camera_names))

    n_fused = len(points)
    idx = subsample(n_fused, args.max_points)
    points, rgb, body_ids = points[idx], rgb[idx], body_ids[idx]
    if len(points) < n_fused:
        print("Subsampled to {} points for plotting.".format(len(points)))

    title = "{} | {} | {} pts | cams: {}".format(
        env_args["env_name"], ep, len(points), "+".join(camera_names)
    )
    fig = build_figure(points, rgb, body_ids, env.sim, args.marker_size, title)

    output = args.output or os.path.join(args.folder, "pointcloud_{}.html".format(ep))
    fig.write_html(output)
    print("Saved interactive point cloud to {}".format(output))
    f.close()
