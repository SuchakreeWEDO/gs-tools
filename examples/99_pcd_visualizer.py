"""COLMAP visualizer

Visualize COLMAP sparse reconstruction outputs. To get demo data, see `./assets/download_colmap_garden.sh`.
"""

import random
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as onp
import tyro
import viser
import viser.transforms as tf
from tqdm.auto import tqdm
from plyfile import PlyData, PlyElement
from viser.extras.colmap import (

    read_points3d_binary,
)

# plydata = PlyData.read('./assets/point_cloud.ply')
# x = plydata['vertex']['x'][:50000].reshape(-1,1)
# y = plydata['vertex']['y'][:50000].reshape(-1,1)
# z = plydata['vertex']['z'][:50000].reshape(-1,1)
# points = xyz = onp.concatenate([x,y,z], axis = 1)
# print(xyz.shape)

# C0 = 0.28209479177387814
# r = 0.5 + C0 * plydata['vertex']['f_dc_0'][:50000].reshape(-1,1)
# g = 0.5 + C0 * plydata['vertex']['f_dc_1'][:50000].reshape(-1,1)
# b = 0.5 + C0 * plydata['vertex']['f_dc_2'][:50000].reshape(-1,1)
# colors = rgb = onp.concatenate([r,g,b], axis = 1)
# print(rgb.shape)

def main(
    colmap_path: Path = Path(__file__).parent / "assets/colmap_garden/sparse/0",
    # images_path: Path = Path(__file__).parent / "assets/colmap_garden/images_8",
    # downsample_factor: int = 2,
) -> None:
    """Visualize COLMAP sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        images_path: Path to the COLMAP images directory.
        downsample_factor: Downsample factor for the images.
    """
    server = viser.ViserServer()
    server.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Load the colmap info.
    points3d = read_points3d_binary(colmap_path / "points3D.bin")
    gui_reset_up = server.add_gui_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ onp.array(
            [0.0, -1.0, 0.0]
        )

    gui_points = server.add_gui_slider(
        "Max points",
        min=1,
        max=len(points3d),
        step=1,
        initial_value=min(len(points3d), 50_000),
    )

    gui_point_size = server.add_gui_number("Point size", initial_value=0.05)

    def visualize_colmap() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""
        # Set the point cloud.
        points = onp.array([points3d[p_id].xyz for p_id in points3d]) # (50000, 3)
        colors = onp.array([points3d[p_id].rgb for p_id in points3d]) # (50000, 3)
        points_selection = onp.random.choice(
            points.shape[0], gui_points.value, replace=False
        )
        points = points[points_selection]
        colors = colors[points_selection]

        print(onp.min(points), onp.max(points)) # -39.752517951310864 51.277373837644504
        print(onp.min(colors), onp.max(colors)) # 0 255

        server.add_point_cloud(
            name="/colmap/pcd",
            points=points,
            colors=colors,
            point_size=gui_point_size.value,
        )

    need_update = True

    @gui_points.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    while True:
        if need_update:
            need_update = False

            server.reset_scene()
            visualize_colmap()

        time.sleep(1e-3)


if __name__ == "__main__":
    
    tyro.cli(main)
