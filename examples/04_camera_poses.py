"""Camera poses

Example showing how we can detect new clients and read camera poses from them.
"""

import time
import viser
import numpy as np

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

server = viser.ViserServer()
server.world_axes.visible = True


@server.on_client_connect
def _(client: viser.ClientHandle) -> None:
    print("new client!")

    # This will run whenever we get a new camera!
    @client.camera.on_update
    def _(_: viser.CameraHandle) -> None:
        print(f"New camera on client {client.client_id}!")

    # Show the client ID in the GUI.
    gui_info = client.add_gui_text("Client ID", initial_value=str(client.client_id))
    gui_info.disabled = True


while True:
    # Get all currently connected clients.
    clients = server.get_clients()
    print("Connected client IDs", clients.keys())

    for id, client in clients.items():

        R = np.transpose(qvec2rotmat(client.camera.wxyz))

        print(f"Camera pose for client {id}")
        print(f"\twxyz (QW QX QY QZ): {client.camera.wxyz}")
        print(f"\tR: {R}")
        print(f"\tposition (T): {client.camera.position}")
        print(f"\tfov: {client.camera.fov}")
        print(f"\taspect: {client.camera.aspect}")
        print(f"\tlast update: {client.camera.update_timestamp}")

    time.sleep(2.0)
