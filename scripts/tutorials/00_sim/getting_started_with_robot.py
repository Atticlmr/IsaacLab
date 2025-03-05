import sys
print(sys.path)
import sys
sys.path.append('/home/li/IsaacLab/_isaac_sim')


from isaacsim.core.cloner import Cloner    # import Cloner interface
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.cloner import GridCloner
from pxr import UsdGeom
import numpy as np

cube_positions = np.array([
    [0.0, 0.0, 0.0],
    [3.0, 0.0, 0.0],
    [6.0, 0.0, 0.0],
    [9.0, 0.0, 0.0],
])


# create our base environment with one cube
base_env_path = "/World/Cube_0"
UsdGeom.Cube.Define(get_current_stage(), base_env_path)

# create a Cloner instance
cloner = Cloner()

# generate 4 paths that begin with "/World/Cube" - path will be appended with _{index}
target_paths = cloner.generate_paths("/World/Cube", 4)

# clone the cube at target paths
cloner.clone(source_prim_path="/World/Cube_0", prim_paths=target_paths,positions=cube_positions)