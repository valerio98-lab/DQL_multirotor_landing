from pathlib import Path

import rospkg

ASSETS_PATH = (
    Path(rospkg.RosPack().get_path("dql_multirotor_landing")) / ".." / ".." / "assets"
)
