from gymnasium.envs.registration import register

from highway_env.envs.highway_env import *
from highway_env.envs.merge_env import *
from highway_env.envs.parking_env import *
from highway_env.envs.roundabout_env import *
from highway_env.envs.two_way_env import *
from highway_env.envs.intersection_env import *
from highway_env.envs.lane_keeping_env import *
from highway_env.envs.u_turn_env import *
from highway_env.envs.exit_env import *
from highway_env.envs.racetrack_env import *
from highway_env.envs.test_env import *
from highway_env.envs.my_highway_env import *
from highway_env.envs.my_merge_env import *

register(
    id="test-env-v0",
    entry_point="highway_env.envs:TestEnv"
)

register(
    id="highway-platoon-v0",
    entry_point="highway_env.envs:MyHighwayEnv"
)

register(
    id="merge-platoon-v0",
    entry_point="highway_env.envs:MyMergeEnv"
)

