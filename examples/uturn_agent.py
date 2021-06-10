import logging
import math

import gym
import numpy as np

from examples import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType, NeighborhoodVehicles
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.core.utils.math import (
    min_angles_difference_signed,
)
from smarts.core.waypoints import Waypoint, Waypoints

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class UTurnAgent(Agent):
    def __init__(self):
        self._initial_heading = 0
        self._task_is_triggered = False

    def act(self, obs: Observation):
        aggressiveness = 10
        des_speed = 12
        des_lane = 0

        vehicle = self.sim._vehicle_index.vehicles_by_actor_id(AGENT_ID)[0]

        miss = self.sim._vehicle_index.sensor_state_for_vehicle_id(
            vehicle.id
        ).mission_planner

        neighborhood_vehicles = obs.neighborhood_vehicle_states
        neighborhood_vehicles = [
            vehicle for vehicle in neighborhood_vehicles if "ego" in vehicle.id
        ]
        pose = vehicle.pose

        start_lane = miss._road_network.nearest_lane(
            miss._mission.start.position,
            include_junctions=False,
            include_special=False,
        )
        start_edge = miss._road_network.road_edge_data_for_lane_id(start_lane.getID())
        oncoming_edge = start_edge.oncoming_edges[0]
        oncoming_lanes = oncoming_edge.getLanes()
        target_lane_index = miss._mission.task.target_lane_index
        target_lane_index = min(target_lane_index, len(oncoming_lanes) - 1)
        target_lane = oncoming_lanes[target_lane_index + 0]

        offset = miss._road_network.offset_into_lane(start_lane, pose.position[:2])
        oncoming_offset = max(0, target_lane.getLength() - offset)
        paths = miss.paths_of_lane_at(target_lane, oncoming_offset, lookahead=30)

        if len(neighborhood_vehicles) < 1:
            fff = obs.waypoint_paths[int(start_lane.getID().split("_")[-1])]
            self._initial_heading = obs.ego_vehicle_state.heading % (2 * math.pi)
        else:
            target_p = neighborhood_vehicles[0].position[0:2]
            target_l = miss._road_network.nearest_lane(target_p)
            target_offset = miss._road_network.offset_into_lane(target_l, target_p)
            fq = target_lane.getLength() - offset - target_offset

            if (
                fq > (aggressiveness / 10) * 65 + (1 - aggressiveness / 10) * 100
                and self._task_is_triggered is False
            ):
                fff = obs.waypoint_paths[int(start_lane.getID().split("_")[-1])]
                self._initial_heading = obs.ego_vehicle_state.heading % (2 * math.pi)

            else:
                self._task_is_triggered = True
                fff = paths[des_lane]

        lat_error = fff[0].signed_lateral_error(
            [vehicle.position[0], vehicle.position[1]]
        )

        if self._task_is_triggered is True and abs(lat_error) > 0.3:
            des_speed = 8

        look_ahead_wp_num = 3
        look_ahead_dist = 3
        vehicle_look_ahead_pt = [
            obs.ego_vehicle_state.position[0]
            - look_ahead_dist * math.sin(obs.ego_vehicle_state.heading),
            obs.ego_vehicle_state.position[1]
            + look_ahead_dist * math.cos(obs.ego_vehicle_state.heading),
        ]

        reference_heading = fff[look_ahead_wp_num].heading
        heading_error = min_angles_difference_signed(
            (obs.ego_vehicle_state.heading % (2 * math.pi)), reference_heading
        )
        controller_lat_error = fff[look_ahead_wp_num].signed_lateral_error(
            vehicle_look_ahead_pt
        )

        steer = 0.34 * controller_lat_error + 1.2 * heading_error

        throttle = -0.23 * (obs.ego_vehicle_state.speed - (des_speed)) - 1.1 * abs(
            obs.ego_vehicle_state.linear_velocity[1]
        )

        if throttle >= 0:
            brake = 0
        else:
            brake = abs(throttle)
            throttle = 0

        return (throttle, brake, steer)


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.StandardWithAbsoluteSteering,
            max_episode_steps=max_episode_steps,
            neighborhood_vehicles=NeighborhoodVehicles(850),
        ),
        agent_builder=UTurnAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=False,
        sumo_auto_start=False,
        seed=seed,
        # zoo_addrs=[("10.193.241.236", 7432)], # Sample server address (ip, port), to distribute social agents in remote server.
        # envision_record_data_replay_path="./data_replay",
    )
    UTurnAgent.sim = env._smarts
    # print(env._smarts, "::::::::::::::::::::::::::")

    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        agent.sim = env._smarts
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )