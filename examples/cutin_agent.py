import logging
import math

import gym
import numpy as np

from examples import default_argument_parser
import matplotlib.pyplot as plt
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.coordinates import Heading, Pose
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.core.utils.math import evaluate_bezier as bezier
from smarts.core.utils.math import (
    lerp,
    low_pass_filter,
    min_angles_difference_signed,
    radians_to_vec,
    signed_dist_to_line,
    vec_to_radians,
)
from smarts.core.waypoints import Waypoint, Waypoints
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"
class BehaviorAgentState(Enum):
    Virtual_lane_following = 0
    Approach = 1
    Interact = 2


class CutInAgent(Agent):
    def __init__(self,cutin_lateral_gain,maximum_offset,cutin_speed):
        self.vehicle_spee=0
        self.lane_index = 1
        self._initial_heading = 0
        self._task_is_triggered = False
        self._counter = 0
        self.lateral_gain = 0.54 #0.34
        self.heading_gain = 1.2
        self._des_speed=12
        self._position_adjust=0
        self._cutin_agent_state=BehaviorAgentState.Virtual_lane_following
        self._prev_cutin_agent_state=None
        # self._aggressiveness=aggressiveness
        self._maximum_offset=maximum_offset
        self._cutin_lateral_gain=cutin_lateral_gain
        self._cutin_speed=cutin_speed
        self._speed_tracking=0.43 # 0.43
        self._traction_gain=1.1   # 0.1.1

    def act(self, obs: Observation):
        print(">>>>>>>>>>>",self._cutin_agent_state)
        aggressiveness = 5
        aggressiveness = self._aggressiveness

        vehicle = self.sim._vehicle_index.vehicles_by_actor_id("Agent-007")[0]

        miss = self.sim._vehicle_index.sensor_state_for_vehicle_id(
            vehicle.id
        ).mission_planner

        neighborhood_vehicles = self.sim.neighborhood_vehicles_around_vehicle(
            vehicle=vehicle, radius=850
        )
        pose = vehicle.pose

        position = pose.position[:2]
        lane = self.sim.scenario.road_network.nearest_lane(position)

        start_lane = miss._road_network.nearest_lane(
            miss._mission.start.position,
            include_junctions=False,
            include_special=False,
        )

        if len(neighborhood_vehicles)!=0:

            target_p = neighborhood_vehicles[0].pose.position[0:2]
            target_l = miss._road_network.nearest_lane(target_p)
        

        def vehicle_control_commands(fff,look_ahead_wp_num,look_ahead_dist,ref_speed,longitudinal_feed_forward=0):
            vehicle_look_ahead_pt = [
                obs.ego_vehicle_state.position[0]
                - look_ahead_dist * math.sin(obs.ego_vehicle_state.heading),
                obs.ego_vehicle_state.position[1]
                + look_ahead_dist * math.cos(obs.ego_vehicle_state.heading),
            ]
            cum_sum=0
            if len(fff)>10:
                for idx in range(10):
                    cum_sum+=abs(fff[idx+1].heading-fff[idx].heading)
            
            reference_heading = fff[look_ahead_wp_num].heading
            heading_error = min_angles_difference_signed(
                (obs.ego_vehicle_state.heading % (2 * math.pi)), reference_heading
            )
            controller_lat_error = fff[look_ahead_wp_num].signed_lateral_error(
                vehicle_look_ahead_pt
            )
            

            steer = (
                self.lateral_gain * controller_lat_error + 0*self.heading_gain * heading_error
            )
            
            min_dis={}
            max_dis={}
            nv_dict_lower={}
            nv_dict_upper={}
            nv_offlane={}
            ego_offset = miss._road_network.offset_into_lane(lane, pose.position[:2])
            start_edge = miss._road_network.road_edge_data_for_lane_id(lane.getID())
            is_in_junction=":" in start_edge.forward_edges[0].getLanes()[0].getID()
            print(start_edge)
            # oncoming_edge = start_edge.oncoming_edges[0]
            # oncoming_lanes = oncoming_edge.getLanes()
            if len(neighborhood_vehicles)!=0:
                for nv in neighborhood_vehicles:
                    nv_lane = miss._road_network.nearest_lane(nv.pose.position[:2],include_junctions=False,include_special=False,)
                    if lane==nv_lane:
                        nv_offset = miss._road_network.offset_into_lane(nv_lane, nv.pose.position[:2])
                        if nv_offset>=ego_offset:
                            nv_dict_upper[nv_offset]=nv
                        else:
                            nv_dict_lower[nv_offset]=nv
                    # else:
                    nv_dist=np.linalg.norm(vehicle.pose.position-nv.pose.position)
                    nv_offlane[nv_dist]=nv
                    for tt in range(30):
                        min_dis[np.linalg.norm(vehicle.pose.position[:2]+0.1*tt*vehicle.speed*radians_to_vec(vehicle.pose.heading)-nv.pose.position[:2]-0.1*tt*nv.speed*radians_to_vec(nv.pose.heading))]=nv
                        max_dis[np.linalg.norm(vehicle.pose.position[:2]+0.1*tt*(vehicle.speed+0.1*tt*4.5)*radians_to_vec(vehicle.pose.heading)-nv.pose.position[:2]-0.1*tt*nv.speed*radians_to_vec(nv.pose.heading))]=nv

                # print(min(min_dis),"<:<:<:<:<:<:<:",nv.pose.heading)
            
            mod1,mod2=0,0
            thresh=10
            if len(nv_dict_upper)!=0:
                nv_lead=nv_dict_upper[min(nv_dict_upper)]
                if min(nv_dict_upper)-ego_offset<thresh:
                    mod1=-30*(ego_offset-min(nv_dict_upper)+thresh)
            if len(nv_dict_lower)!=0:
                nv_back=nv_dict_lower[max(nv_dict_lower)]
                if ego_offset-max(nv_dict_lower)<thresh:
                    mod2=-1*30*(ego_offset-max(nv_dict_lower)-thresh)
                    # nv_dictlane[]
            # print(mod1,mod2,"?????????????????????????")
                    
            #         for tt in range(10):
            #             min_dis[np.linalg.norm(vehicle.pose.position[:2]+0.1*tt*vehicle.speed*radians_to_vec(vehicle.pose.heading)-nv.pose.position[:2]-0.1*tt*nv.speed*radians_to_vec(nv.pose.heading))]=nv
            # print("::::::::::::;",min(min_dis))
            # if min(min_dis)<2:
            #     return (0,1,0)
            print("ACCC",(self.vehicle_spee-vehicle.speed)/0.1)
            self.vehicle_spee=vehicle.speed

            throttle = (
                # -0.23 * (obs.ego_vehicle_state.speed - (neighborhood_vehicles[0].speed))
                -self._speed_tracking * (obs.ego_vehicle_state.speed - ref_speed)
                - self._traction_gain * abs(obs.ego_vehicle_state.linear_velocity[1])+longitudinal_feed_forward+mod1+mod2
                # + self._position_adjust
                # - 0.2 * (vehicle.speed - neighborhood_vehicles[0].speed)
            )
            if self._cutin_agent_state==BehaviorAgentState.Virtual_lane_following:
                # print(lane.getLength(),ego_offset,":::::::::::::::::::")
                if len(nv_dict_upper)!=0 and ego_offset<0.9*lane.getLength():
                    
                    nv_lead=nv_dict_upper[min(nv_dict_upper)]
                    # if min(nv_dict_upper)-ego_offset<thresh:
                    mod1=-3*(ego_offset-min(nv_dict_upper)+thresh)
                    throttle = (
                    # -0.23 * (obs.ego_vehicle_state.speed - (neighborhood_vehicles[0].speed))
                    -1*self._speed_tracking * (obs.ego_vehicle_state.speed - nv_lead.speed)
                    - self._traction_gain * abs(obs.ego_vehicle_state.linear_velocity[1])+mod1+0*mod2
                    )
                else:

                    repel=0
                    vehicle_point=(vehicle.pose.position[0],vehicle.pose.position[1])
                    first_vec=10*radians_to_vec(vehicle.pose.heading+30*3.14/180)
                    second_vec=10*radians_to_vec(vehicle.pose.heading-30*3.14/180)
                    first_point=(vehicle.pose.position[0]+first_vec[0],vehicle.pose.position[1]+first_vec[1])
                    second_point=(vehicle.pose.position[0]+second_vec[0],vehicle.pose.position[1]+second_vec[1])
                    front_triangle=Polygon([vehicle_point,first_point,second_point,vehicle_point])

                    for i in nv_offlane:
                        nv_point=Point(nv_offlane[i].pose.position[0],nv_offlane[i].pose.position[1])
                        if front_triangle.contains(nv_point)==True:
                            dis=np.linalg.norm(vehicle.pose.position-nv.pose.position)
                            # if dis<thresh:
                            repel+=-100*abs(dis-thresh)
                    #     if nv_offlane[i] in list(nv_dict_lower.values()):
                    #         continue
                    #     if i<8 and np.dot(radians_to_vec(vehicle.pose.heading),radians_to_vec(nv_offlane[i].pose.heading)>0):
                    #         repel+=-10/(i**2)
                    

                    throttle = (
                    # -0.23 * (obs.ego_vehicle_state.speed - (neighborhood_vehicles[0].speed))
                    -1*self._speed_tracking * (obs.ego_vehicle_state.speed - ref_speed)
                    - self._traction_gain * abs(obs.ego_vehicle_state.linear_velocity[1])+repel
                    )
                    print("HEEEEEEEEEEEEERRRRRRRRRREEEEEEEe",mod1,throttle,repel)

            if throttle >= 0:
                brake = 0
            else:
                brake = abs(throttle)
                throttle = 0
            # print(throttle,":::::::::::::::::::::::::",brake,vehicle.speed)
            if vehicle.speed>14:
                throttle=0
                brake=1
            if len(min_dis)!=0 and min(min_dis)<6:
                throttle=0
                brake=1
            if len(max_dis)!=0:
                print(min(max_dis),"MAAAAAAAAAAAAXXXXXXXX")
            
            if min(max_dis)>5 and is_in_junction==True:
                throttle=1
                brake=0
            return (throttle, brake, steer)
            # return (1,0,steer)

        if self._cutin_agent_state==BehaviorAgentState.Virtual_lane_following:
            self._prev_cutin_agent_state=BehaviorAgentState.Virtual_lane_following

            # start_lane = miss._road_network.nearest_lane(
            # position,
            # include_junctions=False,
            # include_special=False,)

            fff = miss._waypoints.waypoint_paths_on_lane_at(
                position, start_lane.getID(), 60
            )[0]
            ll=[]
            for idx in range(len(obs.waypoint_paths)):
                ll.append(len(obs.waypoint_paths[idx]))
            if len(obs.waypoint_paths[0])==max(ll):
                fff=obs.waypoint_paths[0]
            else:
                fff=obs.waypoint_paths[ll.index(max(ll))]
            
            # print(len(obs.waypoint_paths[0]),len(obs.waypoint_paths[1]),len(obs.waypoint_paths[2]))
            look_ahead_wp_num = 3
            look_ahead_dist = 3

            print("I am here",vehicle.speed)
            vehicle_inputs=vehicle_control_commands(fff,look_ahead_wp_num,look_ahead_dist,7)
            print("or I am here")
            # if len(neighborhood_vehicles)!=0:
            #     self._cutin_agent_state=BehaviorAgentState.Approach

            return vehicle_inputs


        if self._cutin_agent_state==BehaviorAgentState.Approach:
            self._prev_cutin_agent_state=BehaviorAgentState.Approach

            offset = miss._road_network.offset_into_lane(start_lane, pose.position[:2])
            oncoming_offset = max(0, target_l.getLength() - offset)

            target_offset = miss._road_network.offset_into_lane(target_l, target_p)
            fq = offset - target_offset

            paths = miss.paths_of_lane_at(target_l, oncoming_offset, lookahead=30)

            if self._task_is_triggered is False:
                self.lane_index = start_lane.getID().split("_")[-1]

            des_lane = 0
            off_des = (aggressiveness / 10) * 15 + (1 - aggressiveness / 10) * 35
            des_speed=neighborhood_vehicles[0].speed
            print(aggressiveness,"????????????")
            # print(fq,off_des,"??????????????????????????")

            if abs(fq - off_des) > 1 and self._task_is_triggered is False:
                fff = miss._waypoints.waypoint_paths_on_lane_at(
                    position, start_lane.getID(), 60
                )[0]
                self._position_adjust = -0.3 * (fq - off_des)
            elif self._counter < 5:
                self._task_is_triggered = True
                fff = miss._waypoints.waypoint_paths_on_lane_at(
                    position, start_lane.getID(), 60
                )[0]
                self._position_adjust = -0.3 * (fq - off_des)
                self._counter += 1
                self.lateral_gain = 0.1
                self.heading_gain = 2.1
            else:
                self._task_is_triggered = True
                fff = miss._waypoints.waypoint_paths_on_lane_at(
                    position, start_lane.getID(), 60
                )[0]
                self._position_adjust = -0.3 * (fq - off_des)
                self._counter += 1
                self.lateral_gain = 0.02
                self.heading_gain = 2.1
                self._cutin_agent_state=BehaviorAgentState.Interact

            look_ahead_wp_num = 3
            look_ahead_dist = 3

            vehicle_inputs=vehicle_control_commands(fff,look_ahead_wp_num,look_ahead_dist,des_speed,longitudinal_feed_forward=self._position_adjust)
            return vehicle_inputs

        if self._cutin_agent_state==BehaviorAgentState.Interact:
            self._prev_cutin_agent_state=BehaviorAgentState.Interact


            offset = miss._road_network.offset_into_lane(start_lane, pose.position[:2])
            oncoming_offset = max(0, target_l.getLength() - offset)

            target_offset = miss._road_network.offset_into_lane(target_l, target_p)
            fq = offset - target_offset

            paths = miss.paths_of_lane_at(target_l, oncoming_offset, lookahead=30)

            if self._task_is_triggered is False:
                self.lane_index = start_lane.getID().split("_")[-1]

            des_lane = 0
            off_des = (aggressiveness / 10) * 15 + (1 - aggressiveness / 10) * 35
            des_speed=neighborhood_vehicles[0].speed

            fff = miss._waypoints.waypoint_paths_on_lane_at(
                position, target_l.getID(), 60
            )[0]
            lat_error = fff[0].signed_lateral_error(
                [vehicle.position[0], vehicle.position[1]]
            )
            des_speed=self._des_speed
            if abs(lat_error) < 0.3:
                self.lateral_gain = 0.34
                self.heading_gain = 1.2
                des_speed=neighborhood_vehicles[0].speed
            self._task_is_triggered = True
            self._position_adjust = -0.3 * (fq - off_des)
            look_ahead_wp_num = 3
            look_ahead_dist = 3

            vehicle_inputs=vehicle_control_commands(fff,look_ahead_wp_num,look_ahead_dist,14,longitudinal_feed_forward=self._position_adjust)

            return vehicle_inputs


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.StandardWithAbsoluteSteering, max_episode_steps=max_episode_steps
        ),
        agent_builder=CutInAgent,
        agent_params=(0.1,25,12)
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
    global vvv
    CutInAgent.sim = env._smarts
    # print(env._smarts, "::::::::::::::::::::::::::")
    xx=[]
    yy=[]
    CutInAgent._aggressiveness=0


    # try:
    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        agent.sim = env._smarts
        CutInAgent._aggressiveness+=2
        if CutInAgent._aggressiveness>10:
            raise Exception("SSSSSSSSSSSSSSSSSSSSS")
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            # if agent_obs.ego_vehicle_state.position[0]>96:
            #     print(agent_obs.ego_vehicle_state.position[0],"<<<<<<<<<<<<<<<<<<<<<<",CutInAgent._aggressiveness)
            #     break
            xx.append(agent_obs.ego_vehicle_state.position[0])
            yy.append(agent_obs.ego_vehicle_state.position[1])
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)
    # except:
    #     plt.plot(xx,yy)
    #     plt.show()

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
