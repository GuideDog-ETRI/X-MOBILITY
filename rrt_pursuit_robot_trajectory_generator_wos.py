from DriverBERT.dataset.PathPlanning.InformedRRTStar.informed_rrt_star_gridmap import InformedRRTStar
from DriverBERT.dataset.PathTracking.pure_pursuit.pure_pursuit import * 

from DriverBERT.dataset.grid_map_numpy import RectangularGridMap
from DriverBERT.dataset.util import is_obstacle_around_target_trajectory_in_gridmap

import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import random 
import copy
import os
show_animation = False
show_gpp_animation = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='./data', help='dataset path')
    parser.add_argument('--dataset_name', default='robot', help='dataset name (ethucy, sdd)')
    parser.add_argument("--dataset_split", default='rrtpp', help='dataset split for ethucy dataset(eth, hotel, univ, zara1, zara2')
    parser.add_argument("--obs_len", type=int, default=8, help="number of observation frames")
    parser.add_argument("--pred_len", type=int, default=12, help="number of prediction frames")
    parser.add_argument("--env_range", type=float, default=20.0, help="physically-aware range")
    parser.add_argument("--env_resol", type=float, default=0.2, help="physically-aware resolution")
    parser.add_argument("--robot_radius", type=float, default=1.0, help="physically-aware resolution")
    parser.add_argument("--safe_margin", type=float, default=0.5, help="physically-aware resolution")
    parser.add_argument("--num_obstacle_rect", type=int, default=10, help="physically-aware resolution")
    parser.add_argument("--num_obstacle_circ", type=int, default=5, help="physically-aware resolution")
    args = parser.parse_args()
    # Global Path Planning
    area_width = 100
    area_height = 100
    area_center_x = 0
    area_center_y = 0
    map_resol = 0.1
    max_iter = 500
    seq_len = args.obs_len + args.pred_len
    obs_len = args.obs_len
    pred_len = args.pred_len
    expand_dis = 2.0
    min_dist_to_dest = 40.0
    num_rows = int(area_height / map_resol)
    num_cols = int(area_width / map_resol)
    num_rows = num_rows if (num_rows % 2) == 0 else num_rows+1
    num_cols = num_cols if (num_cols % 2) == 0 else num_cols+1
    
    rand_area=[area_center_x - area_width/2.0, area_center_x + area_width/2.0, area_center_y - area_height/2.0, area_center_y + area_height/2.0]
    max_target_speed = 5.0 / 3.6  # [m/s]
    T = 200.0  # max simulation time
    safe_robot_radius = args.robot_radius + args.safe_margin
    record_dt = 0.4
    dframe = int(record_dt / dt)
    
    num_train_datasets = 10000
    num_test_datasets = 1000
    num_path = 100
    max_num_rect_obstacle = 8
    max_num_circ_obstacle = 4
    min_num_waypoint = 3
    meta_id = 0
    train_idx = 0
    scene_idx = 0
    train_df = pd.DataFrame()
    # ppname = "rrtpp"
    is_traj_gen = True

    dataset_path = os.path.join(args.dataset_path, args.dataset_name, args.dataset_split)
    dataset_scene_path = os.path.join(dataset_path, "scene")
    os.makedirs(dataset_scene_path, exist_ok=True)
    print("Start to build the train datasets")
    while train_idx < num_train_datasets:
        # Environment Generation ############################
        # Generate Path 
        if train_idx % num_path == 0 and is_traj_gen:
            scene_idx += 1
            gridmap = RectangularGridMap(width=num_cols, height=num_rows, resolution=map_resol, 
                            center_x=area_center_x, center_y=area_center_y, init_val=0.0, free_val=0.0)
            gridmap_wo_cspace = RectangularGridMap(width=num_cols, height=num_rows, resolution=map_resol, 
                                        center_x=area_center_x, center_y=area_center_y, init_val=0.0, free_val=0.0)

            gridmap_filename = os.path.join(dataset_scene_path, "train" +  str(scene_idx) + "_cspace.png")
            map = plt.imread(gridmap_filename)
            gridmap.grid_map = (map[:, :, 0] * 2).astype(np.int64) # obstacle: 2, cost: 1, free: 0

            gridmap_cspace_filename = os.path.join(dataset_scene_path, "train" +  str(scene_idx) + ".png")
            map_wo_cspace = plt.imread(gridmap_cspace_filename)
            gridmap_wo_cspace.grid_map = (map_wo_cspace[:, :, 0] * 2).astype(np.int64) # obstacle: 2, cost: 1, free: 0
            
            free_xs, free_ys = gridmap.get_all_free_positions()

        
        print(">> Pick Start and Goal Position.")
        while True:
            free_ids = np.random.choice(np.arange(len(free_xs)), 2)
            # goal_free_ids = np.random.choice(np.arange(len(free_xs)), 2)
            dx = free_xs[free_ids[1]]-free_xs[free_ids[0]]
            dy = free_ys[free_ids[1]]-free_ys[free_ids[0]]
            if math.sqrt(dx*dx + dy*dy) > min_dist_to_dest:
                start = [free_xs[free_ids[0]], free_ys[free_ids[0]]]
                goal = [free_xs[free_ids[1]], free_ys[free_ids[1]]]
                break
            # if not is_obstacle_around_target_trajectory_in_gridmap(track_traj[0, :-2, :], gridmap, self.args.obs_len, near_dist=2.0, resol=map_resol) and not is_obstacle_around_target_trajectory_in_gridmap(track_traj[0, -2:, :], gridmap, self.args.obs_len, near_dist=3.0, resol=map_resol):
            #     continue
            

        # Global Path Planning ############################
        print(">> GPP")
        rrt = InformedRRTStar(start=start, goal=goal, 
                            max_iter=max_iter, expand_dis=expand_dis,
                            rand_area=[area_center_x - area_width/2.0, area_center_x + area_width/2.0, area_center_y - area_height/2.0, area_center_y + area_height/2.0],
                            gridmap=gridmap)
        path = rrt.informed_rrt_star_search(animation=show_gpp_animation)
        if path is None:
            print("No path found")
            is_traj_gen = False
            continue
        print(">> CONTROL")
        # Path Tracking ############################
        cx, cy = [x for (x, y) in reversed(path)], [y for (x, y) in reversed(path)]
        if len(cx) < min_num_waypoint:
            print("Not enough waypoint")
            is_traj_gen = False
            continue
        waypoint_ddist = 0.5
        global_path = np.empty([0, 2])                
        for idx, (x, y) in enumerate(zip(cx[1:], cy[1:])):
            px = cx[idx]
            py = cy[idx]
            if math.sqrt((x-px)*(x-px) + (y-py)*(y-py)) > waypoint_ddist:
                theta = math.atan2(y-py, x-px) 
                dxs = np.arange(px, x, waypoint_ddist * math.cos(theta))
                dys = np.arange(py, y, waypoint_ddist * math.sin(theta))
                min_len = min(len(dxs), len(dys))
                dxs = dxs[:min_len]
                dys = dys[:min_len]
                dps = np.stack((dxs.T, dys.T), axis=1)
                global_path = np.vstack((global_path, dps))
            else:
                dps = np.array([[px, py], [x, y]])
                global_path = np.vstack((global_path, dps))
        cx, cy = global_path[:, 0], global_path[:, 1]

        if not is_obstacle_around_target_trajectory_in_gridmap(global_path, gridmap, obs_len, near_dist=2.0, resol=map_resol):
            print("NO Obstacle")
            is_traj_gen = False
            continue
        else:
            print(">> Obstacles are close")

        init_yaw = math.atan2(cy[1] - cy[0], cx[1] - cx[0])
        # initial state
        state = State(x=start[0], y=start[1], yaw=init_yaw, v=0.0)
        lastIndex = len(cx) - 1
        time = 0.0
        states = States()
        states.append(time, state)
        target_course = TargetCourse(cx, cy)
        target_ind, _ = target_course.search_target_index(state)
        target_speed = max_target_speed
        collision=False
        while T >= time and lastIndex >= target_ind:
            # Calc control input
            ai = proportional_control(target_speed, state.v)
            di, target_ind = pure_pursuit_steer_control(
                state, target_course, target_ind)
            state.update(ai, di)  # Control vehicle
            time += dt
            if gridmap.check_occupancy_from_xy_pos(state.x, state.y):
                print("Collision!")
                is_traj_gen = False
                collision = True
                break
            states.append(time, state)

            if show_animation:  # pragma: no cover
                plt.cla()
                plt.plot(start[0], start[1], "xr")
                plt.plot(goal[0], goal[1], "xr")
                plt.imshow(gridmap.grid_map, origin='lower', interpolation='nearest', extent = rand_area, alpha=1.0, cmap="Blues")
                plot_arrow(state.x, state.y, state.yaw)
                plt.plot(state.x, state.y, "bo",markersize=6)
                plt.plot(cx, cy, "-r", label="course")
                plt.plot(states.x, states.y, "-b", label="trajectory")
                plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
                plt.axis(rand_area)
                plt.grid(True)
                plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
                plt.pause(0.0001)
            gdx = state.x - cx[-1]
            gdy = state.y - cy[-1]
            if math.hypot(gdx, gdy) < 1.5:
                target_speed = 0
            # print(lastIndex, target_ind, math.hypot(gdx, gdy), round(abs(state.v)*100.0))
            if round(abs(state.v)*10.0) == 0.0 and math.hypot(gdx, gdy) < 1.5 and lastIndex == target_ind:
                print("Has arrived and stop.")
                break
        if collision:
            is_traj_gen = False
            continue
        # if T < time:
        #     print("Time over")
        #     is_traj_gen = False
        #     continue    

        states.t = states.t[::dframe]
        states.x = states.x[::dframe]
        states.y = states.y[::dframe]
        states.yaw = states.yaw[::dframe]
        states.v = states.v[::dframe]

        states.t = [states.t[0]] * obs_len + states.t + [states.t[-1]] * pred_len
        states.x = [states.x[0]] * obs_len + states.x + [states.x[-1]] * pred_len
        states.y = [states.y[0]] * obs_len + states.y + [states.y[-1]] * pred_len
        states.yaw = [states.yaw[0]] * obs_len + states.yaw + [states.yaw[-1]] * pred_len
        states.v = [states.v[0]] * obs_len + states.v  + [states.v[-1]] * pred_len




        ids = [meta_id] * len(states.v)
        scene = ['train'+ str(scene_idx)] * len(states.v)
        data = {'frame': states.t, 'x': states.x, 'y': states.y, 'yaw' : states.yaw, 'v': states.v, 'trackId': ids, 'metaId': ids, 'sceneId': scene}
        train_df = pd.concat([train_df, pd.DataFrame(data)])
        print("{0}-th dataset is built.".format(meta_id))
        meta_id = meta_id + 1
        train_idx = train_idx + 1
        is_traj_gen = True

    # os.makedirs("./data/robot", exist_ok=True)
    train_df.to_pickle(os.path.join(dataset_path, "train.pkl"))

    print("Start to build the test datasets")
    # assert False
    meta_id = num_train_datasets
    scene_idx = 0
    test_idx = 0
    test_df = pd.DataFrame()
    is_traj_gen = True
    while test_idx < num_test_datasets:
        # Environment Generation ############################
        # Generate Path 
        if test_idx % num_path == 0 and is_traj_gen:
            scene_idx += 1
            gridmap = RectangularGridMap(width=num_cols, height=num_rows, resolution=map_resol, 
                            center_x=area_center_x, center_y=area_center_y, init_val=0.0, free_val=0.0)
            gridmap_wo_cspace = RectangularGridMap(width=num_cols, height=num_rows, resolution=map_resol, 
                                        center_x=area_center_x, center_y=area_center_y, init_val=0.0, free_val=0.0)

            gridmap_filename = os.path.join(dataset_scene_path, "test" +  str(scene_idx) + "_cspace.png")
            map = plt.imread(gridmap_filename)
            gridmap.grid_map = (map[:, :, 0] * 2).astype(np.int64) # obstacle: 2, cost: 1, free: 0

            gridmap_cspace_filename = os.path.join(dataset_scene_path, "test" +  str(scene_idx) + ".png")
            map_wo_cspace = plt.imread(gridmap_cspace_filename)
            gridmap_wo_cspace.grid_map = (map_wo_cspace[:, :, 0] * 2).astype(np.int64) # obstacle: 2, cost: 1, free: 0
            
            free_xs, free_ys = gridmap.get_all_free_positions()

        print(">> Pick Start and Goal Position.")
        while True:
            free_ids = np.random.choice(np.arange(len(free_xs)), 2)
            dx = free_xs[free_ids[1]]-free_xs[free_ids[0]]
            dy = free_ys[free_ids[1]]-free_ys[free_ids[0]]
            if math.sqrt(dx*dx + dy*dy) > min_dist_to_dest:
                start = [free_xs[free_ids[0]], free_ys[free_ids[0]]]
                goal = [free_xs[free_ids[1]], free_ys[free_ids[1]]]
                break
        # free_ids = np.random.choice(np.arange(len(free_xs)), 2)
        # start = [free_xs[free_ids[0]], free_ys[free_ids[0]]]
        # goal = [free_xs[free_ids[1]], free_ys[free_ids[1]]]
        
        # Global Path Planning ############################
        print(">> GPP")
        rrt = InformedRRTStar(start=start, goal=goal, 
                            max_iter=max_iter, expand_dis=expand_dis,
                            rand_area=[area_center_x - area_width/2.0, area_center_x + area_width/2.0, area_center_y - area_height/2.0, area_center_y + area_height/2.0],
                            gridmap=gridmap)
        path = rrt.informed_rrt_star_search(animation=show_gpp_animation)
        if path is None:
            print("No path found")
            is_traj_gen = False
            continue
        print(">> CONTROL")
        # Path Tracking ############################
        cx, cy = [x for (x, y) in reversed(path)], [y for (x, y) in reversed(path)]
        if len(cx) < min_num_waypoint:
            print("Not enough waypoint")
            is_traj_gen = False
            continue
        waypoint_ddist = 0.5
        global_path = np.empty([0, 2])                
        for idx, (x, y) in enumerate(zip(cx[1:], cy[1:])):
            px = cx[idx]
            py = cy[idx]
            if math.sqrt((x-px)*(x-px) + (y-py)*(y-py)) > waypoint_ddist:
                theta = math.atan2(y-py, x-px) 
                dxs = np.arange(px, x, waypoint_ddist * math.cos(theta))
                dys = np.arange(py, y, waypoint_ddist * math.sin(theta))
                min_len = min(len(dxs), len(dys))
                dxs = dxs[:min_len]
                dys = dys[:min_len]
                dps = np.stack((dxs.T, dys.T), axis=1)
                global_path = np.vstack((global_path, dps))
            else:
                dps = np.array([[px, py], [x, y]])
                global_path = np.vstack((global_path, dps))
        cx, cy = global_path[:, 0], global_path[:, 1]

        if not is_obstacle_around_target_trajectory_in_gridmap(global_path, gridmap, obs_len, near_dist=2.0, resol=map_resol):
            print("NO Obstacle")
            is_traj_gen = False
            continue
        else:
            print(">> Obstacles are close")

        init_yaw = math.atan2(cy[1] - cy[0], cx[1] - cx[0])
        # initial state
        state = State(x=start[0], y=start[1], yaw=init_yaw, v=0.0)
        lastIndex = len(cx) - 1
        time = 0.0
        states = States()
        states.append(time, state)
        target_course = TargetCourse(cx, cy)
        target_ind, _ = target_course.search_target_index(state)
        target_speed = max_target_speed
        collision=False
        while T >= time and lastIndex >= target_ind:
            # Calc control input
            ai = proportional_control(target_speed, state.v)
            di, target_ind = pure_pursuit_steer_control(
                state, target_course, target_ind)
            state.update(ai, di)  # Control vehicle
            time += dt
            if gridmap.check_occupancy_from_xy_pos(state.x, state.y):
                print("Collision!")
                collision = True
                is_traj_gen = False
                break
            states.append(time, state)

            if show_animation:  # pragma: no cover
                plt.cla()
                plt.plot(start[0], start[1], "xr")
                plt.plot(goal[0], goal[1], "xr")
                plt.imshow(gridmap.grid_map, origin='lower', interpolation='nearest', extent = rand_area, alpha=1.0, cmap="Blues")
                plot_arrow(state.x, state.y, state.yaw)
                plt.plot(state.x, state.y, "bo",markersize=6)
                plt.plot(cx, cy, "-r", label="course")
                plt.plot(states.x, states.y, "-b", label="trajectory")
                plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
                plt.axis(rand_area)
                plt.grid(True)
                plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
                plt.pause(0.0001)
            gdx = state.x - cx[-1]
            gdy = state.y - cy[-1]
            if math.hypot(gdx, gdy) < 1.5:
                target_speed = 0
            # print(lastIndex, target_ind, math.hypot(gdx, gdy), round(abs(state.v)*100.0))
            if round(abs(state.v)*10.0) == 0.0 and math.hypot(gdx, gdy) < 1.5 and lastIndex == target_ind:
                print("Has arrived and stop.")
                break
        if collision:
            is_traj_gen = False
            continue
        # if T < time:
        #     print("Time over")
        #     is_traj_gen = False
        #     continue    

        states.t = states.t[::dframe]
        states.x = states.x[::dframe]
        states.y = states.y[::dframe]
        states.yaw = states.yaw[::dframe]
        states.v = states.v[::dframe]

        states.t = [states.t[0]] * obs_len + states.t + [states.t[-1]] * pred_len
        states.x = [states.x[0]] * obs_len + states.x + [states.x[-1]] * pred_len
        states.y = [states.y[0]] * obs_len + states.y + [states.y[-1]] * pred_len
        states.yaw = [states.yaw[0]] * obs_len + states.yaw + [states.yaw[-1]] * pred_len
        states.v = [states.v[0]] * obs_len + states.v  + [states.v[-1]] * pred_len




        ids = [meta_id] * len(states.v)
        scene = ['test'+str(scene_idx)] * len(states.v)
        data = {'frame': states.t, 'x': states.x, 'y': states.y, 'yaw' : states.yaw, 'v': states.v, 'trackId': ids, 'metaId': ids, 'sceneId': scene}
        test_df = pd.concat([test_df, pd.DataFrame(data)])
        # gridmap_filename = "./data/robot/scene/rrtpp" + str(meta_id) + ".png"
        # imageio.imwrite(gridmap_filename, gridmap_wo_cspace.grid_map)
        print("{0}-th dataset is built.".format(meta_id))
        meta_id = meta_id + 1
        test_idx = test_idx + 1
        is_traj_gen = True

    # os.makedirs("./data/robot", exist_ok=True)
    # test_df.to_pickle("./data/robot/" + ppname + "_test.pkl")
    test_df.to_pickle(os.path.join(dataset_path, "test.pkl"))


if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    main()
