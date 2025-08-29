from DriverBERT.dataset.PathPlanning.InformedRRTStar.informed_rrt_star_gridmap import InformedRRTStar
from DriverBERT.dataset.PathTracking.pure_pursuit.pure_pursuit import * 

from DriverBERT.dataset.grid_map_numpy import RectangularGridMap

import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import random 
import copy
import os
show_animation = True
show_gpp_animation = False
def random_polygons(map_center_x, map_center_y, map_size_x, map_size_y, num_polygon=1, rectangle=True, robot_radius=1.0):
    min_dist = 4.0
    max_dist = 16.0      
    dtheta = 0.1
    center_x = np.random.choice(np.hstack((np.arange(-map_size_x/2.0 + max_dist, map_size_x/2.0 - max_dist, 0.1))), num_polygon)
    center_y = np.random.choice(np.hstack((np.arange(-map_size_y/2.0 + max_dist, map_size_y/2.0 - max_dist, 0.1))), num_polygon)    
    if rectangle: # rectangle
        length_x = np.random.choice(np.arange(min_dist, max_dist, 0.1), num_polygon) # min_dist + random.random() * (max_dist - min_dist)
        length_y = np.random.choice(np.arange(min_dist, max_dist, 0.1), num_polygon) # min_dist + random.random() * (max_dist - min_dist)
        pxs = np.stack([center_x-length_x/2.0, center_x-length_x/2.0, center_x+length_x/2.0, center_x+length_x/2.0], axis=1)
        pys = np.stack([center_y-length_y/2.0, center_y+length_y/2.0, center_y+length_y/2.0, center_y-length_y/2.0], axis=1)
    

        radius = np.ones((num_polygon))*robot_radius
        dtheta = 0.1
        dthetas_ur = np.tile(np.arange(0, 0.5*np.pi, dtheta), (num_polygon, 1))
        dthetas_ul = np.tile(np.arange(0.5*np.pi, np.pi, dtheta), (num_polygon, 1))
        dthetas_dl = np.tile(np.arange(np.pi, 1.5*np.pi, dtheta), (num_polygon, 1))
        dthetas_dr = np.tile(np.arange(1.5*np.pi, 2*np.pi, dtheta), (num_polygon, 1))

        bndr_pxs = np.cos(dthetas_ur) * radius[:, np.newaxis] + center_x[:, np.newaxis]+length_x[:, np.newaxis]/2.0
        bndr_pys = np.sin(dthetas_ur) * radius[:, np.newaxis] + center_y[:, np.newaxis]+length_y[:, np.newaxis]/2.0

        bndr_pxs = np.concatenate((bndr_pxs, np.cos(dthetas_ul) * radius[:, np.newaxis] + center_x[:, np.newaxis]-length_x[:, np.newaxis]/2.0), axis=1)
        bndr_pys = np.concatenate((bndr_pys, np.sin(dthetas_ul) * radius[:, np.newaxis] + center_y[:, np.newaxis]+length_y[:, np.newaxis]/2.0), axis=1)
        
        bndr_pxs = np.concatenate((bndr_pxs, np.cos(dthetas_dl) * radius[:, np.newaxis] + center_x[:, np.newaxis]-length_x[:, np.newaxis]/2.0), axis=1)
        bndr_pys = np.concatenate((bndr_pys, np.sin(dthetas_dl) * radius[:, np.newaxis] + center_y[:, np.newaxis]-length_y[:, np.newaxis]/2.0), axis=1)
        
        bndr_pxs = np.concatenate((bndr_pxs, np.cos(dthetas_dr) * radius[:, np.newaxis] + center_x[:, np.newaxis]+length_x[:, np.newaxis]/2.0), axis=1)
        bndr_pys = np.concatenate((bndr_pys, np.sin(dthetas_dr) * radius[:, np.newaxis] + center_y[:, np.newaxis]-length_y[:, np.newaxis]/2.0), axis=1)

        return pxs, pys, bndr_pxs, bndr_pys
    else:
        radius = np.random.choice(np.arange(0.5*min_dist, 0.5*max_dist, 0.1), num_polygon)
        dthetas = np.tile(np.arange(0, 2*np.pi, dtheta), (num_polygon, 1))
        pxs = np.cos(dthetas) * radius[:, np.newaxis] + center_y[:, np.newaxis]
        pys = np.sin(dthetas) * radius[:, np.newaxis] + center_x[:, np.newaxis]
        radius = radius + robot_radius
        bndr_pxs = np.cos(dthetas) * (radius[:, np.newaxis]) + center_y[:, np.newaxis]
        bndr_pys = np.sin(dthetas) * (radius[:, np.newaxis])+ center_x[:, np.newaxis]
        return pxs, pys, bndr_pxs, bndr_pys

    

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
    min_dist_to_dest = 20.0
    num_rows = int(area_height / map_resol)
    num_cols = int(area_width / map_resol)
    num_rows = num_rows if (num_rows % 2) == 0 else num_rows+1
    num_cols = num_cols if (num_cols % 2) == 0 else num_cols+1
    
    rand_area=[area_center_x - area_width/2.0, area_center_x + area_width/2.0, area_center_y - area_height/2.0, area_center_y + area_height/2.0]
    target_speed = 5.0 / 3.6  # [m/s]
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
            print("Generate New Environment")
            gridmap = RectangularGridMap(width=num_cols, height=num_rows, resolution=map_resol, 
                                        center_x=area_center_x, center_y=area_center_y, init_val=0.0, free_val=0.0)
            gridmap_wo_cspace = RectangularGridMap(width=num_cols, height=num_rows, resolution=map_resol, 
                                        center_x=area_center_x, center_y=area_center_y, init_val=0.0, free_val=0.0)

            # print("Generate New Rectangle Obstacles")
            rect_pxs, rect_pys, rect_bndr_pxs, rect_bndr_pys = random_polygons(area_center_x, area_center_y, area_width, area_height, max_num_rect_obstacle, rectangle=True, robot_radius=safe_robot_radius)
            for px, py in zip(rect_pxs, rect_pys):
                gridmap.set_value_from_polygon(pol_x=px.tolist(), pol_y=py.tolist(), val=2) 
            # print("Generate New Circle Environment")
            circ_pxs, circ_pys, circ_bndr_pxs, circ_bndr_pys = random_polygons(area_center_x, area_center_y, area_width, area_height, max_num_circ_obstacle, rectangle=False, robot_radius=safe_robot_radius)
            for px, py in zip(circ_pxs, circ_pys):
                gridmap.set_value_from_polygon(pol_x=px.tolist(), pol_y=py.tolist(), val=2) 
            gridmap_wo_cspace = copy.deepcopy(gridmap)
            for px, py in zip(rect_bndr_pxs, rect_bndr_pys):
                gridmap.set_value_from_polygon(pol_x=px.tolist(), pol_y=py.tolist(), val=1, free_only=True)
            for px, py in zip(circ_bndr_pxs, circ_bndr_pys):
                gridmap.set_value_from_polygon(pol_x=px.tolist(), pol_y=py.tolist(), val=1, free_only=True) 
            free_xs, free_ys = gridmap.get_all_free_positions()
            scene_idx += 1

            gridmap_filename = os.path.join(dataset_scene_path, "train" +  str(scene_idx) + ".png")
            plt.imsave(gridmap_filename, gridmap_wo_cspace.grid_map, cmap='gray', pil_kwargs={'compress_level':0})
            gridmap_cspace_filename = os.path.join(dataset_scene_path, "train" + str(scene_idx) + "_cspace.png")
            plt.imsave(gridmap_cspace_filename, gridmap.grid_map, cmap='gray', pil_kwargs={'compress_level':0})
        
        print(">> Pick Start and Goal Position.")
        while True:
            free_ids = np.random.choice(np.arange(len(free_xs)), 2)
            dx = free_xs[free_ids[1]]-free_xs[free_ids[0]]
            dy = free_ys[free_ids[1]]-free_ys[free_ids[0]]
            if math.sqrt(dx*dx + dy*dy) > min_dist_to_dest:
                start = [free_xs[free_ids[0]], free_ys[free_ids[0]]]
                goal = [free_xs[free_ids[1]], free_ys[free_ids[1]]]
                break

            

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
        init_yaw = math.atan2(cy[1] - cy[0], cx[1] - cx[0])
        # initial state
        state = State(x=start[0], y=start[1], yaw=init_yaw, v=0.0)

        lastIndex = len(cx) - 1
        time = 0.0
        states = States()
        states.append(time, state)
        target_course = TargetCourse(cx, cy)
        target_ind, _ = target_course.search_target_index(state)
        while T >= time and lastIndex > target_ind:
            # Deceleration within specific diatance
            print(target_ind, lastIndex, cx[-1], cy[-1], state.x, state.y)
            if lastIndex == target_ind:
                print("last idx")
                target_speed = 0
            # Calc control input
            ai = proportional_control(target_speed, state.v)
            di, target_ind = pure_pursuit_steer_control(
                state, target_course, target_ind)
            state.update(ai, di)  # Control vehicle
            time += dt
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
                plt.pause(0.001)
        if T < time:
            print("Time over")
            is_traj_gen = False
            continue

        states.t = states.t[::dframe]
        states.x = states.x[::dframe]
        states.y = states.y[::dframe]
        states.yaw = states.yaw[::dframe]
        states.v = states.v[::dframe]

        # if len(states.t) < seq_len:
        # add_len = seq_len - len(states.t)
        states.t = [states.t[0]] * obs_len + states.t + [states.t[-1]] * pred_len
        states.x = [states.x[0]] * obs_len + states.x + [states.x[-1]] * pred_len
        states.y = [states.y[0]] * obs_len + states.y + [states.y[-1]] * pred_len
        states.yaw = [states.yaw[0]] * obs_len + states.yaw + [states.yaw[-1]] * pred_len
        states.v = [states.v[0]] * obs_len + states.v  + [states.v[-1]] * pred_len

        # states.t = states.t + [states.t[-1]] * pred_len
        # states.x = states.x + [states.x[-1]] * pred_len
        # states.y = states.y + [states.y[-1]] * pred_len
        # states.yaw = states.yaw + [states.yaw[-1]] * pred_len
        # states.v = states.v + [states.v[-1]] * pred_len
        # start_frame = random.randint(0, len(states.t)-seq_len)
        # states.t = states.t[start_frame:]
        # states.x = states.x[start_frame:]
        # states.y = states.y[start_frame:]
        # states.yaw = states.yaw[start_frame:]
        # states.v = states.v[start_frame:]

        # if len(states.t) < seq_len:
        # add_len = seq_len - len(states.t)


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
            print("Generate New Environment")
            gridmap = RectangularGridMap(width=num_cols, height=num_rows, resolution=map_resol, 
                                        center_x=area_center_x, center_y=area_center_y, init_val=0.0, free_val=0.0)
            gridmap_wo_cspace = RectangularGridMap(width=num_cols, height=num_rows, resolution=map_resol, 
                                        center_x=area_center_x, center_y=area_center_y, init_val=0.0, free_val=0.0)

            # print("Generate New Rectangle Obstacles")
            rect_pxs, rect_pys, rect_bndr_pxs, rect_bndr_pys = random_polygons(area_center_x, area_center_y, area_width, area_height, max_num_rect_obstacle, rectangle=True)
            for px, py in zip(rect_pxs, rect_pys):
                gridmap.set_value_from_polygon(pol_x=px.tolist(), pol_y=py.tolist(), val=2) 
            # print("Generate New Circle Environment")
            circ_pxs, circ_pys, circ_bndr_pxs, circ_bndr_pys = random_polygons(area_center_x, area_center_y, area_width, area_height, max_num_circ_obstacle, rectangle=False)
            for px, py in zip(circ_pxs, circ_pys):
                gridmap.set_value_from_polygon(pol_x=px.tolist(), pol_y=py.tolist(), val=2) 
            gridmap_wo_cspace = copy.deepcopy(gridmap)
            for px, py in zip(rect_bndr_pxs, rect_bndr_pys):
                gridmap.set_value_from_polygon(pol_x=px.tolist(), pol_y=py.tolist(), val=1, free_only=True)
            for px, py in zip(circ_bndr_pxs, circ_bndr_pys):
                gridmap.set_value_from_polygon(pol_x=px.tolist(), pol_y=py.tolist(), val=1, free_only=True) 
            free_xs, free_ys = gridmap.get_all_free_positions()
            scene_idx += 1
            # gridmap_filename = "./data/robot/scene/" + ppname + "/test" + str(scene_idx) + ".png"
            # plt.imsave(gridmap_filename, gridmap_wo_cspace.grid_map, cmap='gray', pil_kwargs={'compress_level':0})
            # gridmap_cspace_filename = "./data/robot/scene/" + ppname + "/test" + str(scene_idx) + "_cspace.png"
            # plt.imsave(gridmap_cspace_filename, gridmap.grid_map, cmap='gray', pil_kwargs={'compress_level':0})

            gridmap_filename = os.path.join(dataset_scene_path, "test" +  str(scene_idx) + ".png")
            plt.imsave(gridmap_filename, gridmap_wo_cspace.grid_map, cmap='gray', pil_kwargs={'compress_level':0})
            gridmap_cspace_filename = os.path.join(dataset_scene_path, "test" + str(scene_idx) + "_cspace.png")
            plt.imsave(gridmap_cspace_filename, gridmap.grid_map, cmap='gray', pil_kwargs={'compress_level':0})
        
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
        # Path Tracking ############################
        print(">> CONTROL")
        cx, cy = [x for (x, y) in reversed(path)], [y for (x, y) in reversed(path)]
        if len(cx) < min_num_waypoint:
            print("Not enough waypoint")
            is_traj_gen = False
            continue
        init_yaw = math.atan2(cy[1] - cy[0], cx[1] - cx[0])
        # initial state
        state = State(x=start[0], y=start[1], yaw=init_yaw, v=0.0)

        lastIndex = len(cx) - 1
        time = 0.0
        states = States()
        states.append(time, state)
        target_course = TargetCourse(cx, cy)
        target_ind, _ = target_course.search_target_index(state)
        while T >= time and lastIndex > target_ind:
            # Calc control input
            ai = proportional_control(target_speed, state.v)
            di, target_ind = pure_pursuit_steer_control(
                state, target_course, target_ind)
            state.update(ai, di)  # Control vehicle
            time += dt
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
                plt.pause(0.001)
        if T < time:
            print("Time over")
            is_traj_gen = False
            continue
        states.t = states.t[::dframe]
        states.x = states.x[::dframe]
        states.y = states.y[::dframe]
        states.yaw = states.yaw[::dframe]
        states.v = states.v[::dframe]

        # if len(states.t) < seq_len:
        #     add_len = seq_len - len(states.t)
        #     states.t = [states.t[0]] * add_len + states.t
        #     states.x = [states.x[0]] * add_len + states.x
        #     states.y = [states.y[0]] * add_len + states.y
        #     states.yaw = [states.yaw[0]] * add_len + states.yaw
        #     states.v = [states.v[0]] * add_len + states.v
        # start_frame = random.randint(0, len(states.t)-seq_len)
        # states.t = states.t[start_frame:]
        # states.x = states.x[start_frame:]
        # states.y = states.y[start_frame:]
        # states.yaw = states.yaw[start_frame:]
        # states.v = states.v[start_frame:]
        # if len(states.t) < seq_len:
        #     add_len = seq_len - len(states.t)
        #     states.t = states.t + [states.t[-1]] * add_len
        #     states.x = states.x + [states.x[-1]] * add_len
        #     states.y = states.y + [states.y[-1]] * add_len
        #     states.yaw = states.yaw + [states.yaw[-1]] * add_len
        #     states.v = states.v + [states.v[-1]] * add_len


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
