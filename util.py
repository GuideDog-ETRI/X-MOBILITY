
import math
import random
import copy
import numpy as np
from scipy.spatial import distance
from DriverBERT.dataset.grid_map_numpy import RectangularGridMap


# def cotrain_collate_fn(keys, batchs):
    

def rad2deg(rad):
    return rad*180/np.pi


def deg2rad(deg):
    return deg*np.pi/180


def to_scene(trajs):
    return trajs.transpose(1, 0, 2)


def to_trajectory(scene):
    return scene.transpose(1, 0, 2)


def shift_xy(xy, center):
    xy = xy + center[np.newaxis, :]
    return xy


def shift_trajs(trajs, center):
    trajs[:, :, 0:2] = trajs[:, :, 0:2] + center[np.newaxis, np.newaxis, 0:2]
    return trajs


def extract_patch_from_map(src_map, patch_size):
    num_width_patch = src_map.width // patch_size
    num_height_patch = src_map.height // patch_size
    patches = []
    patches_xy = []
    attn_mask = np.ones(num_height_patch*num_width_patch)
    pidx = 0
    for i in range(num_height_patch):
        for j in range(num_width_patch):
            patch = src_map.grid_map[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            x_pos = src_map.min_x + (j*patch_size+patch_size/2) * src_map.resolution # + src_map.resolution / 2.0 # src_map.calc_grid_center_position_from_index(j*patch_size+patch_size/2, src_map.min_x) - src_map.resolution/2.0
            y_pos = src_map.min_y + (i*patch_size+patch_size/2) * src_map.resolution # src_map.calc_grid_center_position_from_index(i*patch_size+patch_size/2, src_map.min_y) - src_map.resolution/2.0
            if np.all(patch == src_map.init_val):
                attn_mask[pidx] = 0
            patches.append(patch.flatten())
            patches_xy.append([x_pos, y_pos])
            pidx += 1

    return np.array(patches), np.array(patches_xy), attn_mask

def convert_patches_to_map(patches, patch_size, resol=0.1):
    num_patch, patch_len = patches.shape
    num_side_patch = np.sqrt(num_patch)
    patch_size = np.sqrt(patch_len)
    width = height = int(num_side_patch*patch_size)
    dst_map = RectangularGridMap(width, height, resol, 0, 0)
    for pidx, patch in enumerate(patches):
        px = pidx % num_side_patch
        py = pidx // num_side_patch
        s_ids = np.arange(patch_len)
        sxs = (s_ids % patch_size + px*patch_size).astype(np.int64)
        sys = (s_ids // patch_size + py*patch_size).astype(np.int64)
        dst_map.set_value_from_xy_index(sxs, sys, patch.flatten())
    return dst_map


def expand_map_with_pad(src_map, pad_val, pad_size):
    width = src_map.width+2*pad_size
    height = src_map.height+2*pad_size
    dst_map = RectangularGridMap(width=width, height=height, resolution=src_map.resolution, center_x=src_map.center_x, center_y=src_map.center_y, init_val=pad_val)
    x_ids, y_ids = src_map.get_all_indices()
    vals = src_map.get_value_from_xy_index(x_ids, y_ids)
    dst_map.set_value_from_xy_index(x_ids+pad_size, y_ids+pad_size, vals)
    return dst_map


def extract_map(src_map, range, resol, trans=None, rot=None):
    num_rows = int((2.0 * range) / resol)
    num_cols = int((2.0 * range) / resol)
    num_rows = num_rows if (num_rows % 2) == 0 else num_rows+1
    num_cols = num_cols if (num_cols % 2) == 0 else num_cols+1
    dst_map = RectangularGridMap(num_cols, num_rows, resol, 0, 0)
    xs, ys = dst_map.get_all_positions()
    if np.all(trans is not None) and rot is not None:
        xys = np.stack((xs, ys), axis=-1)
        rxys = rotate_xy(xys, theta=-rot)
        rxys = shift_xy(rxys, center=-trans)
        vals, valid = src_map.get_value_from_xy_pos(rxys[:, 0], rxys[:, 1])
        xys = xys[valid]
        dst_map.set_value_from_xy_pos(xys[:, 0], xys[:, 1], vals, bigger=True)
    else:
        xys = np.stack((xs, ys), axis=-1)
        vals, valid = src_map.get_value_from_xy_pos(xys[:, 0], xys[:, 1])
        xys = xys[valid]
        dst_map.set_value_from_xy_pos(xys[:, 0], xys[:, 1], vals, bigger=True)
    return dst_map


def extract_map(src_map, nrows, ncols, resol, trans=None, rot=None):
    dst_map = RectangularGridMap(ncols, nrows, resol, 0, 0)
    xs, ys = dst_map.get_all_positions()
    if np.all(trans is not None) and rot is not None:
        xys = np.stack((xs, ys), axis=-1)
        rxys = rotate_xy(xys, theta=-rot)
        rxys = shift_xy(rxys, center=-trans)
        vals, valid = src_map.get_value_from_xy_pos(rxys[:, 0], rxys[:, 1])
        xys = xys[valid]
        dst_map.set_value_from_xy_pos(xys[:, 0], xys[:, 1], vals, bigger=True)
    else:
        xys = np.stack((xs, ys), axis=-1)
        vals, valid = src_map.get_value_from_xy_pos(xys[:, 0], xys[:, 1])
        xys = xys[valid]
        dst_map.set_value_from_xy_pos(xys[:, 0], xys[:, 1], vals, bigger=True)
    return dst_map

# def extract_map(src_map, num_grid, resol, trans=None, rot=None):
#     # num_rows = int((2.0 * range) / resol)
#     # num_cols = int((2.0 * range) / resol)
#     # num_rows = num_rows if (num_rows % 2) == 0 else num_rows+1
#     # num_cols = num_cols if (num_cols % 2) == 0 else num_cols+1
#     dst_map = RectangularGridMap(num_cols, num_rows, resol, 0, 0)
#     xs, ys = dst_map.get_all_positions()
#     if np.all(trans is not None) and rot is not None:
#         xys = np.stack((xs, ys), axis=-1)
#         rxys = rotate_xy(xys, theta=-rot)
#         rxys = shift_xy(rxys, center=-trans)
#         vals, valid = src_map.get_value_from_xy_pos(rxys[:, 0], rxys[:, 1])
#         xys = xys[valid]
#         dst_map.set_value_from_xy_pos(xys[:, 0], xys[:, 1], vals, bigger=True)
#     else:
#         xys = np.stack((xs, ys), axis=-1)
#         vals, valid = src_map.get_value_from_xy_pos(xys[:, 0], xys[:, 1])
#         xys = xys[valid]
#         dst_map.set_value_from_xy_pos(xys[:, 0], xys[:, 1], vals, bigger=True)
#     return dst_map
def circle_segment_intersection(p1: tuple, p2: tuple, r: float) -> list:
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]

    dx, dy = x2 - x1, y2 - y1
    dr2 = dx * dx + dy * dy
    D = x1 * y2 - x2 * y1

    # the first element is the point within segment
    d1 = x1 * x1 + y1 * y1
    d2 = x2 * x2 + y2 * y2
    dd = d2 - d1

    delta = math.sqrt(r * r * dr2 - D * D)

    if delta >= 0:
        if (delta == 0):
            return [(D * dy / dr2, -D * dx / dr2)]
        else:
            return [
                ((D * dy + math.copysign(1.0, dd) * dx * delta) / dr2,
                (-D * dx + math.copysign(1.0, dd) * dy * delta) / dr2),
                ((D * dy - math.copysign(1.0, dd) * dx * delta) / dr2,
                (-D * dx - math.copysign(1.0, dd) * dy * delta) / dr2)
            ]
    else:
        return []

# @staticmethod
def clamp(val: float, min_val: float, max_val: float) -> float:
    if val < min_val:
        val = min_val
    if val > max_val :
        val = max_val
    return val
    
def rotate_xy(xy, theta):
    ct = math.cos(theta)
    st = math.sin(theta)
    r = np.array([[ct, st], [-st, ct]])
    return xy.dot(r.transpose())


def rotate_trajs(trajs, theta, traj_dim=2):
    ct = math.cos(theta)
    st = math.sin(theta)
    if traj_dim == 4:
        r = np.array([[ct, st, 0, 0], [-st, ct, 0, 0], [0, 0, ct, st], [0, 0, -st, ct]])
        rtrajs = []
        for idx, traj in enumerate(trajs):
            rtrajs.append(traj.dot(r.transpose()))
    else:
        r = np.array([[ct, st], [-st, ct]])
        rtrajs = []
        for idx, traj in enumerate(trajs):
            rtrajs.append(traj.dot(r.transpose()))
    return np.array(rtrajs)


def aug_random_flip(trajs, tgt_end_pos, full_traj, env=None, prob_thres=0.5):
    prob = random.random()
    end_pos = copy.deepcopy(tgt_end_pos)
    if prob < prob_thres:
        if prob < prob_thres/2.0:
            trajs[:, :, 0] = -trajs[:, :, 0]
            end_pos[0] = -tgt_end_pos[0]
            full_traj[:, 0] = -full_traj[:, 0]
            if env:
                env.flip_y()
        else:  
            trajs[:, :, 1] = -trajs[:, :, 1]
            end_pos[1] = -tgt_end_pos[1]
            full_traj[:, 1] = -full_traj[:, 1]
            if env:
                env.flip_x()
    return trajs, end_pos, full_traj, env


def aug_random_scale(trajs, full_traj, view_range=20.0, obs_len=8, min_scale=0.8, max_scale=1.2, prob_thres=0.5):
    rnd = random.random()
    if rnd < prob_thres:
        scale = min_scale + (max_scale-min_scale)*(rnd-prob_thres)/prob_thres
        tgt_traj = copy.deepcopy(trajs[0])
        tgt_traj = tgt_traj * scale
        tgt_obs_traj = tgt_traj[:obs_len]
        dist = np.linalg.norm(tgt_obs_traj, axis=-1)
        dist = dist[~np.isnan(dist)]
        if np.all(dist < view_range):
            trajs = trajs * scale
            full_traj = full_traj * scale
    return trajs, full_traj


def aug_random_rotation(trajs, tgt_end_pos, full_traj, env=None, prob_thres=0.5):
    prob = random.random()
    theta = prob * 2.0*np.pi
    trajs = rotate_trajs(trajs=trajs, theta=theta)
    full_traj = rotate_xy(xy=full_traj, theta=theta)
    tgt_end_pos = rotate_xy(xy=tgt_end_pos, theta=theta)
    if env:
        env.rotate(theta)
    return trajs, tgt_end_pos, full_traj, env

def neighbor_filtering(trajs, num_nbr, obs_len, pred_len, input_dim=2, view_range=20.0, view_angle=np.pi/3.0, social_range=2.0):
    """
      At the last frame in observation frames
      After Transform, Filtering, only for observation frames,
      For observation frames
    """
    nbr_trajs = copy.deepcopy(trajs[1:, :obs_len+pred_len])
    nbr_trajs = nbr_trajs[~np.all(np.isnan(nbr_trajs[:, :obs_len, 0]), axis=1)]
    if len(nbr_trajs) > 0:
        nbr_dist = np.sqrt(np.sum(np.square(nbr_trajs[:, :, 0:2]), axis=2))
        nbr_dist[np.isnan(nbr_dist)] = view_range
        out_range = nbr_dist >= view_range
        nbr_trajs[out_range, :] = np.nan

        nbr_curr_pos = copy.deepcopy(nbr_trajs[:, obs_len-1])
        nbr_curr_dist = copy.deepcopy(nbr_dist[:, obs_len-1])

        out_social_traj = nbr_curr_dist > social_range # out of interest
        out_sight_traj_angle = np.abs(np.arctan2(nbr_curr_pos[:, 1], nbr_curr_pos[:, 0]))
        out_sight_traj_angle[np.isnan(out_sight_traj_angle)] = view_angle
        out_sight_traj = out_sight_traj_angle > view_angle/2.0
        out_interest_traj = out_social_traj & out_sight_traj
        nbr_curr_pos[out_interest_traj, :] = np.nan
        valid_nbr_trajs = np.array([not np.isnan(pos[0]) for pos in nbr_curr_pos])

        nbr_trajs_in = nbr_trajs[valid_nbr_trajs]

        # Neighbor Importance Filtering
        nbr_curr_dist_in = nbr_curr_dist[valid_nbr_trajs]
        nearest_nbr_in = np.argsort(nbr_curr_dist_in)

        # Maximum number of neighbor
        nbr_trajs_in = nbr_trajs_in[nearest_nbr_in]
        nbr_trajs = nbr_trajs_in
        num_nbr = min(num_nbr, len(nbr_trajs))
        nbr_trajs = nbr_trajs[:num_nbr, :, :]

        # Random ordering
        rand_order = np.arange(len(nbr_trajs))
        np.random.shuffle(rand_order)
        nbr_trajs = nbr_trajs[rand_order]

        out_trajs = np.full((num_nbr + 1, obs_len + pred_len, input_dim), np.nan)
        out_trajs[0] = trajs[0]
        out_trajs[1:, :obs_len+pred_len] = nbr_trajs
        out_trajs = out_trajs[~np.all(np.isnan(out_trajs[:, :obs_len, 0]), axis=1)]
    else:
        out_trajs = np.expand_dims(trajs[0], axis=0)

    return out_trajs


def is_target_trajectory_inbound(tgt_traj, obs_len=8, traj_bound=20.0): 
    center_pos = tgt_traj[obs_len-1]
    tgt_traj = tgt_traj - center_pos[np.newaxis, :]
    dist = np.linalg.norm(tgt_traj, axis=-1)
    dist = dist[~np.isnan(dist)]
    return ~np.any(dist >= traj_bound)

def is_obstacle_around_target_trajectory(tgt_traj, gridmap, obs_len=8, near_dist=2.0, resol=0.1):
    tgt_future_traj = tgt_traj[obs_len:]
    occ_x_pos, occ_y_pos = gridmap.get_all_occupied_positions(dgrid=int(near_dist/resol))
    cdist = distance.cdist(tgt_future_traj, np.stack((occ_x_pos, occ_y_pos), axis=1))
    if np.any(cdist < near_dist):
        return True
    else:
        return False
    
def is_obstacle_around_target_trajectory_in_gridmap(tgt_traj, gridmap, obs_len=8, near_dist=2.0, resol=0.2):
    tgt_future_traj = tgt_traj
    x_ids, y_ids, valid = gridmap.get_xy_index_from_xy_pos(tgt_future_traj[:, 0], tgt_future_traj[:, 1])
    x_ids, y_ids = x_ids[valid], y_ids[valid]
    idx_dist = int(near_dist/resol)
    dx_ids, dy_ids = np.mgrid[slice(-idx_dist, idx_dist+1, int(1/resol)), slice(-idx_dist, idx_dist+1, int(1/resol))]
    sub_x_ids = x_ids[:, np.newaxis] - dx_ids.flatten().transpose()[np.newaxis, :]
    sub_y_ids = y_ids[:, np.newaxis] - dy_ids.flatten().transpose()[np.newaxis, :]
    sub_ids = np.stack((sub_x_ids.flatten(), sub_y_ids.flatten()), axis=1)
    sub_ids = np.unique(sub_ids, axis=0)
    x_valid = (0 <= sub_ids[:, 0]) & (sub_ids[:, 0] < gridmap.width)
    y_valid = (0 <= sub_ids[:, 1]) & (sub_ids[:, 1] < gridmap.height)
    valid = x_valid & y_valid
    sub_ids = sub_ids[valid, :]
    if np.any(gridmap.check_occupancy_from_xy_index(sub_ids[:, 0], sub_ids[:, 1])):
        return True
    else:
        return False


def get_obstacles_in_view_area(gridmap, view_range=15.0, view_angle=2.09):  
    env_occupied_xs, env_occupied_ys = gridmap.get_all_occupied_positions()
    env_occupied_xys = np.stack((env_occupied_xs, env_occupied_ys), axis=1)
    dist = np.linalg.norm(env_occupied_xys, axis=-1)
    env_occupied_xys = env_occupied_xys[dist < view_range]
    out_sight_traj_angle = np.abs(np.arctan2(env_occupied_xys[:, 1], env_occupied_xys[:, 0]))
    env_occupied_xys = env_occupied_xys[out_sight_traj_angle < view_angle/2.0, :]
    return env_occupied_xys

def is_target_trajectory_outbound(tgt_traj, obs_len=8, traj_bound=20.0): 
    # tgt_obs_traj = tgt_traj[:obs_len]
    center_pos = tgt_traj[obs_len-1]
    tgt_traj = tgt_traj - center_pos[np.newaxis, :]
    dist = np.linalg.norm(tgt_traj, axis=-1)
    dist = dist[~np.isnan(dist)]
    return np.any(dist >= traj_bound)

def is_target_outbound(tgt_traj, obs_len=8, traj_bound=20.0): 
    tgt_obs_traj = tgt_traj[:obs_len]
    center_pos = tgt_obs_traj[obs_len-1]
    tgt_obs_traj = tgt_obs_traj - center_pos[np.newaxis, :]
    dist = np.linalg.norm(tgt_obs_traj, axis=-1)
    dist = dist[~np.isnan(dist)]
    return np.any(dist >= traj_bound)


def translation_to_target(obs_trajs):
    center = -obs_trajs[0, -1, 0:2]
    nbr_obs_trajs = shift_trajs(obs_trajs, center)
    return obs_trajs


def transform_to_stationary_target(trajs, full_trajs, obs_len=8, traj_dim=2):
    center = -trajs[0, obs_len-1, 0:2]
    trajs = shift_trajs(trajs, center)
    if traj_dim > 2:
        theta = np.arctan2(trajs[0, obs_len-1, 1], trajs[0, obs_len-1, 0])
        trajs = rotate_trajs(trajs, theta, traj_dim)
    else:
        # 
        if np.any(np.isnan(trajs[0, obs_len-2:obs_len])):
            theta = 0
        else:
            last1_obs = trajs[0, obs_len-1]
            last2_obs = trajs[0, obs_len-2]
            diff = np.array([last1_obs[0] - last2_obs[0], last1_obs[1] - last2_obs[1]])
            theta = np.arctan2(diff[1], diff[0])
            trajs = rotate_trajs(trajs, theta)
    return trajs, center, theta


def transform_to_target(trajs, obs_len=8, traj_dim=2):
    center = -trajs[0, obs_len-1, 0:2]
    trajs = shift_trajs(trajs, center)
    if traj_dim > 2:
        theta = np.arctan2(trajs[0, obs_len-1, 1], trajs[0, obs_len-1, 0])
        trajs = rotate_trajs(trajs, theta, traj_dim)
    else:
        if np.any(np.isnan(trajs[0, obs_len-2:obs_len])):
            theta = 0
        else:
            last1_obs = trajs[0, obs_len-1]
            last2_obs = trajs[0, obs_len-2]
            diff = np.array([last1_obs[0] - last2_obs[0], last1_obs[1] - last2_obs[1]])
            theta = np.arctan2(diff[1], diff[0])
            trajs = rotate_trajs(trajs, theta)
    return trajs, center, theta

def is_near(trajs, radius=6.0):
    """
    derive label for near neighbors closer than r meters from target even at least once
    """
    if len(trajs) == 1:
        return [True]
    else:
        dist = np.sum(np.square(trajs - trajs[0, :]), axis=2)
        return np.nanmin(dist, axis=1) < radius ** 2



def is_collided(trajs, radius=0.4):
    """
    derive label for near neighbors closer than r meters from target even at least once
    """
    if len(trajs) == 1:
        return [False]
    else:
        # print(trajs.shape)
        dist = np.linalg.norm(trajs[1:, :] - trajs[0, :], axis=-1)
        return np.nanmin(dist, axis=1) < radius*2






