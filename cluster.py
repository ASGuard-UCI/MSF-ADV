from __future__ import division
import numpy as np
import math


def rc2grid(r, c):
    return r * 512 + c


class Node:
    def __init__(self):
        self.center_node = (-1, -1)
        self.parent = (-1, -1)
        self.traversed = 0
        self.is_center = False
        self.is_object = False
        self.point_num = 0
        self.obs_id = -1
        self.node_rank = 0


def Traverse(grid, x):
    pts = []
    while grid[x].traversed == 0:
        pts.append(x)
        grid[x].traversed = 2
        x = grid[x].center_node
    if grid[x].traversed == 2:
        for p in pts:
            grid[p].is_center = True
    for p in pts:
        grid[p].traversed = 1
        grid[p].parent = grid[x].parent


# DJS
def DJSMakeSet(grid, x):
    grid[x].parent = x
    grid[x].node_rank = 0


def DJSFindRecursive(grid, x):
    if grid[x].parent != x:
        grid[x].parent = DJSFindRecursive(grid, grid[x].parent)
    return grid[x].parent


def DJSFind(grid, x):
    y = grid[x].parent
    if y == x or grid[y].parent == y:
        return y
    root = DJSFindRecursive(grid, grid[y].parent)
    grid[x].parent = root
    grid[y].parent = root
    return root


def DJSUnion(grid, x, y):
    x = DJSFind(grid, x)
    y = DJSFind(grid, y)
    if x == y:
        return
    if grid[x].node_rank < grid[y].node_rank:
        grid[x].parent = y
    elif grid[x].node_rank > grid[y].node_rank:
        grid[y].parent = x
    else:
        grid[y].parent = x
        grid[x].node_rank += 1


def IsValidCell(x, y):
    if x >= 0 and y >= 0 and x < grid_size and y < grid_size:
        return True
    else:
        return False


def GenerateMask(range_, confident_range):
    range_mask = np.zeros((grid_size, grid_size))
    meter_per_pixel = range_ * 2.0 / grid_size
    half_width = grid_size / 2
    half_height = grid_size / 2
    for r in range(grid_size):
        for c in range(grid_size):
            distance = math.sqrt(pow((r - half_height), 2.0) +
                                 pow((c - half_width), 2.0))
            distance *= meter_per_pixel
            if (distance <= confident_range):
                range_mask[r][c] = 1

    return range_mask


grid_size = 512


def cluster(category, confidence, height_pt, instance, class_score):
    # conf
    object_thresh = 0.5
    confidence_thres = 0.1
    grid_size = 512
    grid = np.empty((grid_size, grid_size), dtype=object)
    grid.flat = [Node() for _ in grid.flat]
    scale = 0.5 * grid_size / 60.0

    category_score = category[0]  # output_data[0][0]
    # class_score = output_data[1] #(1, 5, 512, 512)
    confidence_score = confidence  # output_data[2] # (1, 1, 512, 512)
    # heading_pt = output_data[3] # (1, 2, 512, 512)
    height_pt = height_pt  # output_data[3] #(1, 1, 512, 512)
    instance_pt = instance[0]  # output_data[4] #(1, 2, 512, 512)

    # construct graph
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i, j)
            DJSMakeSet(grid, x)
            grid[x].is_object = (category_score[0][x] >= object_thresh)
            center_row = int(round(instance_pt[0][x] * scale + x[0]))
            center_col = int(round(instance_pt[1][x] * scale + x[1]))
            grid[x].center_node = (max(0, min(grid_size - 1, center_row)), max(0, min(grid_size - 1, center_col)))

    # traverse node
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i, j)
            if grid[x].is_object and grid[x].traversed == 0:
                Traverse(grid, x)
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i, j)
            if not grid[x].is_center:
                continue
            for i2 in [i - 1, i, i + 1]:
                for j2 in [j - 1, j, j + 1]:
                    if (i2 == i or j2 == j) and IsValidCell(i2, j2):
                        x2 = (i2, j2)
                        if grid[x2].is_center:
                            DJSUnion(grid, x, x2)

    obs_list = []
    label_map = np.zeros(grid_size * grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i, j)
            if not grid[x].is_object:
                continue
            root = DJSFind(grid, x)
            if grid[root].obs_id < 0:
                grid[root].obs_id = len(obs_list) + 1
                obs_list.append([])
            label_map[rc2grid(i, j)] = grid[root].obs_id
            obs_list[grid[root].obs_id - 1].append((i, j))
    filtered_obs = []
    range_mask = GenerateMask(60, 58)

    for obs in obs_list:
        # if len(obs) < 3:
        #     continue
        sum_confidence = 0.0
        for pixel in obs:
            sum_confidence += confidence_score[0][0][pixel]
        if len(obs) > 0:
            sum_confidence /= len(obs)
        cluster_confidence = sum_confidence
        if (range_mask[pixel]):
            if cluster_confidence >= confidence_thres:
                filtered_obs.append(obs)
                filtered_obs[-1].append([])
                filtered_obs[-1][-1].append(cluster_confidence)
        else:
            sum_category = 0.0
            for pixel in obs:
                sum_category += category_score[0][pixel]
            if len(obs) > 0:
                sum_category /= len(obs)
            if sum_category >= object_thresh:
                filtered_obs.append(obs)
            cluster_confidence = max(sum_confidence, confidence_thres + 0.01)
            filtered_obs[-1].append([])
            filtered_obs[-1][-1].append(cluster_confidence)

    for obs in filtered_obs:
        height_sum = 0
        class_sum = np.zeros(5)
        for pixel in obs[:-1]:
            for CLASS_IDX in range(5):
                class_sum[CLASS_IDX] += class_score[0][CLASS_IDX][pixel]
            height_sum += height_pt[0][0][pixel]
        root = DJSFind(grid, pixel)
        cluster_id = grid[root].obs_id
        cluster_class = class_sum / (len(obs) - 1)
        cluster_height = height_sum / (len(obs) - 1)
        obs[-1].append(cluster_id)
        obs[-1].append(cluster_class)
        obs[-1].append(cluster_height)

    return filtered_obs, label_map