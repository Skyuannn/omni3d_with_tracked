
# L:node ----- node ----- node 
# R:node ----- node ----- node


import os
import sys
import argparse
import logging
import numpy as np
import torch
import cv2
import math
import copy

from cv2 import getTickCount, getTickFrequency
from collections import OrderedDict, defaultdict
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import box3d_overlap
from pytorch3d.structures import Meshes

# Local imports from cubercnn and detectron2
from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis


# Suppress numpy scientific notation
np.set_printoptions(suppress=True)
logger = logging.getLogger("detectron2")

# -----------------------------------------
# Utility Functions
# -----------------------------------------
def find_intra_frame_duplicates(detections, threshold=0.2):
    """Identify potential duplicates within the same frame."""
    n = len(detections)
    if n <= 1:
        return []  # No duplicates if only one detection

    corners = [det['corners3D'] for det in detections]
    scores = [det['scores_full'] for det in detections]
    
    cost_matrix = compute_cost_matrix(corners, corners, scores, scores)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    duplicates = [(i, j) for i, j in zip(row_ind, col_ind) if i != j and cost_matrix[i, j] < threshold]
    return duplicates

def remove_lowest_score_detection(detections):
    """Remove detection with the lowest score."""
    if len(detections) == 0:
        return detections

    lowest_score_index = min(range(len(detections)), key=lambda i: detections[i]['score'])
    detections.pop(lowest_score_index)

    return detections

def compute_chamfer_distance(boxes1, boxes2):
    """Compute Chamfer Distance between two sets of 3D boxes."""
    chamfer_dists = torch.zeros((len(boxes1), len(boxes2)))
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            points1, points2 = box1.unsqueeze(0), box2.unsqueeze(0)
            chamfer_dist, _ = chamfer_distance(points1, points2)
            chamfer_dists[i, j] = chamfer_dist
    return chamfer_dists

def cosine_sim(vect1, vect2):
    v1 = torch.as_tensor(vect1, dtype=torch.float32).flatten()
    v2 = torch.as_tensor(vect2, dtype=torch.float32).flatten()
    sim = torch.dot(v1, v2) / (v1.norm() * v2.norm() + 1e-8)
    return sim.item()

def get_aabb_from_corners(corners): 
    """Get axis-aligned bounding box from 3D corners."""
    min_vals, _ = torch.min(corners, dim=1)
    max_vals, _ = torch.max(corners, dim=1)
    return torch.cat((min_vals, max_vals), dim=1)

def generalized_box_iou_3d(boxes1, boxes2):
    """Calculate Generalized IoU for 3D boxes."""
    def get_min_max(boxes):
        return torch.min(boxes, dim=1)[0], torch.max(boxes, dim=1)[0]
    
    min1, max1 = get_min_max(boxes1)
    min2, max2 = get_min_max(boxes2)

    inter_min = torch.maximum(min1[:, None, :], min2)
    inter_max = torch.minimum(max1[:, None, :], max2)
    inter_dims = (inter_max - inter_min).clamp(min=0)
    intersection_volume = inter_dims[:, :, 0] * inter_dims[:, :, 1] * inter_dims[:, :, 2]

    volume1 = (max1 - min1).prod(dim=1)
    volume2 = (max2 - min2).prod(dim=1)
    union_volume = volume1[:, None] + volume2 - intersection_volume

    enclosure_min = torch.minimum(min1[:, None, :], min2)
    enclosure_max = torch.maximum(max1[:, None, :], max2)
    enclosure_dims = enclosure_max - enclosure_min
    enclosure_volume = (enclosure_dims[:, :, 0] * enclosure_dims[:, :, 1] * enclosure_dims[:, :, 2])

    iou = intersection_volume / union_volume
    giou = iou - (enclosure_volume - union_volume) / enclosure_volume

    return (giou + 1) / 2

def compute_cost_matrix(corners1, corners2, scores1, scores2):
    """Compute cost matrix using IoU3D, Chamfer Distance, and cosine similarity."""

    # 如果没有任何检测，直接返回空矩阵
    if len(corners1) == 0 or len(corners2) == 0:
        return np.zeros((len(corners1), len(corners2)), dtype=np.float32)

    # corners 转 tensor
    corners1 = np.stack(corners1)
    corners2 = np.stack(corners2)
    corners1 = torch.as_tensor(corners1, dtype=torch.float32)
    corners2 = torch.as_tensor(corners2, dtype=torch.float32)

    # 3D IoU
    vol, iou_3d = box3d_overlap(corners1, corners2)
    iou_3d_cost = 1 - iou_3d

    # GIoU 
    giou = generalized_box_iou_3d(corners1, corners2)
    giou_cost = 1 - giou

    # Chamfer Distance
    chamfer_dists = compute_chamfer_distance(corners1, corners2)

    # Cosine Similarity
    if len(scores1) == 0 or len(scores2) == 0:
        cosine_costs = torch.zeros_like(iou_3d_cost)
    else:
        s1 = torch.stack([s.detach().flatten().cpu() if s.is_cuda else s.detach().flatten() for s in scores1])
        s2 = torch.stack([s.detach().flatten().cpu() if s.is_cuda else s.detach().flatten() for s in scores2])
        s1_norm = s1 / (s1.norm(dim=1, keepdim=True) + 1e-8)
        s2_norm = s2 / (s2.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = torch.mm(s1_norm, s2_norm.T)
        cosine_costs = 1 - cosine_sim

    # 加权融合
    alpha, beta, gamma, delta = 0.2, 0.1, 0.2, 0.4
    cost_matrix = (
        alpha * iou_3d_cost +
        beta * chamfer_dists +
        gamma * cosine_costs +
        delta * giou_cost
    )

    return cost_matrix.detach().cpu().numpy()

# Dictionaries to keep track of high-cost matches across consecutive frames
consecutive_high_scores = {}
consecutive_very_high_scores = {}

def match_detections(detections1, detections2):
    """
    Matches detections between two frames using a cost matrix and handles special cases
    of high-cost matches. Tracks how long detections have been considered high-cost
    and ignores them if thresholds are exceeded.
    
    Args:
        detections1 (list): Detections from the first frame.
        detections2 (list): Detections from the second frame.
    
    Returns:
        matches (list): List of matched indices between the two detection sets.
    
    Examples:
        corners1 = [A1, A2]
        corners2 = [B1, B2, B3]
        cost_matrix = [[0.1, 0.5, 0.7],
                       [0.6, 0.2, 0.9]]
        第一行[0.1, 0.5, 0.7]:
            A1与B1的成本是0.1
            A1与B2的成本是0.5
            A1与B3的成本是0.7
        第二行[0.6, 0.2, 0.9]:
            A2与B1的成本是0.6
            A2与B2的成本是0.2   
            A2与B3的成本是0.9
        row_ind, col_ind = linear_sum_assignment(cost_matrix)从cost_matrix中找到成本最小的方案
        输出：
            # A1匹配B1,A2匹配B2
            row_ind = [0, 1]  # 行索引
            col_ind = [0, 1]  # 列索引
        for i, j in zip(row_ind, col_ind):
            (i, j) = ((0, 0), (1, 1))
    """
    global consecutive_high_scores, consecutive_very_high_scores

    # Extract 3D corner coordinates and detection scores from each detection
    corners1 = [det['corners3D'] for det in detections1]
    corners2 = [det['corners3D'] for det in detections2]
    scores1 = [det['scores_full'] for det in detections1]
    scores2 = [det['scores_full'] for det in detections2]

    # Compute the cost matrix based on corners and scores
    cost_matrix = compute_cost_matrix(corners1, corners2, scores1, scores2)
    
    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []

    # Thresholds for cost values
    threshold_low = 0.2
    threshold_high = 0.5
    threshold_very_high = 0.66
    max_high_frames = 10  # Max consecutive frames to tolerate high-cost matches
    max_very_high_frames = 3  # Max consecutive frames to tolerate very high-cost matches

    for i, j in zip(row_ind, col_ind):
        print("Cost matrix value:", cost_matrix[i, j])

        # Initialize consecutive counts for this pair if not already present
        if (i, j) not in consecutive_high_scores:
            consecutive_high_scores[(i, j)] = 0
        if (i, j) not in consecutive_very_high_scores:
            consecutive_very_high_scores[(i, j)] = 0

        # Handle low-cost matches
        if cost_matrix[i, j] < threshold_low:
            matches.append((i, j))
            consecutive_high_scores[(i, j)] = 0  # Reset the high score count
            consecutive_very_high_scores[(i, j)] = 0  # Reset the very high score count

        # Handle medium-cost matches
        elif threshold_low <= cost_matrix[i, j] < threshold_high:
            consecutive_high_scores[(i, j)] += 1
            consecutive_very_high_scores[(i, j)] = 0  # Reset the very high score count
            if consecutive_high_scores[(i, j)] <= max_high_frames:
                matches.append((i, j))

        # Handle very high-cost matches (ignore them)
        elif cost_matrix[i, j] >= threshold_very_high:
            print(f"Ignoring match due to very high cost: {cost_matrix[i, j]}")
            consecutive_high_scores[(i, j)] = max_high_frames + 1  # Force ignoring this match
            consecutive_very_high_scores[(i, j)] = max_very_high_frames + 1  # Force ignoring this match

        # Handle persistent high-cost matches over time
        else:
            consecutive_very_high_scores[(i, j)] += 1
            consecutive_high_scores[(i, j)] = 0  # Reset high score count
            if consecutive_very_high_scores[(i, j)] <= max_very_high_frames:
                matches.append((i, j))
            elif consecutive_very_high_scores[(i, j)] > max_very_high_frames:
                print(f"New object detected due to very high cost for {max_very_high_frames} consecutive frames: {cost_matrix[i, j]}")

        # Clean up scores for unmatched pairs
        consecutive_high_scores = {k: v for k, v in consecutive_high_scores.items() if v <= max_high_frames}
        consecutive_very_high_scores = {k: v for k, v in consecutive_very_high_scores.items() if v <= max_very_high_frames}

    return matches

def update_tracks(tracks, global_tracks, matches, detections1, detections2, max_age, frame_number, real_time_diff):
    
    global GLOBAL_NEXT_TRACK_ID, GLOBAL_APPEAR_STEPS
    matched_indices = set()
    updated_tracks = {}

    # matched：保持 id，不要共享引用（深拷贝）
    for i, j in matches:
        track_id = int(detections1[i]['track_id'])
        det = copy.deepcopy(detections2[j])
        det['track_id'] = track_id
        det['age'] = 0
        updated_tracks[track_id] = det
        matched_indices.add(j)

    # age & lost
    newly_lost = []
    for tracks_id, tracks in list(tracks.items()):
        if tracks_id not in updated_tracks:
            tracks['age'] = tracks.get('age', 0) + 1
            if tracks['age'] < max_age:
                updated_tracks[tracks_id] = copy.deepcopy(tracks)
            else:
                tracks['start_frame'] = frame_number
                tracks['end_frame'] = frame_number
                newly_lost.append((tracks_id, copy.deepcopy(tracks)))

    # 把真正 lost 的深拷入 global 
    for tracks_id, tracks in newly_lost:
        if tracks_id not in global_tracks:
            global_tracks[tracks_id] = copy.deepcopy(tracks)

    if len(global_tracks) > 0:
        global_id = sorted(global_tracks.keys())
        last_global_id = global_id[-1]
        last_track = global_tracks[last_global_id]

        sf_for_real_time = last_track.get('start_frame', frame_number)
        ef_for_real_time = last_track.get('end_frame', frame_number)
        real_time_diff = ef_for_real_time - sf_for_real_time
        ef_for_real_time += 1
        last_track['end_frame'] = ef_for_real_time
        global_tracks[last_global_id] = last_track
        # 仅计算，不加入 GLOBAL_APPEAR_STEPS
    print(f"[RealTimeDiff] 两门帧数diff: {real_time_diff}")

    # 新检测到的物体分配全局唯一 id
    for idx, det in enumerate(detections2):
        if idx not in matched_indices:
            new_id = GLOBAL_NEXT_TRACK_ID
            GLOBAL_NEXT_TRACK_ID += 1
            dcopy = copy.deepcopy(det)
            dcopy['track_id'] = new_id
            dcopy['age'] = 0
            updated_tracks[new_id] = dcopy
            
            if len(global_tracks) == 0:
                # 第一个门，不计算 diff
                print(f"[Track {new_id}] 第一个门出现，不计算 diff")
                continue

            if len(global_tracks) > 0:
                print(f"[Track {new_id}] 第{new_id}个门出现，开始计算 diff")
                if (real_time_diff > 2):
                    GLOBAL_APPEAR_STEPS.append(real_time_diff)

                print(f"[Track {last_global_id}] 生命周期帧差 = {real_time_diff} (start={sf_for_real_time}, end={ef_for_real_time})")

    if len(GLOBAL_APPEAR_STEPS) > 0:
        y_step = GLOBAL_APPEAR_STEPS[-1]
    else:
        y_step = 1

    if len(global_tracks) > 1 and len(newly_lost) > 0:
        last_gid = sorted(global_tracks.keys())[-1]   
        g = copy.deepcopy(global_tracks[last_gid])
        g['bbox3D'][2] = (3.0 + y_step / 20.0)  
        global_tracks[last_gid] = g

    return updated_tracks, global_tracks, real_time_diff
 
def get_unique_color(track_id):
    """
    Generates a unique color for each track based on its ID.
    
    Args:
        track_id (int): Unique ID of the track.
    
    Returns:
        color (list): A list representing an RGB color.
    """
    np.random.seed(track_id)
    return [np.random.randint(0, 255) for _ in range(3)]

def parse_detections(dets, thres, cats, target_cats):
    """
    Parses detection outputs and filters them based on a score threshold. Optionally, it can filter 
    detections based on a list of target categories.

    Args:
        dets (object): Object containing the detection outputs (bounding boxes, scores, etc.).
        thres (float): Score threshold below which detections are discarded.
        cats (list): List of categories corresponding to the predicted class indices.
        target_cats (list or None): List of target categories to include. If None, no category filtering is applied.

    Returns:
        parsed_detections (list): List of parsed and filtered detections.
    """
    parsed_detections = []
    n_det = len(dets)
    if n_det > 0:
        # Iterate over each detection and its corresponding properties
        for idx, (corners3D, center_cam, dimensions, score, cat_idx, pose, scores_full) in enumerate(
                zip(dets.pred_bbox3D, dets.pred_center_cam, dets.pred_dimensions, dets.scores, 
                    dets.pred_classes, dets.pred_pose, dets.scores_full)):
            
            # Skip detections with scores below the threshold
            if score < thres:
                continue
            
            # Get the category of the detection using the predicted class index
            category = cats[cat_idx]
            
            # Optionally filter detections by target categories (currently commented out)
            if target_cats and category not in target_cats:
                continue
            
            # Create a detection dictionary with relevant properties
            detection = {
                'corners3D': corners3D.cpu().numpy(),  # Convert 3D corners tensor to numpy array
                'pose': pose,  # Pose of the detected object
                'bbox3D': center_cam.tolist() + dimensions.tolist(),  # 3D bounding box center and dimensions
                'score': score,  # Detection score
                'category': category,  # Object category
                'track_id': None,  # Track ID initially set to None
                'scores_full': scores_full  # Full score information
            }
            
            # Add the detection to the parsed detections list
            parsed_detections.append(detection)

    return parsed_detections

def build_meshes_for_frame(im_rgb, K, tracks_dict, gobal_tracks_dict, device, augmentations, model, thres, cats, target_cats, max_track_age, frame_number, side="L"):  # side = "L" or "R"
        
        real_time_diff = 0
        
        # 预处理 
        aug_input = T.AugInput(im_rgb)
        _ = augmentations(aug_input)
        image = aug_input.image
        image_shape = im_rgb.shape[:2]
        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).to(device),
            'height': image_shape[0],
            'width': image_shape[1],
            'K': K
        }]

        # 推理 
        with torch.no_grad():
            dets = model(batched)[0]['instances']

        # 解析检测 
        detections = parse_detections(dets, thres, cats, target_cats)
        n_det = len(dets)

        # 跟踪更新（每路都独立进行）
        if frame_number == 0:
            for idx, detection in enumerate(detections):
                new_id = idx + 1
                detection['track_id'] = new_id
                detection['age'] = 0
                detection['side'] = side
                detection['start_frame'] = frame_number
                tracks_dict[new_id] = detection
        else:
            current_detections = list(tracks_dict.values())
            matches = match_detections(current_detections, detections)
            new_tracks, new_global_tracks, real_time_diff = update_tracks(tracks_dict, gobal_tracks_dict, matches,
                                                 current_detections, detections, max_track_age, frame_number, real_time_diff)
            tracks_dict = new_tracks
            gobal_tracks_dict = new_global_tracks

        # 绘制 
        meshes, meshes_text = [], []
        meshes2, meshes2_text = [], []
        gobal_meshes, gobal_meshes_text = [], []

        # Tracks mesh
        for track_id, track in tracks_dict.items():
            if track.get('age', 0) < max_track_age:
                cat = track['category']
                score = track['score']
                meshes_text.append(f"{side}_door_{track_id}, {cat}, Scr: {score:.2f}")
                bbox = track['bbox3D']
                pose = track['pose']
                color = [c / 255.0 for c in get_unique_color(track_id + 1000 if side == "R" else track_id)]  # 左右颜色不同
                box_mesh = util.mesh_cuboid(bbox, pose.tolist(), color=color)
                meshes.append(box_mesh)
        
        # Gobal mesh
        for track_id, track in gobal_tracks_dict.items():
                gobal_cat = track['category']
                gobal_score = track['score']
                gobal_meshes_text.append(f"{side}_door_{track_id}, {gobal_cat}, Scr: {gobal_score:.2f} (Lost)")
                gobal_bbox = track['bbox3D']
                gobal_pose = track['pose']
                gobal_color = [0.5, 0.5, 0.5]  # 灰色表示丢失的目标
                gobal_box_mesh = util.mesh_cuboid(gobal_bbox, gobal_pose.tolist(), color=gobal_color)
                gobal_meshes.append(gobal_box_mesh)

        # Detections mesh（原始检测叠加）
        if n_det > 0:
            for idx, (corners3D2, center_cam2, _, dimensions2, pose2, score2, cat_idx2) in enumerate(zip(
                dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D,
                dets.pred_dimensions, dets.pred_pose, dets.scores, dets.pred_classes
            )):
                if score2 < thres:
                    continue
                cat2 = cats[int(cat_idx2)]
                if cat2 not in target_cats:
                    continue
                bbox3D2 = center_cam2.tolist() + dimensions2.tolist()
                meshes2_text.append(f"{cat2}, Scr: {score2:.2f}")

                matched_track_id = None
                for tid, trk in tracks_dict.items():
                    if np.allclose(trk['corners3D'], corners3D2.cpu().numpy(), atol=1e-2):
                        matched_track_id = tid
                        break
                color = [c / 255.0 for c in get_unique_color(matched_track_id + 1000 if side == "R" else matched_track_id)] \
                        if matched_track_id else [c / 255.0 for c in util.get_color(idx)]
                box_mesh2 = util.mesh_cuboid(bbox3D2, pose2.tolist(), color=color)
                meshes2.append(box_mesh2)

                # print(f"[{side}] 物体 {cat2} 的位姿: {bbox3D2}")

        return meshes, meshes_text, meshes2, meshes2_text, gobal_meshes, gobal_meshes_text, tracks_dict, gobal_tracks_dict, real_time_diff

def translate_meshes(meshes, shift_x, shift_y, rotate_deg):
    meshes_shifted = []
    if len(meshes) == 0:
        return meshes_shifted
    angle = math.radians(rotate_deg)
    R = torch.tensor([
        [ math.cos(-angle), 0, math.sin(-angle)],
        [ 0,               1, 0],
        [-math.sin(-angle), 0, math.cos(-angle)]
    ], dtype=torch.float32)
    for mesh in meshes:
        device = mesh.device
        verts = mesh.verts_list()[0].clone()
        faces = mesh.faces_list()[0].clone()
        textures = mesh.textures
        center = verts.mean(dim=0, keepdim=True)
        verts = verts - center
        verts = verts @ R.to(device).T
        verts = verts + center
        verts[:, 0] += float(shift_x)
        verts[:, 1] += float(shift_y)
        new_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        meshes_shifted.append(new_mesh)
    return meshes_shifted

# Dictionary to store active tracks
global tracks_left, gobal_tracks_left, tracks_right, gobal_tracks_right, GLOBAL_NEXT_TRACK_ID, GLOBAL_APPEAR_STEPS
tracks_left = {}
gobal_tracks_left = {}
tracks_right = {}
gobal_tracks_right = {}
GLOBAL_NEXT_TRACK_ID = 1
GLOBAL_APPEAR_STEPS = []


def do_test(args, cfg, model):

    # 视频路径
    video_left = getattr(args, 'input_video_left', None)
    video_right = getattr(args, 'input_video_right', None)
    if video_left is None and video_right is None:
        raise RuntimeError("请通过 --input-video-left & --input-video-right 提供左右视频")
    cap_left = cv2.VideoCapture(video_left)
    cap_right = cv2.VideoCapture(video_right)
    if not cap_left.isOpened():
        raise RuntimeError(f"无法打开左路视频: {video_left}")
    if not cap_right.isOpened():
        raise RuntimeError(f"无法打开右路视频: {video_right}")

    # 可选：保存输出视频
    save_video = getattr(args, 'save_video', None)
    out_writer = None
    if save_video:
        w = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_out = cap_left.get(cv2.CAP_PROP_FPS) or 30.0
        output_video_path = os.path.join(save_video, "tracked_output_dual_top.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps_out, (w*3, h))
        print(f"将保存追踪结果到: {output_video_path}")

    # 模型准备
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 参数
    focal_length = args.focal_length
    principal_point = args.principal_point
    thres = args.threshold

    # 数据增强
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    # 类别信息
    category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)
    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']
    target_cats = ['door']

    # Tracking 初始化
    global tracks_left, gobal_tracks_left, tracks_right, gobal_tracks_right, frame_number, GLOBAL_NEXT_TRACK_ID
    tracks_left   = {}
    gobal_tracks_left     = {}
    tracks_right  = {}
    gobal_tracks_right    = {}
    max_track_age = 30
    real_time_diff_l = 0
    real_time_diff_r = 0
    frame_number = 0
    GLOBAL_NEXT_TRACK_ID = 1

    while True:
        loop_start = getTickCount()
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        if (not ret_l) or (not ret_r):
            break

        im_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
        im_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
        h, w = im_l.shape[:2]

        # 相机内参
        f = focal_length if focal_length != 0 else 4.0 * h / 2
        px, py = (principal_point if len(principal_point) > 0 else (w/2, h/2))
        K = np.array([[f,0,px],[0,f,py],[0,0,1]])

        # 左右各自独立检测 + 独立跟踪
        meshes_l, text_l, meshes2_l, text2_l, gobal_meshes_l, gobal_meshes_text_l, tracks_left, gobal_tracks_left, real_time_diff_l = build_meshes_for_frame(
            im_l, K, tracks_left, gobal_tracks_left, device, augmentations, model, thres, cats, target_cats, max_track_age, frame_number, side="L")
        meshes_r, text_r, meshes2_r, text2_r, gobal_meshes_r, gobal_meshes_text_r, tracks_right, gobal_tracks_right, real_time_diff_r = build_meshes_for_frame(
            im_r, K, tracks_right, gobal_tracks_right, device, augmentations, model, thres, cats, target_cats, max_track_age, frame_number, side="R")
        
        # world shift（只平移 tracked boxes，dets 不用移）
        meshes_l_shift = translate_meshes(meshes_l, -2, 0, rotate_deg=-90)
        meshes_r_shift = translate_meshes(meshes_r, 2, 0, rotate_deg=90)
        combined_shifted = meshes_l_shift + meshes_r_shift

        gobal_meshes_l_shift = translate_meshes(gobal_meshes_l, -2, 3.0 + (real_time_diff_l) / 20, rotate_deg=-90)
        gobal_meshes_r_shift = translate_meshes(gobal_meshes_r, 2, 3.0 + (real_time_diff_r) / 20, rotate_deg=90)
        gobal_combined_shifted = gobal_meshes_l_shift + gobal_meshes_r_shift + combined_shifted
        print(f"[Message] 两门距离：{(real_time_diff_l) / 30}")

        # 左摄像头：绘制 left tracked boxes
        im_l_front = im_l.copy()
        if len(meshes2_l) > 0:
            im_l_det, _, _ = vis.draw_scene_view(im_l, K, meshes2_l, text=text2_l, scale=h, blend_weight=0.5, blend_weight_overlay=0.85)
            im_l_front = ((im_l_front.astype(np.float32) + im_l_det.astype(np.float32)) / 2).astype(np.uint8)

        # 右摄像头：绘制 right tracked boxes
        im_r_front = im_r.copy()
        if len(meshes2_r) > 0:
            im_r_det, _, _ = vis.draw_scene_view(im_r, K, meshes2_r, text=text2_r, scale=h, blend_weight=0.5, blend_weight_overlay=0.85)
            im_r_front = ((im_r_front.astype(np.float32) + im_r_det.astype(np.float32)) / 2).astype(np.uint8)

        # Top-down
        # print("全局跟踪目标数（Top-Down 显示）:", len(gobal_combined_shifted))
        if len(gobal_combined_shifted) == 0:
            im_topdown = np.ones((h, w, 3), dtype=np.uint8) * 225
        else:
            xs_all = np.concatenate([m.verts_padded()[0].cpu().numpy()[:,0] for m in gobal_combined_shifted])
            ys_all = np.concatenate([m.verts_padded()[0].cpu().numpy()[:,1] for m in gobal_combined_shifted])
            zs_all = np.concatenate([m.verts_padded()[0].cpu().numpy()[:,2] for m in gobal_combined_shifted])
            pad = 10.0
            ground_level = ys_all.min()
            ground_bounds = (ground_level, xs_all.min() - pad, xs_all.max() + pad,zs_all.min() - pad, zs_all.max() + pad)
            im_topdown, _ = vis.draw_scene_view(im_l, K, gobal_combined_shifted, text=None,
                                               scale=h, mode='novel',
                                               ground_bounds=ground_bounds,
                                               blend_weight=0.5, blend_weight_overlay=0.85)
            im_topdown = cv2.resize(im_topdown, (w, h))

        # 拼接显示
        im_l_front_resized = cv2.resize(im_l_front, (w, h))
        im_r_front_resized = cv2.resize(im_r_front, (w, h))
        im_topdown_resized = cv2.resize(im_topdown, (w, h))
        final_concat = np.concatenate((im_l_front_resized, im_topdown_resized, im_r_front_resized), axis=1)
        final_concat = cv2.cvtColor(final_concat.astype(np.uint8), cv2.COLOR_RGB2BGR)
        fps = int(getTickFrequency() / max(1, (getTickCount() - loop_start)))
        cv2.putText(final_concat, f'FPS: {fps}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Real-time 3D Detection (L | Top | R)', final_concat)
        if save_video and out_writer is not None:
            out_writer.write(final_concat)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(f"处理完第 {frame_number} 帧")

        frame_number += 1

    cap_left.release()
    cap_right.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()
    print("摄像头关闭，程序结束")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    get_cfg_defaults(cfg)

    config_file = args.config_file

   
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )

    # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #     "cubercnn_DLA34_FPN_outdoor.pth", resume=True
    # )

    with torch.no_grad():
        do_test(args, cfg, model)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument('--input-video-left', type=str, help='path to left input video', required=False)
    parser.add_argument('--input-video-right', type=str, help='path to right input video', required=False)

    parser.add_argument('--save-video', type=str, help='是否把带追踪结果的画面保存为视频')
    parser.add_argument("--focal-length", type=float, default=0, help="focal length for image inputs (in px)")
    parser.add_argument("--principal-point", type=float, default=[], nargs=2, help="principal point for image inputs (in px)")
    parser.add_argument("--threshold", type=float, default=0.40, help="threshold on score for visualizing")
    parser.add_argument("--display", default=False, action="store_true", help="Whether to show the images in matplotlib",)
    parser.add_argument("--categories", nargs='*', default= [] , help="List of target categories to detect and track")

    parser.add_argument("--eval-only", default=True, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--device',  type=str, help='Device to use: cpu or gpu', default="gpu")

    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )