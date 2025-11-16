import os
import sys
import argparse
import logging
import numpy as np
import torch
import cv2

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

    # === corners 转 tensor ===
    corners1 = np.stack(corners1)
    corners2 = np.stack(corners2)
    corners1 = torch.as_tensor(corners1, dtype=torch.float32)
    corners2 = torch.as_tensor(corners2, dtype=torch.float32)

    # === 3D IoU ===
    vol, iou_3d = box3d_overlap(corners1, corners2)
    iou_3d_cost = 1 - iou_3d

    # === GIoU ===
    giou = generalized_box_iou_3d(corners1, corners2)
    giou_cost = 1 - giou

    # === Chamfer Distance ===
    chamfer_dists = compute_chamfer_distance(corners1, corners2)

    # === Cosine Similarity ===
    if len(scores1) == 0 or len(scores2) == 0:
        cosine_costs = torch.zeros_like(iou_3d_cost)
    else:
        s1 = torch.stack([s.detach().flatten().cpu() if s.is_cuda else s.detach().flatten() for s in scores1])
        s2 = torch.stack([s.detach().flatten().cpu() if s.is_cuda else s.detach().flatten() for s in scores2])
        s1_norm = s1 / (s1.norm(dim=1, keepdim=True) + 1e-8)
        s2_norm = s2 / (s2.norm(dim=1, keepdim=True) + 1e-8)
        cosine_sim = torch.mm(s1_norm, s2_norm.T)
        cosine_costs = 1 - cosine_sim

    # === 加权融合 ===
    alpha, beta, gamma, delta = 0.3, 0.1, 0.2, 0.4
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

def update_tracks(tracks, lost_tracks, matches, detections1, detections2, max_age=150, max_lost_age=3000000000):
    """
    Updates track dictionary with matched detections, ages existing tracks,
    and adds new objects as tracks when no match is found.

    Args:
        tracks (dict): Current set of active tracks.
        matches (list): List of matched indices between the two detection sets.
        detections1 (list): Detections from the first frame.
        detections2 (list): Detections from the second frame.
        max_age (int): Maximum age before removing a track.
    
    Returns:
        updated_tracks (dict): Updated dictionary of tracks.
    """
    tracks = {k: v.copy() for k, v in tracks.items()}
    updated_lost = {k: v.copy() for k, v in lost_tracks.items()}
    updated_tracks = {}
    matched_indices = set()

    # 1. 处理已匹配的 tracks —— 直接更新
    for i, j in matches:
        track_id = detections1[i]['track_id']

        new_det = detections2[j].copy()
        new_det['track_id'] = track_id
        new_det['age'] = 0  # reset age after match

        updated_tracks[track_id] = new_det
        matched_indices.add(j)

        if track_id in updated_lost:
            del updated_lost[track_id]

    # 2. 更新未匹配的 active tracks —— 增加 age
    for track_id, track in tracks.items():
        if track_id in updated_tracks:   
            continue  # 已匹配，跳过

        t = track.copy()
        t['age'] = t.get('age', 0) + 1
        if t['age'] < max_age:
            updated_tracks[track_id] = t  # 仍保留在 active tracks
        else:
            updated_lost[track_id] = t  # 移动到 lost tracks

    # 3. 更新 lost_tracks —— age 增加并过滤过期 ID
    used_ids = set(updated_tracks.keys()) | set(updated_lost.keys()) | set(tracks.keys()) | set(lost_tracks.keys())
    next_id = max(used_ids) + 1 if used_ids else 1

    for idx, det in enumerate(detections2):
        if idx not in matched_indices:
            new_det = det.copy()
            new_det['track_id'] = next_id
            new_det['age'] = 0
            updated_tracks[next_id] = new_det
            next_id += 1

    return updated_tracks, updated_lost



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



# Dictionary to store active tracks
tracks = {}
lost_tracks = {}

def do_test(args, cfg, model):

    """
    实时摄像头输入 + CubeRCNN 3D检测 + 物体追踪 + 绘制 + 保存
    """

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    # 模型准备
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 参数
    focal_length = args.focal_length
    principal_point = args.principal_point
    thres = args.threshold
    # target_cats = args.target_cats if hasattr(args, 'target_cats') else []
    target_cats = ['cup', 'door']  
    output_dir = args.output_dir if hasattr(args, 'output_dir') else "./output"
    os.makedirs(output_dir, exist_ok=True)
    display = getattr(args, 'display', True)

    # 数据增强配置
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    # 类别信息
    category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)
    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']

    # Tracking 初始化
    global tracks, lost_tracks
    tracks = {}
    lost_tracks = {}
    max_track_age = 50
    max_lost_age = 3000000000
    frame_number = 0

    print("启动实时检测与追踪 (按 'q' 退出)")

    while True:
        loop_start = getTickCount()
        ret, frame = cap.read()
        if not ret:
            break

        # === 图像与相机内参 ===
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]

        if focal_length == 0:
            focal_length = 4.0 * h / 2
        if len(principal_point) == 0:
            px, py = w / 2, h / 2
        else:
            px, py = principal_point

        K = np.array([
            [focal_length, 0.0, px],
            [0.0, focal_length, py],
            [0.0, 0.0, 1.0]
        ])

        # === 预处理 ===
        aug_input = T.AugInput(im)
        _ = augmentations(aug_input)
        image = aug_input.image
        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).to(device),
            'height': h, 'width': w, 'K': K
        }]

        # === 推理 ===
        with torch.no_grad():
            outputs = model(batched)[0]['instances']

        # === 解析检测 ===
        detections = parse_detections(outputs, thres, cats, target_cats)
        detections = remove_lowest_score_detection(detections)

        if frame_number == 0:
            # 初始化tracks
            for idx, detection in enumerate(detections):
                detection['track_id'] = idx + 1
                detection['age'] = 0    #目标存在帧数
                tracks[idx + 1] = detection
        else:
            # 更新tracks
            current_detections = list(tracks.values())
            current_lost_list =  list(lost_tracks.values())
            candidates = current_detections + current_lost_list
            matches = match_detections(candidates, detections)
            tracks, lost_tracks = update_tracks(tracks, lost_tracks, matches, candidates, detections, max_track_age, max_lost_age)

        # === 绘制与输出 ===
        meshes, meshes_text, meshes2, meshes2_text = [], [], [], []

        for track_id, track in tracks.items():
            if track['age'] < max_track_age:
                cat = track['category']
                score = track['score']
                meshes_text.append(f"T-ID: {track_id}, Category: {cat}, Scr: {score:.2f}")
                bbox = track['bbox3D']
                pose = track['pose']
                color = [c / 255.0 for c in get_unique_color(track_id)]
                box_mesh = util.mesh_cuboid(bbox, pose.tolist(), color=color)
                meshes.append(box_mesh)
        for track_id, track in lost_tracks.items():
            cat = track['category']
            score = track['score']
            bbox = track['bbox3D']
            pose = track['pose']
            color = [0.5, 0.5, 0.5]  # 灰色表示丢失的目标
            meshes_text.append(f"L-ID: {track_id}, Category: {cat}, Scr: {score:.2f}")
            box_mesh = util.mesh_cuboid(bbox, pose.tolist(), color=color)
            meshes.append(box_mesh)

        # 原始检测绘制
        for idx, (corners3D2, center_cam2, center_2D2, dimensions2, pose2, score2, cat_idx2) in enumerate(zip(
                outputs.pred_bbox3D, outputs.pred_center_cam, outputs.pred_center_2D, outputs.pred_dimensions,
                outputs.pred_pose, outputs.scores, outputs.pred_classes
        )):
            if score2 < thres:
                continue
            cat2 = cats[cat_idx2]
            if cat2 not in target_cats and len(target_cats) > 0:
                continue
            bbox3D2 = center_cam2.tolist() + dimensions2.tolist()
            meshes2_text.append(f"Category: {cat2}, Scr: {score2:.2f}")

            matched_track_id = None
            for track_id, track in tracks.items():
                if np.allclose(track['corners3D'], corners3D2.cpu().numpy(), atol=1e-2):
                    matched_track_id = track_id
                    break

            if matched_track_id is not None:
                color = [c / 255.0 for c in get_unique_color(matched_track_id)]
            else:
                color = [c / 255.0 for c in util.get_color(idx)]

            box_mesh2 = util.mesh_cuboid(bbox3D2, pose2.tolist(), color=color)
            meshes2.append(box_mesh2)

        # === 可视化 ===
        im_drawn_rgb = im.copy()
        im_topdown = np.zeros_like(im_drawn_rgb)
        h, w = im.shape[:2]  # 以原图大小为基准
        # 绘制 tracked boxes
        if len(meshes) > 0:
            im_drawn_rgb_tracked, im_topdown_tracked, _ = vis.draw_scene_view(
                im, K, meshes, text=meshes_text,
                scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85
            )
            # im_drawn_rgb = im_drawn_rgb_tracked
            im_topdown = im_topdown_tracked

        # 绘制原始检测 boxes
        if len(meshes2) > 0:
            im_drawn_rgb_det, im_topdown_det, _ = vis.draw_scene_view(
                im, K, meshes2, text=meshes2_text,
                scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85
            )
            # 将原始检测叠加在 tracked 图上
            im_drawn_rgb = ((im_drawn_rgb.astype(np.float32) + im_drawn_rgb_det.astype(np.float32)) / 2).astype(np.uint8)
            # 保证尺寸一致
            im_topdown_det_resized = cv2.resize(im_topdown_det, (im_topdown.shape[1], im_topdown.shape[0]))
            # 然后再融合
            im_topdown = ((im_topdown.astype(np.float32) + im_topdown_det_resized.astype(np.float32)) / 2).astype(np.uint8)


        # 拼接显示
        im_concat = np.concatenate((im_drawn_rgb, im_topdown), axis=1)
        im_display = cv2.cvtColor(im_concat.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # === FPS 显示 ===
        fps = int(getTickFrequency() / (getTickCount() - loop_start))
        cv2.putText(im_display, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real-time 3D Detection + Tracking", im_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    cap.release()
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

    # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #     "/home/skyuan/omni3d_with_tracking/cubercnn_DLA34_FPN.pth", resume=True
    # )


    with torch.no_grad():
        do_test(args, cfg, model)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    # parser.add_argument('--input-video', type=str, help='path to input video', required=True)
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



