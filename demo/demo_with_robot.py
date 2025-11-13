#!/usr/bin/env python3


import logging
import os
import argparse
import sys
import numpy as np
import random

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T

logger = logging.getLogger("detectron2")

import tf2_ros
import tf2_geometry_msgs
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import Point, PointStamped

from scipy.spatial import ConvexHull


from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis

import rospy
import cv2
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import torch




from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import box3d_overlap
from sklearn.metrics.pairwise import cosine_similarity


from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
import tf

from sensor_msgs.msg import CameraInfo

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

def find_intra_frame_duplicates(detections, threshold=0.2):
    
    # Identifica potenziali duplicati all'interno dello stesso frame.
    
    n = len(detections)
    if n <= 1:
        return []  # Nessun duplicato se c'è solo una detection

    corners = [det['corners3D'] for det in detections]
    scores = [det['scores_full'] for det in detections]

    # Calcola la matrice dei costi per lo stesso frame
    cost_matrix = compute_cost_matrix(corners, corners, scores, scores)

    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    duplicates = []
    for i, j in zip(row_ind, col_ind):
        if i != j and cost_matrix[i, j] < threshold:
            duplicates.append((i, j))
            print(f"Possible duplicate detected: Detection {i} and {j} with cost {cost_matrix[i, j]:.4f}")
    
    return duplicates


def remove_lowest_score_detection(detections):
    if len(detections) == 0:
        return detections

    # Trova l'indice della detection con lo score più basso
    lowest_score_index = min(range(len(detections)), key=lambda i: detections[i]['score'])

    # Rimuovi la detection con lo score più basso
    detections.pop(lowest_score_index)

    return detections

def compute_chamfer_distance(boxes1, boxes2):
    num_boxes1 = len(boxes1)
    num_boxes2 = len(boxes2)
    chamfer_dists = torch.zeros((num_boxes1, num_boxes2))

    for i in range(num_boxes1):
        for j in range(num_boxes2):
            points1 = boxes1[i].unsqueeze(0).clone().detach()
            points2 = boxes2[j].unsqueeze(0).clone().detach()
            chamfer_dist, _ = chamfer_distance(points1, points2)
            chamfer_dists[i, j] = chamfer_dist

    # sigmoide
    sigmoid_chamfer_dists = torch.sigmoid(chamfer_dists)

    # normalizzo
    normalized_sigmoid_chamfer_dists = (sigmoid_chamfer_dists - 0.5) / (1 - 0.5)

    return normalized_sigmoid_chamfer_dists

def cosine_sim(vect1, vect2):
    if vect1.is_cuda:
        vect1 = vect1.cpu()
    if vect2.is_cuda:
        vect2 = vect2.cpu()

    vect1 = np.array(vect1).reshape(1, -1)  
    vect2 = np.array(vect2).reshape(1, -1)  
    cosine_sim_value = cosine_similarity(vect1, vect2)[0][0]
    
    normalized_cosine_sim = (cosine_sim_value + 1) / 2
    return normalized_cosine_sim





def generalized_box3d_iou(corners1, corners2):
    
    corners1 = corners1.clone().detach()
    corners2 = corners2.clone().detach()


    # corners1 = torch.tensor(corners1, dtype = torch.float32)
    # corners2 = torch.tensor(corners2,  dtype = torch.float32)

    if corners1.shape[0] == 0 or corners2.shape[0] == 0:
        return torch.full((corners1.shape[0], corners2.shape[0]), fill_value=float('inf'))

    union_vol, iou = box3d_overlap(corners1, corners2)


    combined_corners = torch.cat((corners1, corners2), dim = 0)
    enclosing_hull = ConvexHull(combined_corners.cpu().numpy())
    enclosing_vol = enclosing_hull.volume

    giou = iou - (enclosing_vol - union_vol) / enclosing_vol


    return giou.item()


def compute_iou(corners1, corners2):
    
  


    corners1 = torch.tensor(corners1)
    corners2 = torch.tensor(corners2)

    if corners1.shape[0] == 0 or corners2.shape[0] == 0:
        return torch.full((corners1.shape[0], corners2.shape[0]), fill_value=float('inf'))

    union_vol, iou = box3d_overlap(corners1, corners2)

    return iou
def get_aabb_from_corners(corners): 
    min_vals, _ = torch.min(corners, dim=1 )
    max_vals, _ =  torch.max(corners, dim=1 )
    return torch.cat((min_vals, max_vals), dim = 1)




def generalized_box_iou_3d(boxes1, boxes2):
    """
    Generalized IoU for 3D boxes.

    The boxes should be in tensor format of shape [N, 8, 3] where N is the number of boxes
    and each box has 8 corners with (x, y, z) coordinates.
    """
    def get_min_max(boxes):
        # Convert corners to min and max per dimension
        return torch.min(boxes, dim=1)[0], torch.max(boxes, dim=1)[0]

    # Get min and max for both sets of boxes
    min1, max1 = get_min_max(boxes1)
    min2, max2 = get_min_max(boxes2)

    # Compute intersection
    inter_min = torch.maximum(min1[:, None, :], min2)
    inter_max = torch.minimum(max1[:, None, :], max2)
    inter_dims = (inter_max - inter_min).clamp(min=0)
    intersection_volume = inter_dims[:, :, 0] * inter_dims[:, :, 1] * inter_dims[:, :, 2]

    # Compute volumes
    def volume_from_min_max(min_coords, max_coords):
        return ((max_coords - min_coords).prod(dim=1))

    volume1 = volume_from_min_max(min1, max1)
    volume2 = volume_from_min_max(min2, max2)
    
    union_volume = volume1[:, None] + volume2 - intersection_volume

    # Enclosure calculation
    enclosure_min = torch.minimum(min1[:, None, :], min2)
    enclosure_max = torch.maximum(max1[:, None, :], max2)
    enclosure_dims = enclosure_max - enclosure_min
    enclosure_volume = (enclosure_dims[:, :, 0] * enclosure_dims[:, :, 1] * enclosure_dims[:, :, 2])

    # GIoU calculation
    iou = intersection_volume / union_volume
    giou = iou - (enclosure_volume - union_volume) / enclosure_volume

    # Normalize GIoU from [-1, 1] to [0, 1]
    giou_normalized = (giou + 1) / 2

    return giou_normalized

def compute_cost_matrix(corners1, corners2, scores1, scores2):
    corners1 = torch.tensor(corners1)
    corners2 = torch.tensor(corners2)

    


    if corners1.shape[0] == 0 or corners2.shape[0] == 0:
        return torch.full((corners1.shape[0], corners2.shape[0]), fill_value=float('inf'))

    _, iou_3d = box3d_overlap(corners1, corners2)
    iou_3d_cost = 1 - iou_3d

    # giou_3d = generalized_box3d_iou(corners1, corners2)
    # giou_3d_cost = 1 - giou_3d
    # print("GIOU 3D COST : ")
    # print(giou_3d_cost)
    chamfer_dists = compute_chamfer_distance(corners1, corners2)
    giou = generalized_box_iou_3d(corners1, corners2)

    iou_3d_cost = 1 - iou_3d
    giou_cost = 1 - giou 



    cosine_similarities = np.zeros((corners1.shape[0], corners2.shape[0]))
    for i in range(corners1.shape[0]):
        for j in range(corners2.shape[0]):
            cosine_similarities[i, j] = cosine_sim(scores1[i], scores2[j])


    cosine_costs = 1 - cosine_similarities
    # print("Cosine cost: ")


    alpha = 0.3  # Peso per l'IoU 3D
    beta = 0.2 # Peso per la Chamfer Distance
    gamma = 0.1
    d = 0.4 
    # print("IOU3D")
    # print(iou_3d_cost)
    # print("CHAMFER")
    # print(chamfer_dists)
    # cost_matrix = alpha * iou_3d_cost + beta * chamfer_dists + gamma * cosine_costs

    #cost_matrix = alpha * iou_3d_cost  + gamma * cosine_costs
    
    #cost_matrix = alpha * iou_3d_cost + beta * chamfer_dists + gamma * cosine_costs

    cost_matrix = alpha * iou_3d_cost + beta * chamfer_dists + gamma * cosine_costs + d * giou_cost
    return cost_matrix.numpy()


def match_detections(detections1, detections2):
     corners1 = [det['corners3D'] for det in detections1]
     corners2 = [det['corners3D'] for det in detections2]
     scores1 = [det['scores_full'] for det in detections1]
     scores2 = [det['scores_full'] for det in detections2]

     cost_matrix = compute_cost_matrix(corners1, corners2, scores1, scores2)
    #  print("cost_matrix:")
    #  print(cost_matrix)
     row_ind, col_ind = linear_sum_assignment(cost_matrix)
     matches = []
     for i, j in zip(row_ind, col_ind):
         print("cost matrix i-j:", cost_matrix[i, j])
        #  if cost_matrix[i, j] < 0.95 and detections1[i]['category'] == detections2[j]['category']:
         if cost_matrix[i, j] < 0.5:
             matches.append((i, j))
         else:
             print(f"Ignoring match due to high cost: {cost_matrix[i, j]}")
        
     return matches


# consecutive_high_scores = {}
# consecutive_very_high_scores = {}


# def match_detections(detections1, detections2):


#     global consecutive_high_scores, consecutive_very_high_scores

#     corners1 = [det['corners3D'] for det in detections1]
#     corners2 = [det['corners3D'] for det in detections2]
#     scores1 = [det['scores_full'] for det in detections1]
#     scores2 = [det['scores_full'] for det in detections2]



#     cost_matrix = compute_cost_matrix(corners1, corners2, scores1, scores2)
    
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # matches = []

    # threshold_low = 0.2
    # threshold_high = 0.6
    # threshold_very_high = 0.66
    # max_high_frames = 10
    # max_very_high_frames = 3

    # for i, j in zip(row_ind, col_ind):
    #     print("cost matrix:", cost_matrix[i, j])
        
    #     if (i, j) not in consecutive_high_scores:
    #         consecutive_high_scores[(i, j)] = 0

    #     if (i, j) not in consecutive_very_high_scores:
    #         consecutive_very_high_scores[(i, j)] = 0

    #     if cost_matrix[i, j] < threshold_low:
    #         matches.append((i, j))
    #         consecutive_high_scores[(i, j)] = 0  # Reset count if it's a valid match
    #         consecutive_very_high_scores[(i, j)] = 0  # Reset very high score count
    #     elif threshold_low <= cost_matrix[i, j] < threshold_high:
    #         consecutive_high_scores[(i, j)] += 1
    #         consecutive_very_high_scores[(i, j)] = 0  # Reset very high score count
    #         if consecutive_high_scores[(i, j)] <= max_high_frames:
    #             matches.append((i, j))
    #     elif cost_matrix[i, j] >= threshold_very_high:
    #         print(f"Ignoring match due to very high cost: {cost_matrix[i, j]}")
    #         consecutive_high_scores[(i, j)] = max_high_frames + 1  # Force ignore this match
    #         consecutive_very_high_scores[(i, j)] = max_very_high_frames + 1  # Force ignore this match
    #     else:
    #         consecutive_very_high_scores[(i, j)] += 1
    #         consecutive_high_scores[(i, j)] = 0  # Reset high score count
    #         if consecutive_very_high_scores[(i, j)] <= max_very_high_frames:
    #             matches.append((i, j))
    #         elif consecutive_very_high_scores[(i, j)] > max_very_high_frames:
    #             print(f"New object detected due to very high cost for {max_very_high_frames} consecutive frames: {cost_matrix[i, j]}")

    #     # Pulisce i conteggi per le corrispondenze non più valide
    #     consecutive_high_scores = {k: v for k, v in consecutive_high_scores.items() if v <= max_high_frames}
    #     consecutive_very_high_scores = {k: v for k, v in consecutive_very_high_scores.items() if v <= max_very_high_frames}

    # return matches

 
def get_unique_color(track_id):
    np.random.seed(track_id)
    return [np.random.randint(0,255) for _ in range(3)] 




def update_tracks(tracks, visible_tracks, matches, detections1, detections2, max_age=15000):
    matched_indices = set()
    updated_tracks = {}
    updated_visible_tracks = {}
    thresh = 10

    # Aggiorna le tracce abbinate
    for i, j in matches:
        track_id = detections1[i]['track_id']
        detections2[j]['track_id'] = track_id
        updated_tracks[track_id] = detections2[j]
        updated_tracks[track_id]['age'] = 0
        matched_indices.add(j)

        visibility_count = tracks.get(track_id, {}).get('visibility_count', 0) + 1
        updated_tracks[track_id]['visibility_count'] = visibility_count

        if visibility_count > thresh:
            updated_visible_tracks[track_id] = updated_tracks[track_id]



    # Aggiorna le tracce non abbinate
    for track_id, track in tracks.items():
        if track_id not in updated_tracks:
            track['age'] += 1
            if track['age'] < max_age:
                updated_tracks[track_id] = track
                updated_tracks[track_id]['visibility_count'] = track.get('visibility_count', 0)

                if updated_tracks[track_id]['visibility_count'] >= thresh:
                    updated_visible_tracks[track_id] = track

        # Identifica le tracce da rimuovere
    # track_to_remove = []
    # for track_id, track in tracks.items():
    #     if track['visibility_count'] <= thresh and track['age'] > 0:
    #         track_to_remove.append(track_id)
    #         # print("TRACK TO REMOVE;")
    #         # print(track_to_remove)

    # # Rimuovi le tracce identificate
    # for track_id in track_to_remove:
    #     # if len(updated_visible_tracks) != 0:
    #     #     del updated_visible_tracks[track_id]
    #     # del updated_tracks[track_id]
    #     del tracks[track_id]
    #     del updated_tracks[track_id]
    #     if track_id in updated_visible_tracks: 
    #         del updated_visible_tracks[track_id]

    # Aggiungi nuove rilevazioni come nuove tracce
    for idx, det in enumerate(detections2):
        if idx not in matched_indices:
            new_id = max(tracks.keys(), default=0) + 1
            det['track_id'] = new_id
            det['age'] = 0
            det['visibility_count'] = 1
            updated_tracks[new_id] = det

    return updated_tracks, updated_visible_tracks





class Omni3dDetector:

    def __init__(self):
        rospy.loginfo("Inizializzazione del nodo Omni3dDetector")
        
        
        self.bridge = CvBridge()
        config_file = rospy.get_param("/mono_object_detector_3d_node/config_file", "src/mono_object_detector_3d/src/configs/cubercnn_DLA34_FPN.yaml")
        # self.camera_topic_name = rospy.get_param("/mono_object_detector_3d_node/camera_topic", "/camera_face/color/image_raw")
        # self.camera_topic_name = rospy.get_param("/mono_object_detector_3d_node/camera_topic", "/usb_cam/image_raw")
        # self.use_compressed_image = "compressed" in self.camera_topic_name
        # if self.use_compressed_image:
        #     self.camera_sub = rospy.Subscriber(self.camera_topic_name, CompressedImage, self.camera_sub_callback)
        # else:
        #     self.camera_sub = rospy.Subscriber(self.camera_topic_name, Image, self.camera_sub_callback)

         
        #self.detector_pub = rospy.Publisher("mono_object_detector_3d_result", CompressedImage, queue_size=10)
        
        self.detector_pub = rospy.Publisher("mono_object_detector_3d_result", Image, queue_size=10)
        self.marker_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=100)
        self.listener = tf.TransformListener()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.cfg = get_cfg()
        get_cfg_defaults(self.cfg)
        self.cfg.merge_from_file(config_file)
        


        
        if not hasattr(self.cfg, 'CONFIG_FILE'):
            self.cfg.CONFIG_FILE = config_file

        self.cfg.freeze()
        self.model = build_model(self.cfg)
        self.model.to(self.device)
        checkpoint_path = "src/mono_object_detector_3d/src/cubercnn_DLA34_FPN.pth" # modello completo
        # checkpoint_path = "src/mono_object_detector_3d/src/cubercnn_DLA34_FPN_outdoor.pth" #modello outdoor 
        
        DetectionCheckpointer(self.model, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(checkpoint_path, resume=True)

        

        self.tf_buffer = tf2_ros.Buffer()  
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.model.eval()
        
        self.augmentation = T.AugmentationList([T.ResizeShortestEdge(self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST, "choice")])

        self.tracks = {}
        self.track_colors = {}
        self.visible_tracks = {}
        self.frame_number = 0
        self.class_names = self._load_class_names()
        self.created_marker_ids = set()
    


        # self.focal_length = rospy.get_param("/mono_object_detector_3d_node/focal_length", 1012.9616013)

        # self.focal_length = rospy.get_param("/mono_object_detector_3d_node/focal_length", 512)
        self.focal_length = rospy.get_param("/mono_object_detector_3d_node/focal_length", 0)
        
        self.principal_point = rospy.get_param("/mono_object_detector_3d_node/principal_point", [])
        self.thres = rospy.get_param("/mono_object_detector_3d_node/threshold", 0.58)
        self.target_cats = rospy.get_param("/mono_object_detector_3d_node/categories", [])

        #self.camera_info_sub = rospy.Subscriber('/camera/camera_info', CameraInfo, camera_info_callback)
        self.camera_topic_name = rospy.get_param("/mono_object_detector_3d_node/camera_topic", "/usb_cam/image_raw")
        self.use_compressed_image = "compressed" in self.camera_topic_name
        if self.use_compressed_image:
            self.camera_sub = rospy.Subscriber(self.camera_topic_name, CompressedImage, self.camera_sub_callback)
        else:
            self.camera_sub = rospy.Subscriber(self.camera_topic_name, Image, self.camera_sub_callback)




    def _load_class_names(self):
        rospy.loginfo("Caricamento dei nomi delle classi")
        category_path = os.path.join(util.file_parts(self.cfg.CONFIG_FILE)[0], 'category_meta.json')
        if category_path.startswith(util.CubeRCNNHandler.PREFIX):
            category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)
        metadata = util.load_json(category_path)
        return metadata.get("thing_classes", None)




    def get_track_color(self, track_id):
        if track_id not in self.track_colors:
            self.track_colors[track_id] = [random.randint(0, 255) for _ in range(3)]
        return self.track_colors[track_id]

    def parse_detections(self, dets, thres, cats, target_cats):
        parsed_detections = []

        for idx, (corners3D, center_cam, dimensions, score, cat_idx, pose, scores_full) in enumerate(zip(dets.pred_bbox3D, dets.pred_center_cam, dets.pred_dimensions, dets.scores, dets.pred_classes, dets.pred_pose, dets.scores_full)):
            if score < thres:
                continue
            
            category = cats[cat_idx]
            if target_cats and category not in target_cats:
                continue

            detection = {
                'corners3D': corners3D.cpu().numpy(),  # Lista di punti in coordinate locali
                'pose': pose,
                'bbox3D': center_cam.tolist() + dimensions.tolist(),
                'score': score,
                'category': category,
                'track_id': None,
                'dimension': dimensions,
                'center_cam': center_cam,
                'scores_full':scores_full
            }


            try:
                # Recupera la trasformazione
                transformation = self.tf_buffer.lookup_transform("map", "camera_face", rospy.Time(0), rospy.Duration(0.1))

                # Trasforma il centro della camera
                point_stamped = PointStamped()
                point_stamped.header.frame_id = "camera_face"
                # point_stamped.point.x = center_cam[2].item() * 268.01275701 / 800
                point_stamped.point.x = center_cam[2].item()
               
                point_stamped.point.y = -center_cam[0].item()
                point_stamped.point.z = -center_cam[1].item()

                transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transformation)
                detection['center_cam'] = [transformed_point.point.x, transformed_point.point.y, transformed_point.point.z]

                # Trasforma i corner points
                transformed_corners3D = []
                for corner in corners3D:
                    point_stamped = PointStamped()
                    point_stamped.header.frame_id = "camera_face"
                    point_stamped.point.x = corner[2].item()
                   
                    point_stamped.point.y = -corner[0].item()
                    point_stamped.point.z = -corner[1].item()

                    transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transformation)
                    transformed_corners3D.append([transformed_point.point.x, transformed_point.point.y, transformed_point.point.z])

                detection['corners3D'] = transformed_corners3D

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logerr(f"Unable to find the transformation: {e}")
               

            parsed_detections.append(detection)
        
        return parsed_detections

    def process_frame(self, frame, model, device, augmentations, focal_length, principal_point, thres, cats, target_cats, frame_number):
        h, w = frame.shape[:2]
        
        if focal_length == 0:
            focal_length = 4.0 * h / 2  # Default focal length NDC moltiplicato per l'altezza dell'immagine
        if len(principal_point) == 0:
            px, py = w/2, h/2
            # px = 464
            # py = 400
            
        else:
            px, py = principal_point

        K = np.array([
            [focal_length, 0.0, px],
            [0.0, focal_length, py],
            [0.0, 0.0, 1.0]
        ])
        

        # Preprocessing dell'immagine
        aug_input = T.AugInput(frame)
        _ = augmentations(aug_input)
        image = aug_input.image
        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).to(device),
            'height': h, 'width': w, 'K': K
        }]

        # Predizione del modello
        with torch.no_grad():
            outputs = model(batched)[0]['instances']

        


        max_track_age = 150
        min_consecutive_frames = 3

        detections = self.parse_detections(outputs, thres, cats, target_cats)

       
        
        if detections:
            rospy.loginfo(f"Rilevati {len(detections)} oggetti:")
            for detection in detections:
                rospy.loginfo(f"Categoria: {detection['category']}, Score: {detection['score']}")
        else:
            rospy.loginfo("Nessun oggetto rilevato.")

        #detections = remove_lowest_score_detection(detections)
        
        if frame_number == 0:
            # Inizializza i tracciati con le prime detezioni
            for idx, detection in enumerate(detections):
                detection['track_id'] = idx + 1
                detection['age'] = 0 
                detection['visibility_count']= 1
                self.tracks[idx + 1] = detection
            #rospy.loginfo(f"Initialized Tracks: {self.tracks}")
        else:
            # Aggiorna i tracciati esistenti con nuove detezioni
            current_detections = list(self.tracks.values())
            matches = match_detections(current_detections, detections)
            self.tracks, self.visible_tracks = update_tracks(self.tracks, self.visible_tracks, matches, current_detections, detections, max_track_age)
            #self.tracks = update_tracks(self.tracks, matches, current_detections, detections, max_track_age)


            # rospy.loginfo(f"All tracks : {self.tracks}")
            # rospy.loginfo(f"visible Tracks: {self.visible_tracks}")
        print("DETECTIONS")
        print(detections)
        # Stampa i risultati del tracking per il frame corrente
        meshes_text = []
        meshes = []
        meshes2 = []
        meshes2_text = []

        #print(f'Frame {frame_number}:')
        for track_id, track in self.tracks.items():
            #rospy.loginfo("For 1")
            if track['age'] < max_track_age and track['category'] in target_cats: 
            #if track['age'] < max_track_age:
                print(f"Track ID: {track_id}, Category: {track['category']}, Score: {track['score']:.2f}, age: {track['age']}")
                cat = track['category']
                score = track['score']
                meshes_text.append(f"T-ID: {track_id}, Cat: {cat}, Scr: {score:.2f}")

                bbox = track['bbox3D'] 
                pose = track['pose']
                color = [c / 255.0 for c in get_unique_color(track_id)]

                box_mesh = util.mesh_cuboid(bbox, pose.tolist(), color=color)
                meshes.append(box_mesh)
                #rospy.loginfo(f"Added mesh for Track ID: {track_id}")
        
        # Rappresenta tutte le detection


        
        # for idx, (corners3D2, center_cam2, center_2D2, dimensions2, pose2, score2, cat_idx2) in enumerate(zip(
        #             outputs.pred_bbox3D, outputs.pred_center_cam, outputs.pred_center_2D, outputs.pred_dimensions,
        #             outputs.pred_pose, outputs.scores, outputs.pred_classes
        #     )):

            
            
        #     if score2 < thres:
        #         continue

        #     cat2 = cats[cat_idx2]
        #     if cat2 not in target_cats and len(target_cats) > 0:
        #         #rospy.loginfo("cat2 not in target_cats")
        #        continue

        #     bbox3D2 = center_cam2.tolist() + dimensions2.tolist()
        #     meshes2_text.append(f"Cat: {cat2}, Scr: {score2:.2f}")
        #     color = self.get_track_color(track_id for track_id in self.tracks.items())
        #     box_mesh2 = util.mesh_cuboid(bbox3D2, pose2.tolist(), color=color)
        #     meshes2.append(box_mesh2)

        for detection in detections:
            corners3D2 = detection['corners3D']
            center_cam2 = detection['center_cam']
            bbox3D2 = detection['bbox3D']
            dimensions2 = detection['dimension']
            pose2 = detection['pose']
            score2 = detection['score']
            cat2 = detection['category']
            track_id = detection['track_id']  # Assicurati di gestire 'track_id' se necessario

            print("TRACK ID DETECTIONS : ")
            print(track_id)
            
            # Verifica la soglia e il target_cat
            if score2 < thres:
                continue
            
            if target_cats and cat2 not in target_cats:
                continue

            # Aggiungi il testo della mesh
            meshes2_text.append(f"Cat: {cat2}, Scr: {score2:.2f}")

            # Ottieni il colore per il track_id
            color = self.get_track_color(track_id) 
            print("COLOR DETECTION : ")
            print(color)

            # Crea il cuboide della mesh
            box_mesh2 = util.mesh_cuboid(bbox3D2, pose2.tolist(), color=color)
            meshes2.append(box_mesh2)
           


        # Rendering e salvataggio delle immagini
        if len(meshes) > 0 or len(meshes2) > 0:
            if len(meshes2) > 0:
                #im_drawn_rgb, _, _ = vis.draw_scene_view(frame, K, meshes2, device, text=meshes2_text, scale=frame.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)
               im_drawn_rgb, _, _ = vis.draw_scene_view(frame, K, meshes2, device, text=meshes2_text, scale=frame.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)
            #    im_drawn_rgb, _, _ = vis.draw_scene_view(frame, K, meshes2, "cuda:0", text=meshes2_text, scale=frame.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)
               #im_drawn_rgb = cv2.cvtColor(im_drawn_rgb, cv2.)
               im_drawn_rgb = cv2.normalize(im_drawn_rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
               im_drawn_rgb = im_drawn_rgb.astype(np.uint8)
               #rospy.loginfo("Ok ok ")
                
                # Convertire l'immagine da OpenCV a ROS Image e pubblicarla
               ros_image = self.bridge.cv2_to_imgmsg(im_drawn_rgb, "bgr8")
               self.detector_pub.publish(ros_image)

        else:
            
            # Convertire l'immagine da OpenCV a ROS Image e pubblicarla
              ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
              self.detector_pub.publish(ros_image)
        
        
        self.publish_markers(self.visible_tracks)
        

    def publish_markers(self, visible_tracks):
        marker_array = MarkerArray()
        current_time = rospy.Time.now()

        for track_id, track in visible_tracks.items():
            #rospy.loginfo(f"TRACK ID: {track_id}")
            
            
            # if track_id in self.created_marker_ids:
            #     continue #skip se il marker per l'id è stato già creato

            # self.created_marker_ids.add(track_id)
            try:
                common_time = self.tf_buffer.get_latest_common_time("camera_face", "map")
                trans = self.tf_buffer.lookup_transform("map", "camera_face", common_time)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"Could not transform from camera_face to map: {e}")
                continue

            # if track['age'] > 3:  inserisci per tracciare solo cose viste almeno per 3 frame (?)
            # Marker per il cubo
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = current_time
            marker.ns = "cubes"
            marker.id = track_id
            marker.type = Marker.CUBE

            # set della scala (vedere se usare dimension o bbox3d)
            marker.scale.x = track['dimension'][0].item()  
            marker.scale.y = track['dimension'][2].item()  
            marker.scale.z = track['dimension'][1].item()  

            # colore del track id

            color = self.get_track_color(track_id)
            print("COLOR MARKER :")
            print(color)

            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 0.5

            # pose del marker 
            # marker.pose.position.x = track['center_cam'][2].item() * 268.01275701/ 800 
            # marker.pose.position.y = -track['center_cam'][0].item()
            # marker.pose.position.z = -track['center_cam'][1].item()

            # pose della trasformata 
            marker.pose.position.x = track['center_cam'][0]
            marker.pose.position.y = track['center_cam'][1]
            marker.pose.position.z = track['center_cam'][2]

            # converti la rotation matrix in quaternion
            rotation_matrix = track['pose'].cpu().numpy()
  
            rotation_matrix = np.vstack([np.hstack([rotation_matrix, np.array([[0], [0], [0]])]), np.array([0, 0, 0, 1])])
            quaternion = tf.transformations.quaternion_from_matrix(rotation_matrix)
            
            
  
            marker.pose.orientation.x = quaternion[2]
            marker.pose.orientation.y = -quaternion[0]
            marker.pose.orientation.z = -quaternion[1]
            marker.pose.orientation.w = quaternion[3]

            marker_array.markers.append(marker)

            # Marker per il testo
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = current_time
            text_marker.ns = "texts"
            text_marker.id = track_id + 1000  
            text_marker.type = Marker.TEXT_VIEW_FACING

            text_marker.scale.z = 0.6  # altezza del testo

            # text_marker.color.r = color[0] / 255.0
            text_marker.color.r = color[2] / 255.0
            text_marker.color.g = color[1] / 255.0
            # text_marker.color.b = color[2] / 255.0
            text_marker.color.b = color[0] / 255.0
            text_marker.color.a = 1.0

            # text marker position con la trasformata 
            text_marker.pose.position.x = track['center_cam'][0]
            text_marker.pose.position.y = track['center_cam'][1]
            text_marker.pose.position.z = track['center_cam'][2] +1.5


            text_marker.pose.orientation.x = 0.0
            text_marker.pose.orientation.y = 0.0
            text_marker.pose.orientation.z = 0.0
            text_marker.pose.orientation.w = 1.0

            # testo per le categorie
            text_marker.text = f"ID: {track_id}, Category: {track['category']}, Score: {track['score']:.2f}"


            marker_array.markers.append(text_marker)

        

        self.marker_pub.publish(marker_array)





    def camera_sub_callback(self, ros_image):
        rospy.loginfo("Ricevuta immagine dalla telecamera")
        try:
            if self.use_compressed_image:
                np_arr = np.fromstring(ros_image.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        

        self.process_frame(cv_image, self.model, self.device, self.augmentation, self.focal_length, self.principal_point, self.thres, self.class_names , self.target_cats, self.frame_number)
        
        
        rospy.loginfo("Callback della telecamera completato")     
 
        
        self.frame_number += 1

  

if __name__ == "__main__":
    rospy.init_node("mono_object_detector_3d_node")
    rospy.loginfo("Nodo mono_object_detector_3d_node avviato")
    server = Omni3dDetector()
    rospy.spin()
