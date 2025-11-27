import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch
import cv2  # 添加OpenCV库

from cv2 import getTickCount, getTickFrequency
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis

def do_test(args, cfg, model):

    # 视频路径
    video_path = args.input_video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    
    # 可选：保存输出视频
    save_video = getattr(args, 'save_video', False)
    if save_video:
        output_video_path = os.path.join(args.output_dir, "tracked_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))*2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps_out, frame_size)
        print(f"将保存追踪结果到: {output_video_path}")

    model.eval()
    
    focal_length = args.focal_length
    principal_point = args.principal_point
    thres = args.threshold

    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
        
    # store locally if needed
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']
    
    while True:
        loop_start = getTickCount()
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转换颜色空间 BGR到RGB
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_shape = im.shape[:2]  # h, w

        h, w = image_shape
        
        if focal_length == 0:
            focal_length_ndc = 4.0
            focal_length = focal_length_ndc * h / 2

        if len(principal_point) == 0:
            px, py = w/2, h/2
        else:
            px, py = principal_point

        K = np.array([
            [focal_length, 0.0, px], 
            [0.0, focal_length, py], 
            [0.0, 0.0, 1.0]
        ])

        aug_input = T.AugInput(im)
        _ = augmentations(aug_input)
        image = aug_input.image

        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(), 
            'height': image_shape[0], 'width': image_shape[1], 'K': K
        }]

        with torch.no_grad():
            dets = model(batched)[0]['instances']
        n_det = len(dets)

        meshes = []
        meshes_text = []
        target_cats = ['door']

        if n_det > 0:
            for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                    dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions, 
                    dets.pred_pose, dets.scores, dets.pred_classes
                )):

                # skip
                if score < thres:
                    continue
                
                cat = cats[cat_idx]
                if cat not in target_cats:
                    continue
                bbox3D = center_cam.tolist() + dimensions.tolist()
                meshes_text.append('{} {:.2f}'.format(cat, score))
                color = [c/255.0 for c in util.get_color(idx)]
                box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                meshes.append(box_mesh)
                print (f"物体: {cat} 的位姿: {bbox3D}")

        
        if len(meshes) > 0:
            im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(im, K, meshes, text=meshes_text, scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)
            im_concat = np.concatenate((im_drawn_rgb, im_topdown), axis=1)
        else:
            im_concat = im.copy()

        # 转换回BGR格式显示
        im_concat = im_concat.astype(np.uint8)
        im_display = cv2.cvtColor(im_concat, cv2.COLOR_RGB2BGR)
        
        fps = int(getTickFrequency() / (getTickCount() - loop_start))
        cv2.putText(im_display, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-time 3D Detection', im_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = args.config_file

    # store locally if needed
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

    with torch.no_grad():
        do_test(args, cfg, model)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="/home/skyuan/omni3d-main/configs/Base_Omni3D_in.yaml", metavar="FILE", help="path to config file")
    # 移除原有的input-folder参数
    parser.add_argument('--input-video', type=str, help='path to input video', required=True)
    parser.add_argument('--save-video', action='store_true', help='是否把带追踪结果的画面保存为视频')
    parser.add_argument("--focal-length", type=float, default=0, help="focal length for camera (in px)")
    parser.add_argument("--principal-point", type=float, default=[], nargs=2, help="principal point for camera (in px)")
    parser.add_argument("--threshold", type=float, default=0.25, help="threshold on score for visualizing")
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