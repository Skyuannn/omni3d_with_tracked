import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch
import cv2  # 添加OpenCV库
import math

from cv2 import getTickCount, getTickFrequency
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T
from pytorch3d.structures import Meshes

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

output_dir = 'datasets'

def do_test(args, cfg, model):

    # 支持三种传入方式：
    # 1) --input-video-left 和 --input-video-right 同时提供（优先）
    # 2) 只有 --input-video 提供，则左右都用同一路视频（兼容旧参数）
    video_left = getattr(args, 'input_video_left', None)
    video_right = getattr(args, 'input_video_right', None)
    video_single = getattr(args, 'input_video', None)

    if video_left is None and video_right is None and video_single is None:
        raise RuntimeError("请通过 --input-video-left & --input-video-right 提供左右视频，或使用 --input-video （兼容模式）")

    if video_left is None and video_single is not None:
        video_left = video_single
    if video_right is None and video_single is not None:
        video_right = video_single

    cap_left = cv2.VideoCapture(video_left)
    cap_right = cv2.VideoCapture(video_right)
    if not cap_left.isOpened():
        raise RuntimeError(f"无法打开左路视频: {video_left}")
    if not cap_right.isOpened():
        raise RuntimeError(f"无法打开右路视频: {video_right}")

    # 可选：保存输出视频（宽度为单路宽 * 3）
    save_video = getattr(args, 'save_video', False)
    out_writer = None
    if save_video:
        # 以左路视频帧率与分辨率为标准（假定左右一致或可接受）
        w = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_out = cap_left.get(cv2.CAP_PROP_FPS) or 30.0
        output_video_path = os.path.join(output_dir, "tracked_output_dual_top.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (w * 3, h)
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
    target_cats = ['door']  # 保持你原来的筛选

    # helper: 对单帧生成 meshes（不做位移）
    def build_meshes_for_frame(im_rgb, K):
        aug_input = T.AugInput(im_rgb)
        _ = augmentations(aug_input)
        image = aug_input.image
        image_shape = im_rgb.shape[:2]
        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(),
            'height': image_shape[0], 'width': image_shape[1], 'K': K
        }]
        with torch.no_grad():
            dets = model(batched)[0]['instances']
        meshes = []
        meshes_text = []
        n_det = len(dets)
        if n_det > 0:
            for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                    dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions,
                    dets.pred_pose, dets.scores, dets.pred_classes
                )):
                if float(score) < thres:
                    continue
                cat = cats[int(cat_idx)]
                if cat not in target_cats:
                    continue
                bbox3D = center_cam.tolist() + dimensions.tolist()
                meshes_text.append('{} {:.2f}'.format(cat, float(score)))
                color = [c/255.0 for c in util.get_color(idx)]
                box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                meshes.append(box_mesh)
        return meshes, meshes_text

    # helper: 将 mesh 顶点在 x 轴平移 shift_x（米），返回新的 mesh list
    def translate_meshes(meshes, shift_x, rotate_deg=0):
        meshes_shifted = []
        if len(meshes) == 0:
            return meshes_shifted

        # rotation angle
        angle = math.radians(rotate_deg)

        # ---- Y axis rotation matrix ----
        R = torch.tensor([
            [ math.cos(angle), 0,  math.sin(angle)],
            [ 0,               1,  0              ],
            [-math.sin(angle), 0,  math.cos(angle)],
        ], dtype=torch.float32)

        for mesh in meshes:
            device = mesh.device

            verts = mesh.verts_list()[0].clone()     # [V,3]
            faces = mesh.faces_list()[0].clone()     # [F,3]
            textures = mesh.textures                 # keep textures

            # --------------------------------------------------
            # 1. compute mesh center
            # --------------------------------------------------
            center = verts.mean(dim=0, keepdim=True)  # [1,3]

            # --------------------------------------------------
            # 2. rotate around center (pivot rotation)
            # --------------------------------------------------
            verts = verts - center                    # move to local space
            verts = verts @ R.to(device).T            # rotate
            verts = verts + center                    # move back

            # --------------------------------------------------
            # 3. translate along X (your original function)
            # --------------------------------------------------
            verts[:, 0] += float(shift_x)

            # --------------------------------------------------
            # 4. reconstruct new mesh (keep textures)
            # --------------------------------------------------
            new_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
            meshes_shifted.append(new_mesh)

        return meshes_shifted


    # topdown R：正上方向俯视
    R_topdown = util.euler2mat([np.pi/2, 0, 0])

    while True:
        loop_start = getTickCount()
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()
        if (not ret_l) or (not ret_r):
            break

        # 转换颜色空间 BGR->RGB
        im_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
        im_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)

        h, w = im_l.shape[:2]  # 假设左右同分辨率
        # 计算 K（同你原逻辑）
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

        # 左右各自构建 mesh（原位）
        meshes_l, meshes_text_l = build_meshes_for_frame(im_l, K)
        meshes_r, meshes_text_r = build_meshes_for_frame(im_r, K)

        # 生成平移后的 mesh（world x 轴偏移）
        meshes_l_shift = translate_meshes(meshes_l, 2, rotate_deg=90)
        meshes_r_shift = translate_meshes(meshes_r, -2, rotate_deg=-90)
        # 渲染左右 front 视图（不平移）
        if len(meshes_l) > 0:
            # draw front view (returns im_drawn_rgb when mode='front')
            im_l_front = vis.draw_scene_view(im_l, K, meshes_l, text=meshes_text_l, scale=im_l.shape[0], mode='front', blend_weight=0.5, blend_weight_overlay=0.85)
        else:
            im_l_front = im_l.copy()

        if len(meshes_r) > 0:
            im_r_front = vis.draw_scene_view(im_r, K, meshes_r, text=meshes_text_r, scale=im_r.shape[0], mode='front', blend_weight=0.5, blend_weight_overlay=0.85)
        else:
            im_r_front = im_r.copy()

        # 合并所有平移后的 meshes 用于 topdown（保证两侧都在一个场景中）
        combined_shifted = meshes_l_shift + meshes_r_shift

        # 如果没有任何检测，构造灰色 topdown 画布
        if len(combined_shifted) == 0:
            im_topdown = np.ones((h, w, 3), dtype=np.uint8) * 225
        else:
            # 计算 ground_bounds：基于所有顶点的 x,z 范围以及 y 的最大值
            xs_all = []
            zs_all = []
            ys_all = []
            for m in combined_shifted:
                v = m.verts_padded()[0].cpu().numpy()  # (V,3)
                xs_all.append(v[:, 0])
                ys_all.append(v[:, 1])
                zs_all.append(v[:, 2])
            xs_all = np.concatenate(xs_all) if len(xs_all) > 0 else np.array([0.0])
            zs_all = np.concatenate(zs_all) if len(zs_all) > 0 else np.array([0.0])
            ys_all = np.concatenate(ys_all) if len(ys_all) > 0 else np.array([0.0])

            min_x, max_x = float(xs_all.min()), float(xs_all.max())
            min_z, max_z = float(zs_all.min()), float(zs_all.max())
            max_y3d = float(ys_all.max())

            # pad (米) — 根据场景调整（若看不到物体，增大该值）
            pad = 3.0
            x_start = np.floor(min_x - pad)
            x_end   = np.ceil (max_x + pad)
            z_start = np.floor(min_z - pad)
            z_end   = np.ceil (max_z + pad)

            ground_bounds = (max_y3d, x_start, x_end, z_start, z_end)

            # 使用 novel 模式来渲染 topdown（返回 im_novel_view, canvas）
            # 注意 draw_scene_view 在 'novel' 模式返回 (im_novel_view, canvas)
            im_topdown, _ = vis.draw_scene_view(im_l, K, combined_shifted, text=None, scale=h, R=R_topdown, T=None, mode='novel', blend_weight=0.5, blend_weight_overlay=0.85, ground_bounds=ground_bounds)

            # im_topdown shape: (h, h, 3) — 将其 resize 到 (h, w, 3) 以便与左右 front 宽度一致
            im_topdown = cv2.resize(im_topdown, (w, h))

        # 确保左右 front 与 topdown 宽度一致，若必要 resize
        im_l_front_resized = cv2.resize(im_l_front, (w, h))
        im_r_front_resized = cv2.resize(im_r_front, (w, h))
        im_topdown_resized = cv2.resize(im_topdown, (w, h))

        final_concat = np.concatenate((im_l_front_resized, im_topdown_resized, im_r_front_resized), axis=1)

        # 转回 BGR 显示与保存
        final_concat = final_concat.astype(np.uint8)
        im_display = cv2.cvtColor(final_concat, cv2.COLOR_RGB2BGR)

        fps = int(getTickFrequency() / max(1, (getTickCount() - loop_start)))
        cv2.putText(im_display, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-time 3D Detection (L | Top | R)', im_display)

        if save_video and out_writer is not None:
            out_writer.write(im_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    if out_writer is not None:
        out_writer.release()
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
    # 新增双路输入参数；保留旧的 --input-video 以便兼容
    parser.add_argument('--input-video', type=str, help='path to input video (legacy, used for both left & right if others not provided)', required=False)
    parser.add_argument('--input-video-left', type=str, help='path to left input video', required=False)
    parser.add_argument('--input-video-right', type=str, help='path to right input video', required=False)

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
