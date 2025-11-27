我有一个新想法，保证原有代码所有功能的基础上请帮我实现，想法为：
门拓扑排列结构
1：两个视频输入，左、右帧画面同时启动：
    1.1时序维护：
        按照每一帧来维护时序frame_time
    1.2两侧拓扑结构维护：
        左侧摄像头维护左侧拓扑结构，右侧摄像头维护右侧结构，两边两个摄像头各自维护两条排列
    1.3单侧拓扑结构检测id维护：
        保存所有已识别到的门id，永久记录。左侧记录为L_door_1、L_door_2等，右侧记录为R_door_1、R_door_2等。现在只考虑前进建图，也就是门id会随着识别到新的门而累加
2：全局地图添加与绘制可视化时：
    2.1同一侧单帧单门：
        2.1.1全局地图添加：L_door_m->next_door_dis=frame_time_miss*alp, L_door_m->next_door=L_door_(m+1)
            其中：frame_time_miss=(frame_time_find_L_door_(m+1)-frame_time_miss_L_door_m)
        2.1.2绘制可视化：如果一个门从摄像头中经历从识别到消失，我们认为门已经识别完毕，将存储在全局地图中，如果前进还能发现门，扩大绘制的全局地图，在同一侧最后一个加入全局地图的门的前方frame_time_miss*alp米绘制一个新门。
            其中：alp为可调节阈值
    2.2同一侧单帧多门：
        2.2.1全局地图添加：L_door_m->next_door_dis=0.5, L_door_m->next_door=L_door_(m+1)
        2.2.2绘制可视化：同一侧全局地图中上一扇门的右门框，正好紧挨着下一扇门的左门框
    2.3两侧单帧均检测到门：
        2.2.3全局地图添加：L_door_m->opposite_door=R_door_n
        2.2.3绘制可视化：认为两门属于过道的对侧门
    2.4绘制同一侧门时，参考center_cam中物体距离相机的距离，如果与上一个门距离正负差值在0.5m内，将该门与已建立拓扑排列拟合为一条直线
3.最终构建基于双边拓扑结构的全局地图。而拓扑结构用什么数据结构来存储还有待考虑（链表？图论？）
左侧视频输入：L_door_1 L_door_2 L_door_3 ...L_door_m
               |-----------|---------|----------------|
                                过道
               |-----------|---------------|----------|
右侧视频输入：R_door_1 R_door_2 R_door_3 ...R_door_n。
注意!!!：目前我有match_detections和update_tracks来匹配不同帧的物体进行物体跟踪,用draw_scene_view函数绘图
整体流程代码为：def do_test(args, cfg, model):
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
# 模型准备
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 参数
focal_length = args.focal_length
principal_point = args.principal_point
thres = args.threshold
# target_cats = args.target_cats if hasattr(args, 'target_cats') else []
target_cats = ['door']
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
n_det = len(outputs)
# === 解析检测 ===
detections = parse_detections(outputs, thres, cats, target_cats)
# detections = remove_lowest_score_detection(detections)
if frame_number == 0:
# 初始化tracks
for idx, detection in enumerate(detections):
detection['track_id'] = idx + 1
detection['age'] = 0 #目标存在帧数
tracks[idx + 1] = detection
else:
# 更新tracks
current_detections = list(tracks.values())
matches = match_detections(current_detections, detections)
tracks, updated_lost_tracks = update_tracks(tracks, lost_tracks, matches, current_detections, detections, max_track_age)
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
if n_det > 0:
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
print (f"物体: {cat2} 的位姿: {center_cam2.tolist() + dimensions2.tolist()}")
# === 可视化 ===
im_drawn_rgb = im.copy()
im_topdown = np.zeros_like(im_drawn_rgb)
h, w = im.shape[:2] # 以原图大小为基准
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
print("摄像头关闭，程序结束") def update_tracks(tracks, lost_tracks, matches, detections1, detections2, max_age=150, max_lost_age=3000000000):
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
matched_indices = set()
updated_tracks = {}
updated_lost_tracks = {}
# Update matched tracks
for i, j in matches:
track_id = detections1[i]['track_id']
detections2[j]['track_id'] = track_id
updated_tracks[track_id] = detections2[j]
updated_tracks[track_id]['age'] = 0 # Reset track age after a match
matched_indices.add(j)
# Age the tracks that were not updated
for track_id, track in tracks.items():
if track_id not in updated_tracks:
track['age'] += 1
if track['age'] < max_age:
updated_tracks[track_id] = track
else:
updated_lost_tracks[track_id] = track # Move to lost tracks
# Add new objects as tracks
existing_ids = set(list(updated_tracks.keys()) + list(updated_lost_tracks.keys()))
for idx, det in enumerate(detections2):
if idx not in matched_indices:
new_id = max(tracks.keys(), default=0) + 1
det['track_id'] = new_id
det['age'] = 0
updated_tracks[new_id] = det
existing_ids.add(new_id)
return updated_tracks, updated_lost_tracks。主流程代码结束最后用绘图代码draw_scene_view绘图：def draw_scene_view(im, K, meshes, device, text=None, scale=1000, R=None, T=None, zoom_factor=1.0, mode='front_and_novel', blend_weight=0.80, blend_weight_overlay=1.0, ground_bounds=None, canvas=None, zplane=0.05):
    """
    Draws a scene from multiple different modes.
    Args:
        im (array): the image to draw onto
        K (array): the 3x3 matrix for projection to camera to screen
        meshes ([Mesh]): a list of meshes to draw into the scene
        text ([str]): optional strings to draw per mesh
        scale (int): the size of the square novel view canvas (pixels)
        R (array): a single 3x3 matrix defining the novel view
        T (array): a 3x vector defining the position of the novel view
        zoom_factor (float): an optional amount to zoom out (>1) or in (<1)
        mode (str): supports ['2D_only', 'front', 'novel', 'front_and_novel'] where
            front implies the front-facing camera view and novel is based on R,T
        blend_weight (float): blend factor for box edges over the RGB
        blend_weight_overlay (float): blends the RGB image with the rendered meshes
        ground_bounds (tuple): max_y3d, x3d_start, x3d_end, z3d_start, z3d_end for the Ground floor or
            None to let the renderer to estimate the ground bounds in the novel view itself.
        canvas (array): if the canvas doesn't change it can be faster to re-use it. Optional.
        zplane (float): a plane of depth to solve intersection when
            vertex points project behind the camera plane.
    """
if R is None:
R = util.euler2mat([np.pi/3, 0, 0])
if mode == '2D_only':
im_drawn_rgb = deepcopy(im)
# go in order of reverse depth
for mesh_idx in reversed(np.argsort([mesh.verts_padded().cpu().mean(1)[0, 1] for mesh in meshes])):
mesh = meshes[mesh_idx]
verts3D = mesh.verts_padded()[0].numpy()
verts2D = (K @ verts3D.T) / verts3D[:, -1]
color = [min(255, c*255*1.25) for c in mesh.textures.verts_features_padded()[0,0].tolist()]
x1 = verts2D[0, :].min()
y1 = verts2D[1, :].min()
x2 = verts2D[0, :].max()
y2 = verts2D[1, :].max()
draw_2d_box(im_drawn_rgb, [x1, y1, x2-x1, y2-y1], color=color, thickness=max(2, int(np.round(3*im_drawn_rgb.shape[0]/1250))))
if text is not None:
draw_text(im_drawn_rgb, '{}'.format(text[mesh_idx]), [x1, y1], scale=0.50*im_drawn_rgb.shape[0]/500, bg_color=color)
return im_drawn_rgb
else:
# meshes_scene = join_meshes_as_scene(meshes).cuda()
meshes_scene = join_meshes_as_scene(meshes).to(device)
device = meshes_scene.device
meshes_scene.textures = meshes_scene.textures.to(device)
cameras = util.get_camera(K, im.shape[1], im.shape[0]).to(device)
renderer = util.get_basic_renderer(cameras, im.shape[1], im.shape[0], use_color=True).to(device)
if mode in ['front_and_novel', 'front']:
'''
            Render full scene from image view
            '''
im_drawn_rgb = deepcopy(im)
# save memory if not blending the render
if blend_weight > 0:
rendered_img, _ = renderer(meshes_scene)
sil_mask = rendered_img[0, :, :, 3].cpu().numpy() > 0.1
rendered_img = (rendered_img[0, :, :, :3].cpu().numpy() * 255).astype(np.uint8)
im_drawn_rgb[sil_mask] = rendered_img[sil_mask] * blend_weight + im_drawn_rgb[sil_mask] * (1 - blend_weight)
'''
            Draw edges for image view
            '''
# go in order of reverse depth
for mesh_idx in reversed(np.argsort([mesh.verts_padded().cpu().mean(1)[0, 1] for mesh in meshes])):
mesh = meshes[mesh_idx]
verts3D = mesh.verts_padded()[0].cpu().numpy()
verts2D = (K @ verts3D.T) / verts3D[:, -1]
color = [min(255, c*255*1.25) for c in mesh.textures.verts_features_padded()[0,0].tolist()]
draw_3d_box_from_verts(
im_drawn_rgb, K, verts3D, color=color,
thickness=max(2, int(np.round(3*im_drawn_rgb.shape[0]/1250))),
draw_back=False, draw_top=False, zplane=zplane
                )
x1 = verts2D[0, :].min() #min(verts2D[0, (verts2D[0, :] > 0) & (verts2D[0, :] < im_drawn_rgb.shape[1])])
y1 = verts2D[1, :].min() #min(verts2D[1, (verts2D[1, :] > 0) & (verts2D[1, :] < im_drawn_rgb.shape[0])])
if text is not None:
draw_text(im_drawn_rgb, '{}'.format(text[mesh_idx]), [x1, y1], scale=0.50*im_drawn_rgb.shape[0]/500, bg_color=color)
if blend_weight_overlay < 1.0 and blend_weight_overlay > 0.0:
im_drawn_rgb = im_drawn_rgb * blend_weight_overlay + deepcopy(im) * (1 - blend_weight_overlay)
if mode == 'front':
return im_drawn_rgb
elif mode in ['front_and_novel', 'novel']:
'''
            Render from a new view
            '''
has_canvas_already = canvas is not None
if not has_canvas_already:
canvas = np.ones((scale, scale, 3))
view_R = torch.from_numpy(R).float().to(device)
if T is None:
center = (meshes_scene.verts_padded().min(1).values + meshes_scene.verts_padded().max(1).values).unsqueeze(0)/2
else:
center = torch.from_numpy(T).float().to(device).view(1, 1, 3)
verts_rotated = meshes_scene.verts_padded().clone()
verts_rotated -= center
verts_rotated = (view_R @ verts_rotated[0].T).T.unsqueeze(0)
K_novelview = deepcopy(K)
K_novelview[0, -1] *= scale / im.shape[1]
K_novelview[1, -1] *= scale / im.shape[0]
cameras = util.get_camera(K_novelview, scale, scale).to(device)
renderer = util.get_basic_renderer(cameras, scale, scale, use_color=True).to(device)
margin = 0.01
if T is None:
max_trials = 10000
zoom_factor = 100.0
zoom_factor_in = zoom_factor
while max_trials:
zoom_factor_in = zoom_factor_in*0.95
verts = verts_rotated.clone()
verts[:, :, -1] += center[:, :, -1]*zoom_factor_in
verts_np = verts.cpu().numpy()
proj = ((K_novelview @ verts_np[0].T) / verts_np[:, :, -1])
# some vertices are extremely close or negative...
# this implies we have zoomed in too much
if (verts[0, :, -1] < 0.25).any():
break
# left or above image
elif (proj[:2, :] < scale*margin).any():
break
# right or below borders
elif (proj[:2, :] > scale*(1 - margin)).any():
break
# everything is in view.
zoom_factor = zoom_factor_in
max_trials -= 1
zoom_out_bias = center[:, :, -1].item()
else:
zoom_out_bias = 1.0
verts_rotated[:, :, -1] += zoom_out_bias*zoom_factor
meshes_novel_view = meshes_scene.clone().update_padded(verts_rotated)
rendered_img, _ = renderer(meshes_novel_view)
im_novel_view = (rendered_img[0, :, :, :3].cpu().numpy() * 255).astype(np.uint8)
sil_mask = rendered_img[0, :, :, 3].cpu().numpy() > 0.1
center_np = center.cpu().numpy()
view_R_np = view_R.cpu().numpy()
if not has_canvas_already:
if ground_bounds is None:
min_x3d, _, min_z3d = meshes_scene.verts_padded().min(1).values[0, :].tolist()
max_x3d, max_y3d, max_z3d = meshes_scene.verts_padded().max(1).values[0, :].tolist()
# go for grid projection, but with extremely bad guess at bounds
x3d_start = np.round(min_x3d - (max_x3d - min_x3d)*50)
x3d_end = np.round(max_x3d + (max_x3d - min_x3d)*50)
z3d_start = np.round(min_z3d - (max_z3d - min_z3d)*50)
z3d_end = np.round(max_z3d + (max_z3d - min_z3d)*50)
grid_xs = np.arange(x3d_start, x3d_end)
grid_zs = np.arange(z3d_start, z3d_end)
xs_mesh, zs_mesh = np.meshgrid(grid_xs, grid_zs)
ys_mesh = np.ones_like(xs_mesh)*max_y3d
point_mesh = np.concatenate((xs_mesh[:, :, np.newaxis], ys_mesh[:, :, np.newaxis], zs_mesh[:, :, np.newaxis]), axis=2)
point_mesh_orig = deepcopy(point_mesh)
mesh_shape = point_mesh.shape
point_mesh = view_R_np @ (point_mesh - center_np).transpose(2, 0, 1).reshape(3, -1)
point_mesh[-1] += zoom_out_bias*zoom_factor
point_mesh[-1, :] = point_mesh[-1, :].clip(0.25)
point_mesh_2D = (K_novelview @ point_mesh) / point_mesh[-1]
point_mesh_2D[-1] = point_mesh[-1]
point_mesh = point_mesh.reshape(3, mesh_shape[0], mesh_shape[1]).transpose(1, 2, 0)
point_mesh_2D = point_mesh_2D.reshape(3, mesh_shape[0], mesh_shape[1]).transpose(1, 2, 0)
maskx = (point_mesh_2D[:, :, 0].T >= -50) & (point_mesh_2D[:, :, 0].T < scale+50) & (point_mesh_2D[:, :, 2].T > 0)
maskz = (point_mesh_2D[:, :, 1].T >= -50) & (point_mesh_2D[:, :, 1].T < scale+50) & (point_mesh_2D[:, :, 2].T > 0)
# invalid scene?
if (not maskz.any()) or (not maskx.any()):
return im, im, canvas
# go for grid projection again!! but with sensible bounds
x3d_start = np.round(point_mesh[:, :, 0].T[maskx].min() - 10)
x3d_end = np.round(point_mesh[:, :, 0].T[maskx].max() + 10)
z3d_start = np.round(point_mesh_orig[:, :, 2].T[maskz].min() - 10)
z3d_end = np.round(point_mesh_orig[:, :, 2].T[maskz].max() + 10)
else:
max_y3d, x3d_start, x3d_end, z3d_start, z3d_end = ground_bounds
grid_xs = np.arange(x3d_start, x3d_end)
grid_zs = np.arange(z3d_start, z3d_end)
xs_mesh, zs_mesh = np.meshgrid(grid_xs, grid_zs)
ys_mesh = np.ones_like(xs_mesh)*max_y3d
point_mesh = np.concatenate((xs_mesh[:, :, np.newaxis], ys_mesh[:, :, np.newaxis], zs_mesh[:, :, np.newaxis]), axis=2)
mesh_shape = point_mesh.shape
point_mesh = view_R_np @ (point_mesh - center_np).transpose(2, 0, 1).reshape(3, -1)
point_mesh[-1] += zoom_out_bias*zoom_factor
point_mesh[-1, :] = point_mesh[-1, :].clip(0.25)
point_mesh_2D = (K_novelview @ point_mesh) / point_mesh[-1]
point_mesh_2D[-1] = point_mesh[-1]
point_mesh = point_mesh.reshape(3, mesh_shape[0], mesh_shape[1]).transpose(1, 2, 0)
point_mesh_2D = point_mesh_2D.reshape(3, mesh_shape[0], mesh_shape[1]).transpose(1, 2, 0)
bg_color = (225,)*3
line_color = (175,)*3
canvas[:, :, 0] = bg_color[0]
canvas[:, :, 1] = bg_color[1]
canvas[:, :, 2] = bg_color[2]
lines_to_draw = set()
for grid_row_idx in range(1, len(grid_zs)):
pre_z = grid_zs[grid_row_idx-1]
cur_z = grid_zs[grid_row_idx]
for grid_col_idx in range(1, len(grid_xs)):
pre_x = grid_xs[grid_col_idx-1]
cur_x = grid_xs[grid_col_idx]
p1 = point_mesh_2D[grid_row_idx-1, grid_col_idx-1]
valid1 = p1[-1] > 0
p2 = point_mesh_2D[grid_row_idx-1, grid_col_idx]
valid2 = p2[-1] > 0
if valid1 and valid2:
line = (tuple(p1[:2].astype(int).tolist()), tuple(p2[:2].astype(int).tolist()))
lines_to_draw.add(line)
# draw vertical line from the previous row
p1 = point_mesh_2D[grid_row_idx-1, grid_col_idx-1]
valid1 = p1[-1] > 0
p2 = point_mesh_2D[grid_row_idx, grid_col_idx-1]
valid2 = p2[-1] > 0
if valid1 and valid2:
line = (tuple(p1[:2].astype(int).tolist()), tuple(p2[:2].astype(int).tolist()))
lines_to_draw.add(line)
for line in lines_to_draw:
draw_line(canvas, line[0], line[1], color=line_color, thickness=max(1, int(np.round(3*scale/1250))))
im_novel_view[~sil_mask] = canvas[~sil_mask]
'''
            Draw edges for novel view
            '''
# apply novel view to meshes
meshes_novel = []
for mesh in meshes:
mesh_novel = mesh.clone().to(device)
verts_rotated = mesh_novel.verts_padded()
verts_rotated -= center
verts_rotated = (view_R @ verts_rotated[0].T).T.unsqueeze(0)
verts_rotated[:, :, -1] += zoom_out_bias*zoom_factor
mesh_novel = mesh_novel.update_padded(verts_rotated)
meshes_novel.append(mesh_novel)
# go in order of reverse depth
for mesh_idx in reversed(np.argsort([mesh.verts_padded().cpu().mean(1)[0, 1] for mesh in meshes_novel])):
mesh = meshes_novel[mesh_idx]
verts3D = mesh.verts_padded()[0].cpu().numpy()
verts2D = (K_novelview @ verts3D.T) / verts3D[:, -1]
color = [min(255, c*255*1.25) for c in mesh.textures.verts_features_padded()[0,0].tolist()]
draw_3d_box_from_verts(
im_novel_view, K_novelview, verts3D, color=color,
thickness=max(2, int(np.round(3*im_novel_view.shape[0]/1250))),
draw_back=False, draw_top=False, zplane=zplane
                )
x1 = verts2D[0, :].min()
y1 = verts2D[1, :].min()
if text is not None:
draw_text(im_novel_view, '{}'.format(text[mesh_idx]), [x1, y1], scale=0.50*im_novel_view.shape[0]/500, bg_color=color)
if mode == 'front_and_novel':
return im_drawn_rgb, im_novel_view, canvas
else:
return im_novel_view, canvas
else:
raise ValueError('No visualization written for {}'.format(mode))