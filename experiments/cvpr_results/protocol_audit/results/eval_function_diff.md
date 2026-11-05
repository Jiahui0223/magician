# 评估相关函数：官方 e697009 vs fork 9521d69（`scripts/diff_eval_functions.py` 自动生成）

| 文件 | 函数 | 官方 | fork | 结果 | 差异行数 |
|---|---|---|---|---|---|
| `macarons/utility/macarons_utils.py` | `load_scene` | 有 | 有 | 不同 | 2 |
| `macarons/utility/macarons_utils.py` | `get_scene_gt_surface` | 有 | 有 | 逐字相同 | 0 |
| `macarons/utility/macarons_utils.py` | `Scene.scene_coverage` | 有 | 有 | 逐字相同 | 0 |
| `macarons/utility/macarons_utils.py` | `Scene.fill_cells` | 有 | 有 | 逐字相同 | 0 |
| `macarons/utility/macarons_utils.py` | `Scene.get_covered_points` | 有 | 有 | 逐字相同 | 0 |
| `macarons/utility/macarons_utils.py` | `Camera.compute_partial_point_cloud` | 有 | 有 | 逐字相同 | 0 |
| `macarons/utility/macarons_utils.py` | `Camera.update_camera` | 有 | 有 | 逐字相同 | 0 |
| `macarons/utility/macarons_utils.py` | `Camera.capture_image` | 有 | 有 | 逐字相同 | 0 |
| `macarons/utility/macarons_utils.py` | `Camera.get_idx_from_key` | 有 | 有 | 逐字相同 | 0 |
| `macarons/utility/macarons_utils.py` | `Camera.get_key_from_idx` | 有 | 有 | 逐字相同 | 0 |
| `macarons/utility/macarons_utils.py` | `Camera.__init__` | 有 | 有 | 不同 | 32 |
| `macarons/utility/macarons_utils.py` | `Settings.__init__` | 有 | 有 | 逐字相同 | 0 |
| `macarons/utility/macarons_utils.py` | `Camera.get_neighboring_poses` | 有 | 有 | 逐字相同 | 0 |
| `macarons/utility/macarons_utils.py` | `Camera.get_valid_neighbors` | 有 | 有 | 逐字相同 | 0 |
| `macarons/testers/magician_planning.py` | `setup_test_scene` | 有 | 有 | 逐字相同 | 0 |
| `macarons/testers/magician_planning.py` | `setup_test_camera` | 有 | 有 | 不同 | 3 |
| `macarons/testers/magician_planning.py` | `load_current_frame_perfect_depth` | 有 | 有 | 逐字相同 | 0 |
| `macarons/testers/magician_planning.py` | `apply_perfect_depth_simple` | 有 | 有 | 逐字相同 | 0 |
| `macarons/utility/magician_utils.py` | `line_segment_mesh_intersection` | 有 | 有 | 逐字相同 | 0 |
| `macarons/utility/magician_utils.py` | `get_valid_neighbors` | 无 | 无 | 两边都没有该函数 | 0 |
| `macarons/utility/magician_utils.py` | `get_neighboring_poses` | 无 | 无 | 两边都没有该函数 | 0 |

## 不同之处（unified diff，官方 → fork）

### `macarons/utility/macarons_utils.py` :: `load_scene`

```diff
--- official
+++ fork
@@ -28,3 +28,3 @@
             verts[..., axis] = -verts[..., axis]
-    # torch.tensor()
+    torch.tensor()
     
```

### `macarons/utility/macarons_utils.py` :: `Camera.__init__`

```diff
--- official
+++ fork
@@ -5,2 +5,3 @@
                  gathering_factor=0.05,
+                 occupied_pose_data=None,
                  save_dir_path=None,
@@ -26,2 +27,3 @@
         :param gathering_factor: (float) Proportion of points to gather by the sensor in the depth map.
+        :param occupied_pose_data: (Dict) Dictionary containing info about occupied poses.
         """
@@ -137 +139,31 @@
             self.pose_history[str(pose_idx.numpy().tolist())] = False
+
+        self.use_occupied_pose = False
+        if occupied_pose_data is not None:
+            self.use_occupied_pose = True
+            pose_is_occupied = {}
+            for i in range(len(occupied_pose_data['X_idx'])):
+                is_occupied = occupied_pose_data['occupied'][i]
+
+                X_idx = 0. + occupied_pose_data['X_idx'][i]
+                if mirrored_scene:
+                    if mirrored_axis is None:
+                        raise NameError("Please provide the list of mirrored axis.")
+                    else:
+                        for axis in mirrored_axis:
+                            if axis == 0:
+                                bound = self.pose_l - 1
+                            elif axis == 1:
+                                bound = self.pose_w - 1
+                            elif axis == 2:
+                                bound = self.pose_h - 1
+                            else:
+                                raise NameError("Wrong value for axis to be mirrored. "
+                                                "Please choose an integer in [0, 1, 2].")
+                            X_idx[..., axis] = bound - X_idx[..., axis]
+
+                X_key = str(X_idx.long().cpu().numpy().tolist())
+
+                pose_is_occupied[X_key] = is_occupied.item()
+
+            self.pose_is_occupied = pose_is_occupied
```

### `macarons/testers/magician_planning.py` :: `setup_test_camera`

```diff
--- official
+++ fork
@@ -3,2 +3,3 @@
                       settings,
+                      occupied_pose_data,
                       device,
@@ -14,2 +15,3 @@
     :param settings:
+    :param occupied_pose_data:
     :param device:
@@ -44,2 +46,3 @@
                     gathering_factor=params.gathering_factor,
+                    occupied_pose_data=occupied_pose_data,
                     save_dir_path=training_frames_path,
```

