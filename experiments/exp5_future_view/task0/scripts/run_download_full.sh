#!/bin/bash
# EXP5 Task 0 -- one stream of the complete Matterport3D v1 scan release, in two phases (resumable, safe to re-run):
#   phase 1 (geometry; extracted):  cameras undistorted_camera_parameters matterport_mesh house_segmentations
#                                   undistorted_color_images undistorted_depth_images poisson_meshes
#   phase 2 (rest of the release; zips kept, extracted on demand): all other file types
# $1 = stream id, $2 = scan id list
S=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/exp5_future_view/task0/scripts
$S/run_stream.sh full_geo_$1 $2 cameras undistorted_camera_parameters matterport_mesh house_segmentations \
  undistorted_color_images undistorted_depth_images poisson_meshes --extract
$S/run_stream.sh full_rest_$1 $2 matterport_camera_intrinsics matterport_camera_poses region_segmentations \
  image_overlap_data matterport_skybox_images matterport_color_images matterport_hdr_images matterport_depth_images \
  undistorted_normal_images sens
