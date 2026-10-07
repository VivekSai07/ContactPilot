# Graph Report - ContactPilot  (2026-10-07)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 3368 nodes · 7337 edges · 153 communities (108 shown, 45 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 186 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4076ed6f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- m2t2.py
- torch_nn_functional
- sim_grasp/__init__.py
- os
- inference_graspgen.py
- torch
- dataset.py
- generator.py
- meshcat_utils.py
- dataset_utils.py
- trimesh_transformations
- test_obb_planner.py
- models/pointnet2_utils.py
- executor.py
- data.py
- server.py
- demo_scene_pc.py
- pointnet/pointnet2_modules.py
- image_utils.py
- run_sim_grasp_test.py
- interactive_pick.py
- scene_graph.py
- mesh_utils.py
- numpy
- test_inference_perf.py
- AcryonymDataset
- GraspGenSampler
- train_graspgen_franka_panda_dis.sh
- train_graspgen_robotiq_2f_140_dis.sh
- math_utils.py
- train_graspgen_franka_panda_gen.sh
- tutorial_train_dis.sh
- train_graspgen_robotiq_2f_140_gen.sh
- tutorial_train_gen.sh
- benchmark_prompt_selection.py
- load_default_gripper_config
- SceneGenerator
- TableScene
- math
- indoor3d_util.py
- generate_dataset_suction_single_object.py
- PointSequential
- train_partseg.py
- GraspGenClient
- pathlib
- train_classification.py
- Function
- Function
- PlyElement
- eval_utils.py
- main
- visualization_utils_o3d.py
- download_objects.py
- demo_collision_free_grasps.py
- scene_loaders.py
- LiveViewer
- PlyProperty
- save_grasps_to_usd.py
- ptv3.py
- scene_generator.py
- CheckpointIO
- AcronymRender
- default.py
- ContactGraspnet
- PointCloudReader
- pc_utils.py
- renderer.py
- PlyData
- run_graspmoe
- GraspGen
- test_scene_loaders.py
- save_grasps_to_usd
- spatial_relation_resolver.py
- argparse
- object_knowledge.py
- SceneRenderer
- AcronymRenderScene
- PlyParseError
- metrics.py
- instruction_parser.py
- SceneRenderer
- json
- test_inference_100_grasps
- yaml
- Renderer
- matcher.py
- benchmark_query_kd_np_tf.py
- PlyListProperty
- VisionTransformer
- run_grasp_sim_omniverse.py
- test_usd_grasp_pipeline.py
- benchmark.py
- train_semseg.py
- plyfile.py
- vit.py
- sampling_gpu.cu
- eulerangles.py
- load_point_cloud_from_file
- collate
- z_order.py
- interpolate_gpu.cu
- format_xml.py
- perception.py
- PointInfo
- GraspGenZMQServer
- add_gaussian_shifts
- PointnetSAModuleMSG
- test_semseg.py
- test_serving.py
- plot_utils.py
- MockGraspGenServer
- analyze_failures.py
- Point
- Attention
- get_init_weights_vit
- group_points_gpu.cu
- GroupAll
- TestLiveServer
- sys
- cuda_utils.h
- plot_pr.py
- ball_query_gpu.cu
- create_grasp_sim_usd
- realsense_capture.py
- regularize_pc_point_count
- show3d_balls.py
- load_point_cloud_from_mesh
- NormalizeInverse
- install_pointnet.sh
- install_uv_pointnet.sh
- load_T_base_cam
- sam3_worker.py
- GripperModel
- _load_weights
- recursive_key_value_assign
- estimate_normals_cam_from_pc
- load_available_input_data
- run.sh
- run_server.sh
- TestPCDReader
- SinusoidalPosEmb
- PYBIND11_MODULE
- visualizer/build.sh
- docker/build.sh
- run_meshcat.sh
- grasp_gen
- mcp-server-graspgen

## God Nodes (most connected - your core abstractions)
1. `main()` - 39 edges
2. `GraspGenSampler` - 35 edges
3. `get_gripper_info()` - 35 edges
4. `GraspGenClient` - 34 edges
5. `SceneGenerator` - 31 edges
6. `PlyElement` - 31 edges
7. `AcryonymDataset` - 23 edges
8. `PlyData` - 22 edges
9. `main()` - 22 edges
10. `get_logger()` - 22 edges

## Surprising Connections (you probably didn't know these)
- `ContactGraspNetPredictor` --uses--> `GraspEstimator`  [INFERRED]
  mujoco_grasp_sim/sim_grasp/grasp_predictor.py → contact_graspnet_pytorch/contact_graspnet_pytorch/contact_grasp_estimator.py
- `ContactGraspNetPredictor` --uses--> `CheckpointIO`  [INFERRED]
  mujoco_grasp_sim/sim_grasp/grasp_predictor.py → contact_graspnet_pytorch/contact_graspnet_pytorch/checkpoints.py
- `main()` --calls--> `point_cloud_outlier_removal()`  [INFERRED]
  mujoco_grasp_sim/sim_grasp/graspgen_worker.py → GraspGen/grasp_gen/utils/point_cloud_utils.py
- `object_point_clouds()` --calls--> `depth_and_segmentation_to_point_clouds()`  [INFERRED]
  mujoco_grasp_sim/sim_grasp/graspgen_worker.py → GraspGen/grasp_gen/utils/point_cloud_utils.py
- `main()` --uses--> `GraspGenSampler`  [INFERRED]
  mujoco_grasp_sim/sim_grasp/graspgen_worker.py → GraspGen/grasp_gen/grasp_server.py

## Import Cycles
- None detected.

## Communities (153 total, 45 thin omitted)

### Community 0 - "m2t2.py"
Cohesion: 0.04
Nodes (21): ActionDecoder, build_6d_grasp(), build_6d_place(), double_split(), infer_placements(), compute_attention_mask(), ContactDecoder, adds() (+13 more)

### Community 1 - "torch_nn_functional"
Cohesion: 0.05
Nodes (16): get_loss, get_model, get_loss, get_model, get_loss, get_model, get_loss, get_model (+8 more)

### Community 2 - "sim_grasp/__init__.py"
Cohesion: 0.06
Nodes (22): filter_feasible(), capture_fused(), _box_corners(), _grasp_width(), GraspFeasibilityChecker, _gripper_sample_points(), _hand_boxes(), invert_se3() (+14 more)

### Community 3 - "os"
Cohesion: 0.06
Nodes (16): check_scene_contacts(), render_acronym(), render_scene(), build_6d_grasp(), depth2pc(), index_points(), inverse_transform(), reject_median_outliers() (+8 more)

### Community 4 - "inference_graspgen.py"
Cohesion: 0.08
Nodes (26): get_logger(), get_timestamp(), log_worker(), add_to_dict(), build_optimizer(), compute_iou(), get_data_loader(), get_iou() (+18 more)

### Community 5 - "torch"
Cohesion: 0.08
Nodes (29): _angle_from_tan(), _axis_angle_rotation(), axis_angle_to_matrix(), axis_angle_to_quaternion(), _copysign(), euler_angles_to_matrix(), _index_from_letter(), matrix_to_axis_angle() (+21 more)

### Community 6 - "dataset.py"
Cohesion: 0.06
Nodes (21): generate_negative_freespace(), generate_negative_hardnegatives(), generate_negative_retract(), get_cache_path(), get_cache_prefix(), get_denylist_path(), get_pc_setting_name(), is_valid_cache_dir() (+13 more)

### Community 7 - "generator.py"
Cohesion: 0.07
Nodes (17): compute_recall(), GraspGenDiscriminator, DiffusionNoisePredictionNet, GraspGenGenerator, AttentionLayer, break_up_pc(), compute_grasp_loss(), convert_to_ptv3_pc_format() (+9 more)

### Community 8 - "meshcat_utils.py"
Cohesion: 0.07
Nodes (27): visualize_object_grasp_dataset(), adjoint_transform(), color_interpolation(), colorize(), colorize_for_meshcat(), colorized_points(), main_func(), skew() (+19 more)

### Community 9 - "dataset_utils.py"
Cohesion: 0.06
Nodes (18): compute_emd_data(), convert_dict_to_trimesh(), convert_trimesh_to_dict(), GraspJsonDatasetReader, load_grasp_data(), load_object_grasp_acronym(), load_object_grasp_data(), load_object_grasp_datapoint_objaverse() (+10 more)

### Community 10 - "trimesh_transformations"
Cohesion: 0.06
Nodes (24): get_transform_from_base_link_to_tool_tcp(), load_control_points_for_visualization(), load_visualize_control_points_suction(), sample_points(), generate_circle_points(), get_canonical_gripper_control_points(), get_gripper_depth(), get_gripper_info() (+16 more)

### Community 11 - "test_obb_planner.py"
Cohesion: 0.07
Nodes (29): _build_face_candidates(), _compute_obb(), _interior_positions(), _long_axis_positions(), _min_area_rect_xy(), _obb_from_angle(), _resolve_gripper_geometry(), _run_obb_branch() (+21 more)

### Community 12 - "models/pointnet2_utils.py"
Cohesion: 0.07
Nodes (17): get_loss, get_model, get_loss, get_model, get_loss, get_model, get_loss, get_model (+9 more)

### Community 13 - "executor.py"
Cohesion: 0.06
Nodes (9): SpyExecutor, _candidate_hand_orientations(), DiffIK, _ease(), GraspExecutor, IKResult, _pick_best_seed_result(), _r() (+1 more)

### Community 14 - "data.py"
Cohesion: 0.06
Nodes (13): GraspEstimator, center_pc_convert_cam(), depth2pc(), distance_by_translation_point(), farthest_points(), inverse_transform(), load_available_input_data(), load_graspnet_data() (+5 more)

### Community 15 - "server.py"
Cohesion: 0.07
Nodes (22): main(), _format_grasp_results(), GenerateGraspsFromMesh, GenerateGraspsFromPointCloud, _get_or_create_viser(), _get_server_addr(), _get_socket(), _gripper_polyline() (+14 more)

### Community 16 - "demo_scene_pc.py"
Cohesion: 0.10
Nodes (20): main(), parse_args(), visualize_results(), create_visualizer(), get_color_from_score(), get_normals_from_mesh(), is_rotation_matrix(), load_visualization_gripper_points() (+12 more)

### Community 17 - "pointnet/pointnet2_modules.py"
Cohesion: 0.08
Nodes (13): build_shared_mlp(), _get_norm(), PointnetFPModule, PointnetSAModule, _PointnetSAModuleBase, PointnetSAModuleMSG, PointnetSAModuleMSGVarNPts, _PointnetSAModuleVarNPts (+5 more)

### Community 18 - "image_utils.py"
Cohesion: 0.08
Nodes (18): add_gaussian_noise_to_depth(), add_noise_to_xyz(), blend_images(), compress_img(), convert_img_float_uint8(), convert_label_img_to_seg(), decompress_png(), depth2rgb() (+10 more)

### Community 19 - "run_sim_grasp_test.py"
Cohesion: 0.08
Nodes (11): _box_yaw_alignment_bonus(), _object_geom(), predict_clouds_in_subprocess(), predict_in_subprocess(), rank_candidates(), _subprocess_predict(), GraspPrediction, GraspPredictor (+3 more)

### Community 20 - "interactive_pick.py"
Cohesion: 0.09
Nodes (14): main(), _load_and_predict(), _warm_cgn(), ContactGraspNetPredictor, _aabb_half_extent(), BinHeightmap, build_bin_heightmap(), compute_object_footprint() (+6 more)

### Community 21 - "scene_graph.py"
Cohesion: 0.08
Nodes (15): target_bin_for(), BinSpec, build(), Edge, _location(), Node, SceneGraph, _centre() (+7 more)

### Community 22 - "mesh_utils.py"
Cohesion: 0.07
Nodes (8): create_gripper(), grasp_contact_location(), in_collision_with_gripper(), Object, PandaGripper, grasps_contact_info(), read_object_grasp_data_acronym(), save_contact_data()

### Community 23 - "numpy"
Cohesion: 0.08
Nodes (16): pc_normalize(), jitter_point_cloud(), normalize_data(), rotate_perturbation_point_cloud(), rotate_perturbation_point_cloud_with_normal(), rotate_point_cloud(), rotate_point_cloud_by_angle(), rotate_point_cloud_by_angle_with_normal() (+8 more)

### Community 24 - "test_inference_perf.py"
Cohesion: 0.07
Nodes (14): device(), random_seed(), sample_point_cloud(), sample_pose(), sample_rotation_matrix(), _bench_loop(), BenchResult, _make_discriminator_cfg() (+6 more)

### Community 26 - "GraspGenSampler"
Cohesion: 0.11
Nodes (13): GraspGenSampler, load_grasp_cfg(), franka_config_path(), robotiq_config_path(), sample_point_cloud(), test_grasp_sampler_initialization_franka(), test_grasp_sampler_initialization_robotiq(), test_grasp_sampling_basic_franka() (+5 more)

### Community 27 - "train_graspgen_franka_panda_dis.sh"
Cohesion: 0.06
Nodes (34): BACKBONE, BATCH, CACHE_DIR, CHECKPOINT, CODE_DIR, CONSOLE_LOG, DATASET_NAME, DATASET_VERSION (+26 more)

### Community 28 - "train_graspgen_robotiq_2f_140_dis.sh"
Cohesion: 0.06
Nodes (34): BACKBONE, BATCH, CACHE_DIR, CHECKPOINT, CODE_DIR, CONSOLE_LOG, DATASET_NAME, DATASET_VERSION (+26 more)

### Community 29 - "math_utils.py"
Cohesion: 0.11
Nodes (17): compute_pose_distance_batch(), compute_pose_emd(), construct_suction_grasp_from_point_and_vector(), matrix_to_rotation_6d(), rotation_6d_to_matrix(), rotation_from_vectors(), rotation_matrix_from_vectors(), rt_to_matrix() (+9 more)

### Community 30 - "train_graspgen_franka_panda_gen.sh"
Cohesion: 0.06
Nodes (33): BACKBONE, BATCH, CACHE_DIR, CHECKPOINT, CODE_DIR, CONSOLE_LOG, DATASET_NAME, DATASET_VERSION (+25 more)

### Community 31 - "tutorial_train_dis.sh"
Cohesion: 0.06
Nodes (33): BACKBONE, BATCH, CACHE_DIR, CHECKPOINT, CODE_DIR, DATASET_NAME, DATASET_VERSION, GRASP_DATASET_DIR (+25 more)

### Community 32 - "train_graspgen_robotiq_2f_140_gen.sh"
Cohesion: 0.06
Nodes (32): BACKBONE, BATCH, CACHE_DIR, CHECKPOINT, CODE_DIR, CONSOLE_LOG, DATASET_NAME, DATASET_VERSION (+24 more)

### Community 33 - "tutorial_train_gen.sh"
Cohesion: 0.06
Nodes (32): BACKBONE, BATCH, CACHE_DIR, CHECKPOINT, CODE_DIR, DATASET_NAME, DATASET_VERSION, GRASP_DATASET_DIR (+24 more)

### Community 34 - "benchmark_prompt_selection.py"
Cohesion: 0.10
Nodes (10): main(), object_geom(), parse_seeds(), run_one(), rgb_to_color_name(), filter_selection_by_click(), PromptSelector, resolve_sam3_python() (+2 more)

### Community 35 - "load_default_gripper_config"
Cohesion: 0.09
Nodes (10): GripperModel, load_control_points(), load_control_points_for_visualization(), GripperModel, load_control_points(), load_control_points_for_visualization(), load_control_points(), load_control_points_core() (+2 more)

### Community 38 - "math"
Cohesion: 0.14
Nodes (11): box_inertia(), footprint_radius(), load_manifest(), obj_bounds(), parse_manifest(), prop_body_xml(), prop_files(), prop_scale() (+3 more)

### Community 39 - "indoor3d_util.py"
Cohesion: 0.10
Nodes (16): bbox_label_to_obj(), bbox_label_to_obj_room(), collect_bounding_box(), collect_point_bounding_box(), collect_point_label(), point_label_to_obj(), room2blocks(), room2blocks_plus() (+8 more)

### Community 40 - "generate_dataset_suction_single_object.py"
Cohesion: 0.10
Nodes (8): SuctionCupArray, copy_object_to_dataset(), create_grasp_dataset_json(), create_splits_file(), generate_grasps(), main(), parse_args(), setup_directories()

### Community 41 - "PointSequential"
Cohesion: 0.11
Nodes (8): Block, Embedding, MLP, PDNorm, PointModule, PointSequential, SerializedPooling, SerializedUnpooling

### Community 42 - "train_partseg.py"
Cohesion: 0.10
Nodes (11): PartNormalDataset, pc_normalize(), random_scale_point_cloud(), shift_point_cloud(), main(), parse_args(), to_categorical(), inplace_relu() (+3 more)

### Community 44 - "pathlib"
Cohesion: 0.12
Nodes (12): apply_gallery_settings(), auto_camera(), detect_license(), display_name(), live_url(), main(), ModelType, _parse_floats() (+4 more)

### Community 45 - "train_classification.py"
Cohesion: 0.11
Nodes (11): farthest_point_sample(), ModelNetDataLoader, pc_normalize(), random_point_dropout(), main(), parse_args(), test(), inplace_relu() (+3 more)

### Community 46 - "Function"
Cohesion: 0.07
Nodes (6): BallQuery, FurthestPointSampling, GatherOperation, GroupingOperation, ThreeInterpolate, ThreeNN

### Community 47 - "Function"
Cohesion: 0.07
Nodes (6): BallQuery, FurthestPointSampling, GatherOperation, GroupingOperation, ThreeInterpolate, ThreeNN

### Community 49 - "eval_utils.py"
Cohesion: 0.09
Nodes (9): create_scene(), is_empty(), load_from_isaac_grasp_format(), load_h5_handle_empty_case(), load_urdf_scene(), pose_as_dict(), save_to_isaac_grasp_format(), write_info() (+1 more)

### Community 50 - "main"
Cohesion: 0.11
Nodes (6): gpu_vram_gb(), main(), make_crop(), CameraModule, identification_crop(), sort_outcome()

### Community 51 - "visualization_utils_o3d.py"
Cohesion: 0.12
Nodes (7): inference(), draw_grasps(), draw_pc_with_colors(), plot_coordinates(), plot_mesh(), show_image(), visualize_grasps()

### Community 52 - "download_objects.py"
Cohesion: 0.12
Nodes (9): download_objaverse_meshes(), download_with_retry(), load_uuid_list(), process_directory(), process_single_object(), process_single_object_with_uuid(), process_text_file(), simplify_mesh() (+1 more)

### Community 53 - "demo_collision_free_grasps.py"
Cohesion: 0.15
Nodes (11): depth_and_segmentation_to_point_clouds(), filter_colliding_grasps(), knn_points(), point_cloud_outlier_removal(), point_cloud_outlier_removal_with_color(), process_point_cloud(), test_knn_points_basic(), test_knn_points_identical_points() (+3 more)

### Community 54 - "scene_loaders.py"
Cohesion: 0.12
Nodes (15): build_scene_pc_excluding_object(), depth_to_camera_xyz(), _load_realworld_metadata(), load_realworld_scene(), load_scene(), transform_xyz(), _make_realworld_scene(), test_build_scene_pc_excluding_object_realworld() (+7 more)

### Community 55 - "LiveViewer"
Cohesion: 0.12
Nodes (3): compose_mask_overlay(), draw_status_text(), LiveViewer

### Community 57 - "save_grasps_to_usd.py"
Cohesion: 0.13
Nodes (8): main(), _add_basis_curves_prim(), add_grasps_visualization_to_stage(), _get_gripper_wireframe_segments(), _gf_matrix4d_to_numpy(), _load_yaml_grasps(), _numpy_to_gf_matrix4d(), _remove_existing_grasps()

### Community 58 - "ptv3.py"
Cohesion: 0.13
Nodes (5): batch2offset(), offset2batch(), offset2bincount(), RPE, SerializedAttention

### Community 59 - "scene_generator.py"
Cohesion: 0.11
Nodes (6): object_color(), _bin_xml(), _list_mesh_files(), _make_mesh_object(), _make_primitive(), ObjectSpec

### Community 62 - "default.py"
Cohesion: 0.19
Nodes (11): decode(), encode(), hilbert_decode(), hilbert_encode(), z_order_decode(), z_order_encode(), binary2gray(), decode() (+3 more)

### Community 63 - "ContactGraspnet"
Cohesion: 0.12
Nodes (5): ContactGraspnet, ContactGraspnetLoss, log_string(), train(), send_dict_to_device()

### Community 64 - "PointCloudReader"
Cohesion: 0.12
Nodes (3): estimate_normals_cam_from_pc(), PointCloudReader, vectorized_normal_computation()

### Community 65 - "pc_utils.py"
Cohesion: 0.13
Nodes (9): euler2mat(), draw_point_cloud(), point_cloud_three_views(), point_cloud_three_views_demo(), point_cloud_to_volume(), point_cloud_to_volume_batch(), pyplot_draw_point_cloud(), pyplot_draw_volume() (+1 more)

### Community 66 - "renderer.py"
Cohesion: 0.18
Nodes (10): add_depth_noise(), add_edge_noise(), add_gaussian_noise_to_depth(), compute_camera_pose(), depth2points(), fov_and_size_to_intrinsics(), render_images_given_scene(), render_pc() (+2 more)

### Community 68 - "run_graspmoe"
Cohesion: 0.16
Nodes (7): run_graspmoe(), run_graspmoe_batch(), run_planner_on_batch(), run_planner_on_object(), _find_ckpt_yml(), test_run_graspmoe_e2e_smoke(), test_run_graspmoe_topdown_cuboid_on_table()

### Community 69 - "GraspGen"
Cohesion: 0.12
Nodes (3): GraspGen, box_mesh_path(), point_cloud_from_box()

### Community 70 - "test_scene_loaders.py"
Cohesion: 0.17
Nodes (13): filter_colliding_grasps_fast(), collect_scene_items(), detect_format(), test_collect_scene_items_filter(), test_collect_scene_items_realworld(), test_detect_format_json(), test_detect_format_neither(), test_detect_format_realworld() (+5 more)

### Community 71 - "save_grasps_to_usd"
Cohesion: 0.20
Nodes (5): load_grasps_from_usd(), save_grasps_to_usd(), _make_synthetic_grasps(), TestSaveGraspsToUsd, TestUsdGraspPrimStructure

### Community 72 - "spatial_relation_resolver.py"
Cohesion: 0.15
Nodes (6): _camera_view_axis(), resolve(), _EmptyFakeSelector, _FakePromptSelector, _FakeSelectionResult, _TwoCandidateSelector

### Community 73 - "argparse"
Cohesion: 0.14
Nodes (10): make_parser(), convert_mesh_to_usd(), parse_args(), set_usd_mesh_display_color(), parse_args(), parse_args(), parse_args(), parse_args() (+2 more)

### Community 74 - "object_knowledge.py"
Cohesion: 0.15
Nodes (4): _call(), categorize(), identify(), KnowledgeCache

### Community 78 - "metrics.py"
Cohesion: 0.17
Nodes (6): angular_distance_phi3(), compute_metrics_given_two_sets_of_poses(), GeodesicLoss, normalize_quaternion(), OrientationError, quat_multiply()

### Community 79 - "instruction_parser.py"
Cohesion: 0.16
Nodes (7): main(), _wanted(), _load_dotenv(), _parse_and_validate(), parse_instruction(), Step, _nim_transport()

### Community 81 - "json"
Cohesion: 0.18
Nodes (8): call_model(), extract_json(), _load_dotenv(), main(), schema_errors(), load_uuid_list(), load_graspgen_json_scene(), test_load_graspgen_json_scene()

### Community 82 - "test_inference_100_grasps"
Cohesion: 0.17
Nodes (6): _make_discriminator_cfg(), _make_generator_cfg(), _run_inference(), test_inference_100_grasps(), test_model_components(), test_score_grasps_with_random_weights()

### Community 83 - "yaml"
Cohesion: 0.18
Nodes (6): download_checkpoint(), download_test_data(), copy_checkpoint(), create_config_file(), main(), parse_args()

### Community 85 - "matcher.py"
Cohesion: 0.15
Nodes (3): bce_loss_matrix(), dice_loss_matrix(), HungarianMatcher

### Community 86 - "benchmark_query_kd_np_tf.py"
Cohesion: 0.19
Nodes (6): build_file_writers(), build_summary_ops(), top_grasp_acc_summaries(), tf_knn_max_dist(), tf_nn(), tf_queryball()

### Community 89 - "run_grasp_sim_omniverse.py"
Cohesion: 0.17
Nodes (6): _get_articulation_roots_under_world(), _get_closed_joint_positions(), main(), run_grasp_on_play(), _close_grippers(), _on_timestep()

### Community 91 - "test_usd_grasp_pipeline.py"
Cohesion: 0.18
Nodes (5): box_obj_path(), box_usd_path(), TestEndToEndPipeline, TestObjToUsdConversion, tmp_dir()

### Community 92 - "benchmark.py"
Cohesion: 0.23
Nodes (6): effective_identity(), format_location_agreement(), main(), parse_seeds(), run_one(), summarize_sorting()

### Community 93 - "train_semseg.py"
Cohesion: 0.19
Nodes (4): S3DISDataset, inplace_relu(), main(), parse_args()

### Community 94 - "plyfile.py"
Cohesion: 0.17
Nodes (4): read_ply(), make2d(), _open_stream(), _split_line()

### Community 95 - "vit.py"
Cohesion: 0.19
Nodes (4): checkpoint_filter_fn(), _convert_dinov2(), _convert_openai_clip(), resize_pos_embed()

### Community 96 - "sampling_gpu.cu"
Cohesion: 0.21
Nodes (9): furthest_point_sampling(), gather_points(), gather_points_grad(), furthest_point_sampling_kernel(), gather_points_grad_kernel(), gather_points_grad_kernel_wrapper(), gather_points_kernel(), gather_points_kernel_wrapper() (+1 more)

### Community 97 - "eulerangles.py"
Cohesion: 0.21
Nodes (5): angle_axis2euler(), euler2angle_axis(), euler2quat(), mat2euler(), quat2euler()

### Community 98 - "load_point_cloud_from_file"
Cohesion: 0.24
Nodes (3): load_point_cloud_from_file(), _read_pcd_ascii(), TestLoadPointCloudFromFile

### Community 99 - "collate"
Cohesion: 0.18
Nodes (4): collate(), collate_batch_keys(), score_grasps_with_discriminator(), _prepare_batch()

### Community 100 - "z_order.py"
Cohesion: 0.21
Nodes (3): key2xyz(), KeyLUT, xyz2key()

### Community 101 - "interpolate_gpu.cu"
Cohesion: 0.23
Nodes (9): three_interpolate_grad_kernel(), three_interpolate_grad_kernel_wrapper(), three_interpolate_kernel(), three_interpolate_kernel_wrapper(), three_nn_kernel(), three_nn_kernel_wrapper(), three_interpolate(), three_interpolate_grad() (+1 more)

### Community 102 - "format_xml.py"
Cohesion: 0.27
Nodes (7): _attr_str(), _blank_lines_between(), _escape_attr(), _format_open_tag(), format_xml(), main(), _serialize()

### Community 103 - "perception.py"
Cohesion: 0.23
Nodes (4): clean_depth(), recenter_grasp(), remove_depth_speckles(), workspace_crop()

### Community 104 - "PointInfo"
Cohesion: 0.18
Nodes (7): PointInfo, b, g, r, x, y, z

### Community 105 - "GraspGenZMQServer"
Cohesion: 0.27
Nodes (4): main(), parse_args(), __getattr__(), GraspGenZMQServer

### Community 106 - "add_gaussian_shifts"
Cohesion: 0.20
Nodes (5): add_gaussian_shifts(), add_kinect_noise_to_depth(), dropout_random_ellipses(), jitter_gaussian(), mask_object_edge()

### Community 107 - "PointnetSAModuleMSG"
Cohesion: 0.22
Nodes (3): PointnetSAModule, _PointnetSAModuleBase, PointnetSAModuleMSG

### Community 108 - "test_semseg.py"
Cohesion: 0.27
Nodes (4): ScannetDatasetWholeScene, add_vote(), main(), parse_args()

### Community 110 - "plot_utils.py"
Cohesion: 0.29
Nodes (4): get_set_colors(), plot_3D(), plot_mask_3D(), plot_place_mask_3D()

### Community 111 - "MockGraspGenServer"
Cohesion: 0.22
Nodes (3): _make_fake_grasps(), mock_server(), MockGraspGenServer

### Community 112 - "analyze_failures.py"
Cohesion: 0.27
Nodes (3): analyze_run(), classify_pick(), main()

### Community 114 - "Attention"
Cohesion: 0.22
Nodes (3): Attention, Block, LayerScale

### Community 115 - "get_init_weights_vit"
Cohesion: 0.28
Nodes (4): get_init_weights_vit(), init_weights_vit_jax(), init_weights_vit_moco(), init_weights_vit_timm()

### Community 116 - "group_points_gpu.cu"
Cohesion: 0.28
Nodes (6): group_points_grad_kernel(), group_points_grad_kernel_wrapper(), group_points_kernel(), group_points_kernel_wrapper(), group_points(), group_points_grad()

### Community 119 - "sys"
Cohesion: 0.36
Nodes (3): generate_license(), get_base_license(), main()

### Community 120 - "cuda_utils.h"
Cohesion: 0.29
Nodes (3): opt_block_config(), opt_n_threads(), furthest_point_sampling_kernel_wrapper()

### Community 122 - "ball_query_gpu.cu"
Cohesion: 0.25
Nodes (3): ball_query(), query_ball_point_kernel(), query_ball_point_kernel_wrapper()

### Community 123 - "create_grasp_sim_usd"
Cohesion: 0.29
Nodes (3): create_grasp_sim_usd(), _env_offset_index(), _numpy_to_gf_matrix4d()

### Community 124 - "realsense_capture.py"
Cohesion: 0.43
Nodes (3): capture_frame(), main(), segment_table_top()

### Community 126 - "regularize_pc_point_count"
Cohesion: 0.33
Nodes (3): distance_by_translation_point(), farthest_points(), regularize_pc_point_count()

### Community 130 - "install_pointnet.sh"
Cohesion: 0.33
Nodes (5): CC, CUDAHOSTCXX, CXX, install_pointnet.sh script, TORCH_CUDA_ARCH_LIST

### Community 131 - "install_uv_pointnet.sh"
Cohesion: 0.33
Nodes (5): CC, CUDAHOSTCXX, CXX, install_uv_pointnet.sh script, TORCH_CUDA_ARCH_LIST

### Community 140 - "run.sh"
Cohesion: 0.83
Nodes (3): make_absolute_path(), run.sh script, show_usage()

### Community 141 - "run_server.sh"
Cohesion: 0.83
Nodes (3): make_absolute_path(), run_server.sh script, show_usage()

## Knowledge Gaps
- **220 isolated node(s):** `ErrorInfo`, `b`, `g`, `r`, `x` (+215 more)
  These have ≤1 connection - possible missing edges. (Counts symbols only; 1497 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **45 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SceneRenderer` connect `SceneRenderer` to `os`, `AcronymRenderScene`, `AcronymRender`, `mesh_utils.py`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **What connects `ErrorInfo`, `b`, `g` to the rest of the system?**
  _220 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `m2t2.py` be split into smaller, more focused modules?**
  _Cohesion score 0.04280701754385965 - nodes in this community are weakly interconnected._
- **Why does `AcronymRender` connect `AcronymRender` to `SceneRenderer`, `os`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._
- **Should `torch_nn_functional` be split into smaller, more focused modules?**
  _Cohesion score 0.04679089026915114 - nodes in this community are weakly interconnected._
- **Why does `VisionTransformer` connect `VisionTransformer` to `generator.py`, `_load_weights`, `.get_intermediate_layers`, `get_init_weights_vit`, `vit.py`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Should `sim_grasp/__init__.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0574400723654455 - nodes in this community are weakly interconnected._