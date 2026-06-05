/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { ModelItem, LogLine, ApiEndpoint } from './types';

interface LegacyPerson {
  name: string;
  display_name: string;
  embedding_count: number;
  class: string;
  id_tag: string;
  vectors: number;
  auth_level: string;
  avatar_url?: string;
}

// Real-looking avatar links from reliable fallbacks (identicon/svg or UI avatars)
export const INITIAL_PEOPLE: LegacyPerson[] = [
  {
    name: 'Marcus Thorne',
    display_name: 'Marcus Thorne',
    embedding_count: 512,
    class: 'ALFA',
    id_tag: '#RECO-8821',
    vectors: 512,
    auth_level: 'LVL-4',
    avatar_url: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=256&auto=format&fit=crop'
  },
  {
    name: 'Sarah Chen',
    display_name: 'Sarah Chen',
    embedding_count: 1024,
    class: 'DELTA',
    id_tag: '#RECO-9011',
    vectors: 1024,
    auth_level: 'LVL-MAX',
    avatar_url: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?q=80&w=256&auto=format&fit=crop'
  },
  {
    name: 'Aaliyah James',
    display_name: 'Aaliyah James',
    embedding_count: 256,
    class: 'GAMMA',
    id_tag: '#RECO-4452',
    vectors: 256,
    auth_level: 'LVL-2',
    avatar_url: 'https://images.unsplash.com/photo-1517841905240-472988babdf9?q=80&w=256&auto=format&fit=crop'
  }
];

export const INITIAL_MODELS: ModelItem[] = [
  {
    id: 1,
    name: 'Sentinel_X_v2.0',
    path: 'weights/sentinel_x_v2.pt',
    backbone: 'iresnet101-Modified',
    embedding_size: 512,
    framework: 'PyTorch 2.1',
    created_at: '2023-11-24 14:02:11',
    is_active: true
  },
  {
    id: 2,
    name: 'Ghost_Net_v4',
    path: 'weights/ghostnet_v4_eff.pt',
    backbone: 'Mobile-Efficient-B3',
    embedding_size: 128,
    framework: 'TensorFlow 2.15',
    created_at: '2023-11-20 09:15:33',
    is_active: false
  },
  {
    id: 3,
    name: 'Secure_Face_Legacy',
    path: 'weights/vgg16_secure.pth',
    backbone: 'VGG-16',
    embedding_size: 1024,
    framework: 'PyTorch 1.8',
    created_at: '2023-05-12 18:44:02',
    is_active: false
  }
];

export const INITIAL_LOGS: LogLine[] = [
  { timestamp: '14:22:01.09', level: 'INFO', message: 'CORE_INIT: Analysis engine v4.2 started successfully.' },
  { timestamp: '14:22:05.44', level: 'INPUT_STREAM', message: 'Inbound visual data detected from CAM_HUB_8.' },
  { timestamp: '14:22:08.12', level: 'AI_PROC', message: 'Neural mapping of Subject_Unknown in progress. Confidence threshold set to 85%.' },
  { timestamp: '14:22:12.98', level: 'WARNING', message: 'Low confidence match detected in sub-quadrant 4. Rerunning Deep_AI analysis.' },
  { timestamp: '14:22:15.55', level: 'RESULT', message: 'Positive identity match: John Doe (DB_REF: 0911). Verification code ER-XC990.' }
];

export const API_ENDPOINTS: ApiEndpoint[] = [
  // 1.1 Auth
  {
    category: 'AUTH',
    method: 'POST',
    path: '/auth/login',
    summary: 'Authenticate user & retrieve access token',
    permission: 'Public',
    requestSchema: '{\n  "username": "admin",\n  "password": "password123"\n}',
    responseSchema: '{\n  "access_token": "eyJhbGciOiJIUzI1NiIsIn...",\n  "token_type": "bearer",\n  "role": "admin"\n}'
  },
  {
    category: 'AUTH',
    method: 'POST',
    path: '/auth/register',
    summary: 'Register a new admin/developer/viewer account',
    permission: 'First Public, then Admin only',
    requestSchema: '{\n  "username": "coder_x",\n  "password": "secure_pass_99",\n  "email": "coder@sentinel.ai",\n  "role": "developer"\n}',
    responseSchema: '{\n  "id": 4,\n  "username": "coder_x",\n  "email": "coder@sentinel.ai",\n  "role": "developer",\n  "is_active": true,\n  "created_at": "2026-05-26T13:36:45Z"\n}'
  },
  {
    category: 'AUTH',
    method: 'GET',
    path: '/auth/me',
    summary: 'Retrieve currently logged-in user profile',
    permission: 'Authenticated',
    responseSchema: '{\n  "id": 1,\n  "username": "admin",\n  "email": "chetqiqi@gmail.com",\n  "role": "admin",\n  "is_active": true,\n  "created_at": "2026-01-10T11:00:15Z"\n}'
  },
  {
    category: 'AUTH',
    method: 'GET',
    path: '/auth/users',
    summary: 'List all user accounts in authentication DB',
    permission: 'Admin',
    responseSchema: '[\n  { "id": 1, "username": "admin", "role": "admin" },\n  { "id": 2, "username": "dev_sentinel", "role": "developer" }\n]'
  },
  {
    category: 'AUTH',
    method: 'DELETE',
    path: '/auth/users/{user_id}',
    summary: 'Remove a user account (cannot self-delete)',
    permission: 'Admin',
    responseSchema: '{\n  "deleted": true,\n  "user_id": 2\n}'
  },
  // 1.2 Core System
  {
    category: 'CORE/STATS',
    method: 'GET',
    path: '/health',
    summary: 'Check API running status and neural model memory state',
    permission: 'Public',
    responseSchema: '{\n  "status": "ok",\n  "model_loaded": true\n}'
  },
  {
    category: 'CORE/STATS',
    method: 'POST',
    path: '/model/load',
    summary: 'Force loading current model weights directly into active GPU VRAM',
    permission: 'Admin/Developer',
    responseSchema: '{\n  "status": "success",\n  "loaded_model": "Sentinel_X_v2.0",\n  "vram_allocated_mb": 4120\n}'
  },
  {
    category: 'CORE/STATS',
    method: 'GET',
    path: '/stats',
    summary: 'Retrieve global person counts and registered mathematical embedding vectors',
    permission: 'All logged in',
    responseSchema: '{\n  "person_count": 4129,\n  "embedding_count": 18240000000\n}'
  },
  // 1.3 Identity
  {
    category: 'IDENTITY',
    method: 'GET',
    path: '/identity',
    summary: 'List all registered subjects and individual mathematical snapshot counts',
    permission: 'Admin / Developer',
    responseSchema: '{\n  "persons": [\n    { "name": "Marcus Thorne", "embedding_count": 512 },\n    { "name": "Sarah Chen", "embedding_count": 1024 },\n    { "name": "Aaliyah James", "embedding_count": 256 }\n  ]\n}'
  },
  {
    category: 'IDENTITY',
    method: 'POST',
    path: '/identity/add',
    summary: 'Enroll a new subject by compiling facial coordinates from photo uploads (multipart)',
    permission: 'All logged in',
    requestSchema: '// Requires form-data:\n// person_id: "John Doe"\n// files: [jpeg/png files]',
    responseSchema: '{\n  "person_name": "John Doe",\n  "success_count": 5,\n  "fail_count": 0,\n  "success_files": ["face_1.jpg", "face_2.png"],\n  "failed_files": []\n}'
  },
  {
    category: 'IDENTITY',
    method: 'DELETE',
    path: '/identity/{person_name}',
    summary: 'Wipe person node and all matching embedding clusters from SQLite features DB',
    permission: 'Admin',
    responseSchema: '{\n  "deleted": true,\n  "person_name": "Marcus Thorne"\n}'
  },
  {
    category: 'IDENTITY',
    method: 'PUT',
    path: '/identity/rename',
    summary: 'Modify visual alias of a physical identity block',
    permission: 'Admin',
    requestSchema: '{\n  "old_name": "Sarah Chen",\n  "new_name": "Sarah Chen-Thorne"\n}',
    responseSchema: '{\n  "renamed": true,\n  "old_name": "Sarah Chen",\n  "new_name": "Sarah Chen-Thorne"\n}'
  },
  // 1.4 Inference
  {
    category: 'INFERENCE',
    method: 'POST',
    path: '/recognize',
    summary: 'Map uploaded single frame. Performs face alignment + vector comparison',
    permission: 'All logged in',
    requestSchema: '// Multipart Form-Data:\n// file: <face_image>\n// threshold: 0.45',
    responseSchema: '{\n  "results": [{\n    "box": [140, 80, 240, 240],\n    "name": "Sarah Chen",\n    "display_name": "Sarah Chen",\n    "score": 0.984,\n    "raw_score": 0.824,\n    "accepted": true,\n    "label": "Accepted",\n    "support": 1024\n  }],\n  "annotated_image_base64": "data:image/jpeg;base64,...",\n  "matched_pairs": [{\n    "name": "Sarah Chen",\n    "gallery_face_base64": "data:image/jpeg;base64,...",\n    "query_image_base64": "data:image/jpeg;base64..."\n  }]\n}'
  },
  {
    category: 'INFERENCE',
    method: 'POST',
    path: '/recognize/video',
    summary: 'Parse full-length MP4 stream using frame downsampling and trace logs',
    permission: 'All logged in',
    requestSchema: '// Multipart Form-Data:\n// file: <video_file>\n// skip_frames: 5\n// threshold: 0.45',
    responseSchema: '{\n  "video_base64": "data:video/mp4;base64,...",\n  "output_filename": "output_processed_882.mp4",\n  "total_frames": 1200,\n  "processed_frames": 240,\n  "written_frames": 240,\n  "fps": 24,\n  "width": 1920,\n  "height": 1080,\n  "person_counts": { "Sarah Chen": 150, "Marcus Thorne": 42 },\n  "stranger_count": 8\n}'
  },
  // 1.5 Camera
  {
    category: 'CAMERA',
    method: 'POST',
    path: '/camera/start',
    summary: 'Bind video feed to active MTCNN tracker daemon',
    permission: 'All logged in',
    requestSchema: '{\n  "camera_id": 0,\n  "skip_frames": 3,\n  "threshold": 0.45,\n  "mode": "视频中找人"\n}',
    responseSchema: '{\n  "running": true,\n  "message": "Camera bound on system channel dev_0 successfully."\n}'
  },
  {
    category: 'CAMERA',
    method: 'POST',
    path: '/camera/stop',
    summary: 'Shut down active hardware sensor capture loop',
    permission: 'All logged in',
    responseSchema: '{\n  "running": false,\n  "message": "Capture handle released cleanly."\n}'
  },
  {
    category: 'CAMERA',
    method: 'GET',
    path: '/camera/status',
    summary: 'Poll latest active bounding matrices & temporal counts from sensor thread',
    permission: 'All logged in',
    responseSchema: '{\n  "running": true,\n  "results": [{\n    "box": [80, 110, 180, 180],\n    "name": "Unknown",\n    "display_name": "Subject_Unknown",\n    "score": 0.385,\n    "accepted": false,\n    "label": "Rejected"\n  }],\n  "stats": { "Sarah Chen": 124, "Marcus Thorne": 12 },\n  "total_frames": 8420,\n  "processed_frames": 2806,\n  "fps": 28.4\n}'
  },
  // 1.6 Models
  {
    category: 'MODEL',
    method: 'POST',
    path: '/model/switch',
    summary: 'Hot-swap neural architecture definitions instantly without cold starts',
    permission: 'Admin/Developer',
    requestSchema: '{\n  "model_name": "ghostnet_v4",\n  "img_size": 112,\n  "device": "cuda"\n}',
    responseSchema: '{\n  "model_name": "ghostnet_v4",\n  "weights_path": "weights/ghostnet_v4_eff.pt",\n  "img_size": 112,\n  "device": "cuda",\n  "loaded": true,\n  "gallery_size": 4129\n}'
  },
  {
    category: 'MODEL',
    method: 'GET',
    path: '/model/weights',
    summary: 'Scan host weights/ directory for valid serializations',
    permission: 'Admin/Developer',
    responseSchema: '{\n  "files": [\n    { "name": "sentinel_x_v2.pt", "path": "weights/sentinel_x_v2.pt" },\n    { "name": "ghostnet_v4_eff.pt", "path": "weights/ghostnet_v4_eff.pt" }\n  ]\n}'
  },
  {
    category: 'MODEL',
    method: 'GET',
    path: '/model/runtime',
    summary: 'Inspect device configuration state and VRAM profile',
    permission: 'Public',
    responseSchema: '{\n  "model_name": "iresnet101-Modified",\n  "weights_path": "weights/sentinel_x_v2.pt",\n  "img_size": 112,\n  "device": "cuda:0",\n  "loaded": true,\n  "gallery_size": 4129\n}'
  },
  {
    category: 'MODEL',
    method: 'POST',
    path: '/models/upload',
    summary: 'Upload brand new model definitions to deployment registry (multipart)',
    permission: 'Admin/Developer',
    requestSchema: '// FormData:\n// file: <model_binary>\n// name: "ResNet_Plus"\n// backbone: "iresnet100"\n// embedding_size: 512',
    responseSchema: '{\n  "id": 4,\n  "name": "ResNet_Plus",\n  "path": "weights/resnet_plus.pt",\n  "backbone": "iresnet100",\n  "embedding_size": 512,\n  "framework": "PyTorch 2.1",\n  "created_at": "2026-05-26 13:36:45",\n  "is_active": false\n}'
  },
  {
    category: 'MODEL',
    method: 'GET',
    path: '/models/list',
    summary: 'Fetch inventory of registered weights from SQLite system registry',
    permission: 'Admin/Developer',
    responseSchema: '{\n  "models": [\n    { "id": 1, "name": "Sentinel_X_v2.0", "is_active": true },\n    { "id": 2, "name": "Ghost_Net_v4", "is_active": false }\n  ]\n}'
  },
  {
    category: 'MODEL',
    method: 'POST',
    path: '/models/{id}/activate',
    summary: 'Toggle active flag in system registry to routing model target',
    permission: 'Admin/Developer',
    responseSchema: '{\n  "id": 2,\n  "name": "Ghost_Net_v4",\n  "backbone": "Mobile-Efficient-B3",\n  "is_active": true\n}'
  },
  {
    category: 'MODEL',
    method: 'DELETE',
    path: '/models/{id}',
    summary: 'Wipe model from local storage (cannot wipe the current active network)',
    permission: 'Admin/Developer',
    responseSchema: '{\n  "deleted": true,\n  "id": 3\n}'
  },
  // 1.7 Developer / Benchmark
  {
    category: 'DEVELOPER/EVAL',
    method: 'GET',
    path: '/developer/benchmark',
    summary: 'Analyze physical index cluster and identity volumes',
    permission: 'Admin/Developer',
    responseSchema: '{\n  "database_path": "benchmark/YTF_100p.db",\n  "person_count": 4129,\n  "embedding_count": 18240003290,\n  "top_persons": [\n    { "name": "Sarah Chen", "embedding_count": 1024 },\n    { "name": "Marcus Thorne", "embedding_count": 512 }\n  ]\n}'
  },
  {
    category: 'DEVELOPER/EVAL',
    method: 'POST',
    path: '/developer/model-eval',
    summary: 'Process system balance analysis and compile optimization tips',
    permission: 'Admin/Developer',
    responseSchema: '{\n  "runtime": { "model_name": "Sentinel_X" },\n  "gallery_size": 4129,\n  "class_balance_top": [],\n  "recommendation": "Gallery size is optimal. Recommend threshold 0.45 for balanced FP/FN margins."\n}'
  },
  // 1.8 Eval Jobs
  {
    category: 'DEVELOPER/EVAL',
    method: 'POST',
    path: '/developer/eval/lfw',
    summary: 'Launch background subprocess checking standard datasets LFW/CALFW etc',
    permission: 'Admin/Developer',
    requestSchema: '{\n  "weights_path": "weights/sentinel_x_v2.pt",\n  "backbone": "iresnet50",\n  "datasets": ["lfw", "calfw", "cplfw"],\n  "batch_size": 512\n}',
    responseSchema: '{\n  "job_id": "job_lfw_9921",\n  "message": "LFW benchmark sub-evaluator spawned successfully."\n}'
  },
  {
    category: 'DEVELOPER/EVAL',
    method: 'POST',
    path: '/developer/eval/ijb',
    summary: 'Launch long-running evaluation task over IJB-B or IJB-C target suites',
    permission: 'Admin/Developer',
    requestSchema: '{\n  "weights_path": "weights/sentinel_x_v2.pt",\n  "target": "IJBC",\n  "batch_size": 512\n}',
    responseSchema: '{\n  "job_id": "job_ijb_5561",\n  "message": "IJB evaluation worker task compiled."\n}'
  },
  {
    category: 'DEVELOPER/EVAL',
    method: 'POST',
    path: '/developer/eval/threshold-sweep',
    summary: 'Schedule automated threshold sweep scans over custom validation assets',
    permission: 'Admin/Developer',
    requestSchema: '{\n  "weights_path": "weights/sentinel_x_v2.pt",\n  "image_dir": "test_faces/",\n  "thresholds": "0.30,0.40,0.50,0.60"\n}',
    responseSchema: '{\n  "job_id": "job_sweep_3821",\n  "message": "Scanning jobs scheduled across 7 discrete increments."\n}'
  },
  {
    category: 'DEVELOPER/EVAL',
    method: 'GET',
    path: '/developer/eval/status/{job_id}',
    summary: 'Inspect ongoing subprocess output buffers, logs, and completion states',
    permission: 'Admin/Developer',
    responseSchema: '{\n  "job_id": "job_lfw_9921",\n  "job_type": "lfw_verification",\n  "status": "running",\n  "progress": 68.5,\n  "progress_msg": "Parsing CALFW pairs...",\n  "progress_data": {\n    "current": "calfw",\n    "datasets": { "lfw": { "acc": 0.9983, "threshold": 0.7 } }\n  },\n  "elapsed_seconds": 18.2\n}'
  },
  {
    category: 'DEVELOPER/EVAL',
    method: 'GET',
    path: '/developer/eval/jobs',
    summary: 'Fetch list of the 50 most recent background subprocess records',
    permission: 'Admin/Developer',
    responseSchema: '{\n  "jobs": [\n    { "job_id": "job_lfw_9921", "job_type": "lfw_verification", "status": "done" }\n  ]\n}'
  },
  {
    category: 'DEVELOPER/EVAL',
    method: 'POST',
    path: '/developer/eval/cancel/{job_id}',
    summary: 'Send SIGTERM interrupt signal to active background evaluation subprocess',
    permission: 'Admin/Developer',
    responseSchema: '{\n  "message": "Task job_lfw_9921 interrupted successfully."\n}'
  }
];
