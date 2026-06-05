/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * TypeScript types matching the FastAPI Pydantic schemas in api/schemas.py
 */

// ── Auth ──
export type UserRole = 'admin' | 'developer' | 'viewer';

export interface User {
  id: number;
  username: string;
  email: string | null;
  role: UserRole;
  is_active: boolean;
  created_at: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  role: UserRole;
}

export interface RegisterRequest {
  username: string;
  password: string;
  email?: string | null;
  role?: UserRole;
}

export interface UpdateUserRoleRequest {
  role: UserRole;
}

// ── Core / Stats ──
export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

export interface StatsResponse {
  person_count: number;
  embedding_count: number;
}

// ── Identity ──
export interface IdentityItem {
  name: string;
  embedding_count: number;
  gender: string;
  birth_date: string;
}

export interface IdentityListResponse {
  persons: IdentityItem[];
}

export interface AddIdentityResponse {
  person_name: string;
  success_count: number;
  fail_count: number;
  success_files: string[];
  failed_files: { filename: string; reason: string }[];
}

export interface DeleteIdentityResponse {
  deleted: boolean;
  person_name: string;
}

export interface RenameIdentityRequest {
  old_name: string;
  new_name: string;
}

export interface RenameIdentityResponse {
  renamed: boolean;
  old_name: string;
  new_name: string;
}

export interface UpdateIdentityRequest {
  gender?: string | null;
  birth_date?: string | null;
  new_name?: string | null;
}

export interface UpdateIdentityResponse {
  updated: boolean;
  old_name: string;
  new_name: string;
}

export interface PersonDetailResponse {
  name: string;
  embedding_count: number;
  gender: string;
  birth_date: string;
}

// ── Recognition ──
export interface RecognitionResult {
  box: number[];
  name: string;
  display_name: string;
  score: number;
  raw_score: number;
  accepted: boolean;
  label: string;
  support: number;
}

export interface MatchedPair {
  name: string;
  query_image_base64?: string;
  gallery_face_base64?: string;
  gallery_error?: string;
}

export interface PersonCard {
  name: string;
  gender: string;
  birth_date: string;
  embedding_count: number;
  gallery_face_base64: string | null;
  score: number;
  accepted: boolean;
}

export interface ImageRecognitionResponse {
  results: RecognitionResult[];
  annotated_image_base64: string | null;
  matched_pairs: MatchedPair[];
  person_cards: PersonCard[];
}

export interface VerifyResult {
  target: string;
  passed: boolean;
  frames: number;
}

export interface VideoRecognitionResponse {
  video_base64: string;
  output_filename: string;
  total_frames: number;
  processed_frames: number;
  written_frames: number;
  fps: number;
  width: number;
  height: number;
  person_counts: Record<string, number>;
  stranger_count: number;
  verify_result: VerifyResult | null;
  person_cards: PersonCard[];
}

// ── Camera ──
export interface CameraStartRequest {
  camera_id: number;
  skip_frames: number;
  threshold: number;
  stable_frames: number;
  mode: string;
  verify_target?: string | null;
}

export interface CameraActionResponse {
  running: boolean;
  message: string;
}

export interface CameraStatusResponse {
  running: boolean;
  results: RecognitionResult[];
  stats: Record<string, number>;
  total_frames: number;
  processed_frames: number;
  fps: number;
  verify_status: string | null;
  verify_count: number;
  person_cards: PersonCard[];
}

// ── Model Management ──
export interface ModelSwitchRequest {
  model_name: string;
  weights_path?: string | null;
  img_size?: number | null;
  device?: string | null;
}

export interface ModelRuntimeInfoResponse {
  model_name: string;
  weights_path: string;
  framework?: string;
  img_size: number;
  device: string;
  loaded: boolean;
  gallery_size: number;
}

export interface ModelWeightItem {
  name: string;
  path: string;
}

export interface ModelWeightsResponse {
  files: ModelWeightItem[];
}

export interface ModelItem {
  id: number;
  name: string;
  path: string;
  backbone: string;
  embedding_size: number;
  framework: string;
  created_at: string;
  is_active: boolean;
}

export interface ModelListResponse {
  models: ModelItem[];
}

export type ModelActivateResponse = ModelItem;

export interface ModelDeleteResponse {
  deleted: boolean;
  id: number;
  reason?: string | null;
}

// ── Developer / Benchmark ──
export interface BenchmarkPersonItem {
  name: string;
  embedding_count: number;
}

export interface BenchmarkSummaryResponse {
  database_path: string;
  person_count: number;
  embedding_count: number;
  top_persons: BenchmarkPersonItem[];
}

export interface ModelEvalResponse {
  runtime: ModelRuntimeInfoResponse;
  gallery_size: number;
  class_balance_top: BenchmarkPersonItem[];
  class_balance_bottom: BenchmarkPersonItem[];
  recommendation: string;
}

// ── Eval Jobs ──
export interface LFWEvalRequest {
  weights_path: string;
  backbone: string;
  data_root: string;
  datasets?: string[];
  batch_size: number;
}

export interface IJBEvalRequest {
  weights_path: string;
  backbone: string;
  image_path: string;
  target: string;
  batch_size: number;
  use_norm_score: boolean;
  use_detector_score: boolean;
  use_flip_test: boolean;
  result_dir: string;
}

export interface ThresholdSweepRequest {
  weights_path: string;
  backbone: string;
  image_dir: string;
  db_path: string;
  thresholds: string;
  device: string;
}

export interface EvalJobStatus {
  job_id: string;
  job_type: string;
  status: 'pending' | 'running' | 'done' | 'error';
  model_name: string;
  weights_path: string;
  params: Record<string, unknown>;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  progress: number;
  progress_msg: string;
  progress_data: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string | null;
  elapsed_seconds: number;
}

export interface EvalJobSubmitResponse {
  job_id: string;
  message: string;
}

export interface EvalJobListResponse {
  jobs: EvalJobStatus[];
}

// ── API Simulator (frontend-only documentation types) ──

// ── AI Portrait Generation ──
export interface PortraitStyleItem {
  key: string;
  label: string;
}

export interface PortraitStyleListResponse {
  styles: PortraitStyleItem[];
}

export interface PortraitPersonItem {
  name: string;
  image_paths: string[];
}

export interface PortraitPersonsResponse {
  persons: PortraitPersonItem[];
}

export interface PortraitPersonImageResponse {
  name: string;
  image_path: string;
  image_base64: string;
  mime: string;
}

export interface PortraitGenerateRequest {
  person_name: string;
  image_path: string;
  style: string;
  seed?: number | null;
  num_inference_steps?: number | null;
  guidance_scale?: number | null;
  start_merge_step?: number | null;
}

export interface PortraitGenerateResponse {
  style: string;
  style_label: string;
  reference_image_path: string;
  result_image_base64: string;
  output_path: string;
  prompt_used: string;
  seed: number;
  generation_time_seconds: number;
  width: number;
  height: number;
  message: string;
}

export interface PortraitStatusResponse {
  loaded: boolean;
  device: string;
  available_styles: PortraitStyleItem[];
}
export interface ApiEndpoint {
  category: 'AUTH' | 'CORE/STATS' | 'IDENTITY' | 'INFERENCE' | 'CAMERA' | 'MODEL' | 'DEVELOPER/EVAL';
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  summary: string;
  permission: string;
  requestSchema?: string;
  responseSchema?: string;
}

// ── Frontend-only types ──
export interface LogLine {
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'ERROR' | 'RESULT' | 'INPUT_STREAM' | 'AI_PROC';
  message: string;
}
