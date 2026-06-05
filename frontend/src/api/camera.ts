import { apiGet, apiPost } from './client';
import type { CameraActionResponse, CameraStatusResponse, CameraStartRequest } from '../types';

export function startCamera(params: CameraStartRequest): Promise<CameraActionResponse> {
  return apiPost<CameraActionResponse>('/camera/start', params);
}

export function stopCamera(): Promise<CameraActionResponse> {
  return apiPost<CameraActionResponse>('/camera/stop');
}

export function clearCamera(): Promise<CameraActionResponse> {
  return apiPost<CameraActionResponse>('/camera/clear');
}

export function getCameraStatus(): Promise<CameraStatusResponse> {
  return apiGet<CameraStatusResponse>('/camera/status');
}
