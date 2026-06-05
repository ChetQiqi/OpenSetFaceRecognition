import { apiGet, apiPost, apiDelete, apiUpload, apiUploadWithProgress } from './client';
import type {
  ModelRuntimeInfoResponse,
  ModelWeightsResponse,
  ModelSwitchRequest,
  ModelListResponse,
  ModelActivateResponse,
  ModelDeleteResponse,
} from '../types';

export function getRuntime(): Promise<ModelRuntimeInfoResponse> {
  return apiGet<ModelRuntimeInfoResponse>('/model/runtime');
}

export function getWeights(): Promise<ModelWeightsResponse> {
  return apiGet<ModelWeightsResponse>('/model/weights');
}

export function loadModel(): Promise<{ status: string; model_loaded: boolean }> {
  return apiPost<{ status: string; model_loaded: boolean }>('/model/load');
}

export function switchModel(params: ModelSwitchRequest): Promise<ModelRuntimeInfoResponse> {
  return apiPost<ModelRuntimeInfoResponse>('/model/switch', params);
}

export function listModels(): Promise<ModelListResponse> {
  return apiGet<ModelListResponse>('/models/list');
}

export function uploadModel(
  file: File,
  name: string,
  backbone: string = 'iresnet50',
  embeddingSize: number = 512,
): Promise<ModelActivateResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('name', name);
  formData.append('backbone', backbone);
  formData.append('embedding_size', String(embeddingSize));
  return apiUpload<ModelActivateResponse>('/models/upload', formData);
}

export function uploadModelWithProgress(
  file: File,
  name: string,
  backbone: string,
  embeddingSize: number,
  onProgress: (pct: number) => void,
): Promise<ModelActivateResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('name', name);
  formData.append('backbone', backbone);
  formData.append('embedding_size', String(embeddingSize));
  return apiUploadWithProgress<ModelActivateResponse>('/models/upload', formData, onProgress);
}

export function activateModel(modelId: number): Promise<ModelActivateResponse> {
  return apiPost<ModelActivateResponse>(`/models/${modelId}/activate`);
}

export function deleteModel(modelId: number): Promise<ModelDeleteResponse> {
  return apiDelete<ModelDeleteResponse>(`/models/${modelId}`);
}
