/**
 * AI Portrait Generation API
 *
 * 对应后端端点:
 *   GET  /portrait/styles
 *   GET  /portrait/persons
 *   GET  /portrait/person/{name}/image
 *   POST /portrait/generate
 *   POST /portrait/unload
 *   GET  /portrait/status
 */

import { apiGet, apiPost } from './client';
import type {
  PortraitStyleListResponse,
  PortraitPersonsResponse,
  PortraitPersonImageResponse,
  PortraitGenerateRequest,
  PortraitGenerateResponse,
  PortraitStatusResponse,
} from '../types';

/** 获取可用的风格列表 */
export function getStyles(): Promise<PortraitStyleListResponse> {
  return apiGet<PortraitStyleListResponse>('/portrait/styles');
}

/** 获取已注册人员及其照片信息 */
export function getPersons(): Promise<PortraitPersonsResponse> {
  return apiGet<PortraitPersonsResponse>('/portrait/persons');
}

/** 获取指定人员的原始照片 (base64) */
export function getPersonImage(name: string): Promise<PortraitPersonImageResponse> {
  return apiGet<PortraitPersonImageResponse>(`/portrait/person/${encodeURIComponent(name)}/image`);
}

/** 提交 AI 肖像生成请求 */
export function generate(payload: PortraitGenerateRequest): Promise<PortraitGenerateResponse> {
  return apiPost<PortraitGenerateResponse>('/portrait/generate', payload);
}

/** 卸载肖像生成模型以释放 GPU 显存 */
export function unloadModel(): Promise<{ message: string }> {
  return apiPost<{ message: string }>('/portrait/unload');
}

/** 查询 PortraitService 加载状态 */
export function getStatus(): Promise<PortraitStatusResponse> {
  return apiGet<PortraitStatusResponse>('/portrait/status');
}
