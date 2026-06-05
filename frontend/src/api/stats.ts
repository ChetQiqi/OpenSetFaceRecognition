import { apiGet } from './client';
import type { HealthResponse, StatsResponse } from '../types';

export function getHealth(): Promise<HealthResponse> {
  return apiGet<HealthResponse>('/health');
}

export function getStats(): Promise<StatsResponse> {
  return apiGet<StatsResponse>('/stats');
}
