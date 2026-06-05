import { apiGet, apiPost } from './client';
import type {
  BenchmarkSummaryResponse,
  ModelEvalResponse,
  LFWEvalRequest,
  IJBEvalRequest,
  ThresholdSweepRequest,
  EvalJobStatus,
  EvalJobSubmitResponse,
  EvalJobListResponse,
} from '../types';

export function getBenchmark(): Promise<BenchmarkSummaryResponse> {
  return apiGet<BenchmarkSummaryResponse>('/developer/benchmark');
}

export function runModelEval(topn: number = 10): Promise<ModelEvalResponse> {
  return apiPost<ModelEvalResponse>(`/developer/model-eval?topn=${topn}`);
}

export function submitLfwEval(params: LFWEvalRequest): Promise<EvalJobSubmitResponse> {
  return apiPost<EvalJobSubmitResponse>('/developer/eval/lfw', params);
}

export function submitIjbEval(params: IJBEvalRequest): Promise<EvalJobSubmitResponse> {
  return apiPost<EvalJobSubmitResponse>('/developer/eval/ijb', params);
}

export function submitThresholdSweep(params: ThresholdSweepRequest): Promise<EvalJobSubmitResponse> {
  return apiPost<EvalJobSubmitResponse>('/developer/eval/threshold-sweep', params);
}

export function getEvalStatus(jobId: string): Promise<EvalJobStatus> {
  return apiGet<EvalJobStatus>(`/developer/eval/status/${jobId}`);
}

export function listEvalJobs(): Promise<EvalJobListResponse> {
  return apiGet<EvalJobListResponse>('/developer/eval/jobs');
}

export function cancelEvalJob(jobId: string): Promise<{ message: string }> {
  return apiPost<{ message: string }>(`/developer/eval/cancel/${jobId}`);
}
