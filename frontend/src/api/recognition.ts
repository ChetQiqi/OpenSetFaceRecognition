import { apiPost } from './client';
import type { ImageRecognitionResponse, VideoRecognitionResponse } from '../types';

export function recognizeImage(file: File, threshold: number = 0.45): Promise<ImageRecognitionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('threshold', String(threshold));
  return apiPost<ImageRecognitionResponse>('/recognize', formData);
}

export function recognizeVideo(
  file: File,
  skipFrames: number = 5,
  threshold: number = 0.45,
  stableFrames: number = 3,
  mode: string = '视频中找人',
  verifyTarget?: string,
): Promise<VideoRecognitionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('skip_frames', String(skipFrames));
  formData.append('threshold', String(threshold));
  formData.append('stable_frames', String(stableFrames));
  formData.append('mode', mode);
  if (verifyTarget) {
    formData.append('verify_target', verifyTarget);
  }
  return apiPost<VideoRecognitionResponse>('/recognize/video', formData);
}
