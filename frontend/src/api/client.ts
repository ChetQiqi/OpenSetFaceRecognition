/// <reference types="vite/client" />

/**
 * HTTP client wrapper for the FastAPI backend.
 * Automatically attaches Bearer token and handles errors.
 */

const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem('access_token');
  if (token) {
    return { Authorization: `Bearer ${token}` };
  }
  return {};
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // not JSON
    }
    throw new ApiError(response.status, detail);
  }
  const text = await response.text();
  if (!text) return {} as T;
  return JSON.parse(text) as T;
}

export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'GET',
    headers: { ...getAuthHeaders() },
    signal,
  });
  return handleResponse<T>(response);
}

export async function apiPost<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const isFormData = body instanceof FormData;
  const headers: Record<string, string> = { ...getAuthHeaders() };
  if (!isFormData) {
    headers['Content-Type'] = 'application/json';
  }
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers,
    body: isFormData ? (body as FormData) : JSON.stringify(body),
    signal,
  });
  return handleResponse<T>(response);
}

export async function apiPut<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'PUT',
    headers: { ...getAuthHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  return handleResponse<T>(response);
}

export async function apiDelete<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'DELETE',
    headers: { ...getAuthHeaders() },
    signal,
  });
  return handleResponse<T>(response);
}

export async function apiUpload<T>(path: string, formData: FormData, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { ...getAuthHeaders() },
    body: formData,
    signal,
  });
  return handleResponse<T>(response);
}

/**
 * Upload with real progress tracking via XMLHttpRequest.
 * Calls `onProgress` with 0–100 as the upload proceeds.
 */
export function apiUploadWithProgress<T>(
  path: string,
  formData: FormData,
  onProgress: (pct: number) => void,
  signal?: AbortSignal,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}${path}`);

    const headers = getAuthHeaders();
    Object.entries(headers).forEach(([k, v]) => xhr.setRequestHeader(k, v));

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    if (signal) {
      signal.addEventListener('abort', () => xhr.abort());
    }

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(xhr.responseText ? JSON.parse(xhr.responseText) as T : {} as T);
        } catch {
          resolve({} as T);
        }
      } else {
        let detail = xhr.statusText;
        try { detail = JSON.parse(xhr.responseText).detail ?? detail; } catch {}
        reject(new ApiError(xhr.status, detail));
      }
    });

    xhr.addEventListener('error', () => reject(new ApiError(0, 'Network error')));
    xhr.addEventListener('abort', () => reject(new ApiError(0, 'Upload cancelled')));

    xhr.send(formData);
  });
}
