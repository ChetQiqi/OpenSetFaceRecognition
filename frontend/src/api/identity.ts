import { apiGet, apiPost, apiDelete, apiPut } from './client';
import type {
  IdentityListResponse, AddIdentityResponse, DeleteIdentityResponse,
  RenameIdentityRequest, RenameIdentityResponse,
  UpdateIdentityRequest, UpdateIdentityResponse, PersonDetailResponse,
} from '../types';

export function listIdentities(): Promise<IdentityListResponse> {
  return apiGet<IdentityListResponse>('/identity');
}

export function addIdentity(
  personId: string,
  files: File[],
  gender?: string,
  birth_date?: string,
): Promise<AddIdentityResponse> {
  const formData = new FormData();
  formData.append('person_id', personId);
  if (gender) formData.append('gender', gender);
  if (birth_date) formData.append('birth_date', birth_date);
  files.forEach((file) => formData.append('files', file));
  return apiPost<AddIdentityResponse>('/identity/add', formData);
}

export function deleteIdentity(personName: string): Promise<DeleteIdentityResponse> {
  return apiDelete<DeleteIdentityResponse>(`/identity/${encodeURIComponent(personName)}`);
}

export function renameIdentity(oldName: string, newName: string): Promise<RenameIdentityResponse> {
  return apiPut<RenameIdentityResponse>('/identity/rename', { old_name: oldName, new_name: newName });
}

export function getIdentityDetail(personName: string): Promise<PersonDetailResponse> {
  return apiGet<PersonDetailResponse>(`/identity/${encodeURIComponent(personName)}/detail`);
}

export function updateIdentity(
  personName: string,
  data: UpdateIdentityRequest,
): Promise<UpdateIdentityResponse> {
  return apiPut<UpdateIdentityResponse>(`/identity/${encodeURIComponent(personName)}`, data);
}
