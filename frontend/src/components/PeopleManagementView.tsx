/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Plus,
  Trash2,
  Upload,
  Database,
  RefreshCw,
  AlertTriangle,
  User,
  Pencil,
  Search,
  X,
} from 'lucide-react';
import { LogLine, IdentityItem } from '../types';
import { Language, LOCALES } from '../locales';
import { listIdentities, addIdentity, deleteIdentity, renameIdentity, getIdentityDetail, updateIdentity } from '../api/identity';
import { ApiError } from '../api/client';

interface PeopleManagementViewProps {
  addLog: (level: LogLine['level'], msg: string) => void;
  language?: Language;
}

function getInitialsColor(name: string): string {
  const hue = name.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0) % 360;
  return `hsl(${hue}, 55%, 40%)`;
}

export default function PeopleManagementView({
  addLog,
  language = 'en'
}: PeopleManagementViewProps) {
  const [peopleList, setPeopleList] = useState<IdentityItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeSubTab, setActiveSubTab] = useState<'list' | 'register' | 'add-embeddings' | 'edit-metadata' | 'delete'>('list');
  const [searchQuery, setSearchQuery] = useState('');
  const t = LOCALES[language];

  const filteredPeople = searchQuery.trim()
    ? peopleList.filter(p => p.name.toLowerCase().includes(searchQuery.toLowerCase().trim()))
    : peopleList;

  // Register form state
  const [newName, setNewName] = useState('');
  const [newGender, setNewGender] = useState<string>('unspecified');
  const [newBirthDate, setNewBirthDate] = useState('');
  const [enrollmentFiles, setEnrollmentFiles] = useState<FileList | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // Add embeddings form state
  const [embeddingsPerson, setEmbeddingsPerson] = useState('');
  const [embeddingsFiles, setEmbeddingsFiles] = useState<FileList | null>(null);
  const [isAddingEmbeddings, setIsAddingEmbeddings] = useState(false);

  // Edit metadata form state
  const [editPerson, setEditPerson] = useState('');
  const [editGender, setEditGender] = useState('unspecified');
  const [editBirthDate, setEditBirthDate] = useState('');
  const [editNewName, setEditNewName] = useState('');
  const [isSavingMetadata, setIsSavingMetadata] = useState(false);

  // Rename form state (kept for legacy, used internally by edit-metadata)
  const [renameTarget, setRenameTarget] = useState<string>('');
  const [renameNewName, setRenameNewName] = useState('');

  const fetchPeople = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await listIdentities();
      setPeopleList(data.persons);
      addLog('INFO', `IDENTITY_DB: Loaded ${data.persons.length} registered subjects.`);
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Failed to load identities';
      setError(msg);
      addLog('ERROR', `IDENTITY_DB: ${msg}`);
    } finally {
      setIsLoading(false);
    }
  }, [addLog]);

  useEffect(() => {
    fetchPeople();
  }, [fetchPeople]);

  const handleEnrollSubject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;
    if (!enrollmentFiles || enrollmentFiles.length === 0) {
      setSubmitError(language === 'zh' ? '请选择至少一张照片' : 'Please select at least one photo');
      return;
    }

    setIsSubmitting(true);
    setSubmitError(null);
    try {
      const files: File[] = [];
      for (let i = 0; i < enrollmentFiles.length; i++) {
        files.push(enrollmentFiles[i]);
      }
      const result = await addIdentity(newName, files, newGender, newBirthDate);
      addLog('RESULT', `POST /identity/add: ${result.person_name} — ${result.success_count} success, ${result.fail_count} failed`);
      setNewName('');
      setNewGender('unspecified');
      setNewBirthDate('');
      setEnrollmentFiles(null);
      setSubmitError(null);
      setActiveSubTab('list');
      await fetchPeople();
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Enrollment failed';
      setSubmitError(msg);
      addLog('ERROR', `IDENTITY_DB: ${msg}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDeleteSubject = async (name: string) => {
    try {
      await deleteIdentity(name);
      addLog('WARNING', `DELETE /identity/${name}: Entity expunged.`);
      setPeopleList(prev => prev.filter(p => p.name !== name));
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Delete failed';
      addLog('ERROR', `DELETE /identity/${name}: ${msg}`);
    }
  };

  const loadPersonDetail = useCallback(async (personName: string) => {
    try {
      const detail = await getIdentityDetail(personName);
      setEditGender(detail.gender || 'unspecified');
      setEditBirthDate(detail.birth_date || '');
      setEditNewName(detail.name);
    } catch {
      setEditGender('unspecified');
      setEditBirthDate('');
      setEditNewName(personName);
    }
  }, []);

  const handleAddEmbeddings = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!embeddingsPerson) return;
    if (!embeddingsFiles || embeddingsFiles.length === 0) {
      setSubmitError(t.people.noPhotosSelected);
      return;
    }
    setIsAddingEmbeddings(true);
    setSubmitError(null);
    try {
      const files: File[] = [];
      for (let i = 0; i < embeddingsFiles.length; i++) files.push(embeddingsFiles[i]);
      const result = await addIdentity(embeddingsPerson, files);
      addLog('RESULT', `POST /identity/add (embeddings): ${result.person_name} — ${result.success_count} success, ${result.fail_count} failed`);
      setEmbeddingsFiles(null);
      setActiveSubTab('list');
      await fetchPeople();
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Failed to add embeddings';
      setSubmitError(msg);
      addLog('ERROR', `IDENTITY_DB: ${msg}`);
    } finally {
      setIsAddingEmbeddings(false);
    }
  };

  const handleSaveMetadata = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editPerson) return;
    setIsSavingMetadata(true);
    try {
      const result = await updateIdentity(editPerson, {
        gender: editGender,
        birth_date: editBirthDate,
        new_name: editNewName || null,
      });
      addLog('INFO', `PUT /identity/${result.old_name}: ${result.updated ? 'updated' : 'not found'}${result.new_name !== result.old_name ? `, renamed to ${result.new_name}` : ''}`);
      setEditPerson('');
      setActiveSubTab('list');
      await fetchPeople();
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Failed to update metadata';
      setSubmitError(msg);
      addLog('ERROR', `IDENTITY_DB: ${msg}`);
    } finally {
      setIsSavingMetadata(false);
    }
  };

  const handleRenameSubject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!renameTarget || !renameNewName.trim()) return;
    try {
      const result = await renameIdentity(renameTarget, renameNewName);
      addLog('INFO', `PUT /identity/rename: ${result.old_name} → ${result.new_name}`);
      setRenameTarget('');
      setRenameNewName('');
      await fetchPeople();
    } catch (err) {
      const msg = err instanceof ApiError ? err.detail : 'Rename failed';
      addLog('ERROR', `PUT /identity/rename: ${msg}`);
    }
  };

  return (
    <div className="p-6 flex flex-col overflow-y-auto h-[calc(100vh-4rem)] bg-[#090b11] text-slate-300 font-sans select-none">
      <div className="space-y-6">
        {/* Title bar */}
        <div className="flex items-center justify-between border-b border-[#212733] pb-4">
          <div className="flex items-center gap-3">
            <h3 className="text-lg font-bold text-white font-mono tracking-wide flex items-center gap-2">
              <Database className="w-5 h-5 text-[#06b6d4]" />
              {t.people.title}
            </h3>
            <span className="text-[9px] font-mono text-cyan-400 border border-cyan-500/20 bg-cyan-950/20 px-2.5 py-1 rounded-sm tracking-widest font-extrabold">
              {peopleList.length.toLocaleString()} {language === 'zh' ? '人' : 'persons'}
            </span>
          </div>
          <button
            onClick={fetchPeople}
            disabled={isLoading}
            className="flex items-center gap-1 px-3 py-1.5 text-[10px] font-mono text-slate-400 hover:text-white bg-[#1e2533] hover:bg-[#293245] rounded-sm transition-colors cursor-pointer"
          >
            <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} />
            {language === 'zh' ? '刷新' : 'Refresh'}
          </button>
        </div>

        {/* Tab bar */}
        <div className="flex gap-1.5 border-b border-[#1f2533] pb-px">
          {[
            ['list', language === 'zh' ? '人员列表' : 'Identity List'],
            ['register', t.people.enrollBtn],
            ['add-embeddings', t.people.addEmbeddings],
            ['edit-metadata', t.people.editMetadata],
            ['delete', t.people.deleteBtn],
          ].map(([id, label]) => (
            <button
              key={id}
              onClick={() => setActiveSubTab(id as typeof activeSubTab)}
              className={`px-5 py-2 text-xs font-mono tracking-wider font-bold transition-all relative cursor-pointer ${
                activeSubTab === id
                  ? 'text-[#22d3ee] border-b-2 border-[#06b6d4] font-black'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Loading state */}
        {isLoading && (
          <div className="flex items-center justify-center py-20">
            <div className="flex flex-col items-center gap-3">
              <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin" />
              <p className="text-xs text-slate-500 font-mono uppercase tracking-wider">
                {language === 'zh' ? '正在加载...' : 'Loading identity database...'}
              </p>
            </div>
          </div>
        )}

        {/* Error state */}
        {!isLoading && error && (
          <div className="flex items-center justify-center py-20">
            <div className="flex flex-col items-center gap-3 text-center max-w-md">
              <AlertTriangle className="w-8 h-8 text-red-400" />
              <p className="text-xs text-red-400 font-mono">{error}</p>
              <button
                onClick={fetchPeople}
                className="px-4 py-2 bg-red-950/30 border border-red-500/30 text-red-300 hover:text-white rounded-sm text-xs font-mono cursor-pointer transition-colors"
              >
                {language === 'zh' ? '重试' : 'Retry'}
              </button>
            </div>
          </div>
        )}

        {/* Content */}
        {!isLoading && !error && (
          <>
            {/* Subtab 1: Identity Cards */}
            {activeSubTab === 'list' && (
              <>
                {/* Search bar */}
                {peopleList.length > 0 && (
                  <div className="relative max-w-md">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      placeholder={t.people.searchPlaceholder}
                      className="w-full bg-[#121622] border border-[#1f2533] focus:border-cyan-500 text-slate-200 pl-9 pr-8 py-2 rounded-sm text-xs outline-none font-mono transition-colors"
                    />
                    {searchQuery && (
                      <button
                        onClick={() => setSearchQuery('')}
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white cursor-pointer"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                )}

                {peopleList.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-20 text-slate-500 gap-3">
                    <User className="w-12 h-12" />
                    <p className="text-xs font-mono uppercase tracking-wider">
                      {language === 'zh' ? '暂无登记人员' : 'No identities registered'}
                    </p>
                    <button
                      onClick={() => setActiveSubTab('register')}
                      className="px-4 py-2 bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 rounded-sm text-xs font-mono cursor-pointer hover:bg-cyan-500/20 transition-colors"
                    >
                      {language === 'zh' ? '注册第一个人员' : 'Register first identity'}
                    </button>
                  </div>
                ) : filteredPeople.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-20 text-slate-500 gap-3">
                    <Search className="w-10 h-10 text-slate-600" />
                    <p className="text-xs font-mono uppercase tracking-wider">
                      {language === 'zh'
                        ? `未找到匹配 "${searchQuery}" 的人员`
                        : `No identities matching "${searchQuery}"`}
                    </p>
                    <button
                      onClick={() => setSearchQuery('')}
                      className="px-4 py-2 bg-[#1e2533] hover:bg-[#293245] text-slate-400 hover:text-white rounded-sm text-xs font-mono cursor-pointer transition-colors"
                    >
                      {language === 'zh' ? '清除搜索' : 'Clear search'}
                    </button>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
                    {filteredPeople.map((person) => (
                      <div
                        key={person.name}
                        className="bg-[#121622] rounded-sm p-5 border border-[#1f2533] hover:border-cyan-500/30 flex flex-col justify-between group transition-all duration-300 h-[240px]"
                      >
                        {/* Avatar + name */}
                        <div className="flex items-start gap-3">
                          <div
                            className="w-12 h-12 rounded-sm flex items-center justify-center text-white font-bold text-lg border border-slate-700 shrink-0"
                            style={{ backgroundColor: getInitialsColor(person.name) }}
                          >
                            {person.name.charAt(0).toUpperCase()}
                          </div>
                          <div className="min-w-0 flex-1">
                            <h4 className="text-sm font-bold text-white tracking-tight group-hover:text-cyan-400 transition-colors truncate">
                              {person.name}
                            </h4>
                            <p className="text-[10px] text-slate-500 font-mono mt-0.5">
                              {language === 'zh' ? '特征向量' : 'Embeddings'}: {person.embedding_count}
                            </p>
                            <p className="text-[10px] text-slate-500 font-mono mt-0.5">
                              {t.people.lblGender}: {person.gender === 'male' ? t.people.genderMale : person.gender === 'female' ? t.people.genderFemale : t.people.genderUnspecified}
                              {person.birth_date ? `  |  ${person.birth_date}` : ''}
                            </p>
                          </div>
                        </div>

                        {/* Stats */}
                        <div className="border-t border-[#1e2330] pt-4 space-y-2 font-mono text-[11px]">
                          <div className="flex justify-between">
                            <span className="text-slate-500 uppercase">
                              {language === 'zh' ? '嵌入维度' : 'Vectors'}
                            </span>
                            <span className="text-white font-bold">{person.embedding_count.toLocaleString()}</span>
                          </div>
                        </div>

                        {/* Quick action */}
                        <button
                          onClick={() => { setEditPerson(person.name); loadPersonDetail(person.name); setActiveSubTab('edit-metadata'); }}
                          className="mt-3 w-full bg-[#1e2533] hover:bg-[#2e374d] text-xs font-mono uppercase font-bold tracking-wider py-2 text-slate-300 rounded-sm cursor-pointer select-none transition-colors flex items-center justify-center gap-1"
                        >
                          <Pencil className="w-3 h-3" />
                          {t.people.editBtn}
                        </button>
                      </div>
                    ))}

                    {/* Add new card */}
                    <button
                      onClick={() => setActiveSubTab('register')}
                      className="bg-[#0c0e14] rounded-sm p-5 border-2 border-dashed border-[#1f2533] hover:border-cyan-500/40 hover:bg-cyan-500/5 cursor-pointer flex flex-col items-center justify-center text-center gap-3 transition-all h-[240px] group select-none"
                    >
                      <div className="w-10 h-10 rounded-full bg-[#121622] border border-slate-700 flex items-center justify-center text-slate-400 group-hover:bg-cyan-500 group-hover:border-cyan-400 group-hover:text-slate-950 transition-all">
                        <Plus className="w-5 h-5" />
                      </div>
                      <p className="text-xs font-bold text-slate-400 font-mono uppercase tracking-wider group-hover:text-cyan-400 transition-colors">
                        {t.people.enrollBtn}
                      </p>
                    </button>
                  </div>
                )}
              </>
            )}

            {/* Subtab 2: Register */}
            {activeSubTab === 'register' && (
              <div className="max-w-xl mx-auto bg-[#121622] border border-[#1f2533] rounded-sm p-6 space-y-6">
                <div className="border-b border-[#212733] pb-3">
                  <h4 className="text-xs font-bold font-mono text-white uppercase tracking-wider">
                    {t.people.modalTitle}
                  </h4>
                </div>

                {submitError && (
                  <div className="bg-red-950/30 border border-red-500/30 rounded-sm p-3 flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                    <p className="text-xs text-red-300 font-mono">{submitError}</p>
                  </div>
                )}

                <form onSubmit={handleEnrollSubject} className="space-y-4 text-xs font-mono">
                  <div className="space-y-2">
                    <label className="text-slate-400 uppercase font-bold">
                      {language === 'zh' ? '人员名称' : 'Person Name'}
                    </label>
                    <input
                      type="text"
                      required
                      placeholder={language === 'zh' ? '例如：张三' : 'e.g. John Doe'}
                      value={newName}
                      onChange={(e) => setNewName(e.target.value)}
                      className="w-full bg-[#0c0e14] border border-[#1f2533] focus:border-[#0891b2] text-slate-200 px-3 py-2 rounded-sm outline-none transition-colors"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-slate-400 uppercase font-bold">{t.people.lblGender}</label>
                    <select
                      value={newGender}
                      onChange={(e) => setNewGender(e.target.value)}
                      className="w-full bg-[#0c0e14] border border-[#1f2533] focus:border-[#0891b2] text-slate-200 px-3 py-2 rounded-sm outline-none transition-colors cursor-pointer"
                    >
                      <option value="unspecified">{t.people.genderUnspecified}</option>
                      <option value="male">{t.people.genderMale}</option>
                      <option value="female">{t.people.genderFemale}</option>
                    </select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-slate-400 uppercase font-bold">{t.people.lblBirthDate}</label>
                    <input
                      type="text"
                      placeholder="YYYY-MM"
                      value={newBirthDate}
                      onChange={(e) => setNewBirthDate(e.target.value)}
                      maxLength={7}
                      className="w-full bg-[#0c0e14] border border-[#1f2533] focus:border-[#0891b2] text-slate-200 px-3 py-2 rounded-sm outline-none transition-colors"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-slate-400 uppercase font-bold">
                      {language === 'zh' ? '人脸照片 (.jpg / .png)' : 'Face Photos (.jpg / .png)'}
                    </label>
                    <div className="border border-dashed border-[#1f2533] bg-[#0c0e14] p-4 text-center rounded-sm relative">
                      <input
                        type="file"
                        accept="image/*"
                        multiple
                        onChange={(e) => setEnrollmentFiles(e.target.files)}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                      />
                      <div className="space-y-1 py-1 pointer-events-none">
                        <Upload className="w-5 h-5 mx-auto text-slate-500" />
                        <p className="text-slate-500 text-[10px] uppercase">
                          {enrollmentFiles && enrollmentFiles.length > 0
                            ? `${enrollmentFiles.length} file(s) selected`
                            : language === 'zh'
                              ? '拖曳或点击选择照片'
                              : 'Drag-and-drop or click to select'}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="pt-4 flex gap-3">
                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className="flex-1 py-3 bg-cyan-500 hover:bg-cyan-600 disabled:bg-cyan-800 disabled:cursor-not-allowed font-bold uppercase text-slate-950 font-mono tracking-wider rounded-sm cursor-pointer transition-colors shadow-lg shadow-cyan-500/10"
                    >
                      {isSubmitting ? (
                        <span className="flex items-center justify-center gap-2">
                          <RefreshCw className="w-3 h-3 animate-spin" />
                          {language === 'zh' ? '注册中...' : 'Enrolling...'}
                        </span>
                      ) : (
                        t.people.modalExecute
                      )}
                    </button>
                    <button
                      type="button"
                      onClick={() => { setActiveSubTab('list'); setSubmitError(null); }}
                      className="w-1/3 py-3 bg-[#1e2533] hover:bg-[#293245] font-bold uppercase text-slate-300 rounded-sm cursor-pointer transition-colors"
                    >
                      {t.people.modalCancel}
                    </button>
                  </div>
                </form>
              </div>
            )}

            {/* Subtab 3: Add Embeddings */}
            {activeSubTab === 'add-embeddings' && (
              <div className="max-w-xl mx-auto bg-[#121622] border border-[#1f2533] rounded-sm p-6 space-y-6">
                <div className="border-b border-[#212733] pb-3">
                  <h4 className="text-xs font-bold font-mono text-white uppercase tracking-wider">
                    {t.people.addEmbeddings}
                  </h4>
                </div>
                {submitError && (
                  <div className="bg-red-950/30 border border-red-500/30 rounded-sm p-3 flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                    <p className="text-xs text-red-300 font-mono">{submitError}</p>
                  </div>
                )}
                <form onSubmit={handleAddEmbeddings} className="space-y-4 text-xs font-mono">
                  <div className="space-y-2">
                    <label className="text-slate-400 uppercase font-bold">{t.people.selectPerson}</label>
                    <select
                      value={embeddingsPerson}
                      onChange={(e) => setEmbeddingsPerson(e.target.value)}
                      className="w-full bg-[#0c0e14] border border-[#1f2533] focus:border-[#0891b2] text-slate-200 px-3 py-2 rounded-sm outline-none transition-colors cursor-pointer"
                    >
                      <option value="">--</option>
                      {peopleList.map((p) => (
                        <option key={p.name} value={p.name}>{p.name} ({p.embedding_count} embeddings)</option>
                      ))}
                    </select>
                  </div>
                  <div className="border border-dashed border-[#1f2533] bg-[#0c0e14] p-4 text-center rounded-sm relative">
                    <input
                      type="file" accept="image/*" multiple
                      onChange={(e) => setEmbeddingsFiles(e.target.files)}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    />
                    <div className="space-y-1 py-1 pointer-events-none">
                      <Upload className="w-5 h-5 mx-auto text-slate-500" />
                      <p className="text-slate-500 text-[10px] uppercase">
                        {embeddingsFiles && embeddingsFiles.length > 0
                          ? `${embeddingsFiles.length} file(s) selected`
                          : language === 'zh' ? '拖曳或点击选择新照片' : 'Drop or click to add photos'}
                      </p>
                    </div>
                  </div>
                  <div className="pt-4 flex gap-3">
                    <button
                      type="submit" disabled={isAddingEmbeddings}
                      className="flex-1 py-3 bg-cyan-500 hover:bg-cyan-600 disabled:bg-cyan-800 disabled:cursor-not-allowed font-bold uppercase text-slate-950 font-mono tracking-wider rounded-sm cursor-pointer transition-colors"
                    >
                      {isAddingEmbeddings ? t.people.addingEmbeddings : t.people.btnAddEmbeddings}
                    </button>
                    <button
                      type="button"
                      onClick={() => { setActiveSubTab('list'); setSubmitError(null); }}
                      className="w-1/3 py-3 bg-[#1e2533] hover:bg-[#293245] font-bold uppercase text-slate-300 rounded-sm cursor-pointer transition-colors"
                    >
                      {t.people.modalCancel}
                    </button>
                  </div>
                </form>
              </div>
            )}

            {/* Subtab 4: Edit Metadata */}
            {activeSubTab === 'edit-metadata' && (
              <div className="max-w-xl mx-auto bg-[#121622] border border-[#1f2533] rounded-sm p-6 space-y-6">
                <div className="border-b border-[#212733] pb-3">
                  <h4 className="text-xs font-bold font-mono text-white uppercase tracking-wider">
                    {t.people.editMetadataTitle}
                  </h4>
                </div>
                {submitError && (
                  <div className="bg-red-950/30 border border-red-500/30 rounded-sm p-3 flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                    <p className="text-xs text-red-300 font-mono">{submitError}</p>
                  </div>
                )}
                <form onSubmit={handleSaveMetadata} className="space-y-4 text-xs font-mono">
                  <div className="space-y-2">
                    <label className="text-slate-400 uppercase font-bold">{t.people.selectPerson}</label>
                    <select
                      value={editPerson}
                      onChange={(e) => { setEditPerson(e.target.value); if (e.target.value) loadPersonDetail(e.target.value); }}
                      className="w-full bg-[#0c0e14] border border-[#1f2533] focus:border-[#0891b2] text-slate-200 px-3 py-2 rounded-sm outline-none transition-colors cursor-pointer"
                    >
                      <option value="">--</option>
                      {peopleList.map((p) => (
                        <option key={p.name} value={p.name}>{p.name}</option>
                      ))}
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-slate-400 uppercase font-bold">{t.people.lblNewName}</label>
                    <input
                      type="text"
                      value={editNewName}
                      onChange={(e) => setEditNewName(e.target.value)}
                      className="w-full bg-[#0c0e14] border border-[#1f2533] focus:border-[#0891b2] text-slate-200 px-3 py-2 rounded-sm outline-none transition-colors"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-slate-400 uppercase font-bold">{t.people.lblGender}</label>
                    <select
                      value={editGender}
                      onChange={(e) => setEditGender(e.target.value)}
                      className="w-full bg-[#0c0e14] border border-[#1f2533] focus:border-[#0891b2] text-slate-200 px-3 py-2 rounded-sm outline-none transition-colors cursor-pointer"
                    >
                      <option value="unspecified">{t.people.genderUnspecified}</option>
                      <option value="male">{t.people.genderMale}</option>
                      <option value="female">{t.people.genderFemale}</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-slate-400 uppercase font-bold">{t.people.lblBirthDate}</label>
                    <input
                      type="text" placeholder="YYYY-MM" value={editBirthDate}
                      onChange={(e) => setEditBirthDate(e.target.value)} maxLength={7}
                      className="w-full bg-[#0c0e14] border border-[#1f2533] focus:border-[#0891b2] text-slate-200 px-3 py-2 rounded-sm outline-none transition-colors"
                    />
                  </div>
                  <div className="pt-4 flex gap-3">
                    <button
                      type="submit" disabled={isSavingMetadata}
                      className="flex-1 py-3 bg-cyan-500 hover:bg-cyan-600 disabled:bg-cyan-800 disabled:cursor-not-allowed font-bold uppercase text-slate-950 font-mono tracking-wider rounded-sm cursor-pointer transition-colors"
                    >
                      {isSavingMetadata ? t.people.savingMetadata : t.people.btnSaveMetadata}
                    </button>
                    <button
                      type="button"
                      onClick={() => { setActiveSubTab('list'); setSubmitError(null); }}
                      className="w-1/3 py-3 bg-[#1e2533] hover:bg-[#293245] font-bold uppercase text-slate-300 rounded-sm cursor-pointer transition-colors"
                    >
                      {t.people.modalCancel}
                    </button>
                  </div>
                </form>
              </div>
            )}

            {/* Subtab 5: Delete */}
            {activeSubTab === 'delete' && (
              <div className="max-w-2xl mx-auto bg-[#121622] border border-[#1f2533] p-6 rounded-sm space-y-5">
                <h4 className="text-xs font-bold font-mono text-white uppercase tracking-wider border-b border-[#212733] pb-3">
                  {language === 'zh' ? '删除人员及特征向量' : 'Delete Identities & Embeddings'}
                </h4>
                {peopleList.length === 0 ? (
                  <p className="text-xs text-slate-500 font-mono text-center py-8">
                    {language === 'zh' ? '暂无人员可删除' : 'No identities to delete'}
                  </p>
                ) : (
                  <div className="space-y-3">
                    {peopleList.map((person) => (
                      <div
                        key={person.name}
                        className="flex items-center justify-between p-3.5 bg-[#0c0e14] border border-[#1f222b] rounded-sm hover:border-red-500/30 transition-all font-mono text-xs"
                      >
                        <div className="flex items-center gap-3">
                          <div
                            className="w-8 h-8 rounded-sm flex items-center justify-center text-white font-bold text-xs border border-slate-700"
                            style={{ backgroundColor: getInitialsColor(person.name) }}
                          >
                            {person.name.charAt(0).toUpperCase()}
                          </div>
                          <div>
                            <p className="text-white font-bold">{person.name}</p>
                            <p className="text-[10px] text-slate-500">
                              {person.embedding_count} {language === 'zh' ? '个特征' : 'embeddings'}
                            </p>
                          </div>
                        </div>
                        <button
                          onClick={() => handleDeleteSubject(person.name)}
                          className="px-3.5 py-1.5 bg-red-950/20 hover:bg-red-800 border border-red-500/30 text-red-200 hover:text-white rounded-sm text-[10px] font-bold uppercase transition-colors flex items-center gap-1.5 cursor-pointer select-none"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                          <span>{t.people.deleteBtn}</span>
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* Footer */}
      <footer className="border-t border-[#1e2330] mt-8 pt-4 flex flex-wrap items-center justify-between font-mono text-[10px] text-slate-500">
        <div className="flex items-center gap-5">
          <span className="flex items-center gap-1.5 uppercase font-bold text-emerald-500 bg-emerald-950/20 px-2.5 py-1 border border-emerald-900/30 rounded-sm">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping inline-block" />
            {language === 'zh' ? '数据库在线' : 'Database Online'}
          </span>
          <span>{language === 'zh' ? '总人数:' : 'TOTAL:'} {peopleList.length}</span>
        </div>
        <span>E-RECOGNITION SYSTEMS v4.12</span>
      </footer>
    </div>
  );
}
