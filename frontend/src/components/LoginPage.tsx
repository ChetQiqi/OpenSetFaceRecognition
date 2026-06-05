import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Shield, User, Lock, Mail, ArrowRight, AlertTriangle } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { ApiError } from '../api/client';

export default function LoginPage() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      if (mode === 'login') {
        await login(username, password);
      } else {
        await register(username, password, email || undefined);
        // After registration, log in
        await login(username, password);
      }
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError(mode === 'login' ? 'Login failed' : 'Registration failed');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen w-screen flex items-center justify-center bg-[#090b11] overflow-hidden">
      {/* Background grid pattern */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none"
        style={{
          backgroundImage: 'linear-gradient(rgba(6,182,212,0.8) 1px, transparent 1px), linear-gradient(90deg, rgba(6,182,212,0.8) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />

      {/* Animated scan line */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden opacity-10">
        <div className="absolute w-full h-[2px] bg-cyan-400 animate-[scanLine_4s_linear_infinite]"
          style={{
            animation: 'scanLine 4s linear infinite',
            boxShadow: '0 0 20px rgba(6,182,212,0.5)',
          }}
        />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md relative z-10"
      >
        {/* Brand header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-3">
            <Shield className="w-8 h-8 text-cyan-400" />
            <h1 className="text-2xl font-black tracking-widest text-white font-mono">
              <span className="text-cyan-400">E-</span>RECOGNITION
            </h1>
          </div>
          <p className="text-xs text-slate-500 font-mono uppercase tracking-[0.2em]">
            AI Sentinel System v4.2.1
          </p>
        </div>

        {/* Card */}
        <div className="bg-[#121622] border border-[#262b35] rounded-sm p-8 shadow-2xl shadow-black/40">
          {/* Mode tabs */}
          <div className="flex mb-6 bg-[#181d26] rounded-sm p-0.5 border border-[#262b35]">
            <button
              onClick={() => { setMode('login'); setError(''); }}
              className={`flex-1 py-2 text-xs font-mono font-bold uppercase tracking-wider rounded-sm transition-all ${
                mode === 'login' ? 'bg-cyan-500 text-slate-950' : 'text-slate-400 hover:text-white'
              }`}
            >
              Login
            </button>
            <button
              onClick={() => { setMode('register'); setError(''); }}
              className={`flex-1 py-2 text-xs font-mono font-bold uppercase tracking-wider rounded-sm transition-all ${
                mode === 'register' ? 'bg-cyan-500 text-slate-950' : 'text-slate-400 hover:text-white'
              }`}
            >
              Register
            </button>
          </div>

          {/* Error alert */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-4 bg-red-950/50 border border-red-500/30 rounded-sm p-3 flex items-start gap-2"
              >
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <p className="text-xs text-red-300 font-mono">{error}</p>
              </motion.div>
            )}
          </AnimatePresence>

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Username */}
            <div>
              <label className="block text-[10px] font-mono font-bold uppercase tracking-widest text-slate-400 mb-1.5">
                Username
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  minLength={3}
                  className="w-full bg-[#181d26] border border-[#262b35] focus:border-cyan-500 text-slate-200 pl-10 pr-4 py-2.5 rounded-sm text-sm outline-none focus:ring-1 focus:ring-cyan-900 transition-all font-mono"
                  placeholder="Enter username"
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="block text-[10px] font-mono font-bold uppercase tracking-widest text-slate-400 mb-1.5">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={6}
                  className="w-full bg-[#181d26] border border-[#262b35] focus:border-cyan-500 text-slate-200 pl-10 pr-4 py-2.5 rounded-sm text-sm outline-none focus:ring-1 focus:ring-cyan-900 transition-all font-mono"
                  placeholder="Enter password"
                />
              </div>
            </div>

            {/* Register-only fields */}
            <AnimatePresence>
              {mode === 'register' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-4 overflow-hidden"
                >
                  {/* Email */}
                  <div>
                    <label className="block text-[10px] font-mono font-bold uppercase tracking-widest text-slate-400 mb-1.5">
                      Email (optional)
                    </label>
                    <div className="relative">
                      <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                      <input
                        type="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        className="w-full bg-[#181d26] border border-[#262b35] focus:border-cyan-500 text-slate-200 pl-10 pr-4 py-2.5 rounded-sm text-sm outline-none focus:ring-1 focus:ring-cyan-900 transition-all font-mono"
                        placeholder="email@example.com"
                      />
                    </div>
                  </div>

                </motion.div>
              )}
            </AnimatePresence>

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 bg-cyan-500 hover:bg-cyan-400 disabled:bg-cyan-800 disabled:cursor-not-allowed text-slate-950 font-extrabold uppercase rounded-sm text-xs tracking-widest transition-all cursor-pointer flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/10 active:scale-[0.98]"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <span className="w-3 h-3 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  Processing...
                </span>
              ) : (
                <>
                  {mode === 'login' ? 'Authenticate' : 'Register'}
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>
        </div>

        {/* Footer */}
        <p className="text-center mt-6 text-[10px] text-slate-600 font-mono uppercase tracking-wider">
          FastAPI Backend &bull; JWT Bearer Auth &bull; HS256
        </p>
      </motion.div>
    </div>
  );
}
