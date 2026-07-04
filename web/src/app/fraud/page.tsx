'use client';

import { useState, useCallback } from 'react';
import Link from 'next/link';
import { Shield, AlertTriangle, CheckCircle, Zap, Lock, ChevronRight } from 'lucide-react';
import { runFraudDetection, FraudDetectionResult } from '@/lib/crucible-wasm';

interface FormState {
  amount: string;
  oldBalanceOrig: string;
  newBalanceOrig: string;
  oldBalanceDest: string;
  newBalanceDest: string;
  type: 'CASH_OUT' | 'TRANSFER' | 'OTHER';
}

const PRESETS = [
  {
    label: 'Normal Transfer ₹5,000',
    description: 'Typical low-value transfer',
    icon: '✅',
    values: {
      amount: '5000',
      oldBalanceOrig: '120000',
      newBalanceOrig: '115000',
      oldBalanceDest: '45000',
      newBalanceDest: '50000',
      type: 'TRANSFER' as const,
    },
  },
  {
    label: 'Suspicious Large Transfer ₹4,200,000',
    description: 'Large transfer draining account',
    icon: '🚨',
    values: {
      amount: '4200000',
      oldBalanceOrig: '4210000',
      newBalanceOrig: '0',
      oldBalanceDest: '8000',
      newBalanceDest: '4208000',
      type: 'TRANSFER' as const,
    },
  },
  {
    label: 'Account Draining Pattern',
    description: 'Cash-out zeroing balance',
    icon: '⚠️',
    values: {
      amount: '750000',
      oldBalanceOrig: '752000',
      newBalanceOrig: '0',
      oldBalanceDest: '3000',
      newBalanceDest: '753000',
      type: 'CASH_OUT' as const,
    },
  },
];

const DEFAULT_FORM: FormState = {
  amount: '',
  oldBalanceOrig: '',
  newBalanceOrig: '',
  oldBalanceDest: '',
  newBalanceDest: '',
  type: 'TRANSFER',
};

export default function FraudPage() {
  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [result, setResult] = useState<FraudDetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = useCallback(
    (field: keyof FormState) =>
      (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        setForm((prev) => ({ ...prev, [field]: e.target.value }));
        setResult(null);
        setError(null);
      },
    []
  );

  const applyPreset = useCallback((preset: (typeof PRESETS)[0]) => {
    setForm(preset.values);
    setResult(null);
    setError(null);
  }, []);

  const analyze = useCallback(async () => {
    setError(null);
    setResult(null);
    setLoading(true);

    const isFirstRun = !result;
    if (isFirstRun) setModelLoading(true);

    try {
      const res = await runFraudDetection({
        amount: parseFloat(form.amount) || 0,
        oldBalanceOrig: parseFloat(form.oldBalanceOrig) || 0,
        newBalanceOrig: parseFloat(form.newBalanceOrig) || 0,
        oldBalanceDest: parseFloat(form.oldBalanceDest) || 0,
        newBalanceDest: parseFloat(form.newBalanceDest) || 0,
        type: form.type,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Inference failed');
    } finally {
      setLoading(false);
      setModelLoading(false);
    }
  }, [form, result]);

  const pct = result ? Math.round(result.probability * 100) : 0;
  const isFraud = result?.label === 'FRAUD';

  return (
    <div className="fraud-page">
      {/* ── Header ───────────────────────────────────────────────── */}
      <header className="header">
        <div className="logo font-mono">CRUCIBLE</div>
        <nav className="nav">
          <Link href="/">Home</Link>
          <Link href="/playground">Playground</Link>
          <Link href="/benchmark">Benchmark</Link>
          <Link href="/fraud" className="active">Fraud Demo</Link>
          <Link href="/docs">Docs</Link>
        </nav>
      </header>

      <main className="fraud-main">
        {/* ── Hero ─────────────────────────────────────────────────── */}
        <div className="fraud-hero">
          <div className="fraud-badge">
            <Lock size={14} />
            Privacy-First ML
          </div>
          <h1 className="fraud-title">
            Fraud Detection —
            <span className="gradient-text"> In Your Browser</span>
          </h1>
          <p className="fraud-subtitle">
            Powered by Crucible WebAssembly. Your transaction data{' '}
            <strong>never reaches a server.</strong>
          </p>
        </div>

        <div className="fraud-layout">
          {/* ── Left: Form ──────────────────────────────────────────── */}
          <div className="fraud-card">
            {/* Presets */}
            <div className="fraud-section">
              <h3 className="fraud-section-title">Quick Fill</h3>
              <div className="preset-list">
                {PRESETS.map((p) => (
                  <button
                    key={p.label}
                    id={`preset-${p.label.replace(/\s+/g, '-').toLowerCase()}`}
                    className="preset-btn"
                    onClick={() => applyPreset(p)}
                  >
                    <span className="preset-icon">{p.icon}</span>
                    <span className="preset-text">
                      <span className="preset-label">{p.label}</span>
                      <span className="preset-desc">{p.description}</span>
                    </span>
                    <ChevronRight size={14} className="preset-arrow" />
                  </button>
                ))}
              </div>
            </div>

            {/* Inputs */}
            <div className="fraud-section">
              <h3 className="fraud-section-title">Transaction Details</h3>
              <div className="fraud-form">
                <div className="form-group">
                  <label htmlFor="txn-type">Transaction Type</label>
                  <select
                    id="txn-type"
                    value={form.type}
                    onChange={handleChange('type')}
                    className="fraud-select"
                  >
                    <option value="TRANSFER">TRANSFER</option>
                    <option value="CASH_OUT">CASH_OUT</option>
                    <option value="OTHER">OTHER</option>
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="txn-amount">Transaction Amount (₹)</label>
                  <input
                    id="txn-amount"
                    type="number"
                    placeholder="e.g. 5000"
                    value={form.amount}
                    onChange={handleChange('amount')}
                    className="fraud-input"
                  />
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label htmlFor="orig-bal-before">Origin Balance Before (₹)</label>
                    <input
                      id="orig-bal-before"
                      type="number"
                      placeholder="e.g. 120000"
                      value={form.oldBalanceOrig}
                      onChange={handleChange('oldBalanceOrig')}
                      className="fraud-input"
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor="orig-bal-after">Origin Balance After (₹)</label>
                    <input
                      id="orig-bal-after"
                      type="number"
                      placeholder="e.g. 115000"
                      value={form.newBalanceOrig}
                      onChange={handleChange('newBalanceOrig')}
                      className="fraud-input"
                    />
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label htmlFor="dest-bal-before">Dest. Balance Before (₹)</label>
                    <input
                      id="dest-bal-before"
                      type="number"
                      placeholder="e.g. 45000"
                      value={form.oldBalanceDest}
                      onChange={handleChange('oldBalanceDest')}
                      className="fraud-input"
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor="dest-bal-after">Dest. Balance After (₹)</label>
                    <input
                      id="dest-bal-after"
                      type="number"
                      placeholder="e.g. 50000"
                      value={form.newBalanceDest}
                      onChange={handleChange('newBalanceDest')}
                      className="fraud-input"
                    />
                  </div>
                </div>

                <button
                  id="analyze-btn"
                  className={`analyze-btn ${loading ? 'loading' : ''}`}
                  onClick={analyze}
                  disabled={loading || !form.amount}
                >
                  {modelLoading ? (
                    <>
                      <span className="spinner" />
                      Loading model…
                    </>
                  ) : loading ? (
                    <>
                      <span className="spinner" />
                      Analyzing…
                    </>
                  ) : (
                    <>
                      <Shield size={18} />
                      Analyze Transaction
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* ── Right: Result + Explainer ───────────────────────────── */}
          <div className="fraud-right">
            {/* Result card */}
            {error && (
              <div className="result-card error-card">
                <AlertTriangle size={24} />
                <p>{error}</p>
              </div>
            )}

            {result && (
              <div className={`result-card ${isFraud ? 'fraud-result' : 'legit-result'}`}>
                <div className="result-header">
                  {isFraud ? (
                    <AlertTriangle size={32} className="result-icon fraud-icon" />
                  ) : (
                    <CheckCircle size={32} className="result-icon legit-icon" />
                  )}
                  <div className="result-badge">
                    {isFraud ? '🚨 FRAUDULENT' : '✅ LEGITIMATE'}
                  </div>
                </div>

                <div className="prob-section">
                  <div className="prob-label">
                    <span>Fraud Probability</span>
                    <span className={`prob-value ${isFraud ? 'fraud-text' : 'legit-text'}`}>
                      {pct}%
                    </span>
                  </div>
                  <div className="prob-bar-track">
                    <div
                      className={`prob-bar-fill ${isFraud ? 'bar-fraud' : 'bar-legit'}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>

                <div className="result-meta">
                  <div className="meta-item">
                    <Zap size={14} />
                    <span>
                      Analyzed in{' '}
                      <strong>{result.latencyMs.toFixed(2)}ms</strong>
                    </span>
                  </div>
                  <div className="meta-item privacy-note">
                    <Lock size={14} />
                    <span>Processed entirely in your browser. Zero data transmitted.</span>
                  </div>
                </div>
              </div>
            )}

            {!result && !error && (
              <div className="result-placeholder">
                <Shield size={48} className="placeholder-icon" />
                <p>Fill in transaction details and click <strong>Analyze</strong></p>
              </div>
            )}

            {/* Explainer */}
            <div className="explainer-card">
              <h3>
                <Lock size={16} />
                Why in-browser?
              </h3>
              <p>
                Financial institutions often cannot send raw transaction data to external APIs due to
                compliance requirements (GDPR, PCI-DSS). Crucible enables running ML models
                inside your own application — <strong>no data ever leaves the device.</strong>
              </p>
              <p>
                At ~3MB, Crucible&apos;s WebAssembly runtime is 16× smaller than ONNX Runtime,
                making it practical to bundle directly with a web application for client-side inference.
              </p>
              <div className="tech-pills">
                <span className="tech-pill">Pure Rust WASM</span>
                <span className="tech-pill">MatMul + Sigmoid</span>
                <span className="tech-pill">ONNX opset 13</span>
                <span className="tech-pill">0 network requests</span>
              </div>
            </div>
          </div>
        </div>
      </main>

      <style>{`
        .fraud-page {
          min-height: 100vh;
          background: var(--bg-primary, #0a0a0f);
          color: var(--text-primary, #e2e8f0);
        }

        .fraud-main {
          max-width: 1200px;
          margin: 0 auto;
          padding: 2rem 1.5rem 4rem;
        }

        /* Hero */
        .fraud-hero {
          text-align: center;
          padding: 3rem 0 2.5rem;
        }
        .fraud-badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: rgba(99,102,241,0.15);
          border: 1px solid rgba(99,102,241,0.35);
          color: #818cf8;
          font-size: 0.78rem;
          font-weight: 600;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          padding: 5px 14px;
          border-radius: 100px;
          margin-bottom: 1.25rem;
        }
        .fraud-title {
          font-size: clamp(2rem, 5vw, 3.2rem);
          font-weight: 800;
          line-height: 1.1;
          margin-bottom: 1rem;
          letter-spacing: -0.02em;
        }
        .gradient-text {
          background: linear-gradient(135deg, #818cf8 0%, #a78bfa 50%, #38bdf8 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        .fraud-subtitle {
          font-size: 1.1rem;
          color: #94a3b8;
          max-width: 520px;
          margin: 0 auto;
          line-height: 1.6;
        }

        /* Layout */
        .fraud-layout {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1.5rem;
          margin-top: 2rem;
        }
        @media (max-width: 900px) {
          .fraud-layout { grid-template-columns: 1fr; }
        }

        /* Cards */
        .fraud-card {
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 16px;
          overflow: hidden;
        }
        .fraud-right {
          display: flex;
          flex-direction: column;
          gap: 1.25rem;
        }

        /* Section */
        .fraud-section {
          padding: 1.5rem;
          border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .fraud-section:last-child { border-bottom: none; }
        .fraud-section-title {
          font-size: 0.75rem;
          font-weight: 700;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          color: #64748b;
          margin-bottom: 1rem;
        }

        /* Presets */
        .preset-list { display: flex; flex-direction: column; gap: 8px; }
        .preset-btn {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 12px 14px;
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.07);
          border-radius: 10px;
          cursor: pointer;
          text-align: left;
          transition: all 0.2s;
          width: 100%;
          color: inherit;
        }
        .preset-btn:hover {
          background: rgba(99,102,241,0.1);
          border-color: rgba(99,102,241,0.3);
          transform: translateX(2px);
        }
        .preset-icon { font-size: 1.2rem; flex-shrink: 0; }
        .preset-text { flex: 1; display: flex; flex-direction: column; gap: 2px; }
        .preset-label { font-size: 0.85rem; font-weight: 600; color: #e2e8f0; }
        .preset-desc { font-size: 0.75rem; color: #64748b; }
        .preset-arrow { color: #475569; flex-shrink: 0; }

        /* Form */
        .fraud-form { display: flex; flex-direction: column; gap: 1rem; }
        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .form-group { display: flex; flex-direction: column; gap: 6px; }
        .form-group label {
          font-size: 0.75rem;
          font-weight: 600;
          color: #94a3b8;
          letter-spacing: 0.03em;
        }
        .fraud-input, .fraud-select {
          background: rgba(0,0,0,0.3);
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 8px;
          padding: 10px 12px;
          color: #e2e8f0;
          font-size: 0.9rem;
          outline: none;
          transition: border-color 0.2s, box-shadow 0.2s;
          width: 100%;
          box-sizing: border-box;
        }
        .fraud-select { appearance: none; cursor: pointer; }
        .fraud-input:focus, .fraud-select:focus {
          border-color: rgba(99,102,241,0.5);
          box-shadow: 0 0 0 3px rgba(99,102,241,0.1);
        }
        .fraud-input::placeholder { color: #475569; }

        /* Analyze button */
        .analyze-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          padding: 14px 24px;
          background: linear-gradient(135deg, #6366f1, #8b5cf6);
          border: none;
          border-radius: 10px;
          color: white;
          font-size: 1rem;
          font-weight: 700;
          cursor: pointer;
          transition: all 0.2s;
          margin-top: 0.5rem;
        }
        .analyze-btn:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 8px 24px rgba(99,102,241,0.35);
        }
        .analyze-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .spinner {
          width: 16px;
          height: 16px;
          border: 2px solid rgba(255,255,255,0.3);
          border-top-color: white;
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Result cards */
        .result-card {
          border-radius: 16px;
          padding: 1.75rem;
          border: 1px solid;
          animation: fadeInUp 0.3s ease;
        }
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(12px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .fraud-result {
          background: rgba(239,68,68,0.08);
          border-color: rgba(239,68,68,0.3);
        }
        .legit-result {
          background: rgba(34,197,94,0.08);
          border-color: rgba(34,197,94,0.3);
        }
        .error-card {
          background: rgba(245,158,11,0.08);
          border-color: rgba(245,158,11,0.3);
          display: flex;
          align-items: center;
          gap: 12px;
          color: #fbbf24;
        }

        .result-header {
          display: flex;
          align-items: center;
          gap: 14px;
          margin-bottom: 1.5rem;
        }
        .result-badge {
          font-size: 1.3rem;
          font-weight: 800;
          letter-spacing: 0.05em;
        }
        .fraud-icon { color: #ef4444; }
        .legit-icon { color: #22c55e; }

        /* Probability bar */
        .prob-section { margin-bottom: 1.25rem; }
        .prob-label {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
          font-size: 0.85rem;
          color: #94a3b8;
        }
        .prob-value { font-size: 1.5rem; font-weight: 800; }
        .fraud-text { color: #ef4444; }
        .legit-text { color: #22c55e; }
        .prob-bar-track {
          height: 10px;
          background: rgba(255,255,255,0.08);
          border-radius: 100px;
          overflow: hidden;
        }
        .prob-bar-fill {
          height: 100%;
          border-radius: 100px;
          transition: width 0.6s cubic-bezier(0.34,1.56,0.64,1);
        }
        .bar-fraud {
          background: linear-gradient(90deg, #f97316, #ef4444);
        }
        .bar-legit {
          background: linear-gradient(90deg, #22c55e, #10b981);
        }

        /* Meta */
        .result-meta { display: flex; flex-direction: column; gap: 8px; }
        .meta-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 0.82rem;
          color: #94a3b8;
        }
        .privacy-note {
          color: #4ade80;
          background: rgba(34,197,94,0.08);
          border-radius: 8px;
          padding: 8px 12px;
          border: 1px solid rgba(34,197,94,0.15);
        }

        /* Placeholder */
        .result-placeholder {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 12px;
          padding: 3rem 1.5rem;
          background: rgba(255,255,255,0.02);
          border: 1px dashed rgba(255,255,255,0.1);
          border-radius: 16px;
          color: #475569;
          text-align: center;
          font-size: 0.9rem;
        }
        .placeholder-icon { color: #334155; }

        /* Explainer */
        .explainer-card {
          background: rgba(99,102,241,0.06);
          border: 1px solid rgba(99,102,241,0.15);
          border-radius: 16px;
          padding: 1.5rem;
        }
        .explainer-card h3 {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 0.95rem;
          font-weight: 700;
          color: #818cf8;
          margin-bottom: 0.75rem;
        }
        .explainer-card p {
          font-size: 0.85rem;
          color: #94a3b8;
          line-height: 1.7;
          margin-bottom: 0.75rem;
        }
        .explainer-card p:last-of-type { margin-bottom: 1rem; }
        .tech-pills { display: flex; flex-wrap: wrap; gap: 6px; }
        .tech-pill {
          font-size: 0.72rem;
          font-weight: 600;
          padding: 4px 10px;
          border-radius: 100px;
          background: rgba(99,102,241,0.15);
          border: 1px solid rgba(99,102,241,0.25);
          color: #a5b4fc;
        }
      `}</style>
    </div>
  );
}
