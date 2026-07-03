import React from 'react';
import Link from 'next/link';
import { BookOpen, CheckCircle2, XCircle } from 'lucide-react';
import { getSupportedOps } from '../../lib/api';

export default function DocsPage() {
  const ops = getSupportedOps();

  return (
    <div className="playground-container">
      <header className="header">
        <div className="logo font-mono">CRUCIBLE <span className="logo-accent">/ DOCS</span></div>
        <nav className="nav">
          <Link href="/">Home</Link>
          <Link href="/playground">Playground</Link>
          <Link href="/benchmark">Benchmarks</Link>
          <Link href="/docs" className="active">Docs</Link>
        </nav>
      </header>

      <main className="playground-main">
        <div className="playground-header">
          <div className="title-row">
            <BookOpen size={24} className="title-icon" />
            <h2>Supported ONNX Operators</h2>
          </div>
          <p className="subtitle">
            Crucible supports a specialized subset of 13 operators, optimized using Eigen BLAS kernels for C++ and safe element-wise math in WebAssembly.
          </p>
        </div>

        <div className="card" style={{ marginBottom: '30px' }}>
          <h3 className="card-title">Operator Kernel Architecture</h3>
          <p className="card-description" style={{ marginBottom: '0', lineHeight: '1.6' }}>
            Each operator is built directly from scratch to maximize math performance on the CPU. The C++17 engine maps standard row-major float arrays into <strong>Eigen::Map</strong> matrices, utilizing vectorization (AVX2/NEON) under the hood. The WebAssembly implementation replicates the math behavior client-side in pure Rust, utilizing a subset of operators to execute model graphs in the browser sandbox.
          </p>
        </div>

        {/* 13 Supported Operators Table */}
        <div className="card">
          <h3 className="card-title">13 Core Operators</h3>
          <p className="card-description">
            Complete registry of Crucible operator definitions, their backends, and client-side WASM execution support.
          </p>

          <table className="predictions-table" style={{ marginTop: '10px' }}>
            <thead>
              <tr>
                <th>Operator Name</th>
                <th>ONNX op_type</th>
                <th>Category</th>
                <th>Description</th>
                <th>C++ Backend</th>
                <th style={{ textAlign: 'center' }}>Client WASM</th>
              </tr>
            </thead>
            <tbody>
              {ops.map((op) => (
                <tr key={op.opType}>
                  <td>
                    <span className="label-text" style={{ fontSize: '0.95rem' }}>{op.name}</span>
                  </td>
                  <td>
                    <code className="font-mono" style={{ background: 'rgba(255, 255, 255, 0.05)', padding: '2px 6px', borderRadius: '4px', fontSize: '0.85rem' }}>
                      {op.opType}
                    </code>
                  </td>
                  <td style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                    {op.category}
                  </td>
                  <td style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', maxWidth: '300px', lineHeight: '1.4' }}>
                    {op.description}
                  </td>
                  <td className="font-mono" style={{ fontSize: '0.75rem', color: 'var(--color-primary)' }}>
                    {op.backend}
                  </td>
                  <td style={{ textAlign: 'center' }}>
                    {op.wasmSupported ? (
                      <span className="badge badge-success" style={{ gap: '4px', fontSize: '0.65rem' }}>
                        <CheckCircle2 size={12} />
                        Native WASM
                      </span>
                    ) : (
                      <span className="badge badge-warning" style={{ gap: '4px', fontSize: '0.65rem' }}>
                        <XCircle size={12} />
                        WASM Emulated
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </main>

      <footer className="footer">
        <p>© 2026 Crucible Engine. Comprehensive operator library.</p>
      </footer>
    </div>
  );
}
