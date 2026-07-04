import Link from 'next/link';
import { Play, Cpu, Zap, Code, ShieldCheck } from 'lucide-react';

export default function Home() {
  return (
    <div className="home-container">
      <header className="header">
        <div className="logo font-mono">CRUCIBLE</div>
        <nav className="nav">
          <Link href="/" className="active">Home</Link>
          <Link href="/fraud">Fraud Demo</Link>
          <Link href="/playground">Playground</Link>
          <Link href="/benchmark">Benchmark</Link>
          <Link href="/docs">Docs</Link>
        </nav>
      </header>

      <main className="hero">
        <div className="badge badge-glow">Milestone 4 Live Demo</div>
        <h1 className="hero-title">From-Scratch ONNX Inference Engine</h1>
        <p className="hero-subtitle">
          Crucible compiles C++17 operator kernels, FFI bindings, and pure-Rust WebAssembly interpreter running end-to-end inference directly inside the browser.
        </p>

        <div className="hero-actions">
          <Link href="/fraud" className="btn btn-primary btn-large">
            <ShieldCheck size={20} />
            Try Fraud Detection
          </Link>
          <Link href="/playground" className="btn btn-secondary btn-large">
            <Play size={20} />
            WASM Playground
          </Link>
          <a 
            href="https://github.com/tanmay-alpha/Crucible" 
            className="btn btn-secondary btn-large" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            <Code size={20} />
            View Repository
          </a>
        </div>

        <div className="features-grid">
          <div className="feature-card">
            <Cpu size={32} className="feature-icon" />
            <h4>C++17 Engine Core</h4>
            <p>From-scratch implementation of tensors, operators (Conv2D, Pooling, Norm, Linear), and Kahn's topological sort execution.</p>
          </div>
          <div className="feature-card">
            <Zap size={32} className="feature-icon" />
            <h4>Pure Rust WASM</h4>
            <p>Standalone compiler and runtime implementation of active mathematical operators, loaded client-side via wasm-pack with 0 latency.</p>
          </div>
          <div className="feature-card">
            <ShieldCheck size={32} className="feature-icon" />
            <h4>No Server Overhead</h4>
            <p>Upload model definitions, execute inputs, analyze layers, and evaluate results client-side with 0 network calls.</p>
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>© 2026 Crucible Engine. Created by Tanmay (CS batch 2028).</p>
      </footer>
    </div>
  );
}
