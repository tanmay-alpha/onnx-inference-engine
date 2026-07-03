'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { LayoutGrid, ArrowLeft } from 'lucide-react';
import ModelUploader from '../../components/ModelUploader';
import InferenceRunner from '../../components/InferenceRunner';
import TensorViewer from '../../components/TensorViewer';

interface InferenceResult {
  outputData: Float32Array;
  shape: number[];
  latencyMs: number;
  isEmulated: boolean;
  modelName: string;
}

export default function Playground() {
  const [modelBytes, setModelBytes] = useState<Uint8Array | null>(null);
  const [modelName, setModelName] = useState<string | null>(null);
  const [inferenceResult, setInferenceResult] = useState<InferenceResult | null>(null);

  const handleModelLoaded = (name: string, bytes: Uint8Array) => {
    setModelName(name);
    setModelBytes(bytes);
    setInferenceResult(null); // Clear previous results when new model is loaded
  };

  const handleInferenceCompleted = (result: InferenceResult) => {
    setInferenceResult(result);
  };

  return (
    <div className="playground-container">
      <header className="header">
        <div className="logo-container">
          <Link href="/" className="back-link">
            <ArrowLeft size={16} />
          </Link>
          <div className="logo font-mono">CRUCIBLE <span className="logo-accent">/ PLAYGROUND</span></div>
        </div>
        <nav className="nav">
          <Link href="/">Home</Link>
          <Link href="/playground" className="active">Playground</Link>
        </nav>
      </header>

      <main className="playground-main">
        <div className="playground-header">
          <div className="title-row">
            <LayoutGrid size={24} className="title-icon" />
            <h2>WASM Client-side Playground</h2>
          </div>
          <p className="subtitle">
            Upload an ONNX model file. Tensors are parsed and executed natively in WebAssembly (0 server network roundtrips).
          </p>
        </div>

        <div className="playground-grid">
          <div className="playground-sidebar">
            <ModelUploader 
              onModelLoaded={handleModelLoaded} 
              loadedModelName={modelName} 
            />
            
            <InferenceRunner 
              modelBytes={modelBytes} 
              modelName={modelName}
              onInferenceCompleted={handleInferenceCompleted} 
            />
          </div>

          <div className="playground-content">
            <TensorViewer result={inferenceResult} />
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>© 2026 Crucible Engine. Client-side WASM sandboxed interpreter.</p>
      </footer>
    </div>
  );
}
