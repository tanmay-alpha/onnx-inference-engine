'use client';

import React, { useState, useRef } from 'react';
import { UploadCloud, CheckCircle } from 'lucide-react';

interface ModelUploaderProps {
  onModelLoaded: (name: string, bytes: Uint8Array) => void;
  loadedModelName: string | null;
}

export default function ModelUploader({ onModelLoaded, loadedModelName }: ModelUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const processFile = async (file: File) => {
    if (!file.name.endsWith('.onnx')) {
      setError('Please upload a valid .onnx model file.');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const arrayBuffer = await file.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);
      onModelLoaded(file.name, bytes);
    } catch (err: any) {
      setError('Failed to read ONNX file: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      await processFile(e.target.files[0]);
    }
  };

  const onZoneClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="card uploader-card">
      <h3 className="card-title">1. ONNX Model Upload</h3>
      <p className="card-description">Drag and drop your .onnx model file to load it client-side into the browser.</p>
      
      <div 
        className={`dropzone ${isDragging ? 'dragging' : ''} ${loadedModelName ? 'success' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={onZoneClick}
      >
        <input 
          type="file" 
          ref={fileInputRef} 
          style={{ display: 'none' }} 
          accept=".onnx" 
          onChange={handleFileChange} 
        />
        {loading ? (
          <div className="uploader-status">
            <div className="spinner"></div>
            <p>Reading binary bytes...</p>
          </div>
        ) : loadedModelName ? (
          <div className="uploader-status success">
            <CheckCircle size={40} className="status-icon" />
            <p className="model-name">{loadedModelName}</p>
            <span className="badge badge-success">Loaded Successfully</span>
          </div>
        ) : (
          <div className="uploader-prompt">
            <UploadCloud size={40} className="prompt-icon" />
            <p className="main-prompt">Drag & drop your model or <span>browse</span></p>
            <p className="sub-prompt">Supports .onnx files (e.g. mobilenet_v2.onnx)</p>
          </div>
        )}
      </div>
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}
