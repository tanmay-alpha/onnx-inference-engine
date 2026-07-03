import React from 'react';
import Link from 'next/link';
import { BarChart3, Award } from 'lucide-react';
import { getBenchmarkResults, getChartData } from '../../lib/api';
import LatencyChart from '../../components/LatencyChart';

export default function BenchmarkPage() {
  const benchmarkData = getBenchmarkResults();
  const chartData = getChartData();

  // Find individual engine results
  const crucible = benchmarkData.results.find((r) => r.engine === 'crucible');
  const onnxruntime = benchmarkData.results.find((r) => r.engine === 'onnxruntime');
  const pytorch = benchmarkData.results.find((r) => r.engine === 'pytorch');

  return (
    <div className="playground-container">
      <header className="header">
        <div className="logo font-mono">CRUCIBLE <span className="logo-accent">/ BENCHMARK</span></div>
        <nav className="nav">
          <Link href="/">Home</Link>
          <Link href="/playground">Playground</Link>
          <Link href="/benchmark" className="active">Benchmarks</Link>
          <Link href="/docs">Docs</Link>
        </nav>
      </header>

      <main className="playground-main">
        <div className="playground-header">
          <div className="title-row">
            <BarChart3 size={24} className="title-icon" />
            <h2>Performance Benchmarks</h2>
          </div>
          <p className="subtitle">
            Latency comparison on MobileNetV2 (1, 3, 224, 224 input) over 100 runs on standard CPU.
          </p>
        </div>

        {/* 3 Metric Cards */}
        <div className="stats-grid" style={{ marginBottom: '40px' }}>
          <div className="stat-box" style={{ padding: '24px', borderLeft: '4px solid var(--color-primary)' }}>
            <span className="stat-label">Crucible Core (Eigen)</span>
            <span className="stat-val font-mono" style={{ fontSize: '1.8rem', margin: '8px 0' }}>
              {crucible?.stats.mean_ms.toFixed(1)} ms
            </span>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
              Throughput: <strong>{crucible?.stats.throughput_inf_per_sec.toFixed(1)}</strong> inf/sec
            </span>
          </div>

          <div className="stat-box" style={{ padding: '24px', borderLeft: '4px solid var(--color-warning)' }}>
            <span className="stat-label">ONNX Runtime CPU</span>
            <span className="stat-val font-mono" style={{ fontSize: '1.8rem', margin: '8px 0' }}>
              {onnxruntime?.stats.mean_ms.toFixed(1)} ms
            </span>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
              Throughput: <strong>{onnxruntime?.stats.throughput_inf_per_sec.toFixed(1)}</strong> inf/sec
            </span>
          </div>

          <div className="stat-box" style={{ padding: '24px', borderLeft: '4px solid var(--color-accent)' }}>
            <span className="stat-label">PyTorch CPU</span>
            <span className="stat-val font-mono" style={{ fontSize: '1.8rem', margin: '8px 0' }}>
              {pytorch?.stats.mean_ms.toFixed(1)} ms
            </span>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
              Throughput: <strong>{pytorch?.stats.throughput_inf_per_sec.toFixed(1)}</strong> inf/sec
            </span>
          </div>
        </div>

        {/* Recharts Chart Section */}
        <div className="card" style={{ marginBottom: '40px' }}>
          <h3 className="card-title">Latency vs. Model Parameters</h3>
          <p className="card-description">
            Scaling latency comparison across model parameter configurations (Tiny to Huge sizes).
          </p>
          <LatencyChart data={chartData} />
        </div>

        {/* Detailed Stats Table */}
        <div className="card">
          <h3 className="card-title">Comprehensive Run Statistics</h3>
          <p className="card-description">
            Complete distribution of run latencies gathered from test-suite execution.
          </p>

          <table className="predictions-table" style={{ marginTop: '10px' }}>
            <thead>
              <tr>
                <th>Engine</th>
                <th>Backend Provider</th>
                <th>Mean Latency</th>
                <th>Median</th>
                <th>Min / Max</th>
                <th>P95 / P99</th>
                <th>Throughput</th>
                <th>Speedup vs Torch</th>
              </tr>
            </thead>
            <tbody>
              {benchmarkData.results.map((res) => {
                const ratio = pytorch && res.stats.mean_ms > 0 
                  ? (pytorch.stats.mean_ms / res.stats.mean_ms) 
                  : 1.0;
                  
                return (
                  <tr key={res.engine} className={res.engine === 'crucible' ? 'highlight-row' : ''}>
                    <td>
                      <span className="label-text" style={{ textTransform: 'capitalize' }}>
                        {res.engine === 'crucible' ? 'Crucible (C++)' : res.engine === 'onnxruntime' ? 'ONNX Runtime' : 'PyTorch'}
                      </span>
                    </td>
                    <td className="font-mono" style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {res.backend}
                    </td>
                    <td className="font-mono"><strong>{res.stats.mean_ms.toFixed(2)} ms</strong></td>
                    <td className="font-mono">{res.stats.median_ms.toFixed(2)} ms</td>
                    <td className="font-mono" style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {res.stats.min_ms.toFixed(1)} / {res.stats.max_ms.toFixed(1)} ms
                    </td>
                    <td className="font-mono" style={{ fontSize: '0.8rem' }}>
                      {res.stats.p95_ms.toFixed(1)} / {res.stats.p99_ms.toFixed(1)} ms
                    </td>
                    <td className="font-mono"><strong>{res.stats.throughput_inf_per_sec.toFixed(1)}</strong> ips</td>
                    <td>
                      <span 
                        className={`badge ${ratio >= 1.0 ? 'badge-success' : 'badge-warning'}`}
                        style={{ fontSize: '0.7rem' }}
                      >
                        {ratio >= 1.0 ? `${ratio.toFixed(2)}x Faster` : `${(1/ratio).toFixed(2)}x Slower`}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          
          <div className="info-badge" style={{ marginTop: '24px' }}>
            <Award size={16} />
            <span>
              <strong>Benchmark Note:</strong> {benchmarkData.summary.note}
            </span>
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>© 2026 Crucible Engine. High-performance benchmarking suite.</p>
      </footer>
    </div>
  );
}
