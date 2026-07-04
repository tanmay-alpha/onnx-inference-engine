'use client';

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';

const FOOTPRINT_DATA = [
  { runtime: 'Crucible', binary_mb: 3.1, startup_ms: 48, browser: true,  color: '#818cf8' },
  { runtime: 'TFLite',   binary_mb: 2.1, startup_ms: 35, browser: false, color: '#f59e0b' },
  { runtime: 'ONNX RT',  binary_mb: 51.2, startup_ms: 820, browser: false, color: '#ef4444' },
  { runtime: 'PyTorch',  binary_mb: 756,  startup_ms: 2100, browser: false, color: '#f87171' },
];

interface TooltipPayload {
  dataKey: string;
  value: number;
  name: string;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  const row = FOOTPRINT_DATA.find((d) => d.runtime === label);
  return (
    <div style={{
      background: 'rgba(15,15,25,0.97)',
      border: '1px solid rgba(255,255,255,0.12)',
      borderRadius: 10,
      padding: '12px 16px',
      fontSize: '0.82rem',
      lineHeight: 1.8,
    }}>
      <div style={{ fontWeight: 700, marginBottom: 6, color: '#e2e8f0' }}>{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: '#94a3b8' }}>
          {p.name === 'binary_mb' ? '📦 Binary Size' : '⚡ Startup Time'}:{' '}
          <strong style={{ color: '#e2e8f0' }}>
            {p.name === 'binary_mb' ? `${p.value} MB` : `${p.value} ms`}
          </strong>
        </div>
      ))}
      <div style={{ marginTop: 6, color: row?.browser ? '#4ade80' : '#ef4444', fontWeight: 600 }}>
        {row?.browser ? '✅ Browser capable' : '❌ Server-only'}
      </div>
    </div>
  );
}

export default function FootprintChart() {
  return (
    <div style={{ width: '100%' }}>
      {/* Binary size */}
      <div style={{ marginBottom: 8, fontSize: '0.82rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        Binary Size (MB)
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={FOOTPRINT_DATA} margin={{ top: 4, right: 20, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis dataKey="runtime" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} unit=" MB" width={60} />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
          <Bar dataKey="binary_mb" name="binary_mb" radius={[6, 6, 0, 0]}>
            {FOOTPRINT_DATA.map((entry) => (
              <Cell key={entry.runtime} fill={entry.color} fillOpacity={entry.browser ? 1 : 0.5} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Startup time */}
      <div style={{ marginTop: 24, marginBottom: 8, fontSize: '0.82rem', color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        Cold-Start Time (ms)
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={FOOTPRINT_DATA} margin={{ top: 4, right: 20, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis dataKey="runtime" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} unit=" ms" width={60} />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
          <Bar dataKey="startup_ms" name="startup_ms" radius={[6, 6, 0, 0]}>
            {FOOTPRINT_DATA.map((entry) => (
              <Cell key={entry.runtime} fill={entry.color} fillOpacity={entry.browser ? 1 : 0.5} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Browser capability table */}
      <div style={{
        marginTop: 24,
        background: 'rgba(255,255,255,0.03)',
        borderRadius: 10,
        overflow: 'hidden',
        border: '1px solid rgba(255,255,255,0.07)',
      }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
              {['Runtime', 'Binary Size', 'Startup', 'Browser Capable'].map((h) => (
                <th key={h} style={{ padding: '10px 16px', textAlign: 'left', color: '#64748b', fontWeight: 600, fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {FOOTPRINT_DATA.map((row, i) => (
              <tr key={row.runtime} style={{ borderBottom: i < FOOTPRINT_DATA.length - 1 ? '1px solid rgba(255,255,255,0.05)' : 'none', background: row.browser ? 'rgba(99,102,241,0.05)' : 'transparent' }}>
                <td style={{ padding: '10px 16px', fontWeight: row.browser ? 700 : 400, color: row.browser ? '#818cf8' : '#94a3b8' }}>{row.runtime}</td>
                <td style={{ padding: '10px 16px', fontFamily: 'monospace', color: '#e2e8f0' }}>{row.binary_mb} MB</td>
                <td style={{ padding: '10px 16px', fontFamily: 'monospace', color: '#e2e8f0' }}>{row.startup_ms} ms</td>
                <td style={{ padding: '10px 16px' }}>
                  <span style={{ fontWeight: 700, color: row.browser ? '#4ade80' : '#ef4444' }}>
                    {row.browser ? '✅ Yes' : '❌ No'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
