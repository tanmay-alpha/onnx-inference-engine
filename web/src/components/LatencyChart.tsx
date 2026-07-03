'use client';

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface ChartDataPoint {
  size: string;
  crucible: number;
  onnxruntime: number;
  pytorch: number;
}

interface LatencyChartProps {
  data: ChartDataPoint[];
}

export default function LatencyChart({ data }: LatencyChartProps) {
  // Custom tooltips to blend in with our premium dark design
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div 
          style={{
            background: 'rgba(15, 15, 22, 0.95)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '8px',
            padding: '12px 16px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5)',
            backdropFilter: 'blur(10px)',
          }}
        >
          <p style={{ margin: '0 0 8px 0', fontSize: '0.85rem', fontWeight: 600, color: '#9ca3af' }}>
            Model Size: {label}
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {payload.map((entry: any) => (
              <div 
                key={entry.name} 
                style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '8px', 
                  fontSize: '0.9rem',
                  color: '#f3f4f6'
                }}
              >
                <span 
                  style={{ 
                    display: 'inline-block', 
                    width: '10px', 
                    height: '2px', 
                    background: entry.color 
                  }}
                />
                <span style={{ fontWeight: 500, textTransform: 'capitalize' }}>
                  {entry.name === 'crucible' ? 'Crucible (C++)' : entry.name === 'onnxruntime' ? 'ONNX Runtime' : 'PyTorch'}:
                </span>
                <span style={{ fontFamily: 'monospace', fontWeight: 700 }}>
                  {entry.value} ms
                </span>
              </div>
            ))}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div style={{ width: '100%', height: 400, marginTop: '20px' }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{
            top: 20,
            right: 30,
            left: 10,
            bottom: 20,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" />
          <XAxis 
            dataKey="size" 
            stroke="#9ca3af" 
            fontSize={12} 
            tickLine={false} 
            dy={10}
          />
          <YAxis 
            stroke="#9ca3af" 
            fontSize={12} 
            tickLine={false} 
            dx={-10}
            label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft', fill: '#9ca3af', dy: 40, offset: 0 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            verticalAlign="top" 
            height={40} 
            iconSize={12}
            formatter={(value) => {
              const labelMap: Record<string, string> = {
                crucible: 'Crucible Core (Solid)',
                onnxruntime: 'ONNX Runtime (Dashed)',
                pytorch: 'PyTorch (Dotted)',
              };
              return <span style={{ color: '#f3f4f6', fontSize: '0.85rem', fontWeight: 500, paddingRight: '15px' }}>{labelMap[value]}</span>;
            }}
          />
          
          {/* Crucible: Solid Line */}
          <Line
            type="monotone"
            dataKey="crucible"
            name="crucible"
            stroke="#6366f1"
            strokeWidth={3}
            dot={{ r: 4, strokeWidth: 2 }}
            activeDot={{ r: 6 }}
          />
          
          {/* ONNX Runtime: Dashed Line */}
          <Line
            type="monotone"
            dataKey="onnxruntime"
            name="onnxruntime"
            stroke="#f59e0b"
            strokeWidth={2.5}
            strokeDasharray="6 4"
            dot={{ r: 4, strokeWidth: 2 }}
            activeDot={{ r: 6 }}
          />
          
          {/* PyTorch: Dotted Line */}
          <Line
            type="monotone"
            dataKey="pytorch"
            name="pytorch"
            stroke="#8b5cf6"
            strokeWidth={2}
            strokeDasharray="2 4"
            dot={{ r: 3, strokeWidth: 2 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
