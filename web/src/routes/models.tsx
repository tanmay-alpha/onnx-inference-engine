import { useState, useEffect, useRef, useCallback } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import {
  Upload,
  Trash2,
  Play,
  FileUp,
  HardDrive,
  Clock,
  Cpu,
  RefreshCw,
  Inbox,
  ArrowRight,
  Check,
} from "lucide-react";
import { uploadModel, deleteModel, fetchModels } from "../lib/api";
import type { RegisteredModel } from "../lib/api";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import { CrucibleLayout } from "../components/crucible/Layout";

const fmtBytes = (b: number) => {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(2)} MB`;
};

const fmtDate = (iso: string) => {
  try {
    return new Date(iso).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
};

function ModelsPage() {
  const navigate = useNavigate();
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchModels();
      setModels(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load models");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(`Uploading ${file.name}...`);
    setError(null);

    try {
      const shapeStr = prompt(
        "Enter input shape as comma-separated dimensions (e.g., 1,3,224,224):",
        "1,3,224,224",
      );
      if (!shapeStr) {
        setUploading(false);
        setUploadProgress(null);
        return;
      }

      const inputShape = shapeStr
        .split(",")
        .map((s) => parseInt(s.trim(), 10))
        .filter((n) => !Number.isNaN(n) && n > 0);

      if (inputShape.length === 0) {
        throw new Error("Invalid shape. Enter numbers separated by commas.");
      }

      setUploadProgress(`Validating and uploading ${file.name}...`);
      const result = await uploadModel(file, inputShape);
      setUploadProgress(`Uploaded ${result.name} successfully!`);
      await loadModels();
      setTimeout(() => setUploadProgress(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setUploadProgress(null);
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleDelete = async (modelId: string, modelName: string) => {
    if (!confirm(`Delete model "${modelName}"? This cannot be undone.`)) return;
    setDeleting(modelId);
    try {
      await deleteModel(modelId);
      await loadModels();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setDeleting(null);
    }
  };

  const handleUseForInference = (model: RegisteredModel) => {
    navigate({
      to: "/playground",
      search: { modelId: model.id, modelName: model.name },
    });
  };

  return (
    <CrucibleLayout>
      <section className="c-container" style={{ paddingTop: 48, paddingBottom: 56 }}>
        {/* Header */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            flexWrap: "wrap",
            gap: 16,
            marginBottom: 32,
          }}
        >
          <div>
            <div className="c-eyebrow">MODEL MANAGEMENT</div>
            <h1 className="c-h2" style={{ fontSize: 36 }}>
              ONNX Models
            </h1>
            <p className="c-muted" style={{ marginTop: 8 }}>
              Upload, manage, and run inference on your ONNX models.
            </p>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <Button variant="secondary" onClick={loadModels} disabled={loading}>
              <RefreshCw size={15} />
              Refresh
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".onnx"
              style={{ display: "none" }}
              onChange={handleUpload}
              disabled={uploading}
            />
            <Button onClick={() => fileInputRef.current?.click()} disabled={uploading}>
              <Upload size={15} />
              Upload Model
            </Button>
          </div>
        </div>

        {uploadProgress && (
          <div
            style={{
              padding: "10px 14px",
              borderRadius: 2,
              background: "#ddeedf",
              border: "1px solid #b4d8be",
              color: "#166534",
              fontSize: 13,
              marginBottom: 20,
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            {uploading && <RefreshCw size={14} className="c-spin" />}
            {!uploading && <Check size={14} />}
            {uploadProgress}
          </div>
        )}

        {error && (
          <div
            style={{
              padding: "10px 14px",
              borderRadius: 2,
              background: "#fadcdc",
              border: "1px solid #e4b4b4",
              color: "#b91c1c",
              fontSize: 13,
              marginBottom: 20,
            }}
            role="alert"
          >
            {error}
          </div>
        )}

        {loading ? (
          <div style={{ textAlign: "center", padding: 60 }}>
            <RefreshCw size={24} className="c-spin" style={{ color: "var(--ink-muted)" }} />
            <p className="c-muted" style={{ marginTop: 12 }}>
              Loading models...
            </p>
          </div>
        ) : models.length === 0 ? (
          <Card style={{ textAlign: "center", padding: 60, borderStyle: "dashed" }}>
            <Inbox size={40} style={{ color: "var(--ink-subtle)", marginBottom: 16 }} />
            <h3 className="c-h3" style={{ fontSize: 18, marginBottom: 8 }}>
              No models uploaded yet
            </h3>
            <p className="c-muted" style={{ maxWidth: 420, margin: "0 auto 20px" }}>
              Upload an ONNX model file to start running inferences. Supported operators include
              MatMul, Relu, Sigmoid, Softmax, and more.
            </p>
            <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap" }}>
              <Button onClick={() => fileInputRef.current?.click()}>
                <Upload size={15} />
                Upload your first model
              </Button>
              <Link to="/playground">
                <Button variant="secondary">
                  Try the playground <ArrowRight size={15} />
                </Button>
              </Link>
            </div>
          </Card>
        ) : (
          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Model Name</TableHead>
                  <TableHead style={{ textAlign: "right" }}>Size</TableHead>
                  <TableHead style={{ textAlign: "right" }}>Operators</TableHead>
                  <TableHead style={{ textAlign: "right" }}>Input Shape</TableHead>
                  <TableHead style={{ textAlign: "right" }}>Uploaded</TableHead>
                  <TableHead style={{ textAlign: "right" }}>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {models.map((model) => (
                  <TableRow key={model.id}>
                    <TableCell>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <FileUp size={14} style={{ color: "var(--trace)", flexShrink: 0 }} />
                        <span style={{ fontWeight: 500 }}>{model.name}</span>
                        {!model.all_supported && (
                          <span
                            style={{
                              fontSize: 10,
                              padding: "1px 6px",
                              borderRadius: 2,
                              background: "#f3e7c4",
                              border: "1px solid #dcc58a",
                              color: "#b45309",
                              fontFamily: "var(--f-mono)",
                            }}
                          >
                            partial
                          </span>
                        )}
                      </div>
                    </TableCell>
                    <TableCell style={{ textAlign: "right" }}>
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                        <HardDrive size={12} style={{ color: "var(--ink-muted)" }} />
                        {fmtBytes(model.file_size_bytes)}
                      </span>
                    </TableCell>
                    <TableCell style={{ textAlign: "right" }}>
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                        <Cpu size={12} style={{ color: "var(--ink-muted)" }} />
                        {model.operators.length}
                      </span>
                    </TableCell>
                    <TableCell
                      style={{
                        textAlign: "right",
                        fontFamily: "var(--f-mono)",
                        fontSize: 12,
                      }}
                    >
                      [{model.input_shape?.join(", ") || "—"}]
                    </TableCell>
                    <TableCell style={{ textAlign: "right" }}>
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                        <Clock size={12} style={{ color: "var(--ink-muted)" }} />
                        {fmtDate(model.created_at)}
                      </span>
                    </TableCell>
                    <TableCell style={{ textAlign: "right" }}>
                      <div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleUseForInference(model)}
                          title="Run inference"
                        >
                          <Play size={14} />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDelete(model.id, model.name)}
                          disabled={deleting === model.id}
                          title="Delete model"
                          style={{ color: "var(--risk)" }}
                        >
                          {deleting === model.id ? (
                            <RefreshCw size={14} className="c-spin" />
                          ) : (
                            <Trash2 size={14} />
                          )}
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        )}
      </section>

      <style>{`
        .c-spin {
          animation: c-spin 1s linear infinite;
        }
        @keyframes c-spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </CrucibleLayout>
  );
}

export const Route = {
  component: ModelsPage,
};
