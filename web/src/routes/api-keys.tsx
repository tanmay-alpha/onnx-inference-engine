import { useState, useEffect } from "react";
import {
  Key,
  Plus,
  Trash2,
  Copy,
  Check,
  RefreshCw,
  AlertTriangle,
  Shield,
  Info,
} from "lucide-react";
import { listApiKeys, createApiKey, revokeApiKey } from "../lib/api";
import type { ApiKeyInfo, ApiKeyCreated } from "../lib/api";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "../components/ui/dialog";
import { CrucibleLayout } from "../components/crucible/Layout";

function ApiKeysPage() {
  const [keys, setKeys] = useState<ApiKeyInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [newKeyName, setNewKeyName] = useState("");
  const [newKeyExpiry, setNewKeyExpiry] = useState("");
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [createdKey, setCreatedKey] = useState<ApiKeyCreated | null>(null);
  const [copied, setCopied] = useState(false);
  const [revoking, setRevoking] = useState<string | null>(null);

  const loadKeys = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listApiKeys();
      setKeys(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load API keys");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadKeys();
  }, []);

  const handleCreate = async () => {
    if (!newKeyName.trim()) return;
    setCreating(true);
    setError(null);
    try {
      const result = await createApiKey(
        newKeyName.trim(),
        newKeyExpiry ? parseInt(newKeyExpiry, 10) : undefined,
      );
      setCreatedKey(result);
      await loadKeys();
      setNewKeyName("");
      setNewKeyExpiry("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create API key");
    } finally {
      setCreating(false);
    }
  };

  const handleRevoke = async (keyId: string, keyName: string) => {
    if (!confirm(`Revoke API key "${keyName}"? This cannot be undone.`)) return;
    setRevoking(keyId);
    try {
      await revokeApiKey(keyId);
      await loadKeys();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to revoke key");
    } finally {
      setRevoking(null);
    }
  };

  const copyKey = () => {
    if (createdKey?.key) {
      navigator.clipboard.writeText(createdKey.key);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const fmtDate = (iso: string | null) => {
    if (!iso) return "Never";
    try {
      return new Date(iso).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      });
    } catch {
      return iso;
    }
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
            <div className="c-eyebrow">API AUTHENTICATION</div>
            <h1 className="c-h2" style={{ fontSize: 36 }}>
              API Keys
            </h1>
            <p className="c-muted" style={{ marginTop: 8 }}>
              Manage API keys for programmatic access to the inference engine.
            </p>
          </div>
          <Button onClick={() => setShowCreateDialog(true)}>
            <Plus size={15} />
            Create API Key
          </Button>
        </div>

        {/* Info card */}
        <Card style={{ marginBottom: 24, borderStyle: "dashed" }}>
          <CardContent style={{ display: "flex", gap: 12, padding: 16 }}>
            <Info size={18} style={{ color: "var(--forge)", flexShrink: 0, marginTop: 2 }} />
            <div>
              <p className="c-muted" style={{ fontSize: 13 }}>
                Use API keys to authenticate programmatic requests. Include your key in the{" "}
                <code
                  style={{
                    fontFamily: "var(--f-mono)",
                    fontSize: 12,
                    background: "var(--paper-2)",
                    padding: "1px 6px",
                    borderRadius: 2,
                  }}
                >
                  X-API-Key
                </code>{" "}
                header. Keys are shown once at creation — save them securely. Never expose keys in
                client-side code.
              </p>
            </div>
          </CardContent>
        </Card>

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
              Loading API keys...
            </p>
          </div>
        ) : keys.length === 0 ? (
          <Card style={{ textAlign: "center", padding: 60, borderStyle: "dashed" }}>
            <Key size={40} style={{ color: "var(--ink-subtle)", marginBottom: 16 }} />
            <h3 className="c-h3" style={{ fontSize: 18, marginBottom: 8 }}>
              No API keys yet
            </h3>
            <p className="c-muted" style={{ maxWidth: 400, margin: "0 auto 20px" }}>
              Create an API key to authenticate programmatic requests to the inference engine, model
              upload, and fraud detection endpoints.
            </p>
            <Button onClick={() => setShowCreateDialog(true)}>
              <Plus size={15} />
              Create your first API key
            </Button>
          </Card>
        ) : (
          <Card>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead style={{ textAlign: "right" }}>Prefix</TableHead>
                  <TableHead style={{ textAlign: "right" }}>Created</TableHead>
                  <TableHead style={{ textAlign: "right" }}>Expires</TableHead>
                  <TableHead style={{ textAlign: "center" }}>Status</TableHead>
                  <TableHead style={{ textAlign: "right" }}>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {keys.map((key) => (
                  <TableRow key={key.id}>
                    <TableCell>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <Shield size={14} style={{ color: "var(--trace)" }} />
                        <span style={{ fontWeight: 500 }}>{key.name}</span>
                      </div>
                    </TableCell>
                    <TableCell
                      style={{
                        textAlign: "right",
                        fontFamily: "var(--f-mono)",
                        fontSize: 12,
                        color: "var(--ink-muted)",
                      }}
                    >
                      {key.prefix}...
                    </TableCell>
                    <TableCell style={{ textAlign: "right", fontSize: 13 }}>
                      {fmtDate(key.created_at)}
                    </TableCell>
                    <TableCell
                      style={{
                        textAlign: "right",
                        fontSize: 13,
                        color: key.expires_at ? "var(--ink-muted)" : "var(--ok)",
                      }}
                    >
                      {key.expires_at ? fmtDate(key.expires_at) : "Never"}
                    </TableCell>
                    <TableCell style={{ textAlign: "center" }}>
                      {key.is_active ? (
                        <span
                          style={{
                            fontSize: 11,
                            padding: "2px 8px",
                            borderRadius: 2,
                            background: "#ddeedf",
                            border: "1px solid #b4d8be",
                            color: "#166534",
                            fontFamily: "var(--f-mono)",
                          }}
                        >
                          Active
                        </span>
                      ) : (
                        <span
                          style={{
                            fontSize: 11,
                            padding: "2px 8px",
                            borderRadius: 2,
                            background: "#fadcdc",
                            border: "1px solid #e4b4b4",
                            color: "#b91c1c",
                            fontFamily: "var(--f-mono)",
                          }}
                        >
                          Revoked
                        </span>
                      )}
                    </TableCell>
                    <TableCell style={{ textAlign: "right" }}>
                      {key.is_active && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleRevoke(key.id, key.name)}
                          disabled={revoking === key.id}
                          title="Revoke key"
                          style={{ color: "var(--risk)" }}
                        >
                          {revoking === key.id ? (
                            <RefreshCw size={14} className="c-spin" />
                          ) : (
                            <Trash2 size={14} />
                          )}
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Card>
        )}

        {/* Create API Key Dialog */}
        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create API Key</DialogTitle>
              <DialogDescription>Generate a new API key for programmatic access.</DialogDescription>
            </DialogHeader>
            <div style={{ display: "flex", flexDirection: "column", gap: 14, marginTop: 12 }}>
              <div>
                <Label htmlFor="keyName">Key name</Label>
                <Input
                  id="keyName"
                  placeholder="e.g., production-client"
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleCreate()}
                />
              </div>
              <div>
                <Label htmlFor="keyExpiry">Expiry (optional, days)</Label>
                <Input
                  id="keyExpiry"
                  type="number"
                  min="1"
                  placeholder="Leave empty for no expiry"
                  value={newKeyExpiry}
                  onChange={(e) => setNewKeyExpiry(e.target.value)}
                />
              </div>
              {error && !createdKey && (
                <div
                  style={{
                    padding: "8px 12px",
                    borderRadius: 2,
                    background: "#fadcdc",
                    border: "1px solid #e4b4b4",
                    color: "#b91c1c",
                    fontSize: 13,
                  }}
                  role="alert"
                >
                  {error}
                </div>
              )}
            </div>
            <DialogFooter>
              <Button variant="secondary" onClick={() => setShowCreateDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreate} disabled={!newKeyName.trim() || creating}>
                {creating ? (
                  <>
                    <RefreshCw size={14} className="c-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Plus size={14} />
                    Create
                  </>
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Show created key dialog */}
        <Dialog open={!!createdKey} onOpenChange={(open) => !open && setCreatedKey(null)}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>API Key Created</DialogTitle>
              <DialogDescription>Save this key now — it will not be shown again.</DialogDescription>
            </DialogHeader>
            {createdKey && (
              <div style={{ marginTop: 12 }}>
                <div
                  style={{
                    padding: 12,
                    borderRadius: 2,
                    background: "var(--paper-2)",
                    border: "1px solid var(--rule)",
                    fontFamily: "var(--f-mono)",
                    fontSize: 13,
                    wordBreak: "break-all",
                    color: "var(--ink)",
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  <code style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
                    {createdKey.key}
                  </code>
                  <button
                    onClick={copyKey}
                    style={{
                      background: "none",
                      border: "1px solid var(--rule)",
                      borderRadius: 2,
                      padding: "4px 8px",
                      cursor: "pointer",
                      color: copied ? "var(--ok)" : "var(--ink-muted)",
                      display: "flex",
                      alignItems: "center",
                      gap: 4,
                      fontSize: 12,
                      flexShrink: 0,
                    }}
                  >
                    {copied ? <Check size={14} /> : <Copy size={14} />}
                    {copied ? "Copied" : "Copy"}
                  </button>
                </div>
                <div
                  style={{
                    marginTop: 12,
                    display: "flex",
                    gap: 8,
                    padding: "10px 12px",
                    borderRadius: 2,
                    background: "#f3e7c4",
                    border: "1px solid #dcc58a",
                    fontSize: 12,
                    color: "#92400e",
                    alignItems: "flex-start",
                  }}
                >
                  <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: 1 }} />
                  <span>
                    This key is only shown once. Copy it to a safe place before closing this dialog.
                  </span>
                </div>
              </div>
            )}
            <DialogFooter>
              <Button
                onClick={() => {
                  setCreatedKey(null);
                  setShowCreateDialog(false);
                }}
              >
                Done
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <style>{`
          .c-spin {
            animation: c-spin 1s linear infinite;
          }
          @keyframes c-spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}</style>
      </section>
    </CrucibleLayout>
  );
}

export const Route = {
  component: ApiKeysPage,
};
