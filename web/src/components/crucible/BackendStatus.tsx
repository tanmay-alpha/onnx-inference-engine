import { ServerOff, RefreshCw } from "lucide-react";
import { API_BASE } from "../../lib/api";
import type { BackendHealth } from "../../lib/use-backend-health";

/**
 * Explicit "backend offline" banner. Renders nothing while checking or when
 * the API answered /health.
 */
export function BackendOfflineNotice({
  health,
  onRetry,
  children,
}: {
  health: BackendHealth;
  onRetry?: () => void;
  children?: React.ReactNode;
}) {
  if (health.status !== "offline") return null;
  return (
    <div
      role="alert"
      className="c-card"
      style={{
        display: "flex",
        gap: 12,
        alignItems: "flex-start",
        borderColor: "#E4B4B4",
        background: "#FBEFEF",
        marginBottom: 20,
      }}
    >
      <ServerOff size={18} color="var(--risk)" style={{ marginTop: 2, flexShrink: 0 }} />
      <div style={{ flex: 1, fontSize: 13 }}>
        <div style={{ fontWeight: 600, color: "var(--risk)" }}>Backend offline</div>
        <div className="c-muted" style={{ marginTop: 4 }}>
          The Crucible API at <span className="mono">{API_BASE}</span> did not answer{" "}
          <span className="mono">GET /health</span>. Server features (login, model registry, native
          inference, analytics) are unavailable until it is running.
        </div>
        {children && (
          <div className="c-muted" style={{ marginTop: 4 }}>
            {children}
          </div>
        )}
      </div>
      {onRetry && (
        <button
          className="c-btn c-btn-ghost"
          onClick={onRetry}
          style={{ padding: "6px 10px", display: "inline-flex", gap: 6, alignItems: "center" }}
        >
          <RefreshCw size={13} /> Retry
        </button>
      )}
    </div>
  );
}

/** Native engine availability reported by /health (server up, engine missing). */
export function NativeUnavailableNotice({ health }: { health: BackendHealth }) {
  if (health.status !== "online" || health.info.native_available) return null;
  return (
    <div
      role="alert"
      className="c-card"
      style={{ borderColor: "#DCC58A", background: "#FBF4E2", marginBottom: 20, fontSize: 13 }}
    >
      <div style={{ fontWeight: 600, color: "var(--warn)" }}>Native engine unavailable</div>
      <div className="c-muted" style={{ marginTop: 4 }}>
        The API is up but reports <span className="mono">engine: {health.info.engine}</span> — the
        compiled <span className="mono">crucible_py</span> module is not loaded, so server inference
        returns 503. There is no fallback engine.
      </div>
    </div>
  );
}
