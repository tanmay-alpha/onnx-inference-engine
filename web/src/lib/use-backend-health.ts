import { useCallback, useEffect, useState } from "react";
import { getHealth, getToken, type HealthInfo } from "./api";

export type BackendHealth =
  | { status: "checking" }
  | { status: "online"; info: HealthInfo }
  | { status: "offline"; error: string };

/** Polls GET /health once on mount (and on demand) so pages can show an explicit offline state. */
export function useBackendHealth(): BackendHealth & { recheck: () => void } {
  const [health, setHealth] = useState<BackendHealth>({ status: "checking" });

  const recheck = useCallback(() => {
    setHealth({ status: "checking" });
    getHealth().then(
      (info) => setHealth({ status: "online", info }),
      (e: unknown) =>
        setHealth({ status: "offline", error: e instanceof Error ? e.message : String(e) }),
    );
  }, []);

  useEffect(recheck, [recheck]);

  return { ...health, recheck };
}

/** Client-side login state (read after mount so SSR and hydration agree). */
export function useIsLoggedIn(): boolean | null {
  const [loggedIn, setLoggedIn] = useState<boolean | null>(null);
  useEffect(() => setLoggedIn(!!getToken()), []);
  return loggedIn;
}
