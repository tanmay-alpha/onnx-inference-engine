import { redirect } from "@tanstack/react-router";
import { getToken } from "./api";

/**
 * `beforeLoad` guard for protected routes: redirects to /login when no JWT
 * is stored. Guarded routes set `ssr: false` so this runs in the browser,
 * where the token (localStorage) lives.
 */
export function requireAuth() {
  if (!getToken()) {
    throw redirect({ to: "/login" });
  }
}
