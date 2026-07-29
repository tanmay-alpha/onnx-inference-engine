import { createFileRoute, redirect } from "@tanstack/react-router";
import { getToken } from "../lib/api";

/**
 * Auth guard for protected routes. Redirects to /login if not authenticated.
 */
export function requireAuth() {
  if (!getToken()) {
    throw redirect({ to: "/login" });
  }
}
