"""Webhook notification system for Crucible.

Manages webhook registrations and dispatches events:
  - Fraud detection alerts (high probability)
  - Inference completion notifications
  - Model conversion events
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional

import requests

from server.database import get_session_factory
from server.metrics import record_error


# ---------------------------------------------------------------------------
# Webhook event types
# ---------------------------------------------------------------------------
class WebhookEvent:
    FRAUD_DETECTED = "fraud_detected"
    INFERENCE_COMPLETE = "inference_complete"
    MODEL_CONVERTED = "model_converted"
    MODEL_DELETED = "model_deleted"
    SYSTEM_ERROR = "system_error"


# ---------------------------------------------------------------------------
# Webhook management
# ---------------------------------------------------------------------------
def register_webhook(
    url: str,
    events: list[str],
    secret: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """Register a new webhook endpoint.

    Args:
        url: The endpoint URL to POST events to
        events: List of event types to subscribe to
        secret: Optional signing secret (generated if not provided)
        user_id: Optional user ID for ownership tracking

    Returns:
        Webhook registration record
    """
    webhook_id = uuid.uuid4().hex
    if secret is None:
        secret = secrets.token_hex(32)

    return {
        "id": webhook_id,
        "url": url,
        "events": events,
        "secret": secret,
        "user_id": user_id,
        "is_active": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_triggered": None,
        "failure_count": 0,
        "max_failures": 5,
    }


def generate_signature(payload: str, secret: str) -> str:
    """Generate HMAC-SHA256 signature for webhook payload."""
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def send_webhook(webhook: dict, event_type: str, payload: dict) -> bool:
    """Send a webhook notification.

    Implements retry with exponential backoff:
    - 3 retries with 1s, 2s, 4s delays
    - Tracks failures, disables webhook after max_failures consecutive failures

    Args:
        webhook: Webhook registration record
        event_type: Type of event being sent
        payload: Event payload data

    Returns:
        True if webhook was delivered successfully
    """
    if not webhook.get("is_active", False):
        return False

    if event_type not in webhook.get("events", []):
        return False

    envelope = {
        "id": uuid.uuid4().hex,
        "event": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }

    payload_json = json.dumps(envelope, default=str)
    signature = generate_signature(payload_json, webhook["secret"])

    headers = {
        "Content-Type": "application/json",
        "X-Crucible-Event": event_type,
        "X-Crucible-Signature": signature,
        "X-Crucible-Delivery": envelope["id"],
    }

    from server.config import get_settings
    settings = get_settings()

    max_retries = settings.WEBHOOK_MAX_RETRIES
    delays = [1, 2, 4]  # exponential backoff

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                webhook["url"],
                data=payload_json,
                headers=headers,
                timeout=settings.WEBHOOK_TIMEOUT_SEC,
            )

            if resp.status_code < 400:
                # Success — reset failure count
                webhook["failure_count"] = 0
                webhook["last_triggered"] = datetime.now(timezone.utc).isoformat()
                return True

            # Non-retryable errors (4xx except 429)
            if resp.status_code < 500 and resp.status_code != 429:
                return False

        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.ConnectionError:
            pass
        except Exception:
            pass

        # Wait before retry (except on last attempt)
        if attempt < max_retries - 1:
            time.sleep(delays[attempt])

    # All retries failed
    webhook["failure_count"] = webhook.get("failure_count", 0) + 1
    if webhook["failure_count"] >= webhook.get("max_failures", settings.WEBHOOK_MAX_FAILURES):
        webhook["is_active"] = False

    record_error("webhook_delivery")
    return False
