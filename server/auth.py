"""Authentication for the Crucible server.

One dependency, :func:`require_principal`, protects every non-public route.
It accepts, in this order:

1. ``Authorization: Bearer <JWT>`` issued by ``POST /auth/login``;
2. ``X-API-Key: <user key>`` created via ``POST /auth/api-key`` (stored as a
   SHA-256 hash; inactive/expired keys are rejected);
3. ``X-API-Key: <CRUCIBLE_API_KEY>`` — the optional operator key from the
   environment. The operator is not a user and can see every user's data.

Missing or invalid credentials -> 401. An unset CRUCIBLE_API_KEY simply
disables operator access; it never blocks user authentication.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select

from server.config import get_settings

settings = get_settings()

SECRET_KEY = settings.CRUCIBLE_SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.CRUCIBLE_TOKEN_EXPIRE_MINUTES
API_KEY_PREFIX = settings.CRUCIBLE_API_KEY_PREFIX
KEY_PREFIX_DISPLAY_LEN = 12

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(default=None, max_length=255)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    created_at: str


class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Human-readable name")
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    is_active: bool
    rate_limit: int
    last_used: Optional[str]
    created_at: str
    expires_at: Optional[str]


class ApiKeyCreated(ApiKeyResponse):
    full_key: str  # shown once at creation


@dataclass(frozen=True)
class Principal:
    """The authenticated caller."""
    user_id: Optional[str]          # None for the operator key
    email: Optional[str]
    is_operator: bool = False
    api_key_id: Optional[str] = None
    api_key_rate_limit: Optional[int] = None  # requests/minute for this key (/infer, /inference/batch)

    @property
    def all_owners(self) -> bool:
        """True when the caller may see every user's resources."""
        return self.is_operator


# ---------------------------------------------------------------------------
# Password / token / key helpers
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


# Hash of a random password, computed once at import. Login verifies against
# it when the email is unknown so both paths cost one bcrypt check (no
# account enumeration by timing).
DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))


def create_access_token(user_id: str, email: str, is_admin: bool = False) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "admin": is_admin,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_user_id(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
    sub = payload.get("sub")
    return sub if isinstance(sub, str) and sub else None


def generate_api_key() -> tuple[str, str]:
    """Return (full_key, sha256_hex)."""
    raw = API_KEY_PREFIX + secrets.token_hex(32)
    return raw, hash_api_key(raw)


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _operator_key() -> str:
    # Read the environment at request time so the key can be rotated by a
    # restart-free env change in tests; fall back to settings (.env file).
    return os.environ.get("CRUCIBLE_API_KEY", settings.CRUCIBLE_API_KEY)


_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated: send 'Authorization: Bearer <token>' or 'X-API-Key: <key>'",
    headers={"WWW-Authenticate": "Bearer"},
)


async def _user_from_jwt(token: str) -> Principal:
    from server.database import User, get_session_factory

    user_id = _decode_user_id(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"})
    async with get_session_factory()() as session:
        user = (await session.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return Principal(user_id=user.id, email=user.email)


async def _user_from_api_key(raw_key: str) -> Optional[Principal]:
    from server.database import ApiKey, User, get_session_factory

    key_hash = hash_api_key(raw_key)
    async with get_session_factory()() as session:
        row = (await session.execute(
            select(ApiKey, User).join(User, User.id == ApiKey.user_id).where(ApiKey.key_hash == key_hash)
        )).first()
        if row is None:
            return None
        key, user = row
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        expires = key.expires_at.replace(tzinfo=None) if key.expires_at else None
        if not key.is_active or not user.is_active or (expires is not None and expires <= now):
            raise HTTPException(status_code=401, detail="API key revoked, expired, or owner inactive")
        key.last_used = now
        await session.commit()
        return Principal(user_id=user.id, email=user.email, api_key_id=key.id,
                         api_key_rate_limit=key.rate_limit)


async def require_principal(
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    api_key: Optional[str] = Depends(_api_key_header),
) -> Principal:
    """The single auth dependency for every protected route."""
    if bearer is not None and bearer.credentials:
        return await _user_from_jwt(bearer.credentials)
    if api_key:
        principal = await _user_from_api_key(api_key)
        if principal is not None:
            return principal
        operator_key = _operator_key()
        if operator_key and hmac.compare_digest(api_key.encode(), operator_key.encode()):
            return Principal(user_id=None, email=None, is_operator=True)
        raise HTTPException(status_code=401, detail="Invalid API key")
    raise _UNAUTHORIZED


async def require_user(principal: Principal = Depends(require_principal)) -> Principal:
    """Like require_principal, but rejects the operator key (account routes)."""
    if principal.user_id is None:
        raise HTTPException(status_code=403, detail="This route needs a user account, not the operator key")
    return principal
