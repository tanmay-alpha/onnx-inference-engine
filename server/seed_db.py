"""Seed a development database with a demo account and the repo's real models.

Creates (idempotently) one user and registers the ONNX files that ship in
``models/`` under that user, analysed by the same pipeline as POST /convert
(onnx.checker + native loader/checker when crucible_py is importable).
Nothing else is generated: no synthetic inference logs, fraud cases or
benchmark numbers.

Usage (repo root):
    SEED_EMAIL=demo@example.com SEED_PASSWORD='...' python -m server.seed_db
"""
from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from server import database, validator  # noqa: E402
from server.auth import hash_password  # noqa: E402
from server.config import get_settings  # noqa: E402

DEMO_MODELS = [
    REPO_ROOT / "models" / "fraud" / "fraud_detector.onnx",
    REPO_ROOT / "models" / "mobilenet_v2.onnx",
    REPO_ROOT / "models" / "resnet18.onnx",
]


def _ensure_user(email: str, password: str) -> str:
    with Session(database._get_sync_engine()) as session:
        user = session.execute(select(database.User).where(database.User.email == email)).scalar_one_or_none()
        if user is None:
            user = database.User(id=uuid.uuid4().hex, email=email,
                                 hashed_password=hash_password(password), full_name="Demo User")
            session.add(user)
            session.commit()
            print(f"created user {email}")
        else:
            print(f"user {email} already exists")
        return user.id


def _register(src: Path, owner_id: str, model_dir: Path) -> None:
    existing = {m["name"] for m in database.list_models(owner_id=owner_id)}
    if src.name in existing:
        print(f"  {src.name}: already registered")
        return
    model_id = uuid.uuid4().hex
    dest = model_dir / f"{model_id}.onnx"
    shutil.copyfile(src, dest)
    try:
        analysis = validator.analyze_model_file(dest)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        print(f"  {src.name}: rejected ({exc})")
        return
    database.save_model(
        model_id=model_id, name=src.name, file_path=str(dest), size_bytes=dest.stat().st_size,
        owner_id=owner_id, opset=analysis.opset, operators_used=analysis.operators_used,
        unsupported_ops=analysis.unsupported_ops, unsupported_reasons=analysis.unsupported_reasons,
        all_supported=analysis.all_supported, inputs=analysis.inputs, outputs=analysis.outputs,
        native_checked=analysis.native_checked,
    )
    status = "supported" if analysis.all_supported else f"unsupported: {analysis.unsupported_reasons}"
    print(f"  {src.name}: registered as {model_id} ({status})")


def main() -> None:
    email = os.environ.get("SEED_EMAIL", "demo@example.com").lower()
    password = os.environ.get("SEED_PASSWORD")
    if not password:
        sys.exit("Set SEED_PASSWORD (and optionally SEED_EMAIL) to seed the demo account.")
    database.init_db()
    owner_id = _ensure_user(email, password)
    model_dir = Path(os.environ.get("CRUCIBLE_MODEL_DIR", get_settings().CRUCIBLE_MODEL_DIR))
    model_dir.mkdir(parents=True, exist_ok=True)
    for src in DEMO_MODELS:
        if src.is_file():
            _register(src, owner_id, model_dir.resolve())
        else:
            print(f"  {src.relative_to(REPO_ROOT)}: not present, skipped")


if __name__ == "__main__":
    main()
