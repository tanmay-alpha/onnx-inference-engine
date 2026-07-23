-- Initial Schema Migration Script for Crucible ONNX Inference Engine
-- Aligned with server/database.py SQLAlchemy ORM Models

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITHOUT TIME ZONE
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email ON users(email);

-- 2. API Keys Table
CREATE TABLE IF NOT EXISTS api_keys (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(64) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    rate_limit INTEGER NOT NULL DEFAULT 100,
    last_used TIMESTAMP WITHOUT TIME ZONE,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITHOUT TIME ZONE
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS ix_api_keys_user_id ON api_keys(user_id);

-- 3. Models Table
CREATE TABLE IF NOT EXISTS models (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    model_type VARCHAR(100) NOT NULL DEFAULT 'generic',
    framework VARCHAR(100) NOT NULL DEFAULT 'onnx',
    file_path VARCHAR NOT NULL,
    file_size INTEGER NOT NULL DEFAULT 0,
    version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    input_schema TEXT,
    output_schema TEXT,
    metadata_json TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_by VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP WITHOUT TIME ZONE,
    usage_count INTEGER NOT NULL DEFAULT 0,
    operators_supported TEXT
);

CREATE INDEX IF NOT EXISTS ix_models_name ON models(name);
CREATE INDEX IF NOT EXISTS ix_models_created_by ON models(created_by);

-- 4. Inference Logs Table
CREATE TABLE IF NOT EXISTS inference_logs (
    id VARCHAR(36) PRIMARY KEY,
    model_id VARCHAR(36) NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    api_key_id VARCHAR(36) REFERENCES api_keys(id) ON DELETE SET NULL,
    user_id VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL,
    input_shape TEXT,
    output_shape TEXT,
    latency_ms DOUBLE PRECISION NOT NULL,
    engine VARCHAR(100) NOT NULL DEFAULT 'unknown',
    status VARCHAR(20) NOT NULL DEFAULT 'success',
    error_message TEXT,
    ip_address VARCHAR(45),
    user_agent VARCHAR(512),
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_inference_logs_created_at ON inference_logs(created_at);
CREATE INDEX IF NOT EXISTS ix_inference_logs_model_id ON inference_logs(model_id);
CREATE INDEX IF NOT EXISTS ix_inference_logs_api_key_id ON inference_logs(api_key_id);
CREATE INDEX IF NOT EXISTS ix_inference_logs_user_id ON inference_logs(user_id);
CREATE INDEX IF NOT EXISTS ix_inference_logs_engine ON inference_logs(engine);
CREATE INDEX IF NOT EXISTS ix_inference_logs_status ON inference_logs(status);
CREATE INDEX IF NOT EXISTS idx_inference_logs_model_id_created_at ON inference_logs(model_id, created_at);

-- 5. Fraud Cases Table
CREATE TABLE IF NOT EXISTS fraud_cases (
    id VARCHAR(36) PRIMARY KEY,
    inference_log_id VARCHAR(36),
    transaction_id VARCHAR(100) NOT NULL UNIQUE,
    amount DOUBLE PRECISION NOT NULL,
    fraud_probability DOUBLE PRECISION NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    risk_level VARCHAR(20) NOT NULL DEFAULT 'low',
    features TEXT,
    reviewed BOOLEAN NOT NULL DEFAULT FALSE,
    reviewed_by VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL,
    review_notes TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP WITHOUT TIME ZONE
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_fraud_cases_transaction_id ON fraud_cases(transaction_id);
CREATE INDEX IF NOT EXISTS ix_fraud_cases_created_at ON fraud_cases(created_at);
CREATE INDEX IF NOT EXISTS ix_fraud_cases_is_fraud ON fraud_cases(is_fraud);
CREATE INDEX IF NOT EXISTS ix_fraud_cases_risk_level ON fraud_cases(risk_level);
CREATE INDEX IF NOT EXISTS ix_fraud_cases_reviewed_by ON fraud_cases(reviewed_by);

-- 6. Benchmarks Table
CREATE TABLE IF NOT EXISTS benchmarks (
    id VARCHAR(36) PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    engine VARCHAR(100) NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    memory_mb DOUBLE PRECISION,
    device VARCHAR(50),
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_benchmarks_created_at ON benchmarks(created_at);
CREATE INDEX IF NOT EXISTS ix_benchmarks_engine ON benchmarks(engine);
