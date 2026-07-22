import asyncio
import json
import os
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Ensure root .env is loaded
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.auth import hash_password, generate_api_key
from server.database import (
    ApiKey,
    Benchmark,
    FraudCase,
    InferenceLog,
    ModelRecord,
    User,
    get_session_factory,
)
from sqlalchemy import delete


async def seed_database():
    print("Starting database seeding for Crucible on Supabase PostgreSQL...")
    session_factory = get_session_factory()
    
    async with session_factory() as session:
        # Clear existing data in correct dependency order
        print("Clearing existing data...")
        await session.execute(delete(FraudCase))
        await session.execute(delete(InferenceLog))
        await session.execute(delete(Benchmark))
        await session.execute(delete(ApiKey))
        await session.execute(delete(ModelRecord))
        await session.execute(delete(User))
        await session.commit()
        
        # 1. Seed Users (1 admin + 7 regular users)
        print("Seeding Users...")
        users = []
        admin_user = User(
            id=uuid.uuid4().hex,
            email="admin@crucible.ai",
            hashed_password=hash_password("AdminSecure2026!"),
            full_name="Crucible Admin",
            is_active=True,
            is_admin=True,
            created_at=datetime.utcnow() - timedelta(days=60),
            last_login=datetime.utcnow() - timedelta(hours=2),
        )
        users.append(admin_user)

        user_data = [
            ("alice@example.com", "Alice Smith", False),
            ("bob@example.com", "Bob Jones", False),
            ("charlie@finance.com", "Charlie Risk", False),
            ("diana@mlops.io", "Diana Prince", False),
            ("evan@security.org", "Evan Wright", False),
            ("fiona@data.co", "Fiona Gallagher", False),
            ("inactive@test.com", "Inactive User", False),
        ]

        for email, full_name, is_admin in user_data:
            u = User(
                id=uuid.uuid4().hex,
                email=email,
                hashed_password=hash_password("UserPassword123!"),
                full_name=full_name,
                is_active=(email != "inactive@test.com"),
                is_admin=is_admin,
                created_at=datetime.utcnow() - timedelta(days=random.randint(10, 50)),
                last_login=datetime.utcnow() - timedelta(hours=random.randint(1, 48)),
            )
            users.append(u)

        session.add_all(users)
        await session.commit()
        print(f"Created {len(users)} users.")

        # 2. Seed API Keys
        print("Seeding API Keys...")
        api_keys = []
        key_names = ["Dev Environment Key", "Production Service Account", "Staging ML Pipeline", "CLI Execution Key", "Deprecated Legacy Key"]
        
        for i, u in enumerate(users[:5]):
            full_key, key_hash = generate_api_key()
            is_act = i != 4  # one inactive key
            key = ApiKey(
                id=uuid.uuid4().hex,
                user_id=u.id,
                key_hash=key_hash,
                name=key_names[i],
                is_active=is_act,
                rate_limit=1000 if u.is_admin else random.choice([60, 120, 300]),
                last_used=datetime.utcnow() - timedelta(minutes=random.randint(5, 500)),
                created_at=datetime.utcnow() - timedelta(days=30),
                expires_at=datetime.utcnow() + timedelta(days=90) if is_act else datetime.utcnow() - timedelta(days=1),
            )
            api_keys.append(key)

        session.add_all(api_keys)
        await session.commit()
        print(f"Created {len(api_keys)} API keys.")

        # 3. Seed Models
        print("Seeding Models...")
        model_records = []
        models_metadata = [
            {
                "name": "fraud_detection_xgb_v1",
                "description": "Gradient boosted fraud detection model trained on Kaggle creditcard dataset.",
                "model_type": "fraud_detection",
                "framework": "onnx",
                "file_path": "/tmp/models/fraud_detection_xgb_v1.onnx",
                "file_size": 14_500_000,
                "version": "1.0.0",
                "input_schema": json.dumps({"type": "float32", "shape": [1, 29]}),
                "output_schema": json.dumps({"type": "float32", "shape": [1, 2]}),
                "operators_supported": json.dumps(["MatMul", "Add", "Relu", "Sigmoid"]),
            },
            {
                "name": "resnet18_classifier",
                "description": "ResNet-18 computer vision classifier model.",
                "model_type": "vision",
                "framework": "pytorch_onnx",
                "file_path": "/tmp/models/resnet18_classifier.onnx",
                "file_size": 44_700_000,
                "version": "2.1.0",
                "input_schema": json.dumps({"type": "float32", "shape": [1, 3, 224, 224]}),
                "output_schema": json.dumps({"type": "float32", "shape": [1, 1000]}),
                "operators_supported": json.dumps(["Conv", "BatchNormalization", "Relu", "MaxPool", "GlobalAveragePool", "Gemm"]),
            },
            {
                "name": "bert_tiny_intent",
                "description": "Compact BERT model for user intent classification.",
                "model_type": "nlp",
                "framework": "huggingface_onnx",
                "file_path": "/tmp/models/bert_tiny_intent.onnx",
                "file_size": 18_200_000,
                "version": "1.2.0",
                "input_schema": json.dumps({"type": "int64", "shape": [1, 128]}),
                "output_schema": json.dumps({"type": "float32", "shape": [1, 5]}),
                "operators_supported": json.dumps(["Gather", "LayerNormalization", "MatMul", "Softmax"]),
            },
            {
                "name": "transaction_risk_mlp",
                "description": "Multi-layer perceptron for real-time transaction risk scoring.",
                "model_type": "fraud_detection",
                "framework": "onnx",
                "file_path": "/tmp/models/transaction_risk_mlp.onnx",
                "file_size": 2_100_000,
                "version": "1.0.0",
                "input_schema": json.dumps({"type": "float32", "shape": [1, 15]}),
                "output_schema": json.dumps({"type": "float32", "shape": [1, 1]}),
                "operators_supported": json.dumps(["Gemm", "Relu", "Sigmoid"]),
            },
        ]

        for m_data in models_metadata:
            m = ModelRecord(
                id=uuid.uuid4().hex,
                name=m_data["name"],
                description=m_data["description"],
                model_type=m_data["model_type"],
                framework=m_data["framework"],
                file_path=m_data["file_path"],
                file_size=m_data["file_size"],
                version=m_data["version"],
                input_schema=m_data["input_schema"],
                output_schema=m_data["output_schema"],
                metadata_json=json.dumps({"author": "Crucible Team", "accuracy": 0.984}),
                is_active=True,
                created_by=admin_user.id,
                created_at=datetime.utcnow() - timedelta(days=random.randint(5, 20)),
                usage_count=random.randint(150, 1200),
                operators_supported=m_data["operators_supported"],
            )
            model_records.append(m)

        session.add_all(model_records)
        await session.commit()
        print(f"Created {len(model_records)} models.")

        # 4. Seed Inference Logs (~250 logs)
        print("Seeding ~250 Inference Logs...")
        logs = []
        statuses = ["success"] * 9 + ["error"]  # 90% success, 10% error
        ips = ["192.168.1.10", "10.0.4.15", "172.16.0.8", "54.210.12.89", "35.180.2.14"]
        user_agents = [
            "CruciblePythonClient/1.0.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "CrucibleRustCLI/0.2.1",
            "curl/7.81.0",
        ]

        primary_model = model_records[0]
        for i in range(250):
            st = random.choice(statuses)
            m = random.choice(model_records)
            k = random.choice(api_keys)
            u = random.choice(users)
            
            created_time = datetime.utcnow() - timedelta(hours=random.randint(0, 168), minutes=random.randint(0, 59))
            
            log = InferenceLog(
                id=uuid.uuid4().hex,
                model_id=m.id,
                api_key_id=k.id,
                user_id=u.id,
                input_shape=json.dumps([1, 29] if m.model_type == "fraud_detection" else [1, 3, 224, 224]),
                output_shape=json.dumps([1, 2] if m.model_type == "fraud_detection" else [1, 1000]),
                latency_ms=round(random.uniform(1.2, 45.8) if st == "success" else random.uniform(0.1, 5.0), 3),
                status=st,
                error_message=None if st == "success" else "Input shape mismatch: expected [1, 29], got [1, 10]",
                ip_address=random.choice(ips),
                user_agent=random.choice(user_agents),
                created_at=created_time,
            )
            logs.append(log)

        session.add_all(logs)
        await session.commit()
        print(f"Created {len(logs)} inference logs.")

        # 5. Seed Fraud Cases (40 cases: low, medium, high risk)
        print("Seeding 40 Fraud Cases...")
        fraud_cases = []
        risk_levels = ["low", "medium", "high"]
        reviewers = [u for u in users if u.email in ("admin@crucible.ai", "charlie@finance.com", "evan@security.org")]

        for i in range(40):
            tx_id = f"tx_2026_{1000 + i}"
            amount = round(random.uniform(10.0, 4500.0), 2)
            prob = round(random.uniform(0.01, 0.99), 4)
            
            if prob < 0.35:
                risk = "low"
                is_f = False
            elif prob < 0.75:
                risk = "medium"
                is_f = random.choice([True, False])
            else:
                risk = "high"
                is_f = True

            is_rev = random.choice([True, False])
            rev_by = random.choice(reviewers).id if is_rev else None
            notes = f"Transaction flagged as {risk} risk. Confirmed by reviewer." if is_rev else None

            fc = FraudCase(
                id=uuid.uuid4().hex,
                inference_log_id=logs[i].id if i < len(logs) else None,
                transaction_id=tx_id,
                amount=amount,
                fraud_probability=prob,
                is_fraud=is_f,
                risk_level=risk,
                features=json.dumps({"V1": round(random.uniform(-3, 3), 3), "V2": round(random.uniform(-3, 3), 3), "Amount": amount}),
                reviewed=is_rev,
                reviewed_by=rev_by,
                review_notes=notes,
                created_at=datetime.utcnow() - timedelta(hours=random.randint(1, 120)),
                reviewed_at=datetime.utcnow() - timedelta(hours=random.randint(0, 48)) if is_rev else None,
            )
            fraud_cases.append(fc)

        session.add_all(fraud_cases)
        await session.commit()
        print(f"Created {len(fraud_cases)} fraud cases.")

        # 6. Seed Benchmarks (crucible-cpp, onnxruntime, crucible-wasm)
        print("Seeding Benchmarks...")
        benchmarks = []
        engines = ["crucible-cpp", "onnxruntime", "crucible-wasm"]
        bench_models = ["fraud_model.onnx", "resnet18.onnx", "mobilenetv2.onnx", "bert_tiny.onnx"]

        for b_model in bench_models:
            for eng in engines:
                for _ in range(3):
                    lat = round(random.uniform(0.8, 12.5) if eng == "crucible-cpp" else random.uniform(1.2, 15.0) if eng == "onnxruntime" else random.uniform(2.5, 25.0), 3)
                    mem = round(random.uniform(12.5, 45.0) if eng != "crucible-wasm" else random.uniform(5.0, 15.0), 2)
                    b = Benchmark(
                        id=uuid.uuid4().hex,
                        model_name=b_model,
                        engine=eng,
                        latency_ms=lat,
                        memory_mb=mem,
                        device="CPU (x86_64)" if eng != "crucible-wasm" else "Browser (WASM/V8)",
                        created_at=datetime.utcnow() - timedelta(days=random.randint(1, 14)),
                    )
                    benchmarks.append(b)

        session.add_all(benchmarks)
        await session.commit()
        print(f"Created {len(benchmarks)} benchmark records.")

    print("Successfully finished database seeding!")

if __name__ == "__main__":
    asyncio.run(seed_database())
