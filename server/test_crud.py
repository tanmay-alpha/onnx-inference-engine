import asyncio
import os
import sys
import uuid
from datetime import datetime
from dotenv import load_dotenv

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
from sqlalchemy import select, update, delete

async def validate_crud():
    print("Beginning Phase 5 CRUD Validation on Supabase PostgreSQL...")
    session_factory = get_session_factory()
    
    async with session_factory() as session:
        # 1. User CRUD
        print("Testing User CRUD...")
        user_id = uuid.uuid4().hex
        test_user = User(
            id=user_id,
            email=f"crud_test_{user_id[:8]}@example.com",
            hashed_password=hash_password("CrudPass123!"),
            full_name="CRUD Test User",
            is_active=True,
            is_admin=False,
        )
        session.add(test_user)
        await session.commit()
        
        # Read
        res = await session.execute(select(User).where(User.id == user_id))
        read_user = res.scalar_one_or_none()
        assert read_user is not None and read_user.full_name == "CRUD Test User"
        
        # Update
        read_user.full_name = "Updated CRUD User"
        await session.commit()
        
        res = await session.execute(select(User).where(User.id == user_id))
        updated_user = res.scalar_one_or_none()
        assert updated_user.full_name == "Updated CRUD User"
        
        # 2. ApiKey CRUD
        print("Testing ApiKey CRUD...")
        key_id = uuid.uuid4().hex
        _, key_hash = generate_api_key()
        test_key = ApiKey(
            id=key_id,
            user_id=user_id,
            key_hash=key_hash,
            name="Test CRUD Key",
            rate_limit=500,
        )
        session.add(test_key)
        await session.commit()
        
        # Read & Update
        res = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
        k = res.scalar_one()
        k.rate_limit = 1000
        await session.commit()
        
        # 3. Model CRUD
        print("Testing Model CRUD...")
        model_id = uuid.uuid4().hex
        test_model = ModelRecord(
            id=model_id,
            name="test_crud_model",
            description="Model created for CRUD testing",
            model_type="testing",
            framework="onnx",
            file_path="/tmp/models/test_crud.onnx",
            file_size=1024,
            created_by=user_id,
        )
        session.add(test_model)
        await session.commit()
        
        res = await session.execute(select(ModelRecord).where(ModelRecord.id == model_id))
        m = res.scalar_one()
        m.usage_count += 1
        await session.commit()
        
        # 4. Inference Log CRUD
        print("Testing InferenceLog CRUD...")
        log_id = uuid.uuid4().hex
        test_log = InferenceLog(
            id=log_id,
            model_id=model_id,
            api_key_id=key_id,
            user_id=user_id,
            latency_ms=12.34,
            status="success",
        )
        session.add(test_log)
        await session.commit()
        
        # 5. Fraud Case CRUD
        print("Testing FraudCase CRUD...")
        fraud_id = uuid.uuid4().hex
        test_fraud = FraudCase(
            id=fraud_id,
            inference_log_id=log_id,
            transaction_id=f"tx_crud_{fraud_id[:8]}",
            amount=99.99,
            fraud_probability=0.88,
            is_fraud=True,
            risk_level="high",
        )
        session.add(test_fraud)
        await session.commit()
        
        # Update fraud case review
        test_fraud.reviewed = True
        test_fraud.reviewed_by = user_id
        test_fraud.review_notes = "Verified in CRUD test"
        await session.commit()
        
        # 6. Benchmark CRUD
        print("Testing Benchmark CRUD...")
        bench_id = uuid.uuid4().hex
        test_bench = Benchmark(
            id=bench_id,
            model_name="crud_bench_model",
            engine="crucible-cpp",
            latency_ms=4.56,
            memory_mb=12.8,
        )
        session.add(test_bench)
        await session.commit()
        
        # Cleanup CRUD test entities
        print("Cleaning up CRUD test entities...")
        await session.delete(test_bench)
        await session.delete(test_fraud)
        await session.delete(test_log)
        await session.delete(test_model)
        await session.delete(test_key)
        await session.delete(test_user)
        await session.commit()

    print("[OK] All CRUD operations (Create, Read, Update, Delete) passed successfully!")

if __name__ == "__main__":
    asyncio.run(validate_crud())
