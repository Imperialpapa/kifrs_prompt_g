
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.learning_service import LearningService
from services.rule_service import RuleService
from database.supabase_client import supabase

# Mock AI Interpreter
class MockAIInterpreter:
    def interpret_rule(self, rule_text, field_name):
        return {
            "rule_type": "ai_guessed_type",
            "rule_id": "AI_123",
            "parameters": {},
            "error_message": "AI guessed error",
            "confidence_score": 0.5,
            "interpretation_summary": "AI guess",
            "model_version": "mock-ai"
        }

async def test_learning_cycle():
    print("\n[Test] Starting Learning Cycle Integration Test")
    
    # 1. Initialize Services
    learning_service = LearningService(supabase_client=supabase)
    mock_ai = MockAIInterpreter()
    
    # Test Data
    import uuid
    unique_suffix = str(uuid.uuid4())[:8]
    rule_text = f"테스트 규칙: 값은 100 이상이어야 함 ({unique_suffix})"
    field_name = "test_field"
    
    # 2. Smart Interpret (Before Learning)
    # Should use AI (mock) because no pattern exists yet
    print("[Test] 1. Smart Interpret (Before Learning)...")
    result, source = await learning_service.smart_interpret(
        rule_text, field_name, mock_ai, use_learning=True
    )
    print(f"Result: {result['rule_type']}, Source: {source}")
    
    assert source == "ai", "Should use AI before learning"
    assert result["rule_type"] == "ai_guessed_type"
    
    # 3. Learn Pattern (User Correction)
    print("[Test] 2. Saving User Correction (Learning)...")
    # User corrects it to 'range' type
    learned_pattern = await learning_service.save_learned_pattern(
        rule_text=rule_text,
        field_name=field_name,
        ai_rule_type="range",
        ai_parameters={"min_value": 100},
        ai_error_message="100 이상이어야 합니다.",
        confidence_boost=0.1
    )
    print(f"Saved Pattern ID: {learned_pattern.get('id')}")
    
    # 4. Smart Interpret (After Learning)
    # Should now use the learned pattern
    print("[Test] 3. Smart Interpret (After Learning)...")
    result_learned, source_learned = await learning_service.smart_interpret(
        rule_text, field_name, mock_ai, use_learning=True
    )
    print(f"Result: {result_learned['rule_type']}, Source: {source_learned}")
    
    assert source_learned == "learned", "Should use learned pattern"
    assert result_learned["rule_type"] == "range"
    assert result_learned["parameters"]["min_value"] == 100
    
    # 4.5 Create Dummy Rule in DB (for FK constraint)
    print("[Test] 4.5 Creating Dummy Rule in DB...")
    
    # Create Rule File
    file_data = {
        "file_name": "test_learning_cycle.xlsx",
        "total_rules_count": 1,
        "sheet_count": 1,
        "status": "active"
    }
    file_res = supabase.table('rule_files').insert(file_data).execute()
    file_id = file_res.data[0]['id']
    
    # Create Rule
    rule_data = {
        "rule_file_id": file_id,
        "sheet_name": "TestSheet",
        "field_name": field_name,
        "rule_text": rule_text,
        "row_number": 1,
        "column_letter": "A",
        "ai_rule_type": "range", # Matches learned pattern
        "ai_parameters": {"min_value": 100},
        "is_active": True
    }
    rule_res = supabase.table('rules').insert(rule_data).execute()
    real_rule_id = rule_res.data[0]['id']
    
    # 5. Record Feedback
    print("[Test] 4. Recording Validation Feedback...")
    pattern_id = learned_pattern["id"]
    
    await learning_service.record_validation_result(
        rule_id=real_rule_id,
        pattern_id=pattern_id,
        total_rows=100,
        error_count=0 # Success
    )
    
    # 6. Verify Stats
    print("[Test] 5. Verifying Statistics...")
    stats = await learning_service.get_pattern_effectiveness(pattern_id)
    print(f"Stats: {stats}")
    
    assert stats["success_count"] >= 1
    assert stats["usage_count"] >= 1
    
    # Cleanup
    print("[Test] 6. Cleaning up...")
    supabase.table('rules').delete().eq('id', real_rule_id).execute()
    supabase.table('rule_files').delete().eq('id', file_id).execute()
    supabase.table('rule_patterns').delete().eq('id', pattern_id).execute()
    
    print("\n[Test] Cycle Completed Successfully!")

if __name__ == "__main__":
    asyncio.run(test_learning_cycle())
