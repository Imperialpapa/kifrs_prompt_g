
import asyncio
import sys
import os
import pytest
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

class MockSupabaseClient:
    def __init__(self):
        self._data = {
            "rule_patterns": [],
            "rule_files": [],
            "rules": [],
            "pattern_feedback": []
        }
        
    def table(self, table_name):
        self.current_table = table_name
        return self
        
    def select(self, *args, **kwargs):
        return self
        
    def insert(self, data):
        if isinstance(data, dict):
            # Assign ID if not present
            if 'id' not in data:
                data['id'] = str(uuid.uuid4())
            self._data[self.current_table].append(data)
            self._last_data = [data]
        elif isinstance(data, list):
             for item in data:
                if 'id' not in item:
                    item['id'] = str(uuid.uuid4())
             self._data[self.current_table].extend(data)
             self._last_data = data
        return self
        
    def update(self, data):
        # Very basic mock update - updates last inserted or all (not accurate but enough for this test flow)
        # For more accuracy we need `eq` handling, implemented below
        self._update_data = data
        return self
        
    def eq(self, column, value):
        # Handle select/update filtering
        # Return filtered data for select, or apply update
        if hasattr(self, '_update_data'):
            # Update logic
            for item in self._data[self.current_table]:
                if item.get(column) == value:
                    item.update(self._update_data)
            del self._update_data
        
        # Filter logic for select/delete
        self._filter_column = column
        self._filter_value = value
        return self

    def gte(self, column, value):
        return self
        
    def order(self, *args, **kwargs):
        return self
        
    def limit(self, *args, **kwargs):
        return self

    def single(self):
        self._is_single = True
        return self

    def execute(self):
        # Return result with .data
        class Result:
            pass
        res = Result()
        
        # Filter data if filter was set
        if hasattr(self, '_filter_column'):
            col = self._filter_column
            val = self._filter_value
            filtered = [item for item in self._data.get(self.current_table, []) if item.get(col) == val]
            # Reset filter
            del self._filter_column
            del self._filter_value
        elif hasattr(self, '_last_data'):
             filtered = self._last_data
             del self._last_data
        else:
             filtered = self._data.get(self.current_table, [])
             
        if hasattr(self, '_is_single') and self._is_single:
            res.data = filtered[0] if filtered else None
            del self._is_single
        else:
            res.data = filtered
             
        return res
        
    def delete(self):
        return self

import uuid

@pytest.mark.asyncio
async def test_learning_cycle():
    print("\n[Test] Starting Learning Cycle Integration Test")
    
    # 1. Initialize Services
    # Use real supabase if available, else mock
    client = supabase if supabase else MockSupabaseClient()
    learning_service = LearningService(supabase_client=client)
    mock_ai = MockAIInterpreter()
    
    # Test Data
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
    file_res = client.table('rule_files').insert(file_data).execute()
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
    rule_res = client.table('rules').insert(rule_data).execute()
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
    client.table('rules').delete().eq('id', real_rule_id).execute()
    client.table('rule_files').delete().eq('id', file_id).execute()
    client.table('rule_patterns').delete().eq('id', pattern_id).execute()
    
    print("\n[Test] Cycle Completed Successfully!")

if __name__ == "__main__":
    asyncio.run(test_learning_cycle())
