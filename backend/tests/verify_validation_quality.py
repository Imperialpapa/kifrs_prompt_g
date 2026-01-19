
import unittest
import pandas as pd
import sys
import os
import json
from datetime import datetime

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rule_engine import RuleEngine, KIFRS_RuleEngine
from models import ValidationRule, RuleSource

class TestValidationQuality(unittest.TestCase):
    
    def setUp(self):
        self.engine = RuleEngine()
        
    def test_required_validation(self):
        print("\n[Test] Testing 'Required' Validation...")
        data = pd.DataFrame({
            "name": ["Alice", "", None, "Bob"]
        })
        rule = ValidationRule(
            rule_id="R1",
            field_name="name",
            rule_type="required",
            parameters={},
            error_message_template="Name is required",
            source=RuleSource(original_text="Name required", sheet_name="test", row_number=1),
            ai_interpretation_summary="required",
            confidence_score=1.0
        )
        errors = self.engine.validate(data, [rule])
        print(f"Errors found: {len(errors)}")
        self.assertEqual(len(errors), 2) # Empty string and None should be errors
        self.assertEqual(errors[0].row, 3) # Row 3 (Excel index for 2nd item)
        self.assertEqual(errors[1].row, 4) # Row 4 (Excel index for 3rd item)

    def test_no_duplicates_validation(self):
        print("\n[Test] Testing 'No Duplicates' Validation...")
        data = pd.DataFrame({
            "id": ["101", "102", "101", "103", "102"]
        })
        rule = ValidationRule(
            rule_id="R2",
            field_name="id",
            rule_type="no_duplicates",
            parameters={},
            error_message_template="Duplicate ID",
            source=RuleSource(original_text="Unique ID", sheet_name="test", row_number=1),
            ai_interpretation_summary="unique",
            confidence_score=1.0
        )
        errors = self.engine.validate(data, [rule])
        print(f"Errors found: {len(errors)}")
        # Duplicates: 101 (at 0, 2) and 102 (at 1, 4). All 4 should be flagged?
        # The engine uses: data[data.duplicated(subset=[field], keep=False)]
        # keep=False marks ALL duplicates as True.
        self.assertEqual(len(errors), 4)

    def test_format_validation_regex(self):
        print("\n[Test] Testing 'Format (Regex)' Validation...")
        data = pd.DataFrame({
            "date": ["20230101", "2023-01-01", "invalid", "20231231"]
        })
        rule = ValidationRule(
            rule_id="R3",
            field_name="date",
            rule_type="format",
            parameters={"regex": "^[0-9]{8}$"},
            error_message_template="Invalid date format",
            source=RuleSource(original_text="YYYYMMDD", sheet_name="test", row_number=1),
            ai_interpretation_summary="format",
            confidence_score=1.0
        )
        errors = self.engine.validate(data, [rule])
        print(f"Errors found: {len(errors)}")
        self.assertEqual(len(errors), 2) # "2023-01-01" and "invalid" are errors

    def test_range_validation(self):
        print("\n[Test] Testing 'Range' Validation...")
        data = pd.DataFrame({
            "age": [25, 15, 65, 70, "invalid"]
        })
        rule = ValidationRule(
            rule_id="R4",
            field_name="age",
            rule_type="range",
            parameters={"min_value": 20, "max_value": 60},
            error_message_template="Age out of range",
            source=RuleSource(original_text="Age 20-60", sheet_name="test", row_number=1),
            ai_interpretation_summary="range",
            confidence_score=1.0
        )
        errors = self.engine.validate(data, [rule])
        print(f"Errors found: {len(errors)}")
        # 15 < 20 (Error)
        # 65 > 60 (Error)
        # 70 > 60 (Error)
        # "invalid" -> Exception caught -> Error
        self.assertEqual(len(errors), 4)

    def test_date_logic_validation(self):
        print("\n[Test] Testing 'Date Logic' Validation...")
        data = pd.DataFrame({
            "birth_date": ["20000101", "20000101", "20000101"],
            "hire_date":  ["20200101", "19990101", "20000101"]
        })
        # Rule: Hire Date > Birth Date
        rule = ValidationRule(
            rule_id="R5",
            field_name="hire_date",
            rule_type="date_logic",
            parameters={"compare_field": "birth_date", "operator": "greater_than"},
            error_message_template="Hire date must be after birth date",
            source=RuleSource(original_text="Hire > Birth", sheet_name="test", row_number=1),
            ai_interpretation_summary="logic",
            confidence_score=1.0
        )
        errors = self.engine.validate(data, [rule])
        print(f"Errors found: {len(errors)}")
        # Row 1: 2020 > 2000 (Pass)
        # Row 2: 1999 < 2000 (Error)
        # Row 3: 2000 == 2000 (Error, strictly greater)
        self.assertEqual(len(errors), 2)

    def test_composite_validation(self):
        print("\n[Test] Testing 'Composite' Validation...")
        data = pd.DataFrame({
            "code": ["ABC", "", "123", "AB"]
        })
        # Rule: Required AND Regex(^[A-Z]{3}$)
        rule = ValidationRule(
            rule_id="R6",
            field_name="code",
            rule_type="composite",
            parameters={
                "validations": [
                    {"type": "required", "error_message": "Required"},
                    {"type": "format", "parameters": {"regex": "^[A-Z]{3}$"}, "error_message": "Must be 3 uppercase letters"}
                ]
            },
            error_message_template="Composite Error",
            source=RuleSource(original_text="Required, 3 Upper", sheet_name="test", row_number=1),
            ai_interpretation_summary="composite",
            confidence_score=1.0
        )
        errors = self.engine.validate(data, [rule])
        print(f"Errors found: {len(errors)}")
        # "ABC": Pass
        # "": Fail Required
        # "123": Fail Format
        # "AB": Fail Format
        self.assertEqual(len(errors), 3)

    def test_kifrs_engine(self):
        print("\n[Test] Testing 'K-IFRS' Engine...")
        
        # Create 20 normal employees
        data_list = []
        for i in range(20):
            data_list.append({
                "employee_code": f"E{i}",
                "employee_name": f"Name{i}",
                "birth_date": "19800101",
                "hire_date": "20100101",
                "average_wage": 5000000 + (i * 10000) # 5.0M ~ 5.2M
            })
        
        # Add 1 extreme outlier
        data_list.append({
            "employee_code": "E99",
            "employee_name": "Outlier",
            "birth_date": "19800101",
            "hire_date": "20100101",
            "average_wage": 100000000 # 100M
        })
        
        data = pd.DataFrame(data_list)
        
        engine = KIFRS_RuleEngine(data)
        errors = engine.run_all_checks()
        print(f"K-IFRS Errors Found: {len(errors)}")
        for e in errors:
            print(f" - [{e.rule_id}] {e.message}")
        
        # Check for outlier error
        outlier_errors = [e for e in errors if "KIFRS_OUTLIER" in e.rule_id]
        self.assertTrue(len(outlier_errors) > 0, "Should detect wage outliers")
        print("Outlier detection passed.")

if __name__ == '__main__':
    unittest.main()
