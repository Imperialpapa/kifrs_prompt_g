"""
Migration 006 Runner - Add Pattern Status
=========================================
패턴 상태 관리 컬럼 추가 마이그레이션 실행

Usage:
    python run_migration_006.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from database.supabase_client import supabase


def run_migration():
    """Execute migration 006"""
    print("=" * 60)
    print("Migration 006: Add Pattern Status")
    print("=" * 60)

    if not supabase:
        print("ERROR: Supabase client not initialized")
        return False

    # SQL statements to execute
    statements = [
        # 1. Add status column
        """
        ALTER TABLE rule_patterns
        ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'active';
        """,

        # 2. Initialize status based on is_active
        """
        UPDATE rule_patterns SET status = CASE
            WHEN is_active = true THEN 'active'
            ELSE 'inactive'
        END;
        """,

        # 3. Create index for status
        """
        CREATE INDEX IF NOT EXISTS idx_rule_patterns_status ON rule_patterns(status);
        """
    ]

    print("\nIMPORTANT: Supabase Python client cannot execute raw DDL directly.")
    print("Please run the following SQL in the Supabase SQL Editor:")
    print("-" * 60)
    
    for sql in statements:
        print(sql.strip())
        print(";")
    
    print("-" * 60)
    print("Editor URL: https://supabase.com/dashboard/project/_/sql")
    print("=" * 60)

    return True


if __name__ == "__main__":
    run_migration()
