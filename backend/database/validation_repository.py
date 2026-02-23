"""
K-IFRS 1019 DBO Validation System - Validation Repository
=========================================================
Data access layer for validation sessions and errors
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from database.supabase_client import supabase


class ValidationRepository:
    """
    Repository for validation-related database operations
    """

    def __init__(self):
        """Initialize repository with Supabase client"""
        if not supabase:
            raise ValueError("Supabase client not initialized.")
        self.client = supabase

    async def create_session(self, session_data: Dict[str, Any]) -> str:
        """
        Create a new validation session

        Args:
            session_data: Session metadata and summary
        
        Returns:
            str: UUID of created session
        """
        try:
            result = self.client.table('validation_sessions').insert(session_data).execute()
            if result.data and len(result.data) > 0:
                return result.data[0]['id']
            raise Exception("Failed to create validation session")
        except Exception as e:
            print(f"[ValidationRepository] Error creating session: {str(e)}")
            raise

    async def create_errors_batch(self, errors: List[Dict[str, Any]]) -> int:
        """
        Batch insert validation errors

        Args:
            errors: List of error dictionaries
        
        Returns:
            int: Number of errors created
        """
        if not errors:
            return 0
            
        try:
            # Split into chunks if too many errors (Supabase/PostgREST limit)
            chunk_size = 1000
            total_created = 0
            
            for i in range(0, len(errors), chunk_size):
                chunk = errors[i:i + chunk_size]
                result = self.client.table('validation_errors').insert(chunk).execute()
                total_created += len(result.data) if result.data else 0
                
            return total_created
        except Exception as e:
            print(f"[ValidationRepository] Error creating errors batch: {str(e)}")
            raise

    async def get_session(self, session_id: UUID) -> Optional[Dict]:
        """
        Retrieve session details
        """
        try:
            result = self.client.table('validation_sessions') \
                .select('*') \
                .eq('id', str(session_id)) \
                .execute()
            
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"[ValidationRepository] Error getting session: {str(e)}")
            return None

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """
        List validation sessions
        """
        try:
            result = self.client.table('validation_sessions') \
                .select('id, employee_file_name, validation_status, total_errors, created_at, rule_file_id') \
                .order('created_at', desc=True) \
                .limit(limit) \
                .offset(offset) \
                .execute()
            
            return result.data if result.data else []
        except Exception as e:
            print(f"[ValidationRepository] Error listing sessions: {str(e)}")
            return []

    async def get_session_errors(self, session_id: UUID) -> List[Dict]:
        """
        Retrieve all errors for a session
        PostgREST 기본 1000건 제한을 우회하여 전체 오류를 페이지네이션으로 조회
        """
        try:
            all_errors = []
            page_size = 1000
            offset = 0

            while True:
                result = self.client.table('validation_errors') \
                    .select('*') \
                    .eq('session_id', str(session_id)) \
                    .order('row_number', desc=False) \
                    .range(offset, offset + page_size - 1) \
                    .execute()

                batch = result.data if result.data else []
                all_errors.extend(batch)

                if len(batch) < page_size:
                    break
                offset += page_size

            return all_errors
        except Exception as e:
            print(f"[ValidationRepository] Error getting session errors: {str(e)}")
            return []

    async def create_false_positive_feedback(self, feedback) -> Dict[str, Any]:
        """
        Create false positive feedback record
        """
        try:
            data = feedback.dict()
            # Convert UUIDs to strings
            data['error_id'] = str(data['error_id'])
            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}
            
            result = self.client.table('false_positive_feedback').insert(data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            print(f"[ValidationRepository] Error creating feedback: {str(e)}")
            raise

    async def create_user_correction(self, correction) -> Dict[str, Any]:
        """
        Create user correction record
        """
        try:
            data = correction.dict()
            # Convert UUIDs to strings
            data['session_id'] = str(data['session_id'])
            if data.get('error_id'):
                data['error_id'] = str(data['error_id'])
            
            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}
            
            result = self.client.table('user_corrections').insert(data).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            print(f"[ValidationRepository] Error creating correction: {str(e)}")
            raise

