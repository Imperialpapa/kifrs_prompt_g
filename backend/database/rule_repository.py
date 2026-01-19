"""
K-IFRS 1019 DBO Validation System - Rule Repository
===================================================
Data access layer for rules and related entities

Usage:
    repo = RuleRepository()
    files = await repo.list_rule_files()
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from database.supabase_client import supabase
from utils.logger import debug, info, error


class RuleRepository:
    """
    Repository for rule-related database operations

    Provides CRUD operations for:
    - Rule files (metadata)
    - Individual rules
    - Rule queries and filtering
    """

    def __init__(self):
        """Initialize repository with Supabase client"""
        if not supabase:
            raise ValueError(
                "Supabase client not initialized. "
                "Please configure SUPABASE_URL and SUPABASE_KEY in .env"
            )
        self.client = supabase

    # =========================================================================
    # Rule File Operations
    # =========================================================================

    async def create_rule_file(self, file_data: Dict[str, Any]) -> str:
        """
        Create a new rule file record

        Args:
            file_data: Dictionary with file metadata
                {
                    "file_name": str,
                    "file_version": str (optional),
                    "uploaded_by": str (optional),
                    "total_rules_count": int,
                    "sheet_count": int,
                    "notes": str (optional)
                }

        Returns:
            str: UUID of created rule file

        Raises:
            Exception: If database operation fails
        """
        try:
            result = self.client.table('rule_files').insert(file_data).execute()
            if result.data and len(result.data) > 0:
                return result.data[0]['id']
            raise Exception("Failed to create rule file: No data returned")
        except Exception as e:
            print(f"[RuleRepository] Error creating rule file: {str(e)}")
            raise

    async def get_rule_file(self, file_id: UUID) -> Optional[Dict]:
        """
        Retrieve rule file metadata

        Args:
            file_id: UUID of the rule file

        Returns:
            Dict or None: Rule file metadata
        """
        try:
            result = self.client.table('rule_files') \
                .select('*') \
                .eq('id', str(file_id)) \
                .execute()

            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
        except Exception as e:
            print(f"[RuleRepository] Error getting rule file: {str(e)}")
            return None

    async def list_rule_files(
        self,
        status: str = 'active',
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """
        List all rule files with filtering

        Args:
            status: Filter by status (default: 'active')
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List[Dict]: List of rule file metadata
        """
        try:
            query = self.client.table('rule_files') \
                .select('*') \
                .eq('status', status) \
                .order('uploaded_at', desc=True) \
                .limit(limit) \
                .offset(offset)

            result = query.execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"[RuleRepository] Error listing rule files: {str(e)}")
            return []

    async def update_rule_file(
        self,
        file_id: UUID,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update rule file metadata

        Args:
            file_id: UUID of the rule file
            updates: Dictionary of fields to update

        Returns:
            bool: True if successful
        """
        try:
            updates['updated_at'] = datetime.now().isoformat()
            result = self.client.table('rule_files') \
                .update(updates) \
                .eq('id', str(file_id)) \
                .execute()

            return len(result.data) > 0
        except Exception as e:
            print(f"[RuleRepository] Error updating rule file: {str(e)}")
            return False

    async def archive_rule_file(self, file_id: UUID) -> bool:
        """
        Archive a rule file (soft delete)

        Args:
            file_id: UUID of the rule file

        Returns:
            bool: True if successful
        """
        return await self.update_rule_file(file_id, {'status': 'archived'})

    async def save_original_file(self, file_id: UUID, content: bytes) -> bool:
        """
        원본 파일 바이너리 저장

        Args:
            file_id: UUID of the rule file
            content: Original file bytes

        Returns:
            bool: True if successful
        """
        try:
            import base64
            # PostgreSQL BYTEA로 저장하기 위해 base64 인코딩
            encoded = base64.b64encode(content).decode('utf-8')

            debug(f"Saving original file: {len(content)} bytes -> {len(encoded)} chars base64", "RuleRepository")
            debug(f"Base64 length mod 4: {len(encoded) % 4}", "RuleRepository")

            result = self.client.table('rule_files') \
                .update({
                    'original_file_content': encoded,
                    'file_size_bytes': len(content),
                    'updated_at': datetime.now().isoformat()
                }) \
                .eq('id', str(file_id)) \
                .execute()

            success = len(result.data) > 0
            if success:
                info(f"Original file saved successfully for file_id: {file_id}", "RuleRepository")
            else:
                error(f"Failed to save original file for file_id: {file_id}", "RuleRepository")
            return success
        except Exception as e:
            error(f"Error saving original file: {str(e)}", "RuleRepository")
            return False

    async def get_original_file(self, file_id: UUID) -> Optional[bytes]:
        """
        저장된 원본 파일 바이너리 조회

        Args:
            file_id: UUID of the rule file

        Returns:
            bytes or None: Original file content
        """
        try:
            result = self.client.table('rule_files') \
                .select('original_file_content') \
                .eq('id', str(file_id)) \
                .execute()

            if result.data and result.data[0].get('original_file_content'):
                import base64
                raw_content = result.data[0]['original_file_content']

                debug(f"Raw content length: {len(raw_content)}, first 50 chars: {raw_content[:50]}...", "RuleRepository")

                # Supabase BYTEA 컬럼은 \x... hex 형식으로 반환
                if raw_content.startswith('\\x'):
                    hex_str = raw_content[2:]  # \x 제거
                    debug(f"Detected hex format, converting {len(hex_str)} hex chars", "RuleRepository")
                    try:
                        # hex -> bytes (이게 base64 문자열)
                        base64_bytes = bytes.fromhex(hex_str)
                        # base64 문자열 -> 원본 파일
                        decoded = base64.b64decode(base64_bytes)
                        info(f"Original file loaded (hex->base64): {len(decoded)} bytes", "RuleRepository")
                        return decoded
                    except Exception as e:
                        error(f"Hex decode failed: {e}", "RuleRepository")
                        return None

                # 일반 base64 문자열인 경우
                encoded = raw_content.strip().replace('\n', '').replace('\r', '').replace(' ', '')
                debug(f"After cleanup length: {len(encoded)}, mod 4: {len(encoded) % 4}", "RuleRepository")

                # Base64 패딩 수정 (4의 배수로 맞춤)
                remainder = len(encoded) % 4
                if remainder == 1:
                    error(f"Invalid base64 length ({len(encoded)}), data may be corrupted", "RuleRepository")
                    return None
                elif remainder == 2:
                    encoded += '=='
                elif remainder == 3:
                    encoded += '='

                decoded = base64.b64decode(encoded)
                info(f"Original file loaded successfully: {len(decoded)} bytes", "RuleRepository")
                return decoded
            else:
                debug(f"No original file content found for file_id: {file_id}", "RuleRepository")
            return None
        except Exception as e:
            error(f"Error getting original file: {str(e)}", "RuleRepository")
            return None

    async def clear_ai_interpretation(self, file_id: UUID) -> int:
        """
        해당 파일의 모든 규칙에서 AI 해석 데이터 초기화

        Args:
            file_id: UUID of the rule file

        Returns:
            int: Number of rules cleared
        """
        try:
            result = self.client.table('rules') \
                .update({
                    'ai_rule_id': None,
                    'ai_rule_type': None,
                    'ai_parameters': None,
                    'ai_error_message': None,
                    'ai_interpretation_summary': None,
                    'ai_confidence_score': None,
                    'ai_interpreted_at': None,
                    'ai_model_version': None,
                    'updated_at': datetime.now().isoformat()
                }) \
                .eq('rule_file_id', str(file_id)) \
                .execute()

            return len(result.data) if result.data else 0
        except Exception as e:
            print(f"[RuleRepository] Error clearing AI interpretation: {str(e)}")
            return 0

    async def update_interpretation_status(
        self,
        file_id: UUID,
        status: str,
        engine: str = None
    ) -> bool:
        """
        파일의 해석 상태 업데이트

        Args:
            file_id: UUID of the rule file
            status: 'pending', 'completed', 'failed'
            engine: 'local', 'openai', 'anthropic', 'gemini'

        Returns:
            bool: True if successful
        """
        try:
            updates = {
                'interpretation_status': status,
                'updated_at': datetime.now().isoformat()
            }
            if status == 'completed':
                updates['last_interpreted_at'] = datetime.now().isoformat()
            if engine:
                updates['interpretation_engine'] = engine

            result = self.client.table('rule_files') \
                .update(updates) \
                .eq('id', str(file_id)) \
                .execute()

            return len(result.data) > 0
        except Exception as e:
            print(f"[RuleRepository] Error updating interpretation status: {str(e)}")
            return False

    # =========================================================================
    # Rule Operations
    # =========================================================================

    async def increment_rule_count(self, file_id: UUID) -> bool:
        """Increment the total_rules_count for a file"""
        try:
            # Get current count
            result = self.client.table('rule_files').select('total_rules_count').eq('id', str(file_id)).execute()
            if not result.data:
                return False
            
            current_count = result.data[0]['total_rules_count']
            
            # Update
            self.client.table('rule_files').update({'total_rules_count': current_count + 1}).eq('id', str(file_id)).execute()
            return True
        except Exception as e:
            print(f"[RuleRepo] Error incrementing rule count: {e}")
            return False

    async def create_rules_batch(self, rules: List[Dict[str, Any]]) -> int:
        """
        Batch insert rules

        Args:
            rules: List of rule dictionaries

        Returns:
            int: Number of rules created

        Raises:
            Exception: If batch insert fails
        """
        try:
            # Supabase batch insert
            result = self.client.table('rules').insert(rules).execute()
            return len(result.data) if result.data else 0
        except Exception as e:
            print(f"[RuleRepository] Error creating rules batch: {str(e)}")
            raise

    async def create_single_rule(self, rule_data: Dict[str, Any]) -> Dict:
        """
        Create a single rule record and return it.

        Args:
            rule_data: Dictionary with rule data.

        Returns:
            Dict: The created rule record including its ID.
        """
        try:
            result = self.client.table('rules').insert(rule_data, returning='representation').execute()
            if result.data and len(result.data) > 0:
                return result.data[0]
            raise Exception("Failed to create single rule: No data returned")
        except Exception as e:
            print(f"[RuleRepository] Error creating single rule: {str(e)}")
            raise

    async def get_rules_by_file(
        self,
        file_id: UUID,
        active_only: bool = True
    ) -> List[Dict]:
        """
        Get all rules for a specific file

        Args:
            file_id: UUID of the rule file
            active_only: Only return active rules

        Returns:
            List[Dict]: List of rules
        """
        try:
            query = self.client.table('rules') \
                .select('*') \
                .eq('rule_file_id', str(file_id))

            if active_only:
                query = query.eq('is_active', True)

            result = query.execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"[RuleRepository] Error getting rules by file: {str(e)}")
            return []

    async def get_rules_by_sheet(
        self,
        file_id: UUID,
        canonical_sheet_name: str
    ) -> List[Dict]:
        """
        Get rules for a specific sheet

        Args:
            file_id: UUID of the rule file
            canonical_sheet_name: Normalized sheet name

        Returns:
            List[Dict]: List of rules for the sheet
        """
        try:
            result = self.client.table('rules') \
                .select('*') \
                .eq('rule_file_id', str(file_id)) \
                .eq('canonical_sheet_name', canonical_sheet_name) \
                .eq('is_active', True) \
                .execute()

            return result.data if result.data else []
        except Exception as e:
            print(f"[RuleRepository] Error getting rules by sheet: {str(e)}")
            return []

    async def update_rule_ai_interpretation(
        self,
        rule_id: UUID,
        ai_data: Dict[str, Any]
    ) -> bool:
        """
        Cache AI interpretation for a rule

        Args:
            rule_id: UUID of the rule
            ai_data: AI interpretation data
                {
                    "ai_rule_id": str,
                    "ai_rule_type": str,
                    "ai_parameters": dict,
                    "ai_error_message": str,
                    "ai_interpretation_summary": str,
                    "ai_confidence_score": float,
                    "ai_interpreted_at": str (ISO datetime),
                    "ai_model_version": str
                }

        Returns:
            bool: True if successful
        """
        try:
            result = self.client.table('rules') \
                .update(ai_data) \
                .eq('id', str(rule_id)) \
                .execute()

            return len(result.data) > 0
        except Exception as e:
            print(f"[RuleRepository] Error updating AI interpretation: {str(e)}")
            return False

    async def update_rule_by_field(
        self,
        file_id: UUID,
        sheet_name: str,
        field_name: str,
        ai_data: Dict[str, Any]
    ) -> bool:
        """
        Update rule by matching file_id, sheet_name, and field_name

        Used for caching AI interpretation when we don't have the rule UUID

        Args:
            file_id: UUID of the rule file
            sheet_name: Canonical sheet name
            field_name: Field name
            ai_data: AI interpretation data

        Returns:
            bool: True if at least one rule updated
        """
        try:
            result = self.client.table('rules') \
                .update(ai_data) \
                .eq('rule_file_id', str(file_id)) \
                .eq('canonical_sheet_name', sheet_name) \
                .eq('field_name', field_name) \
                .execute()

            return len(result.data) > 0
        except Exception as e:
            print(f"[RuleRepository] Error updating rule by field: {str(e)}")
            return False

    async def get_rule(self, rule_id: UUID) -> Optional[Dict]:
        """
        Retrieve a single rule by ID

        Args:
            rule_id: UUID of the rule

        Returns:
            Dict or None: Rule data
        """
        try:
            result = self.client.table('rules') \
                .select('*') \
                .eq('id', str(rule_id)) \
                .execute()

            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
        except Exception as e:
            print(f"[RuleRepository] Error getting rule: {str(e)}")
            return None

    async def update_rule(self, rule_id: UUID, updates: Dict[str, Any]) -> bool:
        """
        Update an individual rule

        Args:
            rule_id: UUID of the rule
            updates: Dictionary of fields to update

        Returns:
            bool: True if successful
        """
        try:
            updates['updated_at'] = datetime.now().isoformat()
            
            # Increase version if specific fields are updated (optional logic)
            # current_rule = await self.get_rule(rule_id)
            # if current_rule:
            #     updates['version'] = current_rule.get('version', 1) + 1

            result = self.client.table('rules') \
                .update(updates) \
                .eq('id', str(rule_id)) \
                .execute()

            return len(result.data) > 0
        except Exception as e:
            print(f"[RuleRepository] Error updating rule: {str(e)}")
            return False

    async def delete_rule(self, rule_id: UUID) -> bool:
        """
        Permanently delete a rule

        Args:
            rule_id: UUID of the rule

        Returns:
            bool: True if successful
        """
        try:
            result = self.client.table('rules') \
                .delete() \
                .eq('id', str(rule_id)) \
                .execute()

            return True  # If no exception, consider it successful
        except Exception as e:
            print(f"[RuleRepository] Error deleting rule: {str(e)}")
            return False

    async def deactivate_rule(self, rule_id: UUID) -> bool:
        """
        Soft delete a rule

        Args:
            rule_id: UUID of the rule

        Returns:
            bool: True if successful
        """
        try:
            result = self.client.table('rules') \
                .update({'is_active': False}) \
                .eq('id', str(rule_id)) \
                .execute()

            return len(result.data) > 0
        except Exception as e:
            print(f"[RuleRepository] Error deactivating rule: {str(e)}")
            return False

    # =========================================================================
    # Statistics and Analytics
    # =========================================================================

    async def get_file_statistics(self, file_id: UUID) -> Dict[str, Any]:
        """
        Get statistics for a rule file

        Args:
            file_id: UUID of the rule file

        Returns:
            Dict: Statistics including rule count, sheet count, etc.
        """
        try:
            # Get rule count
            rules = await self.get_rules_by_file(file_id, active_only=True)
            rule_count = len(rules)

            # Get unique sheets
            sheets = set(rule.get('display_sheet_name') for rule in rules if rule.get('display_sheet_name'))
            sheet_count = len(sheets)

            # Get rules with AI interpretation
            interpreted_count = sum(1 for rule in rules if rule.get('ai_rule_id'))

            return {
                'total_rules': rule_count,
                'total_sheets': sheet_count,
                'interpreted_rules': interpreted_count,
                'interpretation_rate': interpreted_count / rule_count if rule_count > 0 else 0
            }
        except Exception as e:
            print(f"[RuleRepository] Error getting file statistics: {str(e)}")
            return {}


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test_repository():
        """Test repository operations"""
        print("=" * 70)
        print("RuleRepository Test")
        print("=" * 70)

        try:
            repo = RuleRepository()
            print("✓ Repository initialized")

            # Test list files
            print("\nListing rule files...")
            files = await repo.list_rule_files(limit=5)
            print(f"Found {len(files)} rule files")

            for file in files:
                print(f"  - {file.get('file_name')} (ID: {file.get('id')})")

            print("\n✓ Repository test completed successfully")

        except Exception as e:
            print(f"\n✗ Repository test failed: {str(e)}")

        print("=" * 70)

    # Run test
    asyncio.run(test_repository())
