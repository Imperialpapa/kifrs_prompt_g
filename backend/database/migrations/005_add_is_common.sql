-- Migration: Add is_common column to rules table
-- 규칙의 공통 적용 여부를 저장하는 컬럼 추가

ALTER TABLE rules 
ADD COLUMN IF NOT EXISTS is_common BOOLEAN DEFAULT false;

-- 인덱스 추가 (검증 시 공통 규칙을 빠르게 찾기 위함)
CREATE INDEX IF NOT EXISTS idx_rules_is_common ON rules(is_common);

COMMENT ON COLUMN rules.is_common IS 'True면 시트명과 관계없이 동일한 필드명을 가진 모든 시트에 규칙 적용';
