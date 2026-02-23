-- ============================================================
-- K-IFRS DBO Validation System - Full Database Reset Migration
-- 기존 마이그레이션 001~009 통합 + 최종 상태 반영
-- 생성일: 2026-02-20
-- ============================================================

-- ============================================================
-- 0. 기존 테이블/뷰/함수 삭제 (순서 중요: FK 의존성 역순)
-- ============================================================

DROP VIEW IF EXISTS v_pattern_effectiveness CASCADE;
DROP FUNCTION IF EXISTS increment_false_positives() CASCADE;
DROP FUNCTION IF EXISTS aggregate_daily_learning_stats() CASCADE;

DROP TABLE IF EXISTS pattern_feedback CASCADE;
DROP TABLE IF EXISTS learning_statistics CASCADE;
DROP TABLE IF EXISTS rule_patterns CASCADE;
DROP TABLE IF EXISTS rule_accuracy_metrics CASCADE;
DROP TABLE IF EXISTS false_positive_feedback CASCADE;
DROP TABLE IF EXISTS user_corrections CASCADE;
DROP TABLE IF EXISTS ai_interpretation_logs CASCADE;
DROP TABLE IF EXISTS validation_errors CASCADE;
DROP TABLE IF EXISTS validation_sessions CASCADE;
DROP TABLE IF EXISTS rules CASCADE;
DROP TABLE IF EXISTS rule_files CASCADE;

-- ============================================================
-- 1. rule_files (규칙 파일 관리)
-- ============================================================

CREATE TABLE rule_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name VARCHAR(255) NOT NULL,
    file_version VARCHAR(50),
    uploaded_by VARCHAR(100) DEFAULT 'system',
    uploaded_at TIMESTAMP DEFAULT NOW(),
    file_size_bytes INTEGER,
    sheet_count INTEGER DEFAULT 0,
    total_rules_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    notes TEXT,
    original_file_url TEXT,
    -- migration 004: 원본 파일 저장 + 해석 상태
    original_file_content BYTEA,
    interpretation_status VARCHAR(20) DEFAULT 'pending',
    last_interpreted_at TIMESTAMP,
    interpretation_engine VARCHAR(20),
    -- timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

COMMENT ON COLUMN rule_files.interpretation_status IS 'pending | completed | failed';
COMMENT ON COLUMN rule_files.interpretation_engine IS 'local | openai | anthropic | gemini';

CREATE INDEX idx_rule_files_status ON rule_files(status);
CREATE INDEX idx_rule_files_uploaded_at ON rule_files(uploaded_at DESC);

-- ============================================================
-- 2. rules (개별 규칙)
-- ============================================================

CREATE TABLE rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_file_id UUID NOT NULL REFERENCES rule_files(id) ON DELETE CASCADE,
    version INTEGER DEFAULT 1,
    -- sheet 관련 (DEPRECATED - migration 008에서 nullable 처리, 하위호환용 유지)
    sheet_name VARCHAR(255),
    display_sheet_name VARCHAR(255),
    canonical_sheet_name VARCHAR(255),
    -- 필드 기반 규칙 관리 (현재 활성)
    row_number VARCHAR(20),          -- migration 007: INTEGER → VARCHAR(20), "5.1" 등 지원
    column_letter VARCHAR(100),      -- migration 009: VARCHAR(10) → VARCHAR(100)
    field_name VARCHAR(255) NOT NULL,
    rule_text TEXT NOT NULL,
    condition VARCHAR(500),
    note TEXT,
    -- AI 해석 결과
    ai_rule_id VARCHAR(50),
    ai_rule_type VARCHAR(50),
    ai_parameters JSONB DEFAULT '{}',
    ai_error_message TEXT,
    ai_interpretation_summary TEXT,
    ai_confidence_score DECIMAL(3,2),
    ai_interpreted_at TIMESTAMP,
    ai_model_version VARCHAR(50),
    -- 상태 플래그
    is_active BOOLEAN DEFAULT true,
    is_common BOOLEAN DEFAULT false,  -- migration 005: 공통 규칙 여부
    -- timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_rules_rule_file_id ON rules(rule_file_id);
CREATE INDEX idx_rules_field_name ON rules(field_name);
CREATE INDEX idx_rules_is_active ON rules(is_active);
CREATE INDEX idx_rules_is_common ON rules(is_common);
CREATE INDEX idx_rules_ai_rule_type ON rules(ai_rule_type);

-- ============================================================
-- 3. validation_sessions (검증 세션)
-- ============================================================

CREATE TABLE validation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_token VARCHAR(100) UNIQUE,
    employee_file_name VARCHAR(255),
    employee_file_url TEXT,
    rule_source_type VARCHAR(20) DEFAULT 'file',
    rule_file_id UUID REFERENCES rule_files(id),
    total_rows INTEGER DEFAULT 0,
    valid_rows INTEGER DEFAULT 0,
    error_rows INTEGER DEFAULT 0,
    total_errors INTEGER DEFAULT 0,
    rules_applied_count INTEGER DEFAULT 0,
    validation_status VARCHAR(20) DEFAULT 'pending',
    ai_processing_time_seconds DECIMAL(10,3),
    validation_processing_time_seconds DECIMAL(10,3),
    system_version VARCHAR(50),
    ai_model_version VARCHAR(50),
    full_results JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

COMMENT ON COLUMN validation_sessions.rule_source_type IS 'file | manual | api';
COMMENT ON COLUMN validation_sessions.validation_status IS 'pending | processing | completed | failed';

CREATE INDEX idx_validation_sessions_created_at ON validation_sessions(created_at DESC);
CREATE INDEX idx_validation_sessions_rule_file_id ON validation_sessions(rule_file_id);

-- ============================================================
-- 4. validation_errors (검증 오류)
-- ============================================================

CREATE TABLE validation_errors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES validation_sessions(id) ON DELETE CASCADE,
    sheet_name VARCHAR(255),
    row_number INTEGER,
    column_name VARCHAR(255),
    rule_id VARCHAR(50),
    error_message TEXT,
    actual_value TEXT,
    expected_value TEXT,
    source_rule_text TEXT,
    user_corrected BOOLEAN DEFAULT false,
    correction_timestamp TIMESTAMP,
    correction_type VARCHAR(50),
    correction_notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_validation_errors_session_id ON validation_errors(session_id);
CREATE INDEX idx_validation_errors_rule_id ON validation_errors(rule_id);
CREATE INDEX idx_validation_errors_row_number ON validation_errors(row_number);

-- ============================================================
-- 5. ai_interpretation_logs (AI 해석 로그)
-- ============================================================

CREATE TABLE ai_interpretation_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_file_id UUID REFERENCES rule_files(id),
    natural_language_rule TEXT,
    sheet_name VARCHAR(255),
    field_name VARCHAR(255),
    interpreted_rule_type VARCHAR(50),
    interpreted_parameters JSONB,
    confidence_score DECIMAL(3,2),
    ai_model_version VARCHAR(50),
    processing_time_seconds DECIMAL(10,3),
    interpretation_quality VARCHAR(20),
    user_feedback TEXT,
    feedback_timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_ai_logs_rule_file_id ON ai_interpretation_logs(rule_file_id);

-- ============================================================
-- 6. false_positive_feedback (오탐 피드백)
-- ============================================================

CREATE TABLE false_positive_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    error_id UUID REFERENCES validation_errors(id),
    session_id UUID REFERENCES validation_sessions(id),
    rule_id VARCHAR(50),
    field_name VARCHAR(255),
    error_message TEXT,
    actual_value TEXT,
    is_false_positive BOOLEAN DEFAULT true,
    user_explanation TEXT,
    suggested_rule_adjustment TEXT,
    feedback_by VARCHAR(100),
    pattern_identified VARCHAR(255),
    applied_to_improve_rules BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_false_positive_rule_id ON false_positive_feedback(rule_id);
CREATE INDEX idx_false_positive_session_id ON false_positive_feedback(session_id);

-- ============================================================
-- 7. rule_accuracy_metrics (규칙 정확도 지표)
-- ============================================================

CREATE TABLE rule_accuracy_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id VARCHAR(50),
    rule_file_id UUID REFERENCES rule_files(id),
    metric_date DATE DEFAULT CURRENT_DATE,
    times_applied INTEGER DEFAULT 0,
    errors_detected INTEGER DEFAULT 0,
    false_positives_reported INTEGER DEFAULT 0,
    false_positive_rate DECIMAL(5,4) DEFAULT 0,
    confidence_trend DECIMAL(3,2),
    accuracy_score DECIMAL(5,2) DEFAULT 100,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_accuracy_rule_id ON rule_accuracy_metrics(rule_id);
CREATE INDEX idx_accuracy_metric_date ON rule_accuracy_metrics(metric_date);

-- ============================================================
-- 8. user_corrections (사용자 수정 이력)
-- ============================================================

CREATE TABLE user_corrections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES validation_sessions(id),
    error_id UUID REFERENCES validation_errors(id),
    original_rule_id VARCHAR(50),
    original_error_message TEXT,
    correction_action VARCHAR(50),
    old_value TEXT,
    new_value TEXT,
    correction_reason TEXT,
    affects_rule_interpretation BOOLEAN DEFAULT false,
    suggested_rule_change TEXT,
    corrected_by VARCHAR(100) DEFAULT 'user',
    -- migration 003: 학습 시스템 확장 컬럼
    confidence_score DECIMAL(3,2),
    is_ai_suggested BOOLEAN DEFAULT false,
    rule_file_id UUID REFERENCES rule_files(id),
    sheet_name VARCHAR(255),
    column_name VARCHAR(255),
    -- timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_corrections_session_id ON user_corrections(session_id);
CREATE INDEX idx_corrections_rule_id ON user_corrections(original_rule_id);
CREATE INDEX idx_learning_lookup ON user_corrections(original_rule_id, column_name, old_value);

-- ============================================================
-- 9. rule_patterns (학습된 규칙 패턴)
-- ============================================================

CREATE TABLE rule_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_hash VARCHAR(32) NOT NULL,
    normalized_text TEXT,
    original_text TEXT,
    field_name_hint VARCHAR(255),
    ai_rule_type VARCHAR(50),
    ai_parameters JSONB DEFAULT '{}',
    ai_error_message TEXT,
    confidence_score DECIMAL(3,2) DEFAULT 0.95,
    usage_count INTEGER DEFAULT 1,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    source_rule_id UUID REFERENCES rules(id) ON DELETE SET NULL,
    is_active BOOLEAN DEFAULT true,
    status VARCHAR(20) DEFAULT 'active',  -- migration 006
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON COLUMN rule_patterns.status IS 'active | inactive | deprecated';

CREATE INDEX idx_rule_patterns_hash ON rule_patterns(pattern_hash);
CREATE INDEX idx_rule_patterns_active ON rule_patterns(is_active);
CREATE INDEX idx_rule_patterns_usage ON rule_patterns(usage_count DESC);
CREATE INDEX idx_rule_patterns_confidence ON rule_patterns(confidence_score DESC);
CREATE INDEX idx_rule_patterns_status ON rule_patterns(status);

-- ============================================================
-- 10. pattern_feedback (패턴 피드백)
-- ============================================================

CREATE TABLE pattern_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id UUID REFERENCES rules(id) ON DELETE SET NULL,
    pattern_id UUID REFERENCES rule_patterns(id) ON DELETE CASCADE,
    feedback_type VARCHAR(50) NOT NULL,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_pattern_feedback_pattern ON pattern_feedback(pattern_id);
CREATE INDEX idx_pattern_feedback_type ON pattern_feedback(feedback_type);
CREATE INDEX idx_pattern_feedback_created ON pattern_feedback(created_at DESC);

-- ============================================================
-- 11. learning_statistics (학습 통계)
-- ============================================================

CREATE TABLE learning_statistics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stat_date DATE UNIQUE NOT NULL,
    total_patterns INTEGER DEFAULT 0,
    new_patterns INTEGER DEFAULT 0,
    total_interpretations INTEGER DEFAULT 0,
    learned_interpretations INTEGER DEFAULT 0,
    ai_interpretations INTEGER DEFAULT 0,
    total_validations INTEGER DEFAULT 0,
    successful_validations INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    avg_confidence DECIMAL(3,2) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- 12. Functions
-- ============================================================

-- 오탐 카운터 증가 함수
CREATE OR REPLACE FUNCTION increment_false_positives()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_false_positive = true THEN
        UPDATE rule_accuracy_metrics
        SET false_positives_reported = false_positives_reported + 1,
            false_positive_rate = (false_positives_reported + 1)::DECIMAL / GREATEST(times_applied, 1),
            updated_at = NOW()
        WHERE rule_id = NEW.rule_id
          AND metric_date = CURRENT_DATE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 오탐 피드백 트리거
DROP TRIGGER IF EXISTS trigger_increment_false_positives ON false_positive_feedback;
CREATE TRIGGER trigger_increment_false_positives
    AFTER INSERT ON false_positive_feedback
    FOR EACH ROW
    EXECUTE FUNCTION increment_false_positives();

-- 일일 학습 통계 집계 함수
CREATE OR REPLACE FUNCTION aggregate_daily_learning_stats()
RETURNS void AS $$
BEGIN
    INSERT INTO learning_statistics (stat_date, total_patterns, new_patterns, avg_confidence)
    VALUES (
        CURRENT_DATE,
        (SELECT COUNT(*) FROM rule_patterns WHERE is_active = true),
        (SELECT COUNT(*) FROM rule_patterns WHERE DATE(created_at) = CURRENT_DATE),
        (SELECT COALESCE(AVG(confidence_score), 0) FROM rule_patterns WHERE is_active = true)
    )
    ON CONFLICT (stat_date) DO UPDATE SET
        total_patterns = EXCLUDED.total_patterns,
        new_patterns = EXCLUDED.new_patterns,
        avg_confidence = EXCLUDED.avg_confidence;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- 13. Views
-- ============================================================

CREATE OR REPLACE VIEW v_pattern_effectiveness AS
SELECT
    rp.id AS pattern_id,
    rp.normalized_text,
    rp.ai_rule_type,
    rp.confidence_score,
    rp.usage_count,
    rp.success_count,
    rp.failure_count,
    rp.status,
    CASE
        WHEN rp.usage_count > 0
        THEN ROUND(rp.success_count::DECIMAL / rp.usage_count, 2)
        ELSE 0
    END AS success_rate,
    (SELECT COUNT(*) FROM pattern_feedback pf WHERE pf.pattern_id = rp.id AND pf.feedback_type = 'positive') AS positive_feedback,
    (SELECT COUNT(*) FROM pattern_feedback pf WHERE pf.pattern_id = rp.id AND pf.feedback_type = 'negative') AS negative_feedback
FROM rule_patterns rp
WHERE rp.is_active = true;

-- ============================================================
-- 14. RLS 비활성화 (개발 환경)
-- ============================================================

ALTER TABLE rule_files DISABLE ROW LEVEL SECURITY;
ALTER TABLE rules DISABLE ROW LEVEL SECURITY;
ALTER TABLE validation_sessions DISABLE ROW LEVEL SECURITY;
ALTER TABLE validation_errors DISABLE ROW LEVEL SECURITY;
ALTER TABLE ai_interpretation_logs DISABLE ROW LEVEL SECURITY;
ALTER TABLE false_positive_feedback DISABLE ROW LEVEL SECURITY;
ALTER TABLE rule_accuracy_metrics DISABLE ROW LEVEL SECURITY;
ALTER TABLE user_corrections DISABLE ROW LEVEL SECURITY;
ALTER TABLE rule_patterns DISABLE ROW LEVEL SECURITY;
ALTER TABLE pattern_feedback DISABLE ROW LEVEL SECURITY;
ALTER TABLE learning_statistics DISABLE ROW LEVEL SECURITY;

-- ============================================================
-- 완료!
-- 총 11개 테이블, 1개 뷰, 2개 함수, 1개 트리거 생성
-- ============================================================
