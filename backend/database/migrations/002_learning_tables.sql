-- ============================================================================
-- Learning AI System - Database Schema
-- ============================================================================
-- 학습 기반 규칙 해석 시스템을 위한 테이블
--
-- 실행 방법: Supabase SQL Editor에서 실행
-- ============================================================================

-- 1. 학습된 규칙 패턴 테이블
CREATE TABLE IF NOT EXISTS rule_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- 패턴 식별
    pattern_hash VARCHAR(32) NOT NULL,  -- MD5 해시 (빠른 검색용)
    normalized_text TEXT NOT NULL,       -- 정규화된 텍스트
    original_text TEXT NOT NULL,         -- 원본 텍스트
    field_name_hint VARCHAR(255),        -- 필드명 힌트 (참고용)

    -- AI 해석 결과
    ai_rule_type VARCHAR(50) NOT NULL,
    ai_parameters JSONB DEFAULT '{}',
    ai_error_message TEXT,

    -- 신뢰도 및 통계
    confidence_score DECIMAL(3,2) DEFAULT 0.95,
    usage_count INTEGER DEFAULT 1,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,

    -- 추적
    source_rule_id UUID REFERENCES rules(id) ON DELETE SET NULL,

    -- 메타데이터
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_rule_patterns_hash ON rule_patterns(pattern_hash);
CREATE INDEX IF NOT EXISTS idx_rule_patterns_active ON rule_patterns(is_active);
CREATE INDEX IF NOT EXISTS idx_rule_patterns_usage ON rule_patterns(usage_count DESC);
CREATE INDEX IF NOT EXISTS idx_rule_patterns_confidence ON rule_patterns(confidence_score DESC);

-- 2. 패턴 피드백 테이블
CREATE TABLE IF NOT EXISTS pattern_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- 연결
    rule_id UUID REFERENCES rules(id) ON DELETE CASCADE,
    pattern_id UUID REFERENCES rule_patterns(id) ON DELETE CASCADE,

    -- 피드백 정보
    feedback_type VARCHAR(50) NOT NULL,  -- 'success', 'failure', 'false_positive', 'corrected'
    details JSONB DEFAULT '{}',

    -- 메타데이터
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_pattern_feedback_pattern ON pattern_feedback(pattern_id);
CREATE INDEX IF NOT EXISTS idx_pattern_feedback_type ON pattern_feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_pattern_feedback_created ON pattern_feedback(created_at DESC);

-- 3. 학습 통계 테이블 (일별 집계)
CREATE TABLE IF NOT EXISTS learning_statistics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- 기간
    stat_date DATE NOT NULL,

    -- 패턴 통계
    total_patterns INTEGER DEFAULT 0,
    new_patterns INTEGER DEFAULT 0,

    -- 사용 통계
    total_interpretations INTEGER DEFAULT 0,
    learned_interpretations INTEGER DEFAULT 0,  -- 학습 패턴 사용
    ai_interpretations INTEGER DEFAULT 0,        -- AI 새로 해석

    -- 정확도 통계
    total_validations INTEGER DEFAULT 0,
    successful_validations INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,

    -- 신뢰도 평균
    avg_confidence DECIMAL(3,2),

    -- 메타데이터
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(stat_date)
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_learning_stats_date ON learning_statistics(stat_date DESC);

-- 4. 뷰: 패턴 효과성 요약
CREATE OR REPLACE VIEW v_pattern_effectiveness AS
SELECT
    rp.id,
    rp.original_text,
    rp.ai_rule_type,
    rp.confidence_score,
    rp.usage_count,
    rp.success_count,
    rp.failure_count,
    CASE
        WHEN (rp.success_count + rp.failure_count) > 0
        THEN ROUND(rp.success_count::DECIMAL / (rp.success_count + rp.failure_count), 2)
        ELSE NULL
    END AS success_rate,
    CASE
        WHEN (rp.success_count + rp.failure_count) < 5 THEN 'insufficient_data'
        WHEN rp.success_count::DECIMAL / NULLIF(rp.success_count + rp.failure_count, 0) >= 0.95 THEN 'excellent'
        WHEN rp.success_count::DECIMAL / NULLIF(rp.success_count + rp.failure_count, 0) >= 0.80 THEN 'good'
        WHEN rp.success_count::DECIMAL / NULLIF(rp.success_count + rp.failure_count, 0) >= 0.60 THEN 'needs_improvement'
        ELSE 'critical'
    END AS status,
    rp.created_at,
    rp.updated_at
FROM rule_patterns rp
WHERE rp.is_active = true
ORDER BY rp.usage_count DESC;

-- 5. 함수: 일별 통계 집계
CREATE OR REPLACE FUNCTION aggregate_daily_learning_stats()
RETURNS void AS $$
DECLARE
    today DATE := CURRENT_DATE;
BEGIN
    INSERT INTO learning_statistics (
        stat_date,
        total_patterns,
        new_patterns,
        avg_confidence
    )
    SELECT
        today,
        COUNT(*),
        COUNT(*) FILTER (WHERE DATE(created_at) = today),
        AVG(confidence_score)
    FROM rule_patterns
    WHERE is_active = true
    ON CONFLICT (stat_date)
    DO UPDATE SET
        total_patterns = EXCLUDED.total_patterns,
        new_patterns = EXCLUDED.new_patterns,
        avg_confidence = EXCLUDED.avg_confidence;
END;
$$ LANGUAGE plpgsql;

-- 6. RLS 정책 (Row Level Security)
ALTER TABLE rule_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE pattern_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE learning_statistics ENABLE ROW LEVEL SECURITY;

-- 모든 사용자가 읽기 가능 (필요시 조정)
CREATE POLICY "Allow public read on rule_patterns" ON rule_patterns FOR SELECT USING (true);
CREATE POLICY "Allow public insert on rule_patterns" ON rule_patterns FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update on rule_patterns" ON rule_patterns FOR UPDATE USING (true);

CREATE POLICY "Allow public read on pattern_feedback" ON pattern_feedback FOR SELECT USING (true);
CREATE POLICY "Allow public insert on pattern_feedback" ON pattern_feedback FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow public read on learning_statistics" ON learning_statistics FOR SELECT USING (true);
CREATE POLICY "Allow public all on learning_statistics" ON learning_statistics FOR ALL USING (true);

-- ============================================================================
-- 완료 메시지
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE 'Learning AI System tables created successfully!';
    RAISE NOTICE 'Tables: rule_patterns, pattern_feedback, learning_statistics';
    RAISE NOTICE 'View: v_pattern_effectiveness';
END $$;
