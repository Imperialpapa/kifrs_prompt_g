-- Add status column for better lifecycle management
ALTER TABLE rule_patterns
ADD COLUMN status VARCHAR(20) DEFAULT 'active';

-- Initialize status based on is_active
UPDATE rule_patterns SET status = CASE
    WHEN is_active = true THEN 'active'
    ELSE 'inactive'
END;

-- Create index for status
CREATE INDEX IF NOT EXISTS idx_rule_patterns_status ON rule_patterns(status);
