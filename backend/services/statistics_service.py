"""
Statistics Service - 데이터 및 규칙 통계 분석
=============================================
규칙별 성능, 오류 빈도, False Positive 비율 등을 분석하여 제공

최적화:
- 세션 조회 limit 100 → 30 (대시보드용으로 충분)
- 오류 집계를 DB 레벨에서 최대한 처리
- 불필요한 필드 조회 제거
"""

from typing import List, Dict, Any
from collections import Counter
from database.supabase_client import supabase
from utils.logger import get_logger

logger = get_logger("statistics_service")


class StatisticsService:
    """통계 데이터 집계 및 분석 서비스"""

    def __init__(self):
        self.client = supabase

    async def get_dashboard_statistics(self) -> Dict[str, Any]:
        """대시보드용 전체 통계 데이터 조회 (최적화)"""
        try:
            # 1. 세션 통계 (최근 30개만 조회 - 대시보드에 충분)
            sessions_res = self.client.table('validation_sessions') \
                .select('total_rows, total_errors, validation_status, created_at') \
                .order('created_at', desc=True) \
                .limit(30) \
                .execute()

            sessions = sessions_res.data
            total_sessions = len(sessions)
            total_rows_validated = sum(s['total_rows'] or 0 for s in sessions)
            total_errors = sum(s['total_errors'] or 0 for s in sessions)
            avg_error_rate = (total_errors / total_rows_validated * 100) if total_rows_validated > 0 else 0

            # 2. 규칙별 오류 순위 (Top 10) - rule_id만 조회하여 DB 전송량 최소화
            errors_res = self.client.table('validation_errors') \
                .select('rule_id, error_message') \
                .order('created_at', desc=True) \
                .limit(500) \
                .execute()

            # Counter를 사용한 효율적 집계
            rule_counter = Counter()
            rule_sample_msg = {}
            for err in errors_res.data:
                rid = err['rule_id']
                rule_counter[rid] += 1
                if rid not in rule_sample_msg:
                    rule_sample_msg[rid] = err['error_message']

            top_error_rules = [
                {'rule_id': rid, 'count': cnt, 'sample_msg': rule_sample_msg.get(rid, '')}
                for rid, cnt in rule_counter.most_common(10)
            ]

            # 3. False Positive 피드백 (rule_id만 조회)
            fp_counts = {}
            try:
                feedback_res = self.client.table('false_positive_feedback') \
                    .select('rule_id') \
                    .eq('is_false_positive', True) \
                    .limit(500) \
                    .execute()

                fp_counter = Counter(fb['rule_id'] for fb in feedback_res.data)
                fp_counts = dict(fp_counter)
            except Exception as fp_err:
                logger.debug(f"FP feedback query skipped: {fp_err}")

            # FP 병합
            for rule in top_error_rules:
                rule['fp_count'] = fp_counts.get(rule['rule_id'], 0)
                rule['accuracy_score'] = self._calculate_accuracy_score(rule['count'], rule['fp_count'])

            # 4. 최근 추이 (최대 10개 세션)
            recent_trend = [
                {"date": s['created_at'][:10], "errors": s['total_errors'] or 0}
                for s in sessions[:10]
            ][::-1]

            return {
                "overview": {
                    "total_sessions": total_sessions,
                    "total_rows_validated": total_rows_validated,
                    "avg_error_rate": round(avg_error_rate, 2)
                },
                "top_error_rules": top_error_rules,
                "recent_trend": recent_trend
            }

        except Exception as e:
            logger.error(f"Error generating stats: {str(e)}")
            return {"error": str(e)}

    def _calculate_accuracy_score(self, error_count: int, fp_count: int) -> int:
        """단순 정확도 점수 계산 (0-100)"""
        if error_count == 0:
            return 100
        fp_rate = fp_count / error_count
        score = 100 * (1 - fp_rate)
        return round(score)
