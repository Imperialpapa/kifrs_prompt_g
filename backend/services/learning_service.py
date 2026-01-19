"""
Learning AI Service - 학습 기반 규칙 해석 시스템
================================================================
사용자 피드백을 학습하여 규칙 해석 정확도를 지속적으로 개선

핵심 기능:
1. 패턴 학습: 사용자 확정 매핑을 패턴으로 저장
2. 패턴 매칭: 새 규칙 해석 시 학습된 패턴 우선 검색
3. 피드백 수집: 검증 결과(성공/실패)를 수집하여 신뢰도 조정
4. 통계 제공: 학습 현황, 정확도 추이 등

확장 계획:
- 유사도 기반 퍼지 매칭
- 클라우드 AI 모델 파인튜닝 연동
- 조직별 학습 모델 분리
"""

import re
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from uuid import UUID, uuid4


class LearningService:
    """
    학습 기반 규칙 해석 서비스
    """

    def __init__(self, supabase_client=None):
        """
        초기화

        Args:
            supabase_client: Supabase 클라이언트 (없으면 인메모리 모드)
        """
        self.client = supabase_client
        self._memory_patterns: Dict[str, Dict] = {}  # 인메모리 캐시
        self._memory_feedback: List[Dict] = []
        print("[LearningService] Initialized")

    # =========================================================================
    # 1. 패턴 저장 및 검색
    # =========================================================================

    def normalize_rule_text(self, rule_text: str) -> str:
        """
        규칙 텍스트 정규화 (비교용)
        - 소문자 변환
        - 불필요한 공백 제거
        - 특수문자 정규화
        """
        if not rule_text:
            return ""

        normalized = rule_text.lower().strip()
        # 여러 공백을 하나로
        normalized = re.sub(r'\s+', ' ', normalized)
        # 괄호 내용 정규화
        normalized = re.sub(r'\(\s+', '(', normalized)
        normalized = re.sub(r'\s+\)', ')', normalized)

        return normalized

    def compute_pattern_hash(self, rule_text: str, field_name: str = "") -> str:
        """
        패턴 해시 생성 (빠른 검색용)
        """
        normalized = self.normalize_rule_text(rule_text)
        # 필드명은 선택적으로 포함 (범용 패턴 vs 필드 특정 패턴)
        key = f"{normalized}"
        return hashlib.md5(key.encode()).hexdigest()

    async def save_learned_pattern(
        self,
        rule_text: str,
        field_name: str,
        ai_rule_type: str,
        ai_parameters: Dict,
        ai_error_message: str,
        source_rule_id: Optional[str] = None,
        confidence_boost: float = 0.0
    ) -> Dict[str, Any]:
        """
        학습된 패턴 저장

        Args:
            rule_text: 원본 규칙 텍스트
            field_name: 필드명
            ai_rule_type: 검증 유형
            ai_parameters: 파라미터
            ai_error_message: 오류 메시지
            source_rule_id: 원본 규칙 ID (추적용)
            confidence_boost: 신뢰도 보너스

        Returns:
            저장된 패턴 정보
        """
        pattern_hash = self.compute_pattern_hash(rule_text)
        normalized_text = self.normalize_rule_text(rule_text)

        pattern_data = {
            "id": str(uuid4()),
            "pattern_hash": pattern_hash,
            "normalized_text": normalized_text,
            "original_text": rule_text,
            "field_name_hint": field_name,  # 참고용 (엄격 매칭 아님)
            "ai_rule_type": ai_rule_type,
            "ai_parameters": ai_parameters,
            "ai_error_message": ai_error_message,
            "confidence_score": min(1.0, 0.95 + confidence_boost),  # 학습 패턴은 기본 95% + 보너스
            "usage_count": 1,
            "success_count": 0,
            "failure_count": 0,
            "source_rule_id": source_rule_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "is_active": True
        }

        if self.client:
            try:
                # 기존 패턴 확인 (같은 해시)
                existing = self.client.table('rule_patterns') \
                    .select('*') \
                    .eq('pattern_hash', pattern_hash) \
                    .eq('is_active', True) \
                    .execute()

                if existing.data:
                    # 기존 패턴 업데이트 (usage_count 증가)
                    old_pattern = existing.data[0]
                    update_data = {
                        "ai_rule_type": ai_rule_type,
                        "ai_parameters": ai_parameters,
                        "ai_error_message": ai_error_message,
                        "usage_count": old_pattern.get("usage_count", 0) + 1,
                        "confidence_score": min(1.0, old_pattern.get("confidence_score", 0.95) + 0.01),
                        "updated_at": datetime.now().isoformat()
                    }
                    self.client.table('rule_patterns') \
                        .update(update_data) \
                        .eq('id', old_pattern['id']) \
                        .execute()

                    pattern_data = {**old_pattern, **update_data}
                    print(f"[LearningService] Updated pattern: {pattern_hash[:8]}... (usage: {update_data['usage_count']})")
                else:
                    # 새 패턴 저장
                    self.client.table('rule_patterns').insert(pattern_data).execute()
                    print(f"[LearningService] Saved new pattern: {pattern_hash[:8]}...")

            except Exception as e:
                print(f"[LearningService] DB save error, using memory: {e}")
        
        # Always update memory cache
        if pattern_hash in self._memory_patterns:
            # If in cache, update it
            if not self.client: # If no DB, we handle increment here
                self._memory_patterns[pattern_hash]["usage_count"] += 1
                self._memory_patterns[pattern_hash]["confidence_score"] = min(
                    1.0, self._memory_patterns[pattern_hash]["confidence_score"] + 0.01
                )
            # Update content
            self._memory_patterns[pattern_hash].update({
                "ai_rule_type": ai_rule_type,
                "ai_parameters": ai_parameters,
                "ai_error_message": ai_error_message,
                "updated_at": datetime.now().isoformat()
            })
            # Sync with DB data if available
            if self.client:
                 self._memory_patterns[pattern_hash] = pattern_data
        else:
            self._memory_patterns[pattern_hash] = pattern_data

        return pattern_data

    async def find_matching_pattern(
        self,
        rule_text: str,
        field_name: str = "",
        threshold: float = 0.8
    ) -> Optional[Dict[str, Any]]:
        """
        학습된 패턴에서 매칭 검색

        Args:
            rule_text: 검색할 규칙 텍스트
            field_name: 필드명 (참고용)
            threshold: 유사도 임계값

        Returns:
            매칭된 패턴 또는 None
        """
        pattern_hash = self.compute_pattern_hash(rule_text)
        normalized_text = self.normalize_rule_text(rule_text)

        # 1. 인메모리 캐시 우선 확인 (Exact Match)
        if pattern_hash in self._memory_patterns:
            print(f"[LearningService] Cache hit: {pattern_hash[:8]}...")
            return {
                **self._memory_patterns[pattern_hash],
                "match_type": "exact",
                "match_score": 1.0
            }

        # 2. DB 검색 (정확히 일치하는 패턴)
        if self.client:
            try:
                result = self.client.table('rule_patterns') \
                    .select('*') \
                    .eq('pattern_hash', pattern_hash) \
                    .eq('is_active', True) \
                    .execute()

                if result.data:
                    pattern = result.data[0]
                    # Cache valid pattern
                    self._memory_patterns[pattern_hash] = pattern
                    
                    print(f"[LearningService] Exact match found (DB): {pattern['ai_rule_type']} (usage: {pattern.get('usage_count', 1)})")
                    return {
                        **pattern,
                        "match_type": "exact",
                        "match_score": 1.0
                    }

                # 3. 유사 패턴 검색 (DB)
                # TODO: 더 정교한 유사도 알고리즘 (Levenshtein, TF-IDF 등)
                all_patterns = self.client.table('rule_patterns') \
                    .select('*') \
                    .eq('is_active', True) \
                    .order('usage_count', desc=True) \
                    .limit(100) \
                    .execute()

                if all_patterns.data:
                    best_match = None
                    best_score = 0

                    for pattern in all_patterns.data:
                        # Cache loaded patterns for future similar search (optional, maybe too much memory?)
                        # self._memory_patterns[pattern['pattern_hash']] = pattern 
                        
                        score = self._calculate_similarity(
                            normalized_text,
                            pattern.get('normalized_text', '')
                        )
                        if score > best_score and score >= threshold:
                            best_score = score
                            best_match = pattern

                    if best_match:
                        print(f"[LearningService] Similar match found: {best_match['ai_rule_type']} (score: {best_score:.2f})")
                        return {
                            **best_match,
                            "match_type": "similar",
                            "match_score": best_score
                        }

            except Exception as e:
                print(f"[LearningService] DB search error: {e}")

        # 4. 유사 패턴 검색 (인메모리 fallback - DB 없을 때)
        if not self.client:
            for hash_key, pattern in self._memory_patterns.items():
                score = self._calculate_similarity(
                    normalized_text,
                    pattern.get('normalized_text', '')
                )
                if score >= threshold:
                    return {
                        **pattern,
                        "match_type": "similar",
                        "match_score": score
                    }

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트 간의 유사도 계산 (0~1)
        현재: 단순 토큰 기반 Jaccard 유사도
        TODO: 더 정교한 알고리즘으로 교체 (임베딩 기반 등)
        """
        if not text1 or not text2:
            return 0.0

        # 토큰화
        tokens1 = set(re.findall(r'\w+', text1.lower()))
        tokens2 = set(re.findall(r'\w+', text2.lower()))

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard 유사도
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    # =========================================================================
    # 2. 피드백 수집 및 신뢰도 조정
    # =========================================================================

    async def record_feedback(
        self,
        rule_id: str,
        pattern_id: Optional[str],
        feedback_type: str,  # 'success', 'failure', 'false_positive', 'corrected'
        details: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        피드백 기록

        Args:
            rule_id: 규칙 ID
            pattern_id: 사용된 패턴 ID (있는 경우)
            feedback_type: 피드백 유형
            details: 추가 정보

        Returns:
            피드백 기록
        """
        feedback_data = {
            "id": str(uuid4()),
            "rule_id": rule_id,
            "pattern_id": pattern_id,
            "feedback_type": feedback_type,
            "details": details or {},
            "created_at": datetime.now().isoformat()
        }

        if self.client and pattern_id:
            try:
                # 피드백 저장
                self.client.table('pattern_feedback').insert(feedback_data).execute()

                # 패턴 통계 업데이트
                pattern = self.client.table('rule_patterns') \
                    .select('*') \
                    .eq('id', pattern_id) \
                    .single() \
                    .execute()

                if pattern.data:
                    update_data = {"updated_at": datetime.now().isoformat()}

                    if feedback_type == 'success':
                        update_data["success_count"] = pattern.data.get("success_count", 0) + 1
                        # 성공 시 신뢰도 소폭 상승
                        update_data["confidence_score"] = min(
                            1.0,
                            pattern.data.get("confidence_score", 0.95) + 0.005
                        )
                    elif feedback_type in ['failure', 'false_positive']:
                        update_data["failure_count"] = pattern.data.get("failure_count", 0) + 1
                        # 실패 시 신뢰도 하락
                        update_data["confidence_score"] = max(
                            0.5,
                            pattern.data.get("confidence_score", 0.95) - 0.02
                        )
                    elif feedback_type == 'corrected':
                        # 사용자가 수정한 경우 - 새 패턴으로 학습
                        pass

                    self.client.table('rule_patterns') \
                        .update(update_data) \
                        .eq('id', pattern_id) \
                        .execute()

                    print(f"[LearningService] Feedback recorded: {feedback_type} for pattern {pattern_id[:8]}...")

            except Exception as e:
                print(f"[LearningService] Feedback save error: {e}")

        self._memory_feedback.append(feedback_data)
        return feedback_data

    async def record_validation_result(
        self,
        rule_id: str,
        pattern_id: Optional[str],
        total_rows: int,
        error_count: int,
        false_positive_count: int = 0
    ):
        """
        검증 결과 기반 피드백 기록

        Args:
            rule_id: 규칙 ID
            pattern_id: 패턴 ID
            total_rows: 총 검증 행 수
            error_count: 오류 발견 수
            false_positive_count: 오탐 수
        """
        # 성공률 계산
        if total_rows > 0:
            error_rate = error_count / total_rows
            false_positive_rate = false_positive_count / total_rows if false_positive_count else 0

            if false_positive_rate > 0.1:  # 오탐률 10% 초과
                await self.record_feedback(rule_id, pattern_id, 'false_positive', {
                    "total_rows": total_rows,
                    "false_positive_count": false_positive_count,
                    "false_positive_rate": false_positive_rate
                })
            elif error_rate < 0.05:  # 오류율 5% 미만 = 성공적
                await self.record_feedback(rule_id, pattern_id, 'success', {
                    "total_rows": total_rows,
                    "error_count": error_count
                })

    # =========================================================================
    # 3. 통계 및 분석
    # =========================================================================

    async def get_learning_statistics(self) -> Dict[str, Any]:
        """
        학습 통계 조회

        Returns:
            학습 현황 통계
        """
        stats = {
            "total_patterns": 0,
            "active_patterns": 0,
            "total_usage": 0,
            "avg_confidence": 0.0,
            "success_rate": 0.0,
            "top_patterns": [],
            "recent_feedback": [],
            "improvement_trend": []
        }

        if self.client:
            try:
                # 전체 패턴 통계
                patterns = self.client.table('rule_patterns') \
                    .select('*') \
                    .eq('is_active', True) \
                    .execute()

                if patterns.data:
                    stats["total_patterns"] = len(patterns.data)
                    stats["active_patterns"] = len([p for p in patterns.data if p.get('usage_count', 0) > 0])
                    stats["total_usage"] = sum(p.get('usage_count', 0) for p in patterns.data)

                    confidences = [p.get('confidence_score', 0) for p in patterns.data]
                    stats["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0

                    total_success = sum(p.get('success_count', 0) for p in patterns.data)
                    total_failure = sum(p.get('failure_count', 0) for p in patterns.data)
                    total_feedback = total_success + total_failure
                    stats["success_rate"] = total_success / total_feedback if total_feedback > 0 else 0

                    # 상위 패턴 (사용량 기준)
                    sorted_patterns = sorted(patterns.data, key=lambda x: x.get('usage_count', 0), reverse=True)
                    stats["top_patterns"] = [
                        {
                            "id": p["id"],
                            "text_preview": p.get("original_text", "")[:50],
                            "rule_type": p.get("ai_rule_type"),
                            "usage_count": p.get("usage_count", 0),
                            "confidence": p.get("confidence_score", 0)
                        }
                        for p in sorted_patterns[:10]
                    ]

                # 최근 피드백
                feedback = self.client.table('pattern_feedback') \
                    .select('*') \
                    .order('created_at', desc=True) \
                    .limit(20) \
                    .execute()

                if feedback.data:
                    stats["recent_feedback"] = feedback.data

            except Exception as e:
                print(f"[LearningService] Stats error: {e}")

        # 인메모리 통계
        if not stats["total_patterns"] and self._memory_patterns:
            patterns = list(self._memory_patterns.values())
            stats["total_patterns"] = len(patterns)
            stats["total_usage"] = sum(p.get('usage_count', 0) for p in patterns)

        return stats

    async def get_pattern_effectiveness(self, pattern_id: str) -> Dict[str, Any]:
        """
        특정 패턴의 효과성 분석

        Args:
            pattern_id: 패턴 ID

        Returns:
            효과성 분석 결과
        """
        if not self.client:
            return {"error": "DB not connected"}

        try:
            pattern = self.client.table('rule_patterns') \
                .select('*') \
                .eq('id', pattern_id) \
                .single() \
                .execute()

            if not pattern.data:
                return {"error": "Pattern not found"}

            p = pattern.data
            total_feedback = p.get('success_count', 0) + p.get('failure_count', 0)

            return {
                "pattern_id": pattern_id,
                "rule_type": p.get('ai_rule_type'),
                "usage_count": p.get('usage_count', 0),
                "success_count": p.get('success_count', 0),
                "failure_count": p.get('failure_count', 0),
                "success_rate": p.get('success_count', 0) / total_feedback if total_feedback > 0 else None,
                "confidence_score": p.get('confidence_score', 0),
                "recommendation": self._generate_recommendation(p)
            }

        except Exception as e:
            return {"error": str(e)}

    def _generate_recommendation(self, pattern: Dict) -> str:
        """
        패턴 개선 권장사항 생성
        """
        success_count = pattern.get('success_count', 0)
        failure_count = pattern.get('failure_count', 0)
        total = success_count + failure_count

        if total < 5:
            return "데이터 부족 - 더 많은 검증 필요"

        success_rate = success_count / total

        if success_rate >= 0.95:
            return "우수 - 높은 정확도 유지 중"
        elif success_rate >= 0.8:
            return "양호 - 지속적인 모니터링 권장"
        elif success_rate >= 0.6:
            return "개선 필요 - 규칙 파라미터 검토 권장"
        else:
            return "즉시 검토 필요 - 규칙 재설정 권장"

    # =========================================================================
    # 4. 스마트 해석 (학습 + AI 통합)
    # =========================================================================

    async def smart_interpret(
        self,
        rule_text: str,
        field_name: str,
        ai_interpreter,  # AIRuleInterpreter 인스턴스
        use_learning: bool = True,
        similarity_threshold: float = 0.8
    ) -> Tuple[Dict[str, Any], str]:
        """
        스마트 규칙 해석 - 학습 패턴 우선, 없으면 AI 해석

        Args:
            rule_text: 규칙 텍스트
            field_name: 필드명
            ai_interpreter: AI 해석기 인스턴스
            use_learning: 학습 패턴 사용 여부
            similarity_threshold: 유사도 임계값

        Returns:
            Tuple[해석 결과, 해석 소스 ('learned'/'ai')]
        """
        # 1. 학습 패턴 검색
        if use_learning:
            matched_pattern = await self.find_matching_pattern(
                rule_text, field_name, similarity_threshold
            )

            if matched_pattern:
                match_type = matched_pattern.get('match_type', 'exact')
                match_score = matched_pattern.get('match_score', 1.0)

                # 정확 매칭이거나 높은 유사도면 학습 결과 사용
                if match_type == 'exact' or match_score >= 0.9:
                    result = {
                        "rule_type": matched_pattern.get('ai_rule_type'),
                        "rule_id": f"LEARNED_{matched_pattern.get('id', '')[:8]}",
                        "parameters": matched_pattern.get('ai_parameters', {}),
                        "error_message": matched_pattern.get('ai_error_message', ''),
                        "confidence_score": matched_pattern.get('confidence_score', 0.95),
                        "interpretation_summary": f"학습된 패턴 ({match_type}, {match_score:.0%})",
                        "pattern_id": matched_pattern.get('id'),
                        "match_type": match_type,
                        "match_score": match_score
                    }
                    return result, "learned"

        # 2. AI 해석 (학습 패턴 없거나 사용 안 함)
        ai_result = ai_interpreter.interpret_rule(rule_text, field_name)

        return ai_result, "ai"
