"""
Learning AI Service - 학습 기반 규칙 해석 시스템
================================================================
사용자 피드백을 학습하여 규칙 해석 정확도를 지속적으로 개선

핵심 기능:
1. 패턴 학습: 사용자 확정 매핑을 패턴으로 저장 (AI 신뢰도 반영)
2. 패턴 매칭: 
   - 인메모리 캐시 (Exact Match)
   - DB 정밀 매칭
   - 필드명 기반 우선 검색 (Context Aware)
   - 인덱스 기반 전역 유사 검색 (Fast Global Search)
3. 피드백 수집: 검증 결과(성공/실패)를 수집하여 신뢰도 조정
4. 생명주기 관리:
   - 저신뢰 패턴 자동 비활성화 (status='inactive')
   - 패턴 복구 (reactivate)
   - 고성공률 패턴 확정

확장 계획:
- 클라우드 AI 모델 파인튜닝 연동
- 조직별 학습 모델 분리
"""

import re
import json
import hashlib
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid import UUID, uuid4
from collections import Counter
from difflib import SequenceMatcher


class TFIDFCalculator:
    """
    TF-IDF 계산기 - 학습 패턴 코퍼스 기반 텍스트 유사도 계산
    """

    def __init__(self):
        self.document_count = 0
        self.document_frequencies: Dict[str, int] = {}  # 각 토큰이 나타난 문서 수
        self.corpus_tokens: List[set] = []  # 각 문서의 토큰 집합

    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분리"""
        if not text:
            return []
        return re.findall(r'\w+', text.lower())

    def update_corpus(self, texts: List[str]):
        """
        코퍼스 업데이트 - 학습 패턴 추가 시 호출
        """
        self.document_count = len(texts)
        self.document_frequencies = {}
        self.corpus_tokens = []

        for text in texts:
            tokens = set(self.tokenize(text))
            self.corpus_tokens.append(tokens)

            for token in tokens:
                self.document_frequencies[token] = self.document_frequencies.get(token, 0) + 1

    def add_document(self, text: str):
        """단일 문서 추가"""
        tokens = set(self.tokenize(text))
        self.corpus_tokens.append(tokens)
        self.document_count += 1

        for token in tokens:
            self.document_frequencies[token] = self.document_frequencies.get(token, 0) + 1

    def get_idf(self, token: str) -> float:
        """역문서빈도(IDF) 계산"""
        if self.document_count == 0:
            return 0.0

        df = self.document_frequencies.get(token, 0)
        if df == 0:
            return 0.0

        return math.log(self.document_count / df) + 1

    def compute_tfidf_vector(self, text: str) -> Dict[str, float]:
        """TF-IDF 벡터 계산"""
        tokens = self.tokenize(text)
        if not tokens:
            return {}

        # TF 계산 (정규화된 빈도)
        token_counts = Counter(tokens)
        max_count = max(token_counts.values()) if token_counts else 1

        tfidf_vector = {}
        for token, count in token_counts.items():
            tf = count / max_count  # 정규화된 TF
            idf = self.get_idf(token)
            tfidf_vector[token] = tf * idf

        return tfidf_vector

    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """두 TF-IDF 벡터 간 코사인 유사도"""
        if not vec1 or not vec2:
            return 0.0

        # 공통 키에 대해서만 내적 계산
        common_keys = set(vec1.keys()) & set(vec2.keys())
        if not common_keys:
            return 0.0

        dot_product = sum(vec1[k] * vec2[k] for k in common_keys)

        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


def levenshtein_similarity(text1: str, text2: str) -> float:
    """
    Levenshtein 기반 유사도 계산 (difflib.SequenceMatcher 사용)

    Returns:
        0.0 ~ 1.0 사이의 유사도 값
    """
    if not text1 or not text2:
        return 0.0

    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


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
        self._memory_patterns: Dict[str, Dict] = {}  # 인메모리 캐시 (Hash Search용)
        self._memory_feedback: List[Dict] = []
        
        # Pattern Indexing for fast search
        self._cached_pattern_list: List[Dict] = []
        self._last_pattern_sync_time: Optional[datetime] = None
        
        self._tfidf_calculator = TFIDFCalculator()  # TF-IDF 계산기
        self._tfidf_initialized = False  # 코퍼스 초기화 여부
        print("[LearningService] Initialized")

    async def _initialize_tfidf_corpus(self):
        """
        TF-IDF 코퍼스 초기화 - 기존 학습 패턴 로드
        """
        if self._tfidf_initialized:
            return

        try:
            if self.client:
                # DB에서 모든 활성 패턴 로드
                patterns = self.client.table('rule_patterns') \
                    .select('normalized_text') \
                    .eq('is_active', True) \
                    .execute()

                if patterns.data:
                    texts = [p.get('normalized_text', '') for p in patterns.data if p.get('normalized_text')]
                    self._tfidf_calculator.update_corpus(texts)
                    print(f"[LearningService] TF-IDF corpus initialized with {len(texts)} patterns")
            else:
                # 인메모리 모드
                texts = [p.get('normalized_text', '') for p in self._memory_patterns.values() if p.get('normalized_text')]
                if texts:
                    self._tfidf_calculator.update_corpus(texts)

            self._tfidf_initialized = True
        except Exception as e:
            print(f"[LearningService] TF-IDF initialization error: {e}")
            self._tfidf_initialized = True  # 에러나도 다시 시도 안 함

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
        confidence_boost: float = 0.0,
        source_ai_confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        학습된 패턴 저장 또는 업데이트

        기능:
        - 새 패턴 저장: 초기 신뢰도는 AI 신뢰도와 0.95 중 작은 값 사용
        - 기존 패턴 업데이트: 사용 횟수 증가 및 신뢰도 상승
        - 비활성 패턴 복구: 'inactive' 상태인 패턴을 다시 'active'로 전환

        Args:
            rule_text: 원본 규칙 텍스트
            field_name: 필드명
            ai_rule_type: 검증 유형
            ai_parameters: 파라미터
            ai_error_message: 오류 메시지
            source_rule_id: 원본 규칙 ID (추적용)
            confidence_boost: 신뢰도 보너스
            source_ai_confidence: AI 해석 신뢰도 (초기값 반영용)

        Returns:
            저장된 패턴 정보
        """
        pattern_hash = self.compute_pattern_hash(rule_text)
        normalized_text = self.normalize_rule_text(rule_text)

        # 기본 신뢰도 결정 (AI 신뢰도 반영)
        if source_ai_confidence is not None:
            base_confidence = min(0.95, source_ai_confidence)
        else:
            base_confidence = 0.95  # 사용자 확정 기본값

        pattern_data = {
            "id": str(uuid4()),
            "pattern_hash": pattern_hash,
            "normalized_text": normalized_text,
            "original_text": rule_text,
            "field_name_hint": field_name,  # 참고용 (엄격 매칭 아님)
            "ai_rule_type": ai_rule_type,
            "ai_parameters": ai_parameters,
            "ai_error_message": ai_error_message,
            "confidence_score": min(1.0, base_confidence + confidence_boost),
            "usage_count": 1,
            "success_count": 0,
            "failure_count": 0,
            "source_rule_id": source_rule_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "is_active": True,
            "status": "active"
        }

        if self.client:
            try:
                # 기존 패턴 확인 (같은 해시)
                existing = self.client.table('rule_patterns') \
                    .select('*') \
                    .eq('pattern_hash', pattern_hash) \
                    .execute() # is_active 조건 제거하여 비활성 패턴도 검색

                if existing.data:
                    # 기존 패턴 업데이트 (usage_count 증가)
                    old_pattern = existing.data[0]
                    update_data = {
                        "ai_rule_type": ai_rule_type,
                        "ai_parameters": ai_parameters,
                        "ai_error_message": ai_error_message,
                        "usage_count": old_pattern.get("usage_count", 0) + 1,
                        "confidence_score": min(1.0, old_pattern.get("confidence_score", 0.95) + 0.01),
                        "updated_at": datetime.now().isoformat(),
                        "is_active": True, # 재활성화
                        "status": "active" # 상태 업데이트
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
                "updated_at": datetime.now().isoformat(),
                "is_active": True,
                "status": "active"
            })
            # Sync with DB data if available
            if self.client:
                 self._memory_patterns[pattern_hash] = pattern_data
        else:
            self._memory_patterns[pattern_hash] = pattern_data
            # 새 패턴을 TF-IDF 코퍼스에 추가
            if normalized_text and self._tfidf_initialized:
                self._tfidf_calculator.add_document(normalized_text)

        return pattern_data

    async def reactivate_pattern(self, pattern_id: str) -> bool:
        """
        비활성화된 패턴 다시 활성화
        """
        if not self.client:
            return False

        try:
            result = self.client.table('rule_patterns') \
                .update({
                    "status": "active", 
                    "is_active": True,
                    "updated_at": datetime.now().isoformat()
                }) \
                .eq('id', pattern_id) \
                .execute()
            
            if result.data:
                print(f"[LearningService] Reactivated pattern: {pattern_id}")
                return True
        except Exception as e:
            print(f"[LearningService] Reactivate error: {e}")
        
        return False

    def _find_best_match(
        self,
        patterns: List[Dict[str, Any]],
        rule_text: str,
        threshold: float
    ) -> Optional[Dict[str, Any]]:
        """
        주어진 패턴 목록에서 최적의 매칭 패턴 찾기
        """
        if not patterns:
            return None
            
        normalized_text = self.normalize_rule_text(rule_text)
        best_match = None
        best_score = 0.0

        for pattern in patterns:
            score = self._calculate_similarity(
                normalized_text,
                pattern.get('normalized_text', '')
            )
            if score > best_score and score >= threshold:
                best_score = score
                best_match = pattern

        if best_match:
            return {
                **best_match,
                "match_type": "similar",
                "match_score": best_score
            }
        return None

    async def _sync_patterns_from_db(self, max_age_seconds: int = 3600):
        """
        DB에서 패턴을 메모리로 동기화 (주기적 실행)
        """
        now = datetime.now()
        if (self._last_pattern_sync_time and 
            (now - self._last_pattern_sync_time).total_seconds() < max_age_seconds and
            self._cached_pattern_list):
            return

        if self.client:
            try:
                # 활성 패턴 전체 로드 (최대 2000개 제한)
                result = self.client.table('rule_patterns') \
                    .select('*') \
                    .eq('is_active', True) \
                    .gte('confidence_score', 0.5) \
                    .order('usage_count', desc=True) \
                    .limit(2000) \
                    .execute()

                if result.data:
                    self._cached_pattern_list = result.data
                    self._last_pattern_sync_time = now
                    
                    # TF-IDF 코퍼스도 업데이트
                    texts = [p.get('normalized_text', '') for p in result.data if p.get('normalized_text')]
                    self._tfidf_calculator.update_corpus(texts)
                    self._tfidf_initialized = True
                    
                    print(f"[LearningService] Synced {len(self._cached_pattern_list)} patterns to memory index")
            except Exception as e:
                print(f"[LearningService] Pattern sync error: {e}")

    async def find_matching_pattern(
        self,
        rule_text: str,
        field_name: str = "",
        threshold: float = 0.8
    ) -> Optional[Dict[str, Any]]:
        """
        학습된 패턴에서 매칭 검색

        검색 전략:
        1. 인메모리 캐시 확인 (Exact Match)
        2. DB 정확 매칭 확인 (Exact Match)
        3. 필드명 기반 유사 패턴 검색 (Context Match) - 같은 필드의 패턴 우선
        4. 전체 유사 패턴 검색 (Global Match) - 인메모리 인덱스 활용

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
                # 2-1. Exact Match Check
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

                # TF-IDF 코퍼스 초기화 (처음 검색 시 한 번만)
                await self._initialize_tfidf_corpus()

                # 2-2. 같은 필드의 패턴에서 우선 검색 (유사도 매칭)
                if field_name:
                    field_patterns = self.client.table('rule_patterns') \
                        .select('*') \
                        .eq('field_name_hint', field_name) \
                        .eq('is_active', True) \
                        .execute()

                    if field_patterns.data:
                        best_match = self._find_best_match(field_patterns.data, rule_text, threshold)
                        if best_match:
                            print(f"[LearningService] Field match found: {best_match['ai_rule_type']} (score: {best_match['match_score']:.2f})")
                            return best_match

                # 2-3. 전체 유사 패턴 검색 (인메모리 인덱스 활용)
                await self._sync_patterns_from_db()

                if self._cached_pattern_list:
                    best_match = self._find_best_match(self._cached_pattern_list, rule_text, threshold)
                    if best_match:
                        print(f"[LearningService] Global match found (Index): {best_match['ai_rule_type']} (score: {best_match['match_score']:.2f})")
                        return best_match

            except Exception as e:
                print(f"[LearningService] DB search error: {e}")

        # 3. 유사 패턴 검색 (인메모리 fallback - DB 없을 때)
        if not self.client:
            best_match = self._find_best_match(list(self._memory_patterns.values()), rule_text, threshold)
            if best_match:
                return best_match

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트 간의 하이브리드 유사도 계산 (0~1)

        알고리즘:
        - Jaccard 유사도: 30% (토큰 집합 기반)
        - TF-IDF Cosine 유사도: 40% (가중 토큰 기반)
        - Levenshtein 유사도: 30% (문자열 편집 거리 기반)

        이 조합으로 다양한 유형의 유사성을 포착:
        - Jaccard: 동일 단어 포함 여부
        - TF-IDF: 중요 단어에 가중치 부여
        - Levenshtein: 오타/변형 텍스트 처리
        """
        if not text1 or not text2:
            return 0.0

        # 1. Jaccard 유사도 (토큰 기반)
        tokens1 = set(re.findall(r'\w+', text1.lower()))
        tokens2 = set(re.findall(r'\w+', text2.lower()))

        if not tokens1 or not tokens2:
            jaccard_score = 0.0
        else:
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            jaccard_score = intersection / union if union > 0 else 0.0

        # 2. TF-IDF Cosine 유사도
        vec1 = self._tfidf_calculator.compute_tfidf_vector(text1)
        vec2 = self._tfidf_calculator.compute_tfidf_vector(text2)
        tfidf_score = self._tfidf_calculator.cosine_similarity(vec1, vec2)

        # 3. Levenshtein 유사도 (문자열 기반)
        levenshtein_score = levenshtein_similarity(text1, text2)

        # 하이브리드 가중 평균
        # 가중치: Jaccard 30%, TF-IDF 40%, Levenshtein 30%
        hybrid_score = (
            0.30 * jaccard_score +
            0.40 * tfidf_score +
            0.30 * levenshtein_score
        )

        return hybrid_score

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
            학습 현황 통계 (대시보드 차트 데이터 포함)
        """
        stats = {
            "total_patterns": 0,
            "active_patterns": 0,
            "total_usage": 0,
            "avg_confidence": 0.0,
            "success_rate": 0.0,
            "top_patterns": [],
            "recent_feedback": [],
            "improvement_trend": [],
            # 차트용 추가 데이터
            "daily_learning_trend": [],      # 일별 학습 추이 (최근 30일)
            "pattern_type_distribution": {}, # 규칙 유형별 분포
            "confidence_distribution": {},   # 신뢰도 구간별 분포
            "weekly_success_rate": []        # 주간 성공률 추이
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

                    # === 차트용 데이터 계산 ===

                    # 1. 규칙 유형별 분포
                    type_counts = {}
                    for p in patterns.data:
                        rule_type = p.get('ai_rule_type', 'unknown')
                        type_counts[rule_type] = type_counts.get(rule_type, 0) + 1
                    stats["pattern_type_distribution"] = type_counts

                    # 2. 신뢰도 구간별 분포
                    confidence_bins = {
                        "0.6-0.7": 0,
                        "0.7-0.8": 0,
                        "0.8-0.9": 0,
                        "0.9-1.0": 0
                    }
                    for p in patterns.data:
                        conf = p.get('confidence_score', 0)
                        if 0.6 <= conf < 0.7:
                            confidence_bins["0.6-0.7"] += 1
                        elif 0.7 <= conf < 0.8:
                            confidence_bins["0.7-0.8"] += 1
                        elif 0.8 <= conf < 0.9:
                            confidence_bins["0.8-0.9"] += 1
                        elif conf >= 0.9:
                            confidence_bins["0.9-1.0"] += 1
                    stats["confidence_distribution"] = confidence_bins

                    # 3. 일별 학습 추이 (created_at 기준, 최근 30일)
                    daily_counts = {}
                    for p in patterns.data:
                        created_str = p.get('created_at', '')
                        if created_str:
                            # ISO 문자열에서 날짜 부분 추출
                            date_part = created_str[:10]  # YYYY-MM-DD
                            daily_counts[date_part] = daily_counts.get(date_part, 0) + 1

                    # 최근 30일 데이터 생성
                    today = datetime.now().date()
                    daily_trend = []
                    cumulative_total = 0
                    for i in range(29, -1, -1):
                        date = today - timedelta(days=i)
                        date_str = date.isoformat()
                        new_count = daily_counts.get(date_str, 0)
                        cumulative_total += new_count
                        daily_trend.append({
                            "date": date_str,
                            "new_patterns": new_count,
                            "total_patterns": cumulative_total
                        })
                    stats["daily_learning_trend"] = daily_trend

                # 최근 피드백
                feedback = self.client.table('pattern_feedback') \
                    .select('*') \
                    .order('created_at', desc=True) \
                    .limit(20) \
                    .execute()

                if feedback.data:
                    stats["recent_feedback"] = feedback.data

                    # 4. 주간 성공률 추이 (피드백 데이터 기반)
                    weekly_stats = {}
                    for f in feedback.data:
                        created_str = f.get('created_at', '')
                        if created_str:
                            date_part = created_str[:10]
                            # 주 시작일로 변환 (월요일 기준)
                            try:
                                date_obj = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                                week_start = date_obj - timedelta(days=date_obj.weekday())
                                week_key = week_start.strftime('%Y-%m-%d')
                            except:
                                week_key = date_part

                            if week_key not in weekly_stats:
                                weekly_stats[week_key] = {"success": 0, "total": 0}

                            weekly_stats[week_key]["total"] += 1
                            if f.get('feedback_type') == 'success':
                                weekly_stats[week_key]["success"] += 1

                    # 주간 성공률 계산
                    weekly_success_rate = []
                    for week, data in sorted(weekly_stats.items())[-8:]:  # 최근 8주
                        rate = data["success"] / data["total"] if data["total"] > 0 else 0
                        weekly_success_rate.append({
                            "week": week,
                            "success_rate": round(rate, 2),
                            "total_feedback": data["total"]
                        })
                    stats["weekly_success_rate"] = weekly_success_rate

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

    # =========================================================================
    # 5. 자동 학습 시스템
    # =========================================================================

    async def auto_learn_from_validation(
        self,
        rule_id: str,
        rule_text: str,
        field_name: str,
        ai_interpretation: Dict[str, Any],
        validation_success_rate: float,
        total_rows: int
    ) -> Optional[Dict[str, Any]]:
        """
        검증 성공 시 자동 패턴 학습

        조건:
        - 성공률 >= 95%
        - 검증 행 수 >= 10
        - AI 신뢰도 >= 0.8

        Args:
            rule_id: 규칙 ID
            rule_text: 원본 규칙 텍스트
            field_name: 필드명
            ai_interpretation: AI 해석 결과
            validation_success_rate: 검증 성공률 (0.0 ~ 1.0)
            total_rows: 검증된 총 행 수

        Returns:
            저장된 패턴 정보 또는 None (조건 미충족 시)
        """
        # 자동 학습 조건 체크
        if validation_success_rate < 0.95:
            print(f"[AutoLearn] Skip: success rate {validation_success_rate:.1%} < 95%")
            return None

        if total_rows < 10:
            print(f"[AutoLearn] Skip: total rows {total_rows} < 10")
            return None

        ai_confidence = ai_interpretation.get('confidence_score', 0)
        if ai_confidence < 0.8:
            print(f"[AutoLearn] Skip: AI confidence {ai_confidence:.1%} < 80%")
            return None

        # 조건 충족 - 자동 학습
        try:
            pattern = await self.save_learned_pattern(
                rule_text=rule_text,
                field_name=field_name,
                ai_rule_type=ai_interpretation.get('rule_type', 'unknown'),
                ai_parameters=ai_interpretation.get('parameters', {}),
                ai_error_message=ai_interpretation.get('error_message', ''),
                source_rule_id=rule_id,
                confidence_boost=0.05,  # 자동 학습은 낮은 부스트
                source_ai_confidence=ai_confidence  # AI 신뢰도 반영
            )
            print(f"[AutoLearn] Pattern saved: {pattern.get('pattern_hash', '')[:8]}... (success: {validation_success_rate:.1%})")
            return pattern

        except Exception as e:
            print(f"[AutoLearn] Error: {e}")
            return None

    async def deactivate_low_confidence_patterns(
        self,
        threshold: float = 0.6,
        min_feedback: int = 5
    ) -> Dict[str, Any]:
        """
        저신뢰 패턴 자동 비활성화

        조건:
        - 피드백 5회 이상 수집됨
        - 성공률 < 60% (기본값)

        동작:
        - is_active = False 설정
        - status = 'inactive' 설정
        - 메모리 캐시에서 제거

        Args:
            threshold: 성공률 임계값 (기본 0.6 = 60%)
            min_feedback: 최소 피드백 횟수 (기본 5회)

        Returns:
            비활성화된 패턴 통계
        """
        result = {
            "deactivated_count": 0,
            "deactivated_patterns": [],
            "skipped_count": 0
        }

        if not self.client:
            return {"error": "DB not connected"}

        try:
            # 모든 활성 패턴 조회
            patterns = self.client.table('rule_patterns') \
                .select('*') \
                .eq('is_active', True) \
                .execute()

            if not patterns.data:
                return result

            for pattern in patterns.data:
                success_count = pattern.get('success_count', 0)
                failure_count = pattern.get('failure_count', 0)
                total_feedback = success_count + failure_count

                # 최소 피드백 조건 확인
                if total_feedback < min_feedback:
                    result["skipped_count"] += 1
                    continue

                # 성공률 계산
                success_rate = success_count / total_feedback

                # 임계값 미달 시 비활성화
                if success_rate < threshold:
                    self.client.table('rule_patterns') \
                        .update({
                            'is_active': False,
                            'status': 'inactive',
                            'updated_at': datetime.now().isoformat(),
                            'deactivated_reason': f'Low success rate: {success_rate:.1%}'
                        }) \
                        .eq('id', pattern['id']) \
                        .execute()

                    result["deactivated_count"] += 1
                    result["deactivated_patterns"].append({
                        "id": pattern['id'],
                        "text_preview": pattern.get('original_text', '')[:50],
                        "success_rate": success_rate
                    })

                    # 메모리 캐시에서도 제거
                    pattern_hash = pattern.get('pattern_hash')
                    if pattern_hash and pattern_hash in self._memory_patterns:
                        del self._memory_patterns[pattern_hash]

                    print(f"[Maintenance] Deactivated pattern: {pattern['id'][:8]}... (success: {success_rate:.1%})")

            print(f"[Maintenance] Deactivated {result['deactivated_count']} low-confidence patterns")
            return result

        except Exception as e:
            print(f"[Maintenance] Error: {e}")
            return {"error": str(e)}

    async def confirm_high_success_patterns(
        self,
        threshold: float = 0.98,
        consecutive_success: int = 10
    ) -> Dict[str, Any]:
        """
        고성공률 패턴 확정 (신뢰도 1.0 설정)

        조건:
        - 연속 성공 10회 이상
        - 성공률 >= 98%

        Args:
            threshold: 성공률 임계값 (기본 0.98 = 98%)
            consecutive_success: 최소 연속 성공 횟수 (기본 10회)

        Returns:
            확정된 패턴 통계
        """
        result = {
            "confirmed_count": 0,
            "confirmed_patterns": [],
            "skipped_count": 0
        }

        if not self.client:
            return {"error": "DB not connected"}

        try:
            # 활성 패턴 중 아직 확정되지 않은 패턴 (confidence < 1.0)
            patterns = self.client.table('rule_patterns') \
                .select('*') \
                .eq('is_active', True) \
                .lt('confidence_score', 1.0) \
                .execute()

            if not patterns.data:
                return result

            for pattern in patterns.data:
                success_count = pattern.get('success_count', 0)
                failure_count = pattern.get('failure_count', 0)
                total_feedback = success_count + failure_count

                # 최소 성공 횟수 조건 확인
                if success_count < consecutive_success:
                    result["skipped_count"] += 1
                    continue

                # 성공률 계산
                if total_feedback == 0:
                    continue

                success_rate = success_count / total_feedback

                # 임계값 이상이면 확정
                if success_rate >= threshold:
                    self.client.table('rule_patterns') \
                        .update({
                            'confidence_score': 1.0,
                            'updated_at': datetime.now().isoformat()
                        }) \
                        .eq('id', pattern['id']) \
                        .execute()

                    result["confirmed_count"] += 1
                    result["confirmed_patterns"].append({
                        "id": pattern['id'],
                        "text_preview": pattern.get('original_text', '')[:50],
                        "success_rate": success_rate,
                        "success_count": success_count
                    })

                    # 메모리 캐시도 업데이트
                    pattern_hash = pattern.get('pattern_hash')
                    if pattern_hash and pattern_hash in self._memory_patterns:
                        self._memory_patterns[pattern_hash]['confidence_score'] = 1.0

                    print(f"[Maintenance] Confirmed pattern: {pattern['id'][:8]}... (success: {success_rate:.1%}, count: {success_count})")

            print(f"[Maintenance] Confirmed {result['confirmed_count']} high-success patterns")
            return result

        except Exception as e:
            print(f"[Maintenance] Error: {e}")
            return {"error": str(e)}

    async def run_maintenance(self) -> Dict[str, Any]:
        """
        학습 시스템 유지보수 일괄 실행

        - 저신뢰 패턴 비활성화
        - 고성공률 패턴 확정

        Returns:
            유지보수 결과 통계
        """
        print("[Maintenance] Starting learning system maintenance...")

        deactivate_result = await self.deactivate_low_confidence_patterns()
        confirm_result = await self.confirm_high_success_patterns()

        return {
            "deactivation": deactivate_result,
            "confirmation": confirm_result,
            "executed_at": datetime.now().isoformat()
        }
