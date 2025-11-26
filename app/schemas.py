"""
Pydantic 스키마 정의

FastAPI의 요청/응답 데이터 모델을 정의합니다.
Pydantic을 사용하여 자동 검증 및 문서화를 제공합니다.
"""

from __future__ import annotations
from typing import List, Optional, Dict
from datetime import datetime
from pydantic import BaseModel
from enum import Enum


# ===========================
# 기본 추천 스키마 (Week 1)
# ===========================

class RecommendationItem(BaseModel):
    """
    개별 추천 문제 정보

    모든 추천 API에서 공통으로 사용하는 문제 정보 형식
    """
    problem_id: int              # 문제 번호 (예: 1000, 1234)
    title: str                   # 문제 제목 (예: "A+B", "DFS와 BFS")
    difficulty: str              # 난이도 (bronze/silver/gold/platinum/diamond/ruby)
    accuracy: float              # 정답률 (0.0 ~ 1.0 범위)
    score: float                 # 추천 점수 (알고리즘별로 계산 방식 상이)
    reason: str                  # 추천 이유 (사용자에게 보여줄 설명)


class RecommendResponse(BaseModel):
    """
    추천 API 응답 형식

    GET /recommend, /recommend/*, 등 모든 추천 엔드포인트의 표준 응답
    """
    user_id: str                             # 요청한 사용자 ID
    k: int                                   # 요청한 추천 개수
    items: List[RecommendationItem]          # 추천 문제 목록 (최대 k개)


# ===========================
# 배치 추천 스키마 (Week 1)
# ===========================

class BatchRecommendRequest(BaseModel):
    """
    배치 추천 요청 형식

    여러 사용자에 대해 한 번에 추천을 요청할 때 사용
    POST /recommend/batch 엔드포인트에서 사용
    """
    user_ids: List[str]                      # 추천을 받을 사용자 ID 목록
    k: int = 5                               # 사용자당 추천 개수 (기본값: 5)
    exclude_solved: bool = True              # 이미 푼 문제 제외 여부 (기본값: True)
    difficulty_min: Optional[str] = None     # 최소 난이도 필터 (예: "silver")
    difficulty_max: Optional[str] = None     # 최대 난이도 필터 (예: "gold")


class BatchRecommendResponse(BaseModel):
    """
    배치 추천 응답 형식

    각 사용자별로 추천 결과를 딕셔너리 형태로 반환
    """
    results: Dict[str, List[RecommendationItem]]  # user_id를 키로 하는 추천 결과
    total_users: int                              # 처리한 총 사용자 수
    strategy: str                                 # 사용된 추천 전략 (tfidf/popularity/...)


# ===========================
# 취약점 분석 스키마 (Week 2)
# ===========================

class WeakTag(BaseModel):
    """
    취약 태그 정보

    사용자가 어려워하는 알고리즘/주제 태그
    성공률이 낮거나 실패 횟수가 많은 태그를 식별
    """
    tag: str                     # 태그 이름 (예: "dp", "greedy", "graph")
    success_rate: float          # 성공률 (0.0 ~ 100.0 범위, 퍼센트)
    failures: int                # 실패 횟수
    total_attempts: int          # 총 시도 횟수 (성공 + 실패)


class TagStat(BaseModel):
    """
    태그별 통계 정보

    특정 태그에 대한 사용자의 상세한 풀이 통계
    """
    total_attempts: int          # 총 시도 횟수
    successes: int               # 성공 횟수 (AC)
    failures: int                # 실패 횟수 (WA, TLE, 등)
    success_rate: float          # 성공률 (0.0 ~ 100.0 범위, 퍼센트)


class FailedProblem(BaseModel):
    """
    실패한 문제 정보

    사용자가 틀린 문제의 상세 정보
    """
    problem_id: int              # 문제 번호
    title: str                   # 문제 제목
    tags: List[str]              # 문제 태그 목록
    difficulty: str              # 난이도
    verdict: str                 # 채점 결과 (WA, TLE, MLE, RE, 등)


class WeaknessAnalysisResponse(BaseModel):
    """
    취약점 분석 API 응답 형식

    GET /analysis/weakness 엔드포인트에서 사용
    사용자의 약점을 다각도로 분석한 결과를 반환
    """
    user_id: str                              # 분석 대상 사용자 ID
    total_attempts: int                       # 전체 시도 횟수
    total_failures: int                       # 전체 실패 횟수
    weak_tags: List[WeakTag]                  # 취약 태그 목록 (상위 5개)
    tag_stats: Dict[str, TagStat]             # 모든 태그의 통계 (태그명: 통계)
    recent_failures: List[FailedProblem]      # 최근 실패 문제 (최대 5개)


# ===========================
# 스피드런 스키마 (Week 3)
# ===========================

class SpeedrunMode(str, Enum):
    """스피드런 모드"""
    THIRTY_MIN = "30min"    # 30분 모드 (초보)
    SIXTY_MIN = "60min"     # 60분 모드 (초보)
    NINETY_MIN = "90min"    # 90분 모드 (고수)
    ONE_TWENTY_MIN = "120min"  # 120분 모드 (고수)


class SpeedrunDifficulty(str, Enum):
    """스피드런 난이도"""
    BRONZE = "bronze"       # 초보자용
    SILVER = "silver"       # 중급자용
    GOLD = "gold"           # 고수용
    PLATINUM = "platinum"   # 고수용 (상위)


class SpeedrunSessionCreate(BaseModel):
    """
    스피드런 세션 생성 요청

    정각/30분에만 생성 가능
    """
    user_id: str                # 사용자 ID
    mode: SpeedrunMode          # 30분 또는 60분 모드
    difficulty: SpeedrunDifficulty = SpeedrunDifficulty.BRONZE  # 난이도 (기본: bronze)


class SpeedrunSessionResponse(BaseModel):
    """
    스피드런 세션 응답

    세션 정보와 진행 상황을 포함
    """
    session_id: str             # 세션 ID (UUID)
    user_id: str                # 사용자 ID
    mode: str                   # 모드 (30min/60min/90min/120min)
    difficulty: str             # 난이도 (bronze/silver/gold/platinum)
    start_time: str             # 시작 시간 (ISO 8601)
    end_time: str               # 종료 시간 (ISO 8601)
    remaining_seconds: int      # 남은 시간 (초)
    problem_set_id: str         # 문제 세트 ID
    problems: List[int]         # 문제 번호 목록
    problem_scores: Optional[List[int]] = None  # 문제별 점수 (고수 모드)
    solved_problems: List[int]  # 풀은 문제 목록
    solved_count: int           # 푼 문제 수
    total_count: int            # 전체 문제 수
    status: str                 # 상태 (active/completed/expired)
    score: int                  # 현재 점수


class SpeedrunSubmit(BaseModel):
    """
    스피드런 문제 제출

    백준에서 AC를 받았을 때 호출
    """
    session_id: str             # 세션 ID
    problem_id: int             # 문제 번호
    verdict: str                # 채점 결과 (AC/WA/TLE 등)


class SpeedrunSubmitResponse(BaseModel):
    """스피드런 제출 응답"""
    success: bool               # 성공 여부
    score: Optional[int] = None # 현재 점수
    solved_count: Optional[int] = None  # 푼 문제 수
    remaining: Optional[List[int]] = None  # 남은 문제 목록
    error: Optional[str] = None # 에러 메시지


class ProblemDetail(BaseModel):
    """문제 상세 정보"""
    problem_id: int             # 문제 번호
    solved: bool                # 풀이 여부


class SpeedrunResult(BaseModel):
    """
    스피드런 결과

    세션 종료 후 최종 결과
    """
    session_id: str             # 세션 ID
    user_id: str                # 사용자 ID
    mode: str                   # 모드
    total_time: int             # 총 소요 시간 (초)
    solved_count: int           # 푼 문제 수
    total_count: int            # 전체 문제 수
    final_score: int            # 최종 점수
    problems_detail: List[ProblemDetail]  # 각 문제별 상세 정보
    rank: Optional[int]         # 순위 (옵션)
    status: str                 # 최종 상태


class LeaderboardEntry(BaseModel):
    """리더보드 항목"""
    user_id: str                # 사용자 ID
    score: int                  # 점수
    solved_count: int           # 푼 문제 수
    total_count: int            # 전체 문제 수
    completed_at: str           # 완료 시간 (ISO 8601)


class LeaderboardResponse(BaseModel):
    """리더보드 응답"""
    mode: str                           # 모드 (30min/60min)
    entries: List[LeaderboardEntry]     # 순위 목록


# ===========================
# 성장 분석 스키마 (Week 3)
# ===========================

class TagAccuracy(BaseModel):
    """태그별 정답률"""
    tag: str                    # 태그명 (dp, graph 등)
    total: int                  # 총 시도 횟수
    solved: int                 # 정답 횟수
    accuracy: float             # 정답률 (0.0 ~ 100.0)


class WeeklyStats(BaseModel):
    """주간 통계"""
    week: str                   # 주차 (2024-W01 형식)
    total_attempts: int         # 총 시도
    total_solved: int           # 총 정답
    accuracy: float             # 정답률


class DifficultyProgress(BaseModel):
    """난이도별 진행 상황"""
    difficulty: str             # 난이도 (bronze/silver/gold 등)
    total: int                  # 총 시도
    solved: int                 # 정답 수
    accuracy: float             # 정답률


class GrowthReportResponse(BaseModel):
    """
    성장 리포트 응답

    사용자의 전체적인 성장 현황을 보여줍니다.
    """
    user_id: str                            # 사용자 ID
    period_days: int                        # 분석 기간 (일)

    # 전체 요약
    total_attempts: int                     # 총 시도 횟수
    total_solved: int                       # 총 정답 횟수
    overall_accuracy: float                 # 전체 정답률

    # 태그별 분석
    weak_tags: List[TagAccuracy]            # 취약 태그 (정답률 낮은 순)
    strong_tags: List[TagAccuracy]          # 강점 태그 (정답률 높은 순)

    # 난이도별 분석
    difficulty_stats: List[DifficultyProgress]  # 난이도별 통계

    # 시간별 추이
    weekly_progress: List[WeeklyStats]      # 주간 추이


class AccuracyTrendResponse(BaseModel):
    """정답률 변화 추이 응답"""
    user_id: str
    period_days: int
    weekly_trend: List[WeeklyStats]         # 주간 정답률 추이
    improvement: float                       # 향상도 (첫주 대비 마지막주)


class TagAnalysisResponse(BaseModel):
    """태그별 상세 분석 응답"""
    user_id: str
    total_tags: int                         # 분석된 태그 수
    tag_stats: List[TagAccuracy]            # 모든 태그 통계
    most_attempted: List[TagAccuracy]       # 가장 많이 시도한 태그
    needs_improvement: List[TagAccuracy]    # 개선 필요 태그 (시도 많고 정답률 낮음)