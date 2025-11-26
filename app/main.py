"""
백준 문제 추천 시스템 FastAPI 서버

Week 1: 베이스라인 추천 알고리즘 (TF-IDF, Popularity, Random, Hybrid)
Week 2: 취약점 기반 추천 알고리즘 (Weakness)

이 모듈은 FastAPI를 사용하여 RESTful API 서버를 구성하고,
5가지 추천 전략을 제공합니다.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
from traceback import format_exc


from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 추천 알고리즘 구현체 임포트
from app.recommender import (
    TfidfRecommender,         # TF-IDF 기반 콘텐츠 추천
    PopularityRecommender,    # 정답률 기반 인기도 추천
    RandomRecommender,        # 무작위 추천 (A/B 테스트용)
    HybridRecommender,        # TF-IDF + Popularity 하이브리드
    WeaknessRecommender       # 취약점 분석 기반 추천 (Week 2)
)
# Pydantic 스키마 임포트
from app.schemas import (
    RecommendResponse,           # 단일 추천 응답
    BatchRecommendRequest,       # 배치 추천 요청
    BatchRecommendResponse,      # 배치 추천 응답
    WeaknessAnalysisResponse,    # 취약점 분석 응답 (Week 2)
    SpeedrunSessionCreate,       # 스피드런 세션 생성 요청 (Week 3)
    SpeedrunSessionResponse,     # 스피드런 세션 응답
    SpeedrunSubmit,              # 스피드런 문제 제출
    SpeedrunSubmitResponse,      # 스피드런 제출 응답
    SpeedrunResult,              # 스피드런 결과
    LeaderboardResponse,         # 리더보드 응답
    LeaderboardEntry,            # 리더보드 항목
    # 성장 분석 스키마 (Week 3)
    GrowthReportResponse,        # 성장 리포트 응답
    AccuracyTrendResponse,       # 정답률 추이 응답
    TagAnalysisResponse          # 태그별 분석 응답
)

# FastAPI 애플리케이션 초기화
app = FastAPI(title="Baekjoon Recommender MVP")

# CORS 미들웨어 설정 (크로스 오리진 요청 허용)
# 프론트엔드에서 API 호출을 위해 모든 오리진 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # 모든 도메인 허용 (프로덕션에서는 제한 필요)
    allow_credentials=True,      # 쿠키/인증 정보 허용
    allow_methods=["*"],         # 모든 HTTP 메서드 허용 (GET, POST, etc.)
    allow_headers=["*"],         # 모든 HTTP 헤더 허용
)

# ===========================
# 데이터 경로 설정
# ===========================

# 이 파일(app/main.py) 기준 절대경로를 계산
BASE = Path(__file__).resolve().parent
PROB_PATH = BASE / "data" / "problems.csv"      # 문제 정보 CSV 경로
HIST_PATH = BASE / "data" / "user_history.csv"  # 사용자 풀이 이력 CSV 경로


# ===========================
# 추천기 인스턴스 초기화
# ===========================

# 1. TF-IDF 추천기 (Week 1 - 콘텐츠 기반 필터링)
# 사용자가 풀었던 문제의 특징(제목, 태그, 난이도)을 분석하여
# 코사인 유사도로 비슷한 문제를 추천
tfidf_recommender = TfidfRecommender(
    problems_path=str(PROB_PATH),  # 문제 데이터
    history_path=str(HIST_PATH),   # 사용자 이력
)

# 2. 인기도 기반 추천기 (Week 1 - 정답률 기반)
# 정답률(accuracy)이 높은 문제를 우선 추천
# 콜드 스타트 문제에 강건하며 가장 단순한 베이스라인
popularity_recommender = PopularityRecommender(
    problems_path=str(PROB_PATH),
    history_path=str(HIST_PATH),
)

# 3. 랜덤 추천기 (Week 1 - 무작위 샘플링)
# 조건에 맞는 문제 중 무작위로 추천
# A/B 테스트의 대조군 및 Exploration 전략으로 사용
random_recommender = RandomRecommender(
    problems_path=str(PROB_PATH),
    history_path=str(HIST_PATH),
    seed=42  # 재현 가능성을 위한 랜덤 시드 고정
)

# 4. 하이브리드 추천기 (Week 1 - TF-IDF + Popularity 결합)
# 개인화(TF-IDF)와 인기도(Popularity)를 가중 평균으로 결합
# 기본 가중치: TF-IDF 70%, Popularity 30%
hybrid_recommender = HybridRecommender(
    problems_path=str(PROB_PATH),
    history_path=str(HIST_PATH),
    weights={"tfidf": 0.7, "popularity": 0.3}  # 가중치 설정
)

# 5. 취약점 기반 추천기 (Week 2 - 사용자 약점 분석)
# 사용자가 자주 틀리는 태그를 분석하여 약점을 보완할 문제 추천
# 성공률이 60% 미만인 태그를 취약 태그로 식별
weakness_recommender = WeaknessRecommender(
    problems_path=str(PROB_PATH),
    history_path=str(HIST_PATH),
    recent_days=30  # 최근 30일간의 풀이 이력을 분석 대상으로 함
)

# 기본 추천기 설정 (하위 호환성을 위해 TF-IDF를 기본으로 사용)
recommender = tfidf_recommender

# 스피드런 매니저 초기화 (Week 3)
from app.speedrun import SpeedrunManager
speedrun_manager = SpeedrunManager()

# 성장 분석기 초기화 (Week 3)
from app.growth import GrowthAnalyzer
growth_analyzer = GrowthAnalyzer(
    problems_path=str(PROB_PATH),
    history_path=str(HIST_PATH)
)


# ===========================
# API 엔드포인트 정의
# ===========================


@app.get("/health")
def health():
    """
    서버 헬스체크 엔드포인트

    서버가 정상적으로 동작하는지 확인하는 단순 상태 체크용
    반환값: {"status": "ok"}
    """
    return {"status": "ok"}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_id: str = Query(...),                      # 사용자 ID (필수)
    k: int = Query(5, ge=1, le=50),                # 추천 개수 (기본값: 5, 범위: 1-50)
    exclude_solved: bool = Query(True),             # 풀었던 문제 제외 여부 (기본값: True)
    difficulty_min: Optional[str] = Query(None),    # 최소 난이도 필터 (예: "silver")
    difficulty_max: Optional[str] = Query(None),    # 최대 난이도 필터 (예: "gold")
):
    """
    TF-IDF 기반 추천 엔드포인트 (Week 1)

    사용자가 풀었던 문제의 특징을 분석하여 유사한 문제를 추천합니다.

    알고리즘:
    - TF-IDF 벡터화로 문제의 텍스트 특징 추출
    - 사용자 프로필 = 풀었던 문제들의 평균 벡터
    - 코사인 유사도 계산하여 유사한 문제 선정
    - 최종 점수 = 0.9 * 유사도 + 0.1 * 정답률

    예시:
    GET /recommend?user_id=1&k=5&exclude_solved=true
    """
    try:
        # 추천 알고리즘 실행
        items = recommender.recommend(
            user_id=user_id,
            k=k,
            exclude_solved=exclude_solved,
            diff_min=difficulty_min,
            diff_max=difficulty_max
        )
        return {"user_id": user_id, "k": k, "items": items}
    except Exception as e:
        # 에러 발생 시 콘솔에 풀스택 출력하고, 클라이언트에도 에러 메시지 전달
        print("[ERROR]", e)
        print(format_exc())  # 전체 스택 트레이스 출력
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/recommend/popularity", response_model=RecommendResponse)
def recommend_popularity(
    user_id: str = Query(...),                      # 사용자 ID (필수)
    k: int = Query(5, ge=1, le=50),                # 추천 개수 (기본값: 5, 범위: 1-50)
    exclude_solved: bool = Query(True),             # 풀었던 문제 제외 여부
    difficulty_min: Optional[str] = Query(None),    # 최소 난이도 필터
    difficulty_max: Optional[str] = Query(None),    # 최대 난이도 필터
):
    """
    인기도 기반 추천 엔드포인트 (Week 1)

    정답률(accuracy)이 높은 문제를 우선 추천합니다.
    개인화는 없지만 콜드 스타트 문제에 강건합니다.

    알고리즘:
    - 정답률을 기준으로 문제 정렬
    - 상위 k개를 추천
    - 추천 이유: "인기도 기반 추천 (정확도: XX%)"

    예시:
    GET /recommend/popularity?user_id=1&k=5
    """
    try:
        items = popularity_recommender.recommend(
            user_id=user_id,
            k=k,
            exclude_solved=exclude_solved,
            diff_min=difficulty_min,
            diff_max=difficulty_max
        )
        return {"user_id": user_id, "k": k, "items": items}
    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/recommend/random", response_model=RecommendResponse)
def recommend_random(
    user_id: str = Query(...),                      # 사용자 ID (필수)
    k: int = Query(5, ge=1, le=50),                # 추천 개수 (기본값: 5, 범위: 1-50)
    exclude_solved: bool = Query(True),             # 풀었던 문제 제외 여부
    difficulty_min: Optional[str] = Query(None),    # 최소 난이도 필터
    difficulty_max: Optional[str] = Query(None),    # 최대 난이도 필터
):
    """
    랜덤 추천 엔드포인트 (Week 1)

    조건에 맞는 문제 중 무작위로 추천합니다.
    A/B 테스트의 대조군으로 사용하거나 Exploration 전략으로 활용합니다.

    알고리즘:
    - 조건 필터링 후 무작위 샘플링
    - seed=42로 재현 가능성 보장
    - 추천 이유: "랜덤 추천"

    예시:
    GET /recommend/random?user_id=1&k=5
    """
    try:
        items = random_recommender.recommend(
            user_id=user_id,
            k=k,
            exclude_solved=exclude_solved,
            diff_min=difficulty_min,
            diff_max=difficulty_max
        )
        return {"user_id": user_id, "k": k, "items": items}
    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/recommend/hybrid", response_model=RecommendResponse)
def recommend_hybrid(
    user_id: str = Query(...),                      # 사용자 ID (필수)
    k: int = Query(5, ge=1, le=50),                # 추천 개수 (기본값: 5, 범위: 1-50)
    exclude_solved: bool = Query(True),             # 풀었던 문제 제외 여부
    difficulty_min: Optional[str] = Query(None),    # 최소 난이도 필터
    difficulty_max: Optional[str] = Query(None),    # 최대 난이도 필터
):
    """
    하이브리드 추천 엔드포인트 (Week 1)

    TF-IDF 기반 개인화 추천과 인기도 추천을 결합합니다.
    개인화와 인기도의 균형을 맞춘 전략입니다.

    알고리즘:
    - TF-IDF와 Popularity 각각 실행
    - 가중 평균으로 최종 점수 계산 (기본: 70:30)
    - 최종 점수 = 0.7 * TF-IDF 점수 + 0.3 * 인기도 점수
    - 추천 이유: "하이브리드 추천 (TF-IDF: 0.7, 인기도: 0.3)"

    예시:
    GET /recommend/hybrid?user_id=1&k=5
    """
    try:
        items = hybrid_recommender.recommend(
            user_id=user_id,
            k=k,
            exclude_solved=exclude_solved,
            diff_min=difficulty_min,
            diff_max=difficulty_max
        )
        return {"user_id": user_id, "k": k, "items": items}
    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/recommend/weakness", response_model=RecommendResponse)
def recommend_weakness(
    user_id: str = Query(...),                      # 사용자 ID (필수)
    k: int = Query(5, ge=1, le=50),                # 추천 개수 (기본값: 5, 범위: 1-50)
    exclude_solved: bool = Query(True),             # 풀었던 문제 제외 여부
    difficulty_min: Optional[str] = Query(None),    # 최소 난이도 필터
    difficulty_max: Optional[str] = Query(None),    # 최대 난이도 필터
):
    """
    취약점 기반 추천 엔드포인트 (Week 2)

    사용자가 자주 틀리는 알고리즘 태그를 분석하여
    약점을 보완할 수 있는 문제를 추천합니다.

    알고리즘:
    1. 취약점 분석:
       - 최근 30일간 풀이 이력 분석
       - 태그별 성공률 계산
       - 성공률 60% 미만 태그를 취약 태그로 식별

    2. 점수 계산:
       - 매칭 태그 점수 (각 10점)
       - 취약 우선순위 보너스 (최대 100점)
       - 난이도 보너스 (쉬운 문제 우대)
       - 정답률 보너스 (0.1배)

    3. 추천 이유 생성:
       - "dp 알고리즘 보완 추천 (현재 성공률: 33%)"
       - "graph, shortest-path 등 취약 태그 집중 연습"

    예시:
    GET /recommend/weakness?user_id=2&k=5
    """
    try:
        items = weakness_recommender.recommend(
            user_id=user_id,
            k=k,
            exclude_solved=exclude_solved,
            diff_min=difficulty_min,
            diff_max=difficulty_max
        )
        return {"user_id": user_id, "k": k, "items": items}
    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/analysis/weakness", response_model=WeaknessAnalysisResponse)
def analyze_weakness(user_id: str = Query(...)):  # 사용자 ID (필수)
    """
    취약점 분석 엔드포인트 (Week 2)

    사용자의 풀이 이력을 분석하여 상세한 취약점 통계를 반환합니다.
    추천 없이 순수하게 분석 결과만 제공합니다.

    반환 정보:
    - total_attempts: 전체 시도 횟수
    - total_failures: 전체 실패 횟수
    - weak_tags: 취약 태그 목록 (성공률 낮은 상위 5개)
    - tag_stats: 모든 태그의 상세 통계 (성공/실패/성공률)
    - recent_failures: 최근 실패한 문제 목록 (최대 5개)

    예시:
    GET /analysis/weakness?user_id=2

    응답 예시:
    {
      "user_id": "2",
      "total_attempts": 10,
      "total_failures": 5,
      "weak_tags": [
        {
          "tag": "greedy",
          "success_rate": 50.0,
          "failures": 1,
          "total_attempts": 2
        }
      ],
      "tag_stats": {...},
      "recent_failures": [...]
    }
    """
    try:
        # 취약점 분석 실행
        analysis = weakness_recommender.analyze_weakness(user_id)
        return {
            "user_id": user_id,
            **analysis  # 분석 결과 딕셔너리를 펼쳐서 반환
        }
    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/recommend/batch", response_model=BatchRecommendResponse)
def recommend_batch(
    request: BatchRecommendRequest,  # 배치 요청 바디 (user_ids, k, exclude_solved, etc.)
    strategy: str = Query("tfidf", regex="^(tfidf|popularity|random|hybrid|weakness)$")  # 추천 전략 선택
):
    """
    배치 추천 엔드포인트 (Week 1 + Week 2)

    여러 사용자에 대해 한 번에 추천을 생성합니다.
    대량의 사용자를 대상으로 추천을 미리 계산할 때 유용합니다.

    사용 가능한 전략:
    - tfidf: TF-IDF 기반 콘텐츠 추천
    - popularity: 정답률 기반 인기도 추천
    - random: 무작위 추천
    - hybrid: TF-IDF + Popularity 하이브리드
    - weakness: 취약점 기반 추천 (Week 2)

    요청 예시:
    POST /recommend/batch?strategy=weakness
    {
      "user_ids": ["1", "2", "3"],
      "k": 5,
      "exclude_solved": true,
      "difficulty_min": "silver",
      "difficulty_max": "gold"
    }

    응답 예시:
    {
      "results": {
        "1": [...],
        "2": [...],
        "3": [...]
      },
      "total_users": 3,
      "strategy": "weakness"
    }
    """
    try:
        # 전략에 따라 추천기 선택
        strategy_map = {
            "tfidf": tfidf_recommender,
            "popularity": popularity_recommender,
            "random": random_recommender,
            "hybrid": hybrid_recommender,
            "weakness": weakness_recommender
        }
        selected_recommender = strategy_map[strategy]

        # 각 사용자에 대해 추천 생성
        results = {}
        for user_id in request.user_ids:
            items = selected_recommender.recommend(
                user_id=user_id,
                k=request.k,
                exclude_solved=request.exclude_solved,
                diff_min=request.difficulty_min,
                diff_max=request.difficulty_max
            )
            results[user_id] = items  # 사용자 ID를 키로 하여 결과 저장

        return {
            "results": results,                    # 사용자별 추천 결과 딕셔너리
            "total_users": len(request.user_ids),  # 처리한 총 사용자 수
            "strategy": strategy                   # 사용된 추천 전략
        }
    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/__debug")
def debug():
    """
    디버깅 정보 엔드포인트

    서버 상태와 데이터 통계를 확인하는 내부 디버깅용 엔드포인트입니다.
    개발 중 데이터 로딩 상태를 빠르게 확인할 때 유용합니다.

    반환 정보:
    - problems_shape: TF-IDF 행렬의 크기 [문제 수, 특징 차원]
    - n_problems: 전체 문제 수
    - n_history: 전체 풀이 이력 수
    - problem_cols: problems.csv의 컬럼명 목록
    - history_cols: user_history.csv의 컬럼명 목록
    - sample_user_ids: 샘플 사용자 ID 10개 (테스트용)
    - available_strategies: 사용 가능한 추천 전략 목록

    예시:
    GET /__debug

    주의: 프로덕션 환경에서는 보안을 위해 비활성화해야 합니다.
    """
    return {
        "problems_shape": list(tfidf_recommender.X.shape),  # TF-IDF 행렬 크기
        "n_problems": int(tfidf_recommender.problems.shape[0]),  # 문제 개수
        "n_history": int(tfidf_recommender.history.shape[0]),  # 이력 개수
        "problem_cols": list(tfidf_recommender.problems.columns),  # 문제 데이터 컬럼
        "history_cols": list(tfidf_recommender.history.columns),  # 이력 데이터 컬럼
        "sample_user_ids": list(map(str, tfidf_recommender.history["user_id"].unique()[:10])),  # 샘플 유저
        "available_strategies": ["tfidf", "popularity", "random", "hybrid", "weakness"]  # 지원 전략
    }


# ===========================
# 스피드런 API (Week 3)
# ===========================


@app.post("/speedrun/create", response_model=SpeedrunSessionResponse)
def create_speedrun_session(request: SpeedrunSessionCreate):
    """
    스피드런 세션 생성 (정각/30분에만 가능)

    타임 어택 모드를 시작합니다.
    정각 또는 30분에만 세션을 생성할 수 있습니다.

    [초보 모드]
    - mode: "30min" (5문제) 또는 "60min" (10문제)
    - difficulty: "bronze" 또는 "silver"
    - 점수: 문제당 100점 + 시간보너스(50%)

    [고수 모드]
    - mode: "90min" (4문제) 또는 "120min" (6문제)
    - difficulty: "gold" 또는 "platinum"
    - 점수: 골드 150점, 플래티넘 200점 + 시간보너스(30%)

    예시:
    POST /speedrun/create
    {
      "user_id": "1",
      "mode": "90min",
      "difficulty": "gold"
    }
    """
    try:
        # 세션 생성
        session = speedrun_manager.create_session(
            user_id=request.user_id,
            mode=request.mode.value,
            difficulty=request.difficulty.value
        )

        return SpeedrunSessionResponse(**session.to_dict())

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/speedrun/session/{session_id}", response_model=SpeedrunSessionResponse)
def get_speedrun_session(session_id: str):
    """
    스피드런 세션 조회

    현재 진행 중인 세션의 상태를 확인합니다.

    응답:
    - remaining_seconds: 남은 시간 (초)
    - solved_count: 현재까지 푼 문제 수
    - score: 현재 점수
    - status: 세션 상태 (active/completed/expired)

    예시:
    GET /speedrun/session/{session_id}
    """
    try:
        session = speedrun_manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

        return SpeedrunSessionResponse(**session.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/speedrun/submit", response_model=SpeedrunSubmitResponse)
def submit_speedrun_problem(submit: SpeedrunSubmit):
    """
    스피드런 문제 제출

    백준에서 AC를 받았을 때 호출하여 점수를 업데이트합니다.

    요청:
    - session_id: 세션 ID
    - problem_id: 문제 번호
    - verdict: 채점 결과 (AC만 인정)

    응답:
    - success: 성공 여부
    - score: 현재 점수
    - solved_count: 푼 문제 수
    - remaining: 남은 문제 목록

    예시:
    POST /speedrun/submit
    {
      "session_id": "uuid-1234",
      "problem_id": 1000,
      "verdict": "AC"
    }
    """
    try:
        result = speedrun_manager.submit_problem(
            session_id=submit.session_id,
            problem_id=submit.problem_id,
            verdict=submit.verdict
        )

        return SpeedrunSubmitResponse(**result)

    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/speedrun/result/{session_id}", response_model=SpeedrunResult)
def get_speedrun_result(session_id: str):
    """
    스피드런 결과 조회

    세션 종료 후 최종 결과를 확인합니다.

    응답:
    - final_score: 최종 점수
    - solved_count: 푼 문제 수
    - total_time: 총 소요 시간 (초)
    - rank: 순위 (리더보드 기준)
    - problems_detail: 각 문제별 풀이 여부

    예시:
    GET /speedrun/result/{session_id}
    """
    try:
        result = speedrun_manager.get_result(session_id)

        if not result:
            raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다")

        return SpeedrunResult(**result)

    except HTTPException:
        raise
    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/speedrun/leaderboard", response_model=LeaderboardResponse)
def get_speedrun_leaderboard(
    mode: str = Query("30min", regex="^(30min|60min|90min|120min)$"),
    limit: int = Query(10, ge=1, le=100)
):
    """
    스피드런 리더보드 조회

    모드별 상위 랭커 목록을 확인합니다.

    파라미터:
    - mode: "30min", "60min", "90min", "120min" (기본값: 30min)
    - limit: 조회할 인원 수 (기본값: 10, 최대: 100)

    응답:
    - mode: 모드
    - entries: 순위 목록
      - user_id: 사용자 ID
      - score: 점수
      - solved_count: 푼 문제 수
      - completed_at: 완료 시간

    예시:
    GET /speedrun/leaderboard?mode=90min&limit=10
    """
    try:
        entries = speedrun_manager.get_leaderboard(mode=mode, limit=limit)

        return LeaderboardResponse(
            mode=mode,
            entries=[LeaderboardEntry(**e) for e in entries]
        )

    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/speedrun/active/{user_id}", response_model=SpeedrunSessionResponse)
def get_active_speedrun(user_id: str):
    """
    사용자의 활성 세션 조회

    현재 진행 중인 스피드런 세션이 있는지 확인합니다.

    예시:
    GET /speedrun/active/1
    """
    try:
        session = speedrun_manager.get_active_session(user_id)

        if not session:
            raise HTTPException(status_code=404, detail="활성 세션이 없습니다")

        return SpeedrunSessionResponse(**session.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))


# ===========================
# 성장 분석 API (Week 3)
# ===========================


@app.get("/growth/report", response_model=GrowthReportResponse)
def get_growth_report(
    user_id: str = Query(...),
    days: int = Query(30, ge=7, le=365)
):
    """
    종합 성장 리포트 조회

    사용자의 문제 풀이 이력을 분석하여 성장 현황을 보여줍니다.

    파라미터:
    - user_id: 사용자 ID (필수)
    - days: 분석 기간 (기본값: 30일, 범위: 7~365일)

    응답:
    - total_attempts: 총 시도 횟수
    - total_solved: 총 정답 횟수
    - overall_accuracy: 전체 정답률
    - weak_tags: 취약 태그 (정답률 낮은 순 5개)
    - strong_tags: 강점 태그 (정답률 높은 순 5개)
    - difficulty_stats: 난이도별 통계
    - weekly_progress: 주간 추이

    예시:
    GET /growth/report?user_id=1&days=30
    """
    try:
        report = growth_analyzer.analyze_growth(user_id, days)
        return GrowthReportResponse(**report)
    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/growth/accuracy-trend", response_model=AccuracyTrendResponse)
def get_accuracy_trend(
    user_id: str = Query(...),
    days: int = Query(30, ge=7, le=365)
):
    """
    정답률 변화 추이 조회

    주간 정답률 변화와 향상도를 보여줍니다.

    파라미터:
    - user_id: 사용자 ID (필수)
    - days: 분석 기간 (기본값: 30일)

    응답:
    - weekly_trend: 주간 정답률 추이
    - improvement: 향상도 (첫주 대비 마지막주 정답률 차이)

    예시:
    GET /growth/accuracy-trend?user_id=1&days=30
    """
    try:
        trend = growth_analyzer.analyze_accuracy_trend(user_id, days)
        return AccuracyTrendResponse(**trend)
    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/growth/tags", response_model=TagAnalysisResponse)
def get_tag_analysis(
    user_id: str = Query(...),
    days: int = Query(30, ge=7, le=365)
):
    """
    태그별 상세 분석 조회

    각 알고리즘 태그별 정답률과 개선 필요 영역을 보여줍니다.

    파라미터:
    - user_id: 사용자 ID (필수)
    - days: 분석 기간 (기본값: 30일)

    응답:
    - total_tags: 분석된 태그 수
    - tag_stats: 모든 태그 통계
    - most_attempted: 가장 많이 시도한 태그 (상위 5개)
    - needs_improvement: 개선 필요 태그 (시도 3회 이상, 정답률 60% 미만)

    예시:
    GET /growth/tags?user_id=1&days=30
    """
    try:
        analysis = growth_analyzer.analyze_tags(user_id, days)
        return TagAnalysisResponse(**analysis)
    except Exception as e:
        print("[ERROR]", e)
        print(format_exc())
        raise HTTPException(status_code=503, detail=str(e))
