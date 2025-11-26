"""
스피드런 세션 관리 모듈

정각/30분에 시작하는 타임 어택 모드
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import uuid
from pathlib import Path
import random


class SpeedrunSession:
    """스피드런 세션 정보"""

    # 모드별 시간 (초)
    MODE_DURATION = {
        "30min": 1800,
        "60min": 3600,
        "90min": 5400,
        "120min": 7200
    }

    # 고수 모드 여부
    ADVANCED_MODES = ["90min", "120min"]

    def __init__(
        self,
        session_id: str,
        user_id: str,
        mode: str,
        difficulty: str,
        start_time: datetime,
        problem_set_id: str,
        problems: List[int],
        problem_scores: Optional[List[int]] = None
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.mode = mode  # "30min", "60min", "90min", "120min"
        self.difficulty = difficulty  # "bronze", "silver", "gold", "platinum"
        self.start_time = start_time
        self.problem_set_id = problem_set_id
        self.problems = problems
        self.problem_scores = problem_scores  # 고수 모드: 문제별 점수

        # 모드에 따른 종료 시간 계산
        duration = self.MODE_DURATION.get(mode, 1800)
        self.end_time = start_time + timedelta(seconds=duration)
        self.duration_seconds = duration

        # 진행 상황
        self.solved_problems: List[int] = []
        self.status = "active"  # active, completed, expired
        self.score = 0

    def is_expired(self) -> bool:
        """세션이 만료되었는지 확인"""
        return datetime.now() >= self.end_time

    def get_remaining_seconds(self) -> int:
        """남은 시간 (초)"""
        if self.is_expired():
            return 0
        remaining = (self.end_time - datetime.now()).total_seconds()
        return max(0, int(remaining))

    def submit_problem(self, problem_id: int, verdict: str) -> bool:
        """
        문제 제출 처리

        Args:
            problem_id: 문제 번호
            verdict: 채점 결과 (AC, WA, TLE 등)

        Returns:
            성공 여부
        """
        # 이미 푼 문제는 무시
        if problem_id in self.solved_problems:
            return False

        # 세트에 포함된 문제인지 확인
        if problem_id not in self.problems:
            return False

        # AC만 인정
        if verdict.upper() == "AC":
            self.solved_problems.append(problem_id)
            self._calculate_score()
            return True

        return False

    def _calculate_score(self):
        """점수 계산 (초보/고수 모드 분리)"""
        if self.mode in self.ADVANCED_MODES:
            self._calculate_advanced_score()
        else:
            self._calculate_beginner_score()

    def _calculate_beginner_score(self):
        """초보 모드 점수 계산 (브론즈/실버)"""
        # 기본 점수: 문제당 100점
        base_score = len(self.solved_problems) * 100

        # 시간 보너스: 남은 시간 비율 × 50 × 푼 문제 수
        time_ratio = self.get_remaining_seconds() / self.duration_seconds
        time_bonus = time_ratio * 50 * len(self.solved_problems)

        self.score = int(base_score + time_bonus)

    def _calculate_advanced_score(self):
        """고수 모드 점수 계산 (골드/플래티넘)"""
        # 기본 점수: 푼 문제의 난이도별 점수 합계
        base_score = 0
        for problem_id in self.solved_problems:
            idx = self.problems.index(problem_id)
            if self.problem_scores and idx < len(self.problem_scores):
                base_score += self.problem_scores[idx]
            else:
                # 기본 점수 (점수 정보 없으면 난이도별 기본값)
                base_score += 150 if self.difficulty == "gold" else 200

        # 시간 보너스: 남은 시간 비율 × 기본점수 × 0.3
        time_ratio = self.get_remaining_seconds() / self.duration_seconds
        time_bonus = time_ratio * base_score * 0.3

        self.score = int(base_score + time_bonus)

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        result = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "mode": self.mode,
            "difficulty": self.difficulty,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "remaining_seconds": self.get_remaining_seconds(),
            "problem_set_id": self.problem_set_id,
            "problems": self.problems,
            "solved_problems": self.solved_problems,
            "solved_count": len(self.solved_problems),
            "total_count": len(self.problems),
            "status": self.status,
            "score": self.score
        }
        # 고수 모드면 문제별 점수도 포함
        if self.problem_scores:
            result["problem_scores"] = self.problem_scores
        return result


class SpeedrunManager:
    """스피드런 세션 관리자"""

    def __init__(self):
        # 세션 저장소 (메모리 기반 - 나중에 Redis로 교체 가능)
        self.sessions: Dict[str, SpeedrunSession] = {}

        # 문제 세트 로드
        data_dir = Path(__file__).parent / "data"
        sets_file = data_dir / "speedrun_sets.json"

        with open(sets_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.problem_sets = {s["set_id"]: s for s in data["sets"]}

        # 리더보드 (메모리 기반)
        self.leaderboard: Dict[str, List[Dict]] = {
            "30min": [],
            "60min": [],
            "90min": [],
            "120min": []
        }

    def check_time_slot(self) -> bool:
        """
        현재 시각이 정각 또는 30분인지 확인

        Returns:
            True: 시작 가능 (정각 또는 30분)
            False: 시작 불가
        """
        now = datetime.now()
        minute = now.minute

        # 정각(0분) 또는 30분만 허용
        # 실제로는 0~5분, 30~35분 정도 여유를 둘 수도 있음
        return minute == 0 or minute == 30

    def create_session(self, user_id: str, mode: str, difficulty: str = "bronze") -> SpeedrunSession:
        """
        스피드런 세션 생성

        Args:
            user_id: 사용자 ID
            mode: "30min", "60min", "90min", "120min"
            difficulty: 난이도 ("bronze", "silver", "gold", "platinum")

        Returns:
            생성된 세션

        Raises:
            ValueError: 시간대가 맞지 않거나 잘못된 모드
        """
        # 시간 체크
        if not self.check_time_slot():
            raise ValueError("스피드런은 정각 또는 30분에만 시작할 수 있습니다")

        # 모드 검증
        valid_modes = ["30min", "60min", "90min", "120min"]
        if mode not in valid_modes:
            raise ValueError(f"mode는 {valid_modes} 중 하나여야 합니다")

        # 난이도와 모드에 맞는 문제 세트 선택
        available_sets = [
            s for s in self.problem_sets.values()
            if s["mode"] == mode and s["difficulty"] == difficulty
        ]

        if not available_sets:
            raise ValueError(f"{difficulty} {mode} 문제 세트가 없습니다")

        # 랜덤 선택 (또는 로테이션)
        problem_set = random.choice(available_sets)

        # 세션 생성
        session_id = str(uuid.uuid4())
        session = SpeedrunSession(
            session_id=session_id,
            user_id=user_id,
            mode=mode,
            difficulty=difficulty,
            start_time=datetime.now(),
            problem_set_id=problem_set["set_id"],
            problems=problem_set["problems"],
            problem_scores=problem_set.get("problem_scores")  # 고수 모드용
        )

        # 저장
        self.sessions[session_id] = session

        return session

    def get_session(self, session_id: str) -> Optional[SpeedrunSession]:
        """세션 조회"""
        session = self.sessions.get(session_id)

        # 만료된 세션 상태 업데이트
        if session and session.is_expired() and session.status == "active":
            session.status = "expired"
            self._add_to_leaderboard(session)

        return session

    def get_active_session(self, user_id: str) -> Optional[SpeedrunSession]:
        """사용자의 활성 세션 조회"""
        for session in self.sessions.values():
            if session.user_id == user_id and session.status == "active":
                if not session.is_expired():
                    return session
                else:
                    session.status = "expired"
        return None

    def submit_problem(
        self,
        session_id: str,
        problem_id: int,
        verdict: str
    ) -> Dict:
        """
        문제 제출

        Returns:
            제출 결과
        """
        session = self.get_session(session_id)

        if not session:
            return {"success": False, "error": "세션을 찾을 수 없습니다"}

        if session.is_expired():
            return {"success": False, "error": "시간이 종료되었습니다"}

        success = session.submit_problem(problem_id, verdict)

        if success:
            # 모든 문제를 푼 경우
            if len(session.solved_problems) == len(session.problems):
                session.status = "completed"
                self._add_to_leaderboard(session)

            return {
                "success": True,
                "score": session.score,
                "solved_count": len(session.solved_problems),
                "remaining": self._get_remaining_problems(session)
            }
        else:
            return {"success": False, "error": "이미 푼 문제이거나 세트에 없는 문제입니다"}

    def _get_remaining_problems(self, session: SpeedrunSession) -> List[int]:
        """남은 문제 목록"""
        return [p for p in session.problems if p not in session.solved_problems]

    def _add_to_leaderboard(self, session: SpeedrunSession):
        """리더보드에 추가"""
        entry = {
            "user_id": session.user_id,
            "score": session.score,
            "solved_count": len(session.solved_problems),
            "total_count": len(session.problems),
            "completed_at": datetime.now().isoformat()
        }

        self.leaderboard[session.mode].append(entry)

        # 점수 순 정렬 (상위 100명만 유지)
        self.leaderboard[session.mode].sort(key=lambda x: x["score"], reverse=True)
        self.leaderboard[session.mode] = self.leaderboard[session.mode][:100]

    def get_leaderboard(self, mode: str, limit: int = 10) -> List[Dict]:
        """
        리더보드 조회

        Args:
            mode: "30min" or "60min"
            limit: 조회할 인원 수

        Returns:
            상위 랭커 목록
        """
        if mode not in self.leaderboard:
            return []

        return self.leaderboard[mode][:limit]

    def get_result(self, session_id: str) -> Optional[Dict]:
        """
        스피드런 결과 조회

        Returns:
            결과 딕셔너리 (점수, 풀이 시간, 순위 등)
        """
        session = self.get_session(session_id)

        if not session:
            return None

        # 순위 계산
        rank = None
        leaderboard = self.leaderboard[session.mode]
        for i, entry in enumerate(leaderboard, 1):
            if entry["user_id"] == session.user_id and entry["score"] == session.score:
                rank = i
                break

        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "mode": session.mode,
            "total_time": session.duration_seconds - session.get_remaining_seconds(),
            "solved_count": len(session.solved_problems),
            "total_count": len(session.problems),
            "final_score": session.score,
            "problems_detail": [
                {
                    "problem_id": pid,
                    "solved": pid in session.solved_problems
                }
                for pid in session.problems
            ],
            "rank": rank,
            "status": session.status
        }
