"""
성장 분석 모듈

사용자의 문제 풀이 이력을 분석하여 성장 리포트를 생성합니다.
- 태그별 정답률 분석
- 주간 정답률 추이
- 난이도별 진행 상황
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import pandas as pd


class GrowthAnalyzer:
    """성장 분석기"""

    def __init__(self, problems_path: str, history_path: str):
        """
        Args:
            problems_path: 문제 데이터 CSV 경로
            history_path: 사용자 풀이 이력 CSV 경로
        """
        self.problems = pd.read_csv(problems_path)
        self.history = pd.read_csv(history_path)

        # 문제 정보 딕셔너리 생성 (빠른 조회용)
        self.problem_info = {}
        for _, row in self.problems.iterrows():
            self.problem_info[row["problem_id"]] = {
                "title": row.get("title", ""),
                "difficulty": row.get("difficulty", "unknown"),
                "tags": str(row.get("tags", "")).split(",") if pd.notna(row.get("tags")) else []
            }

    def get_user_history(self, user_id: str, days: int = 30) -> pd.DataFrame:
        """사용자의 최근 풀이 이력 조회"""
        user_history = self.history[self.history["user_id"] == int(user_id)].copy()

        if days > 0 and "solved_at" in user_history.columns:
            cutoff = datetime.now() - timedelta(days=days)
            user_history["solved_at"] = pd.to_datetime(user_history["solved_at"], errors="coerce")
            user_history = user_history[user_history["solved_at"] >= cutoff]

        return user_history

    def analyze_growth(self, user_id: str, days: int = 30) -> Dict:
        """
        종합 성장 리포트 생성

        Args:
            user_id: 사용자 ID
            days: 분석 기간 (일)

        Returns:
            성장 리포트 딕셔너리
        """
        user_history = self.get_user_history(user_id, days)

        if user_history.empty:
            return self._empty_report(user_id, days)

        # 기본 통계
        total_attempts = len(user_history)
        total_solved = len(user_history[user_history["verdict"] == "AC"])
        overall_accuracy = (total_solved / total_attempts * 100) if total_attempts > 0 else 0

        # 태그별 분석
        tag_stats = self._analyze_tags(user_history)
        weak_tags = sorted(tag_stats, key=lambda x: x["accuracy"])[:5]
        strong_tags = sorted(tag_stats, key=lambda x: x["accuracy"], reverse=True)[:5]

        # 난이도별 분석
        difficulty_stats = self._analyze_difficulty(user_history)

        # 주간 추이
        weekly_progress = self._analyze_weekly(user_history)

        return {
            "user_id": user_id,
            "period_days": days,
            "total_attempts": total_attempts,
            "total_solved": total_solved,
            "overall_accuracy": round(overall_accuracy, 2),
            "weak_tags": weak_tags,
            "strong_tags": strong_tags,
            "difficulty_stats": difficulty_stats,
            "weekly_progress": weekly_progress
        }

    def analyze_accuracy_trend(self, user_id: str, days: int = 30) -> Dict:
        """정답률 변화 추이 분석"""
        user_history = self.get_user_history(user_id, days)

        if user_history.empty:
            return {
                "user_id": user_id,
                "period_days": days,
                "weekly_trend": [],
                "improvement": 0.0
            }

        weekly = self._analyze_weekly(user_history)

        # 향상도 계산 (첫주 대비 마지막주)
        improvement = 0.0
        if len(weekly) >= 2:
            first_acc = weekly[0]["accuracy"]
            last_acc = weekly[-1]["accuracy"]
            improvement = last_acc - first_acc

        return {
            "user_id": user_id,
            "period_days": days,
            "weekly_trend": weekly,
            "improvement": round(improvement, 2)
        }

    def analyze_tags(self, user_id: str, days: int = 30) -> Dict:
        """태그별 상세 분석"""
        user_history = self.get_user_history(user_id, days)

        if user_history.empty:
            return {
                "user_id": user_id,
                "total_tags": 0,
                "tag_stats": [],
                "most_attempted": [],
                "needs_improvement": []
            }

        tag_stats = self._analyze_tags(user_history)

        # 가장 많이 시도한 태그
        most_attempted = sorted(tag_stats, key=lambda x: x["total"], reverse=True)[:5]

        # 개선 필요 태그 (시도 3회 이상, 정답률 60% 미만)
        needs_improvement = [
            t for t in tag_stats
            if t["total"] >= 3 and t["accuracy"] < 60
        ]
        needs_improvement = sorted(needs_improvement, key=lambda x: x["accuracy"])[:5]

        return {
            "user_id": user_id,
            "total_tags": len(tag_stats),
            "tag_stats": sorted(tag_stats, key=lambda x: x["tag"]),
            "most_attempted": most_attempted,
            "needs_improvement": needs_improvement
        }

    def _analyze_tags(self, history: pd.DataFrame) -> List[Dict]:
        """태그별 통계 계산"""
        tag_stats = defaultdict(lambda: {"total": 0, "solved": 0})

        for _, row in history.iterrows():
            problem_id = row["problem_id"]
            verdict = row["verdict"]

            if problem_id in self.problem_info:
                tags = self.problem_info[problem_id]["tags"]
                for tag in tags:
                    tag = tag.strip()
                    if tag:
                        tag_stats[tag]["total"] += 1
                        if verdict == "AC":
                            tag_stats[tag]["solved"] += 1

        result = []
        for tag, stats in tag_stats.items():
            accuracy = (stats["solved"] / stats["total"] * 100) if stats["total"] > 0 else 0
            result.append({
                "tag": tag,
                "total": stats["total"],
                "solved": stats["solved"],
                "accuracy": round(accuracy, 2)
            })

        return result

    def _analyze_difficulty(self, history: pd.DataFrame) -> List[Dict]:
        """난이도별 통계 계산"""
        diff_stats = defaultdict(lambda: {"total": 0, "solved": 0})

        for _, row in history.iterrows():
            problem_id = row["problem_id"]
            verdict = row["verdict"]

            if problem_id in self.problem_info:
                difficulty = self.problem_info[problem_id]["difficulty"]
                diff_stats[difficulty]["total"] += 1
                if verdict == "AC":
                    diff_stats[difficulty]["solved"] += 1

        # 난이도 순서 정의
        order = ["bronze", "silver", "gold", "platinum", "diamond", "ruby", "unknown"]

        result = []
        for diff in order:
            if diff in diff_stats:
                stats = diff_stats[diff]
                accuracy = (stats["solved"] / stats["total"] * 100) if stats["total"] > 0 else 0
                result.append({
                    "difficulty": diff,
                    "total": stats["total"],
                    "solved": stats["solved"],
                    "accuracy": round(accuracy, 2)
                })

        return result

    def _analyze_weekly(self, history: pd.DataFrame) -> List[Dict]:
        """주간 통계 계산"""
        if "solved_at" not in history.columns:
            return []

        history = history.copy()
        history["solved_at"] = pd.to_datetime(history["solved_at"], errors="coerce")
        history = history.dropna(subset=["solved_at"])

        if history.empty:
            return []

        # 주차 계산
        history["week"] = history["solved_at"].dt.strftime("%Y-W%W")

        weekly_stats = []
        for week, group in history.groupby("week"):
            total = len(group)
            solved = len(group[group["verdict"] == "AC"])
            accuracy = (solved / total * 100) if total > 0 else 0

            weekly_stats.append({
                "week": week,
                "total_attempts": total,
                "total_solved": solved,
                "accuracy": round(accuracy, 2)
            })

        return sorted(weekly_stats, key=lambda x: x["week"])

    def _empty_report(self, user_id: str, days: int) -> Dict:
        """빈 리포트 생성"""
        return {
            "user_id": user_id,
            "period_days": days,
            "total_attempts": 0,
            "total_solved": 0,
            "overall_accuracy": 0.0,
            "weak_tags": [],
            "strong_tags": [],
            "difficulty_stats": [],
            "weekly_progress": []
        }
