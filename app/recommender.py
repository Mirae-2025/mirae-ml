from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Robust CSV Loader
# -----------------------------
def _read_csv_robust(path: str) -> pd.DataFrame:
    """
    견고한 CSV 파일 로더

    기능:
    - UTF-8 BOM 자동 처리 (encoding="utf-8-sig")
    - 구분자 자동 감지 (쉼표, 세미콜론, 탭 등)
    - 컬럼명 표준화 (소문자 + 언더스코어)
    - 일반적인 컬럼명 오타 자동 교정

    Args:
        path: CSV 파일 경로

    Returns:
        정규화된 컬럼명을 가진 DataFrame
    """
    # sep=None으로 구분자 자동 감지
    df = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")

    # 컬럼명 정규화: 소문자, 공백 제거, BOM 제거
    df.columns = [c.strip().lower().replace("\ufeff", "").replace(" ", "_") for c in df.columns]

    # 일반적인 컬럼명 오타 매핑
    rename_map = {
        "userid": "user_id",
        "user-id": "user_id",
        "problemid": "problem_id",
        "problem-id": "problem_id",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    return df


# 백준 난이도 순서 (낮음 -> 높음)
DIFF_ORDER = ["bronze", "silver", "gold", "platinum", "diamond", "ruby"]


def _diff_to_rank(x: str) -> int:
    """
    난이도 문자열을 숫자 순위로 변환

    Args:
        x: 난이도 문자열 (예: "bronze", "gold")

    Returns:
        난이도 순위 (0: bronze, 1: silver, ..., 5: ruby)
        존재하지 않는 난이도인 경우 -1 반환
    """
    if isinstance(x, str):
        x = x.lower()
        if x in DIFF_ORDER:
            return DIFF_ORDER.index(x)
    return -1


# -----------------------------
# Base Recommender Interface
# -----------------------------
class BaseRecommender(ABC):
    """
    모든 추천 시스템의 추상 베이스 클래스

    모든 추천기가 공통으로 사용하는 기능을 제공:
    - 데이터 로드 및 검증
    - 풀이한 문제 조회
    - 난이도 필터링
    """

    def __init__(self, problems_path: str, history_path: str):
        """
        추천기 초기화

        Args:
            problems_path: 문제 정보 CSV 파일 경로
            history_path: 사용자 풀이 이력 CSV 파일 경로
        """
        # CSV 파일 로드
        self.problems: pd.DataFrame = _read_csv_robust(problems_path)
        self.history: pd.DataFrame = _read_csv_robust(history_path)

        # 데이터 검증 및 전처리
        self._validate_and_prepare_data()

    def _validate_and_prepare_data(self):
        """
        데이터 검증 및 공통 전처리 수행

        작업 내용:
        1. 필수 컬럼 존재 여부 확인
        2. 결측값 처리
        3. 데이터 타입 통일
        4. 빠른 조회를 위한 매핑 딕셔너리 생성
        """
        needed_p = {"problem_id", "title", "tags", "difficulty", "accuracy"}
        needed_h = {"user_id", "problem_id", "verdict"}
        if not needed_p.issubset(set(self.problems.columns)):
            raise ValueError(f"problems.csv missing columns: {needed_p - set(self.problems.columns)}")
        if not needed_h.issubset(set(self.history.columns)):
            raise ValueError(f"user_history.csv missing columns: {needed_h - set(self.history.columns)}")

        self.problems["title"] = self.problems["title"].fillna("").astype(str)
        self.problems["tags"] = self.problems["tags"].fillna("").astype(str)
        self.problems["difficulty"] = self.problems["difficulty"].fillna("silver").astype(str)
        self.problems["accuracy"] = pd.to_numeric(self.problems["accuracy"], errors="coerce").fillna(0.0)

        self.history["user_id"] = self.history["user_id"].astype(str)
        self.history["verdict"] = self.history["verdict"].fillna("").astype(str)

        # Common maps
        self.acc_map = self.problems.set_index("problem_id")["accuracy"].to_dict()
        self.diff_map = self.problems.set_index("problem_id")["difficulty"].to_dict()
        self.title_map = self.problems.set_index("problem_id")["title"].to_dict()

    def _get_solved_problems(self, user_id: str) -> set:
        """Get set of solved problem IDs for a user"""
        solved = self.history[
            (self.history["user_id"] == str(user_id)) &
            (self.history["verdict"].str.upper() == "AC")
        ]
        return set(solved["problem_id"].tolist())

    def _within_diff_range(self, pid: int, dmin: Optional[str], dmax: Optional[str]) -> bool:
        """Check if problem difficulty is within specified range"""
        if dmin is None and dmax is None:
            return True
        d = _diff_to_rank(self.diff_map.get(pid, "silver"))
        lo = _diff_to_rank(dmin) if dmin else None
        hi = _diff_to_rank(dmax) if dmax else None
        if lo is not None and d < lo:
            return False
        if hi is not None and d > hi:
            return False
        return True

    @abstractmethod
    def recommend(
        self,
        user_id: str,
        k: int = 5,
        exclude_solved: bool = True,
        diff_min: Optional[str] = None,
        diff_max: Optional[str] = None,
    ) -> List[Dict]:
        """Generate recommendations for a user"""
        pass


class TfidfRecommender(BaseRecommender):
    """Content-based recommender using TF-IDF vectorization"""

    def __init__(self, problems_path: str, history_path: str):
        super().__init__(problems_path, history_path)

        print("[DEBUG] problems columns:", list(self.problems.columns))
        print("[DEBUG] history columns :", list(self.history.columns))

        # TF-IDF vectorization (title + tags + difficulty)
        corpus = (
            self.problems["title"] + " " +
            self.problems["tags"].str.replace(";", " ", regex=False) + " " +
            self.problems["difficulty"]
        )
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.X = self.vectorizer.fit_transform(corpus)

        # ID mappings
        ids = self.problems["problem_id"].tolist()
        self.id2idx = {pid: i for i, pid in enumerate(ids)}
        self.idx2id = {i: pid for pid, i in self.id2idx.items()}

    # -----------------------------
    # Helpers
    # -----------------------------
    def _user_profile(self, user_id: str):
        # AC만 사용
        solved = self.history[
            (self.history["user_id"] == str(user_id)) & (self.history["verdict"].str.upper() == "AC")
        ]

        if solved.empty:
            # 콜드스타트: accuracy 상위 5개 평균
            topk = self.problems.sort_values("accuracy", ascending=False).head(5)["problem_id"].tolist()
            arrs = [self.X[self.id2idx[pid]].toarray() for pid in topk if pid in self.id2idx]
        else:
            arrs = [self.X[self.id2idx[pid]].toarray() for pid in solved["problem_id"].tolist() if pid in self.id2idx]

        if not arrs:
            # (1, n_features) 영벡터
            return np.zeros((1, self.X.shape[1]))

        # (m, 1, n) -> (m, n)
        mat = np.vstack([a.reshape(1, -1) for a in arrs])  # 모두 (1, n)로 맞춤
        prof = mat.mean(axis=0, keepdims=True)             # (1, n)
        return prof

    def _reason(self, user_id: str, pid: int) -> str:
        profile = self._user_profile(user_id)
        
        profile = np.asarray(profile)
        if profile.ndim == 1:
           profile = profile.reshape(1, -1)
# (n_items, n_features) vs (1, n_features) 확인
        assert profile.shape[1] == self.X.shape[1], f"profile dim {profile.shape} vs X {self.X.shape}"

        idx = self.id2idx[pid]
        vec = self.X[idx].toarray().flatten()
        prof = np.asarray(profile).flatten()
        contrib = vec * prof
        if contrib.sum() == 0:
            return "최근 풀이 이력과 유사한 태그 기반 기본 추천"
        top_idx = contrib.argsort()[-3:][::-1]
        terms = np.array(self.vectorizer.get_feature_names_out())[top_idx]
        key_terms = [t for t in terms if len(t) >= 2][:3]
        return f"이전 풀이와 유사한 키워드: {', '.join(key_terms)}"

    # -----------------------------
    # Public
    # -----------------------------
    def recommend(
        self,
        user_id: str,
        k: int = 5,
        exclude_solved: bool = True,
        diff_min: Optional[str] = None,
        diff_max: Optional[str] = None,
    ) -> List[Dict]:
        profile = self._user_profile(user_id)
        if profile.ndim == 1:
            profile = profile.reshape(1, -1)

        sims = cosine_similarity(profile, self.X).flatten()

        solved_ids = set()
        if exclude_solved:
            solved_ids = set(
                self.history[
                    (self.history["user_id"] == str(user_id)) & (self.history["verdict"].str.upper() == "AC")
                ]["problem_id"].tolist()
            )

        candidates: List[Tuple[int, float]] = []
        for idx, s in enumerate(sims):
            pid = self.idx2id[idx]
            if exclude_solved and pid in solved_ids:
                continue
            if not self._within_diff_range(pid, diff_min, diff_max):
                continue
            candidates.append((pid, float(s)))

        # 점수: 유사도(0.9) + 정확도(0.1)
        def score(pid, sim):
            return 0.9 * sim + 0.1 * float(self.acc_map.get(pid, 0.0))

        ranked = sorted([(pid, score(pid, sim)) for pid, sim in candidates],
                        key=lambda x: x[1], reverse=True)[:k]

        results = []
        for pid, sc in ranked:
            results.append({
                "problem_id": int(pid),
                "title": self.title_map.get(pid, ""),
                "difficulty": self.diff_map.get(pid, ""),
                "accuracy": float(self.acc_map.get(pid, 0.0)),
                "score": round(float(sc), 4),
                "reason": self._reason(user_id, pid),
            })
        return results


# -----------------------------
# Popularity-based Recommender
# -----------------------------
class PopularityRecommender(BaseRecommender):
    """Baseline recommender that suggests most popular problems"""

    def recommend(
        self,
        user_id: str,
        k: int = 5,
        exclude_solved: bool = True,
        diff_min: Optional[str] = None,
        diff_max: Optional[str] = None,
    ) -> List[Dict]:
        """Recommend most popular problems (highest accuracy)"""
        solved_ids = set()
        if exclude_solved:
            solved_ids = self._get_solved_problems(user_id)

        # Filter candidates
        candidates = []
        for _, row in self.problems.iterrows():
            pid = row["problem_id"]
            if exclude_solved and pid in solved_ids:
                continue
            if not self._within_diff_range(pid, diff_min, diff_max):
                continue
            candidates.append((pid, float(row["accuracy"])))

        # Sort by popularity (accuracy)
        ranked = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

        # Format results
        results = []
        for pid, acc in ranked:
            results.append({
                "problem_id": int(pid),
                "title": self.title_map.get(pid, ""),
                "difficulty": self.diff_map.get(pid, ""),
                "accuracy": acc,
                "score": acc,
                "reason": f"인기도 기반 추천 (정확도: {acc:.1f}%)",
            })
        return results


# -----------------------------
# Random Recommender
# -----------------------------
class RandomRecommender(BaseRecommender):
    """Baseline recommender that suggests random problems"""

    def __init__(self, problems_path: str, history_path: str, seed: Optional[int] = None):
        super().__init__(problems_path, history_path)
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def recommend(
        self,
        user_id: str,
        k: int = 5,
        exclude_solved: bool = True,
        diff_min: Optional[str] = None,
        diff_max: Optional[str] = None,
    ) -> List[Dict]:
        """Recommend random problems"""
        solved_ids = set()
        if exclude_solved:
            solved_ids = self._get_solved_problems(user_id)

        # Filter candidates
        candidates = []
        for _, row in self.problems.iterrows():
            pid = row["problem_id"]
            if exclude_solved and pid in solved_ids:
                continue
            if not self._within_diff_range(pid, diff_min, diff_max):
                continue
            candidates.append(pid)

        # Random sample
        selected = random.sample(candidates, min(k, len(candidates)))

        # Format results
        results = []
        for pid in selected:
            results.append({
                "problem_id": int(pid),
                "title": self.title_map.get(pid, ""),
                "difficulty": self.diff_map.get(pid, ""),
                "accuracy": float(self.acc_map.get(pid, 0.0)),
                "score": 0.5,
                "reason": "랜덤 추천",
            })
        return results


# -----------------------------
# Hybrid Recommender
# -----------------------------
class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining multiple strategies"""

    def __init__(
        self,
        problems_path: str,
        history_path: str,
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(problems_path, history_path)

        # Initialize sub-recommenders
        self.tfidf = TfidfRecommender(problems_path, history_path)
        self.popularity = PopularityRecommender(problems_path, history_path)

        # Default weights
        self.weights = weights or {"tfidf": 0.7, "popularity": 0.3}

    def recommend(
        self,
        user_id: str,
        k: int = 5,
        exclude_solved: bool = True,
        diff_min: Optional[str] = None,
        diff_max: Optional[str] = None,
    ) -> List[Dict]:
        """Generate hybrid recommendations"""
        # Get recommendations from each strategy
        tfidf_recs = self.tfidf.recommend(user_id, k * 2, exclude_solved, diff_min, diff_max)
        pop_recs = self.popularity.recommend(user_id, k * 2, exclude_solved, diff_min, diff_max)

        # Combine scores
        combined_scores = {}
        for rec in tfidf_recs:
            pid = rec["problem_id"]
            combined_scores[pid] = {
                "info": rec,
                "score": rec["score"] * self.weights["tfidf"]
            }

        for rec in pop_recs:
            pid = rec["problem_id"]
            if pid in combined_scores:
                combined_scores[pid]["score"] += rec["score"] * self.weights["popularity"]
            else:
                combined_scores[pid] = {
                    "info": rec,
                    "score": rec["score"] * self.weights["popularity"]
                }

        # Sort by combined score
        ranked = sorted(
            combined_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )[:k]

        # Format results
        results = []
        for pid, data in ranked:
            info = data["info"]
            info["score"] = round(data["score"], 4)
            info["reason"] = f"하이브리드 추천 (TF-IDF: {self.weights['tfidf']}, 인기도: {self.weights['popularity']})"
            results.append(info)

        return results


# -----------------------------
# Weakness-based Recommender
# -----------------------------
class WeaknessRecommender(BaseRecommender):
    """Recommender that analyzes user weaknesses and suggests problems to improve"""

    def __init__(self, problems_path: str, history_path: str, recent_days: int = 30):
        super().__init__(problems_path, history_path)
        self.recent_days = recent_days

        # Create tag index for problems
        self.problem_tags_map = {}
        for _, row in self.problems.iterrows():
            pid = row["problem_id"]
            tags = row["tags"].split(";") if row["tags"] else []
            self.problem_tags_map[pid] = [t.strip() for t in tags if t.strip()]

    def analyze_weakness(self, user_id: str) -> Dict:
        """Analyze user's weaknesses based on failed/wrong submissions"""
        user_history = self.history[self.history["user_id"] == str(user_id)]

        if user_history.empty:
            return {
                "total_attempts": 0,
                "total_failures": 0,
                "weak_tags": [],
                "tag_stats": {},
                "recent_failures": []
            }

        # Separate successful and failed attempts
        failed = user_history[user_history["verdict"].str.upper() != "AC"]
        solved = user_history[user_history["verdict"].str.upper() == "AC"]

        # Analyze tag statistics
        tag_success = defaultdict(int)
        tag_failure = defaultdict(int)
        tag_total = defaultdict(int)

        # Count successes by tag
        for pid in solved["problem_id"]:
            if pid in self.problem_tags_map:
                for tag in self.problem_tags_map[pid]:
                    tag_success[tag] += 1
                    tag_total[tag] += 1

        # Count failures by tag
        failed_problems = []
        for _, row in failed.iterrows():
            pid = row["problem_id"]
            if pid in self.problem_tags_map:
                failed_problems.append({
                    "problem_id": int(pid),
                    "title": self.title_map.get(pid, ""),
                    "tags": self.problem_tags_map[pid],
                    "difficulty": self.diff_map.get(pid, ""),
                    "verdict": row["verdict"]
                })
                for tag in self.problem_tags_map[pid]:
                    tag_failure[tag] += 1
                    tag_total[tag] += 1

        # Calculate tag statistics
        tag_stats = {}
        for tag in tag_total:
            total = tag_total[tag]
            success = tag_success.get(tag, 0)
            failure = tag_failure.get(tag, 0)
            success_rate = (success / total * 100) if total > 0 else 0

            tag_stats[tag] = {
                "total_attempts": total,
                "successes": success,
                "failures": failure,
                "success_rate": round(success_rate, 1)
            }

        # Find weak tags (low success rate, multiple attempts)
        weak_tags = []
        for tag, stats in tag_stats.items():
            if stats["total_attempts"] >= 2 and stats["success_rate"] < 60:
                weak_tags.append({
                    "tag": tag,
                    "success_rate": stats["success_rate"],
                    "failures": stats["failures"],
                    "total_attempts": stats["total_attempts"]
                })

        # Sort by failure count and success rate
        weak_tags.sort(key=lambda x: (x["failures"], -x["success_rate"]), reverse=True)

        return {
            "total_attempts": len(user_history),
            "total_failures": len(failed),
            "weak_tags": weak_tags[:5],  # Top 5 weak tags
            "tag_stats": tag_stats,
            "recent_failures": failed_problems[-5:]  # Last 5 failures
        }

    def recommend(
        self,
        user_id: str,
        k: int = 5,
        exclude_solved: bool = True,
        diff_min: Optional[str] = None,
        diff_max: Optional[str] = None,
    ) -> List[Dict]:
        """Recommend problems based on user's weaknesses"""

        # Analyze weaknesses
        weakness_analysis = self.analyze_weakness(user_id)

        if not weakness_analysis["weak_tags"]:
            # No weaknesses found, fall back to popularity-based recommendation
            # Use direct filtering instead of creating new recommender instance
            solved_ids = set()
            if exclude_solved:
                solved_ids = self._get_solved_problems(user_id)

            candidates = []
            for _, row in self.problems.iterrows():
                pid = row["problem_id"]
                if exclude_solved and pid in solved_ids:
                    continue
                if not self._within_diff_range(pid, diff_min, diff_max):
                    continue
                candidates.append((pid, float(row["accuracy"])))

            # Sort by popularity
            ranked = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

            results = []
            for pid, acc in ranked:
                results.append({
                    "problem_id": int(pid),
                    "title": self.title_map.get(pid, ""),
                    "difficulty": self.diff_map.get(pid, ""),
                    "accuracy": acc,
                    "score": acc,
                    "reason": "취약점 없음 - 인기도 기반 추천",
                })
            return results

        # Get solved problem IDs
        solved_ids = set()
        if exclude_solved:
            solved_ids = self._get_solved_problems(user_id)

        # Collect candidates from weak tags
        candidates = []
        weak_tag_names = [wt["tag"] for wt in weakness_analysis["weak_tags"]]

        for _, row in self.problems.iterrows():
            pid = row["problem_id"]

            # Skip if already solved
            if exclude_solved and pid in solved_ids:
                continue

            # Skip if not in difficulty range
            if not self._within_diff_range(pid, diff_min, diff_max):
                continue

            # Check if problem has weak tags
            problem_tags = self.problem_tags_map.get(pid, [])
            matching_weak_tags = [t for t in problem_tags if t in weak_tag_names]

            if matching_weak_tags:
                # Calculate priority score
                # Higher score for: more matching weak tags, easier difficulty, higher accuracy
                tag_score = len(matching_weak_tags) * 10

                # Priority for weaker tags (based on order in weak_tags list)
                weakness_priority = sum(
                    (5 - weak_tag_names.index(t)) * 5
                    for t in matching_weak_tags
                    if t in weak_tag_names[:5]
                )

                # Easier problems get higher priority
                diff_rank = _diff_to_rank(row["difficulty"])
                difficulty_bonus = (6 - diff_rank) * 3 if diff_rank >= 0 else 0

                # Higher accuracy gets slight bonus
                accuracy_bonus = row["accuracy"] * 0.1

                score = tag_score + weakness_priority + difficulty_bonus + accuracy_bonus

                candidates.append({
                    "problem_id": pid,
                    "score": score,
                    "matching_tags": matching_weak_tags,
                    "difficulty": row["difficulty"],
                    "accuracy": row["accuracy"]
                })

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Format results
        results = []
        for cand in candidates[:k]:
            pid = cand["problem_id"]
            matching_tags = cand["matching_tags"]

            # Generate detailed reason
            tag_info = []
            for tag in matching_tags[:2]:  # Top 2 matching tags
                tag_stat = weakness_analysis["tag_stats"].get(tag, {})
                failures = tag_stat.get("failures", 0)
                success_rate = tag_stat.get("success_rate", 0)
                tag_info.append(f"{tag} (실패 {failures}회, 성공률 {success_rate:.0f}%)")

            reason_parts = []
            if tag_info:
                reason_parts.append(f"취약 태그: {', '.join(tag_info)}")

            # Add difficulty guidance
            diff = self.diff_map.get(pid, "")
            reason_parts.append(f"{diff.capitalize()} 난이도로 기초 다지기")

            reason = " - ".join(reason_parts)

            results.append({
                "problem_id": int(pid),
                "title": self.title_map.get(pid, ""),
                "difficulty": diff,
                "accuracy": float(self.acc_map.get(pid, 0.0)),
                "score": round(cand["score"], 4),
                "reason": reason,
            })

        return results
