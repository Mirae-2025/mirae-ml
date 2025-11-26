# Week 2: 취약점 기반 추천 API 구현 + 추천 이유 생성

##  개요

사용자의 오답/실패 이력을 분석하여 취약한 부분을 보완할 수 있는 **취약점 기반 추천 시스템**을 구현

## 목표

1.  사용자 취약점 분석 시스템 구축
2.  태그별 성공률/실패율 통계 생성
3.  취약점 기반 문제 추천 알고리즘 구현
4.  상세한 추천 이유 자동 생성
5.  새로운 API 엔드포인트 추가

##  핵심 기능

### 1. 취약점 분석 (WeaknessRecommender)

**알고리즘 개요:**
```
사용자의 풀이 이력을 분석
    ↓
태그별로 성공/실패 집계
    ↓
성공률 60% 미만인 태그를 "취약 태그"로 식별
    ↓
취약 태그가 포함된 문제를 쉬운 난이도부터 추천
```

**파일:** [`app/recommender.py:368-597`](C:\ml_ai\app\recommender.py#L368-L597)

---

### 2. 취약점 분석 로직

#### 2.1. 태그별 통계 계산

```python
def analyze_weakness(self, user_id: str) -> Dict:
    """
    사용자의 취약점을 분석하여 상세 통계 반환

    Returns:
        {
            "total_attempts": 전체 시도 횟수,
            "total_failures": 전체 실패 횟수,
            "weak_tags": 취약 태그 목록 (상위 5개),
            "tag_stats": 모든 태그의 통계,
            "recent_failures": 최근 실패 문제 (최대 5개)
        }
    """
```

**분석 과정:**

1. **성공/실패 문제 분리**
```python
failed = user_history[user_history["verdict"] != "AC"]
solved = user_history[user_history["verdict"] == "AC"]
```

2. **태그별 집계**
```python
for problem in solved:
    for tag in problem.tags:
        tag_success[tag] += 1
        tag_total[tag] += 1

for problem in failed:
    for tag in problem.tags:
        tag_failure[tag] += 1
        tag_total[tag] += 1
```

3. **성공률 계산**
```python
success_rate = (successes / total_attempts) * 100
```

4. **취약 태그 식별**
```python
# 조건: 2회 이상 시도 & 성공률 60% 미만
if total_attempts >= 2 and success_rate < 60:
    weak_tags.append({
        "tag": tag,
        "success_rate": success_rate,
        "failures": failures,
        "total_attempts": total_attempts
    })
```

#### 2.2. 분석 결과 예시

```json
{
  "user_id": "2",
  "total_attempts": 3,
  "total_failures": 1,
  "weak_tags": [
    {
      "tag": "greedy",
      "success_rate": 50.0,
      "failures": 1,
      "total_attempts": 2
    }
  ],
  "tag_stats": {
    "greedy": {
      "total_attempts": 2,
      "successes": 1,
      "failures": 1,
      "success_rate": 50.0
    },
    "implementation": {
      "total_attempts": 1,
      "successes": 1,
      "failures": 0,
      "success_rate": 100.0
    }
  },
  "recent_failures": [
    {
      "problem_id": 1700,
      "title": "멀티탭 스케줄링",
      "tags": ["greedy", "simulation", "queue"],
      "difficulty": "silver",
      "verdict": "WA"
    }
  ]
}
```

---

### 3. 취약점 기반 추천 로직

#### 3.1. 문제 후보 필터링

```python
for problem in all_problems:
    # 이미 푼 문제 제외
    if problem in solved_problems:
        continue

    # 난이도 범위 필터링
    if not in_difficulty_range(problem):
        continue

    # 취약 태그 포함 여부 확인
    matching_tags = [t for t in problem.tags if t in weak_tags]
    if matching_tags:
        candidates.append(problem)
```

#### 3.2. 우선순위 스코어 계산

```python
score = (
    len(matching_tags) * 10 +           # 매칭되는 취약 태그 수 (중요!)
    weakness_priority +                  # 취약도 순위 (1위 태그 > 2위 태그)
    (6 - difficulty_rank) * 3 +         # 쉬운 난이도 가산점
    accuracy * 0.1                       # 정답률 보너스
)
```

**스코어 계산 예시:**

문제: "보석 도둑" (gold, greedy 태그, accuracy=0.34)
- 취약 태그: ["greedy"]
- `tag_score = 1 * 10 = 10`
- `weakness_priority = (5 - 0) * 5 = 25` (1순위 취약 태그)
- `difficulty_bonus = (6 - 2) * 3 = 12` (gold는 2순위)
- `accuracy_bonus = 0.34 * 0.1 = 0.034`
- **총점 = 47.034**

#### 3.3. 추천 결과

```python
# 점수 기준 내림차순 정렬 후 상위 k개 선택
candidates.sort(key=lambda x: x['score'], reverse=True)
return candidates[:k]
```

---

### 4. 추천 이유 자동 생성

#### 4.1. 템플릿 시스템

```python
def _generate_reason(problem, weak_tags, tag_stats):
    """
    문제별로 맞춤형 추천 이유 생성

    형식: "취약 태그: [태그1 (실패 N회, 성공률 X%)], [태그2 ...] - [난이도] 난이도로 기초 다지기"
    """
    tag_info = []
    for tag in problem.matching_tags[:2]:  # 최대 2개 태그
        stat = tag_stats[tag]
        tag_info.append(
            f"{tag} (실패 {stat['failures']}회, 성공률 {stat['success_rate']:.0f}%)"
        )

    reason = f"취약 태그: {', '.join(tag_info)} - {difficulty.capitalize()} 난이도로 기초 다지기"
    return reason
```

#### 4.2. 추천 이유 예시

**케이스 1: 취약점 발견**
```
"취약 태그: greedy (실패 1회, 성공률 50%), simulation (실패 1회, 성공률 0%) - Silver 난이도로 기초 다지기"
```

**케이스 2: 취약점 없음**
```
"취약점 없음 - 인기도 기반 추천"
```

---

##  새로운 API 엔드포인트

### 1. 취약점 분석

```http
GET /analysis/weakness?user_id={id}
```

**응답 스키마:**
```python
class WeaknessAnalysisResponse(BaseModel):
    user_id: str
    total_attempts: int
    total_failures: int
    weak_tags: List[WeakTag]
    tag_stats: Dict[str, TagStat]
    recent_failures: List[FailedProblem]
```

**사용 사례:**
- 사용자 프로필 페이지에 취약 분야 표시
- 학습 대시보드 데이터 제공
- 문제 추천 전 분석 결과 확인

---

### 2. 취약점 기반 추천

```http
GET /recommend/weakness?user_id={id}&k={num}
```

**파라미터:**
- `user_id`: 사용자 ID (필수)
- `k`: 추천 개수 (기본값: 5)
- `exclude_solved`: 풀었던 문제 제외 (기본값: true)
- `difficulty_min`: 최소 난이도 (선택)
- `difficulty_max`: 최대 난이도 (선택)

**응답 예시:**
```json
{
  "user_id": "2",
  "k": 3,
  "items": [
    {
      "problem_id": 1700,
      "title": "멀티탭 스케줄링",
      "difficulty": "silver",
      "accuracy": 0.44,
      "score": 50.044,
      "reason": "취약 태그: greedy (실패 1회, 성공률 50%) - Silver 난이도로 기초 다지기"
    },
    {
      "problem_id": 1200,
      "title": "보석 도둑",
      "difficulty": "gold",
      "accuracy": 0.34,
      "score": 47.034,
      "reason": "취약 태그: greedy (실패 1회, 성공률 50%) - Gold 난이도로 기초 다지기"
    }
  ]
}
```

---

### 3. 배치 추천 (업데이트)

```http
POST /recommend/batch?strategy=weakness
Content-Type: application/json

{
  "user_ids": ["1", "2"],
  "k": 3
}
```

이제 `strategy` 파라미터에 `weakness` 추가 지원!

---

##  아키텍처 개선사항

### 1. Pydantic 스키마 추가

```python
# app/schemas.py

class WeakTag(BaseModel):
    """취약 태그 정보"""
    tag: str
    success_rate: float
    failures: int
    total_attempts: int

class TagStat(BaseModel):
    """태그별 통계"""
    total_attempts: int
    successes: int
    failures: int
    success_rate: float

class FailedProblem(BaseModel):
    """실패한 문제 정보"""
    problem_id: int
    title: str
    tags: List[str]
    difficulty: str
    verdict: str

class WeaknessAnalysisResponse(BaseModel):
    """취약점 분석 응답"""
    user_id: str
    total_attempts: int
    total_failures: int
    weak_tags: List[WeakTag]
    tag_stats: Dict[str, TagStat]
    recent_failures: List[FailedProblem]
```

### 2. 추천기 확장

모든 추천기가 동일한 인터페이스 사용:
```python
recommender.recommend(
    user_id=user_id,
    k=k,
    exclude_solved=True,
    diff_min=difficulty_min,
    diff_max=difficulty_max
)
```

---

##  알고리즘 설명

### 취약도 우선순위 계산

```python
# 취약 태그 순위: [greedy, dp, graph, binary-search, ...]
weakness_priority = sum(
    (5 - weak_tag_names.index(t)) * 5
    for t in matching_tags
    if t in weak_tag_names[:5]
)
```

**예시:**
- 1순위 취약 태그 매칭: `(5-0)*5 = 25점`
- 2순위 취약 태그 매칭: `(5-1)*5 = 20점`
- 3순위 취약 태그 매칭: `(5-2)*5 = 15점`

→ **더 취약한 태그에 더 높은 가중치!**

### 난이도 가산점

```python
difficulty_bonus = (6 - diff_rank) * 3

# diff_rank: bronze=0, silver=1, gold=2, ...
```

**예시:**
- Bronze: `(6-0)*3 = 18점`
- Silver: `(6-1)*3 = 15점`
- Gold: `(6-2)*3 = 12점`

→ **쉬운 문제부터 추천!**

---

##  추천 품질 향상

### Before (Week 1)
```json
{
  "reason": "이전 풀이와 유사한 키워드: greedy, sorting, heap"
}
```

### After (Week 2)
```json
{
  "reason": "취약 태그: greedy (실패 3회, 성공률 25%) - Bronze 난이도로 기초 다지기"
}
```

**개선점:**
-  **구체적**: 실패 횟수와 성공률 명시
-  **실행 가능**: 난이도 수준 제시
-  **개인화**: 사용자별 맞춤 분석
-  **설명 가능**: 추천 근거가 명확함

---

## 사용 시나리오

### 시나리오 1: 학습 초기 단계

**사용자:** 백준 입문자, greedy 문제 여러 번 실패

**분석 결과:**
```json
{
  "weak_tags": [
    {"tag": "greedy", "success_rate": 20.0, "failures": 4}
  ]
}
```

**추천 결과:**
- Bronze 난이도의 greedy 문제
- 정답률이 높은 문제 우선
- **이유:** "greedy (실패 4회, 성공률 20%) - Bronze 난이도로 기초 다지기"

---

### 시나리오 2: 특정 알고리즘 약점

**사용자:** 중급자, DP와 그래프에서 고전

**분석 결과:**
```json
{
  "weak_tags": [
    {"tag": "dp", "success_rate": 40.0, "failures": 3},
    {"tag": "graph", "success_rate": 50.0, "failures": 2}
  ]
}
```

**추천 결과:**
- DP 태그 Silver/Gold 문제
- Graph 태그 Silver 문제
- **이유:** "dp (실패 3회, 성공률 40%), graph (실패 2회, 성공률 50%) - Silver 난이도로 기초 다지기"

---

### 시나리오 3: 취약점 없음

**사용자:** 고급자, 대부분의 문제 해결

**분석 결과:**
```json
{
  "weak_tags": []
}
```

**추천 결과:**
- 인기도 기반으로 fallback
- **이유:** "취약점 없음 - 인기도 기반 추천"

---

##  테스트 

### 1. 취약점 분석 테스트

```bash
# User 2는 greedy에서 50% 성공률
curl "http://127.0.0.1:8000/analysis/weakness?user_id=2"
```

### 2. 취약점 기반 추천 테스트

```bash
# User 2에게 greedy 태그 문제 추천
curl "http://127.0.0.1:8000/recommend/weakness?user_id=2&k=3"
```

### 3. Python 테스트 스크립트

```python
import requests

BASE_URL = "http://127.0.0.1:8000"

# 1. 취약점 분석
analysis = requests.get(
    f"{BASE_URL}/analysis/weakness",
    params={"user_id": "2"}
).json()

print("취약 태그:", analysis["weak_tags"])

# 2. 추천 요청
recommendations = requests.get(
    f"{BASE_URL}/recommend/weakness",
    params={"user_id": "2", "k": 5}
).json()

for item in recommendations["items"]:
    print(f"{item['title']}: {item['reason']}")
```


## 🎓 핵심 학습 포인트

### 1. 데이터 기반 개인화
- 사용자의 실제 행동 데이터(실패 이력) 활용
- 단순 유사도가 아닌 학습 효과 극대화

### 2. 설명 가능한 AI
- 추천 이유를 명확하게 제시
- 사용자 신뢰도 향상

### 3. 적응형 난이도 조절
- 취약한 부분은 쉬운 난이도부터
- 점진적 난이도 상승 전략

### 4. Fallback 전략
- 취약점이 없는 경우 대체 로직
- 견고한 시스템 설계

---

##  향후 개선 방향

### 1. 시간대별 학습 패턴 분석
```python
def analyze_time_patterns(user_id):
    # 학습 시간대별 성공률 분석
    # 최적 학습 시간 추천
```

### 2. 학습 곡선 추적
```python
def track_learning_curve(user_id, tag):
    # 태그별 실력 변화 추이
    # 성장 속도 측정
```

### 3. 문제 간 연관 관계
```python
def find_related_problems(problem_id):
    # 유사 문제 그룹핑
    # 학습 경로 추천
```

### 4. 개인별 학습 속도
```python
def estimate_difficulty_for_user(user_id, problem):
    # 사용자별 체감 난이도 추정
    # 맞춤형 난이도 조절
```

---

