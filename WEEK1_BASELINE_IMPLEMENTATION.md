#  Baseline 추천 파이프라인 구현 + 샘플 API 작성

##  개요
 기본적인 추천 시스템 인프라를 구축하고, 다양한 베이스라인 추천 알고리즘을 구현

##  목표

1.  추천 시스템 기본 구조 설계
2.  베이스라인 추천 알고리즘 4종 구현
3.  RESTful API 서버 구축
4.  각 알고리즘 테스트 및 검증


##  구현된 추천 알고리즘

### 1. TfidfRecommender (콘텐츠 기반 추천)

**알고리즘 설명:**
- TF-IDF 벡터화를 사용한 콘텐츠 기반 필터링
- 사용자가 풀었던 문제의 특징(제목, 태그, 난이도)을 분석
- 코사인 유사도로 유사한 문제 추천

**핵심 로직:**
```python
# 1. 문제의 텍스트 특징을 TF-IDF 벡터로 변환
corpus = problems["title"] + " " + problems["tags"] + " " + problems["difficulty"]
X = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(corpus)

# 2. 사용자 프로필 생성 (풀었던 문제들의 평균 벡터)
user_profile = 풀었던_문제들_벡터.mean()

# 3. 코사인 유사도 계산
similarities = cosine_similarity(user_profile, all_problems)

# 4. 유사도 + 정확도 가중합으로 최종 점수 계산
score = 0.9 * similarity + 0.1 * accuracy
```

**추천 이유 생성:**
- TF-IDF 벡터의 기여도가 높은 상위 3개 키워드 추출
- 예: "이전 풀이와 유사한 키워드: graph, bfs, shortest-path"

**파일:** [`app/recommender.py:109-193`](C:\ml_ai\app\recommender.py#L109-L193)

---

### 2. PopularityRecommender (인기도 기반 추천)

**알고리즘 설명:**
- 정확도(정답률)가 높은 문제를 추천
- 가장 단순하지만 효과적인 베이스라인
- 콜드 스타트 문제에도 강건함

**핵심 로직:**
```python
# 정확도(accuracy) 기준으로 정렬하여 상위 k개 추천
problems.sort_values('accuracy', ascending=False).head(k)
```

**추천 이유:**
- "인기도 기반 추천 (정확도: 88.0%)"

**파일:** [`app/recommender.py:196-239`](C:\ml_ai\app\recommender.py#L196-L239)

---

### 3. RandomRecommender (랜덤 추천)

**알고리즘 설명:**
- 무작위로 문제를 추천
- A/B 테스트의 대조군으로 사용
- Exploration을 위한 기본 전략

**핵심 로직:**
```python
# 조건에 맞는 문제 중 랜덤 샘플링
random.sample(candidate_problems, k)
```

**특징:**
- `seed` 파라미터로 재현 가능성 보장
- 모든 문제에 동일한 기회 부여

**파일:** [`app/recommender.py:242-291`](C:\ml_ai\app\recommender.py#L242-L291)

---

### 4. HybridRecommender (하이브리드 추천)

**알고리즘 설명:**
- TF-IDF와 인기도 방식을 결합
- 가중 평균으로 최종 점수 계산
- 개인화와 인기도의 균형

**핵심 로직:**
```python
# 각 추천기에서 결과 가져오기
tfidf_results = tfidf_recommender.recommend(user_id, k*2)
popularity_results = popularity_recommender.recommend(user_id, k*2)

# 가중 평균 (기본: TF-IDF 70%, 인기도 30%)
final_score = 0.7 * tfidf_score + 0.3 * popularity_score
```

**추천 이유:**
- "하이브리드 추천 (TF-IDF: 0.7, 인기도: 0.3)"

**파일:** [`app/recommender.py:294-362`](C:\ml_ai\app\recommender.py#L294-L362)

##  API 엔드포인트

### 기본 엔드포인트

#### 1. Health Check
```http
GET /health
```
서버 상태 확인

#### 2. Debug Info
```http
GET /__debug
```
데이터 통계 및 사용 가능한 추천 전략 확인

### 추천 엔드포인트

#### 1. TF-IDF 추천
```http
GET /recommend?user_id={id}&k={num}
```

**파라미터:**
- `user_id`: 사용자 ID (필수)
- `k`: 추천 개수 (기본값: 5, 범위: 1-50)
- `exclude_solved`: 풀었던 문제 제외 여부 (기본값: true)
- `difficulty_min`: 최소 난이도 (선택)
- `difficulty_max`: 최대 난이도 (선택)

**응답 예시:**
```json
{
  "user_id": "1",
  "k": 3,
  "items": [
    {
      "problem_id": 1300,
      "title": "K번째 수",
      "difficulty": "gold",
      "accuracy": 0.29,
      "score": 0.1199,
      "reason": "이전 풀이와 유사한 키워드: math, binary-search, sorting"
    }
  ]
}
```

#### 2. 인기도 기반 추천
```http
GET /recommend/popularity?user_id={id}&k={num}
```

#### 3. 랜덤 추천
```http
GET /recommend/random?user_id={id}&k={num}
```

#### 4. 하이브리드 추천
```http
GET /recommend/hybrid?user_id={id}&k={num}
```

#### 5. 배치 추천
```http
POST /recommend/batch?strategy={strategy}
Content-Type: application/json

{
  "user_ids": ["1", "2", "3"],
  "k": 5,
  "exclude_solved": true
}
```

**파라미터:**
- `strategy`: 추천 전략 선택 (tfidf, popularity, random, hybrid)

##  공통 기능

### 1. 데이터 전처리
```python
class BaseRecommender(ABC):
    def _validate_and_prepare_data(self):
        # 필수 컬럼 검증
        # 결측값 처리
        # 데이터 타입 통일
        # 빠른 조회를 위한 매핑 생성
```

### 2. 풀었던 문제 필터링
```python
def _get_solved_problems(self, user_id: str) -> set:
    # verdict == "AC"인 문제들만 추출
    return set(solved_problem_ids)
```

### 3. 난이도 범위 필터링
```python
def _within_diff_range(self, pid, diff_min, diff_max):
    # bronze(0) ~ ruby(5) 숫자로 변환하여 비교
    return diff_min <= problem_difficulty <= diff_max
```

##  데이터 스키마

### problems.csv
| 컬럼 | 타입 | 설명 |
|------|------|------|
| problem_id | int | 문제 번호 |
| title | str | 문제 제목 |
| tags | str | 태그 (세미콜론 구분) |
| difficulty | str | 난이도 (bronze/silver/gold/...) |
| accuracy | float | 정답률 (0.0-1.0) |

### user_history.csv
| 컬럼 | 타입 | 설명 |
|------|------|------|
| user_id | str | 사용자 ID |
| problem_id | int | 문제 번호 |
| verdict | str | 채점 결과 (AC/WA/TLE/...) |
| solved_at | str | 풀이 일시 |

##  테스트 방법

### 1. 서버 실행
```bash
cd c:\ml_ai
.venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

### 2. Swagger UI 접속
브라우저에서 http://127.0.0.1:8000/docs 접속

### 3. curl로 테스트
```bash
# TF-IDF 추천
curl "http://127.0.0.1:8000/recommend?user_id=1&k=3"

# 인기도 추천
curl "http://127.0.0.1:8000/recommend/popularity?user_id=1&k=3"

# 하이브리드 추천
curl "http://127.0.0.1:8000/recommend/hybrid?user_id=1&k=3"
```
