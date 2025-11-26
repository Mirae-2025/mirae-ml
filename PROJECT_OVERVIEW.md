# 백준 문제 추천 시스템 프로젝트 전체 설명

## 📌 프로젝트 개요

이 프로젝트는 백준 온라인 저지(Baekjoon Online Judge)의 알고리즘 문제를 사용자에게 추천하는 머신러닝 기반 시스템입니다. FastAPI를 사용하여 RESTful API 서버를 구축했고, 5가지 추천 알고리즘을 구현했습니다.

**프로젝트 구조:**
```
ml_ai/
├── app/
│   ├── main.py          # FastAPI 서버 및 API 엔드포인트
│   ├── schemas.py       # 데이터 모델 정의 (Pydantic)
│   ├── recommender.py   # 추천 알고리즘 구현
│   └── data/
│       ├── problems.csv      # 문제 정보 (12개)
│       └── user_history.csv  # 사용자 풀이 이력 (6개)
```

---

## 🎯 1. main.py - API 서버의 핵심

`main.py`는 프로젝트의 진입점이자 API 서버. FastAPI 프레임워크를 사용하여 HTTP 요청을 받아 추천 결과를 반환

### 주요 역할

1. **서버 초기화와 설정**
   - FastAPI 애플리케이션을 생성하고 CORS(Cross-Origin Resource Sharing)를 설정합니다
   - CORS는 브라우저에서 실행되는 프론트엔드 애플리케이션이 이 API 서버에 접근할 수 있도록 허용하는 설정입니다

2. **데이터 경로 설정**
   - CSV 파일(problems.csv, user_history.csv)의 위치를 지정합니다
   - 이 파일들은 문제 정보와 사용자 풀이 이력을 담고 있습니다

3. **5가지 추천기 인스턴스 생성**
   서버가 시작될 때 5개의 추천 알고리즘 객체를 미리 만들어둡니다:

   - **TfidfRecommender**: 사용자가 과거에 풀었던 문제의 제목, 태그, 난이도를 분석하여 비슷한 특징을 가진 문제를 추천합니다. "당신이 DP 문제를 많이 풀었다면, 비슷한 DP 문제를 추천해드릴게요" 같은 방식입니다.

   - **PopularityRecommender**: 가장 많은 사람들이 맞춘 문제(정답률이 높은 문제)를 추천합니다. 개인화는 없지만 안전한 선택입니다.

   - **RandomRecommender**: 조건에 맞는 문제 중 무작위로 선택합니다. A/B 테스트에서 대조군으로 사용하거나, 사용자에게 다양한 경험을 제공할 때 유용합니다.

   - **HybridRecommender**: TfidfRecommender와 PopularityRecommender를 70:30 비율로 결합합니다. 개인화된 추천과 인기도를 균형있게 반영합니다.

   - **WeaknessRecommender** (Week 2 구현): 사용자가 자주 틀리는 알고리즘 태그를 분석하여 약점을 보완할 문제를 추천합니다. "당신은 Greedy 문제를 자주 틀리네요. Greedy 연습 문제를 추천해드릴게요" 같은 방식입니다.

4. **9개의 API 엔드포인트 제공**

   각 엔드포인트는 HTTP 요청을 받아 추천 결과를 JSON 형태로 반환합니다:

   - `GET /health`: 서버가 정상 작동하는지 확인
   - `GET /recommend`: TF-IDF 기반 추천
   - `GET /recommend/popularity`: 인기도 기반 추천
   - `GET /recommend/random`: 랜덤 추천
   - `GET /recommend/hybrid`: 하이브리드 추천
   - `GET /recommend/weakness`: 취약점 기반 추천 (Week 2)
   - `GET /analysis/weakness`: 사용자 취약점 분석 (Week 2)
   - `POST /recommend/batch`: 여러 사용자 일괄 추천
   - `GET /__debug`: 디버깅 정보 (개발용)

### 동작 흐름 예시

사용자가 `GET /recommend?user_id=1&k=5`를 요청하면:

1. FastAPI가 `recommend()` 함수를 호출합니다
2. `user_id="1"`, `k=5` (5개 추천) 파라미터가 전달됩니다
3. `tfidf_recommender.recommend()` 메서드를 실행합니다
4. 추천 결과(문제 목록)를 JSON으로 변환하여 반환합니다

```json
{
  "user_id": "1",
  "k": 5,
  "items": [
    {
      "problem_id": 1003,
      "title": "피보나치 함수",
      "difficulty": "silver",
      "accuracy": 0.45,
      "score": 0.8234,
      "reason": "이전 풀이와 유사한 키워드: dp, recursion"
    },
    ...
  ]
}
```

---

## 📦 2. schemas.py - 데이터 구조 정의

`schemas.py`는 API에서 주고받는 데이터의 형식을 정의하는 파일입니다. Pydantic이라는 라이브러리를 사용하여 데이터 검증과 자동 문서화를 제공합니다.

### 주요 역할

**"계약서"를 작성하는 것과 같습니다.** API를 사용하는 개발자가 "이 엔드포인트는 어떤 형식의 데이터를 받고, 어떤 형식으로 응답하는지"를 명확히 알 수 있게 합니다.

### 주요 스키마 설명

#### 1. RecommendationItem (개별 추천 문제)
```python
{
  "problem_id": 1003,        # 문제 번호
  "title": "피보나치 함수",   # 문제 제목
  "difficulty": "silver",    # 난이도
  "accuracy": 0.45,          # 정답률 (45%)
  "score": 0.8234,           # 추천 점수 (높을수록 추천도 높음)
  "reason": "dp 관련 추천"   # 왜 이 문제를 추천하는지 설명
}
```

#### 2. RecommendResponse (추천 API 응답)
```python
{
  "user_id": "1",            # 요청한 사용자
  "k": 5,                    # 요청한 개수
  "items": [...]             # 추천 문제 목록 (RecommendationItem 배열)
}
```

#### 3. BatchRecommendRequest (배치 추천 요청)
여러 사용자를 한 번에 처리할 때 사용합니다:
```python
{
  "user_ids": ["1", "2", "3"],    # 여러 사용자 ID
  "k": 5,                         # 각 사용자당 추천 개수
  "exclude_solved": true,         # 풀었던 문제 제외
  "difficulty_min": "silver",     # 최소 난이도
  "difficulty_max": "gold"        # 최대 난이도
}
```

#### 4. WeakTag (취약 태그 - Week 2)
사용자가 어려워하는 알고리즘 주제를 나타냅니다:
```python
{
  "tag": "greedy",           # 태그 이름
  "success_rate": 50.0,      # 성공률 50%
  "failures": 3,             # 실패 횟수
  "total_attempts": 6        # 총 시도 횟수
}
```

#### 5. WeaknessAnalysisResponse (취약점 분석 응답)
```python
{
  "user_id": "2",
  "total_attempts": 10,           # 전체 시도 횟수
  "total_failures": 5,            # 전체 실패 횟수
  "weak_tags": [                  # 취약 태그 목록 (상위 5개)
    {"tag": "greedy", "success_rate": 50.0, ...}
  ],
  "tag_stats": {                  # 모든 태그의 통계
    "dp": {"successes": 3, "failures": 2, ...}
  },
  "recent_failures": [...]        # 최근 실패한 문제
}
```

### Pydantic의 장점

1. **자동 검증**: 잘못된 형식의 데이터가 들어오면 자동으로 에러를 발생시킵니다
2. **자동 문서화**: Swagger UI에서 자동으로 API 문서를 생성합니다
3. **타입 안정성**: 개발 중 IDE에서 자동완성과 타입 체크를 제공합니다

---

## 🧠 3. recommender.py - 추천 알고리즘의 두뇌

`recommender.py`는 실제 추천 로직이 구현된 핵심 파일입니다. 5가지 추천 알고리즘 클래스가 정의되어 있으며, 각각 다른 방식으로 문제를 추천합니다.

### 전체 구조

```
BaseRecommender (추상 클래스)
├── TfidfRecommender
├── PopularityRecommender
├── RandomRecommender
├── HybridRecommender
└── WeaknessRecommender
```

모든 추천기는 `BaseRecommender`를 상속받아 공통 기능(데이터 로드, 필터링 등)을 재사용합니다.

### BaseRecommender (공통 기능)

모든 추천기가 공유하는 기본 기능을 제공합니다:

1. **CSV 파일 로드**: `_read_csv_robust()` 함수로 문제 데이터와 풀이 이력을 읽어옵니다
2. **데이터 검증**: 필수 컬럼이 있는지 확인하고, 결측값을 처리합니다
3. **풀이한 문제 조회**: `_get_solved_problems()` - 사용자가 AC(Accepted)를 받은 문제 목록을 반환
4. **난이도 필터링**: `_within_diff_range()` - 지정된 난이도 범위 내의 문제만 선택

### 1. TfidfRecommender (콘텐츠 기반 필터링)

**핵심 아이디어**: "비슷한 문제를 풀었으면, 비슷한 문제를 추천하자"

#### 동작 방식

1. **TF-IDF 벡터화**
   - 각 문제의 제목, 태그, 난이도를 텍스트로 합칩니다
   - 예: "피보나치 함수 dp;recursion silver" → 숫자 벡터로 변환
   - TF-IDF는 "중요한 단어"에 높은 가중치를 부여하는 기법입니다

2. **사용자 프로필 생성**
   - 사용자가 풀었던 문제들의 벡터를 평균내어 "사용자 취향"을 표현합니다
   - 예: 사용자가 DP 문제를 많이 풀었다면, 프로필 벡터에서 "dp" 관련 값이 높아집니다

3. **코사인 유사도 계산**
   - 사용자 프로필과 각 문제 벡터 간의 유사도를 계산합니다
   - 유사도가 높을수록 사용자 취향에 맞는 문제입니다

4. **최종 점수 계산**
   ```
   score = 0.9 × 유사도 + 0.1 × 정답률
   ```
   - 유사도를 주로 반영하되, 정답률도 약간 고려합니다

5. **추천 이유 생성**
   - 어떤 키워드가 매칭되었는지 분석하여 이유를 생성합니다
   - 예: "이전 풀이와 유사한 키워드: dp, recursion"

### 2. PopularityRecommender (인기도 기반)

**핵심 아이디어**: "많은 사람이 맞춘 문제는 좋은 문제다"

#### 동작 방식

1. 모든 문제를 정답률(accuracy) 순으로 정렬합니다
2. 사용자가 이미 푼 문제는 제외합니다
3. 난이도 필터가 있다면 해당 범위만 선택합니다
4. 상위 k개를 반환합니다

**장점**: 개인화는 없지만 안정적이고 빠릅니다. 콜드 스타트(신규 사용자) 문제에 강합니다.

### 3. RandomRecommender (무작위 추천)

**핵심 아이디어**: "랜덤으로 선택해서 다양성을 제공하자"

#### 동작 방식

1. 조건(난이도, 풀이 여부 등)에 맞는 문제들을 모읍니다
2. `random.sample()`로 k개를 무작위 선택합니다
3. `seed=42`로 고정하여 재현 가능성을 보장합니다

**용도**: A/B 테스트의 대조군, Exploration 전략

### 4. HybridRecommender (하이브리드)

**핵심 아이디어**: "개인화와 인기도를 함께 고려하자"

#### 동작 방식

1. TfidfRecommender와 PopularityRecommender를 각각 실행합니다
2. 각 문제의 점수를 가중 평균으로 결합합니다:
   ```
   최종 점수 = 0.7 × TF-IDF 점수 + 0.3 × 인기도 점수
   ```
3. 최종 점수 순으로 정렬하여 상위 k개를 반환합니다

**장점**: 개인화의 정확도와 인기도의 안정성을 동시에 얻습니다.

### 5. WeaknessRecommender (취약점 기반 - Week 2)

**핵심 아이디어**: "약점을 분석해서 그 부분을 보완할 문제를 추천하자"

#### 동작 방식

1. **취약점 분석** (`analyze_weakness()`)
   - 사용자의 풀이 이력을 성공(AC)과 실패(WA, TLE 등)로 분류합니다
   - 각 알고리즘 태그별로 성공률을 계산합니다
   - 성공률이 60% 미만인 태그를 "취약 태그"로 식별합니다

   예시:
   ```python
   tag_stats = {
     "dp": {"successes": 5, "failures": 1, "success_rate": 83.3},
     "greedy": {"successes": 1, "failures": 1, "success_rate": 50.0},  # 취약!
     "graph": {"successes": 0, "failures": 3, "success_rate": 0.0}     # 취약!
   }
   ```

2. **문제 추천**
   - 취약 태그를 포함하는 문제를 찾습니다
   - 각 문제에 대해 점수를 계산합니다:

   ```
   점수 = (매칭 태그 수 × 10)
        + (취약도 우선순위 보너스)
        + (난이도 보너스 - 쉬울수록 높음)
        + (정답률 × 0.1)
   ```

   - 쉬운 문제부터 추천하여 기초를 다지도록 유도합니다

3. **상세한 추천 이유 생성**
   ```
   "취약 태그: greedy (실패 3회, 성공률 33%) - Silver 난이도로 기초 다지기"
   ```

#### Fallback 로직

만약 취약점이 없는 사용자라면(모든 태그를 잘 푸는 경우):
- 인기도 기반 추천으로 전환합니다
- 이유: "취약점 없음 - 인기도 기반 추천"

---

## 🔄 전체 시스템 흐름

사용자가 취약점 기반 추천을 요청하는 전체 과정:

```
1. 브라우저/클라이언트
   ↓ GET /recommend/weakness?user_id=2&k=5

2. main.py (FastAPI 서버)
   ↓ recommend_weakness() 함수 실행
   ↓ schemas.py의 RecommendResponse 형식 검증

3. recommender.py (WeaknessRecommender)
   ↓ analyze_weakness("2") - 사용자 2의 취약점 분석
   ↓ - tag_stats 계산 (태그별 성공률)
   ↓ - weak_tags 식별 (성공률 60% 미만)
   ↓
   ↓ recommend("2", k=5) - 추천 생성
   ↓ - 취약 태그 포함 문제 필터링
   ↓ - 점수 계산 및 정렬
   ↓ - 추천 이유 생성

4. main.py
   ↓ JSON 응답 생성

5. 브라우저/클라이언트
   ↓ 추천 결과 표시
```

---

## 🎓 핵심 개념 정리

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)
- **목적**: 텍스트에서 중요한 단어를 찾는 기법
- **원리**:
  - TF: 한 문서에서 자주 나오는 단어 → 중요
  - IDF: 모든 문서에서 자주 나오는 단어 → 덜 중요
  - TF-IDF = TF × IDF
- **예시**: "the", "a" 같은 흔한 단어는 점수가 낮고, "dp", "greedy" 같은 특정 단어는 점수가 높음

### 2. 코사인 유사도 (Cosine Similarity)
- **목적**: 두 벡터가 얼마나 비슷한 방향을 가리키는지 측정
- **범위**: -1 ~ 1 (1에 가까울수록 유사)
- **장점**: 벡터의 크기와 무관하게 방향만 비교

### 3. 콜드 스타트 문제 (Cold Start Problem)
- **문제**: 신규 사용자는 풀이 이력이 없어 개인화 추천이 불가능
- **해결**: 인기도 기반 추천, 랜덤 추천 등 fallback 전략 사용

### 4. 추천 시스템의 Trade-off
- **정확도 vs 다양성**: TF-IDF는 정확하지만 비슷한 문제만 추천, Random은 다양하지만 덜 정확
- **개인화 vs 안정성**: TF-IDF는 개인화되지만 데이터 필요, Popularity는 안정적이지만 개인화 없음
- **Exploitation vs Exploration**: 알려진 것 활용 vs 새로운 것 탐색

---

## 📊 데이터 흐름

### CSV 파일 구조

**problems.csv** (문제 정보):
```csv
problem_id,title,tags,difficulty,accuracy
1000,A+B,math;implementation,bronze,0.89
1003,피보나치 함수,dp;recursion,silver,0.45
```

**user_history.csv** (풀이 이력):
```csv
user_id,problem_id,verdict,solved_at
1,1000,AC,2024-01-01
1,1003,WA,2024-01-02
2,1003,AC,2024-01-03
```

### 데이터 처리 파이프라인

```
1. CSV 파일 로드
   ↓ _read_csv_robust()

2. 데이터 정규화
   ↓ - 컬럼명 소문자 변환
   ↓ - 결측값 처리
   ↓ - 타입 변환

3. 빠른 조회를 위한 매핑
   ↓ - acc_map: {problem_id → accuracy}
   ↓ - diff_map: {problem_id → difficulty}
   ↓ - title_map: {problem_id → title}

4. 추천 알고리즘 적용
   ↓ - 필터링 (난이도, 풀이 여부)
   ↓ - 점수 계산
   ↓ - 정렬 및 상위 k개 선택

5. JSON 응답 생성
```

---

## 🚀 API 사용 예시

### 1. TF-IDF 추천
```bash
GET /recommend?user_id=1&k=5&difficulty_min=silver&difficulty_max=gold
```

### 2. 취약점 분석
```bash
GET /analysis/weakness?user_id=2
```

### 3. 배치 추천
```bash
POST /recommend/batch?strategy=weakness
{
  "user_ids": ["1", "2", "3"],
  "k": 5,
  "exclude_solved": true
}
```

---

## 📈 확장 가능성

이 시스템은 다음과 같이 확장할 수 있습니다:

1. **협업 필터링 추가**: 비슷한 사용자가 푼 문제 추천
2. **딥러닝 모델 적용**: 임베딩 기반 추천
3. **실시간 학습**: 사용자 피드백을 실시간 반영
4. **A/B 테스트**: 여러 전략의 효과 비교
5. **추천 이유 개선**: LLM을 활용한 자연스러운 설명 생성

---

## 🎯 프로젝트 요약

| 파일 | 역할 | 주요 내용 |
|------|------|-----------|
| **main.py** | API 서버 | FastAPI 서버, 9개 엔드포인트, 5개 추천기 초기화 |
| **schemas.py** | 데이터 구조 | Pydantic 모델, 요청/응답 형식 정의 |
| **recommender.py** | 추천 로직 | 5가지 추천 알고리즘, TF-IDF, 취약점 분석 |

**핵심 기능**:
- Week 1: 베이스라인 추천 (TF-IDF, Popularity, Random, Hybrid)
- Week 2: 취약점 기반 추천 (Weakness Analysis)

**기술 스택**:
- FastAPI (웹 프레임워크)
- Pydantic (데이터 검증)
- scikit-learn (TF-IDF, 코사인 유사도)
- pandas (데이터 처리)

이 시스템은 사용자의 문제 풀이 이력을 분석하여 다양한 방식으로 최적의 문제를 추천하며, 특히 Week 2의 취약점 기반 추천은 사용자의 약점을 보완하는 데 초점을 맞춘 혁신적인 기능입니다.
