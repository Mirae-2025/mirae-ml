# Week 3: 스피드런 초보 모드 구현 



**백준 문제 풀이 게임화**: 정각/30분 시작 타임 어택 모드
- 난이도: 브론즈/실버
- 모드: 30분(5문제) / 60분(10문제)
- 점수: 기본점수 + 시간보너스


---

## 시스템 구조

```
사용자 → FastAPI → SpeedrunManager → speedrun_sets.json
                      ├─ 세션 관리 (메모리/Redis)
                      ├─ 점수 계산
                      └─ 리더보드
```

---

## 핵심 구현

### 1. 문제 세트 (speedrun_sets.json)
| 세트 | 난이도 | 모드 | 문제 수 |
|------|--------|------|---------|
| bronze_30min | 브론즈 | 30분 | 5문제 |
| silver_30min | 실버 | 30분 | 5문제 |
| bronze_60min | 브론즈 | 60분 | 10문제 |

**예시**: `[1000, 1001, 10998, 1008, 2557]` (A+B, 사칙연산 등)

---

### 2. 점수 계산 공식
```python
기본점수 = 푼 문제 × 100
시간보너스 = (남은시간/전체시간) × 50 × 푼 문제
최종점수 = 기본점수 + 시간보너스
```

**예시**: 30분 모드, 3문제 풀이, 10분 남음
→ 300 + 50 = **350점**

---

### 3. 6개 API 엔드포인트
```
POST /speedrun/create          # 세션 시작 (정각/30분만 가능)
GET  /speedrun/session/{id}    # 진행 상황
POST /speedrun/submit          # 문제 제출 (AC만 인정)
GET  /speedrun/result/{id}     # 최종 결과
GET  /speedrun/leaderboard     # 상위 100명
GET  /speedrun/active/{uid}    # 활성 세션 확인
```

---

## 사용 시나리오

```
[10:00] 스피드런 시작 → 5문제 할당
[10:05] 1번 AC → 108점
[10:12] 2번 AC → 214점
[10:25] 5번 완료 → 425점 (3위)
```

---

## 기술 스택

- **서버**: FastAPI (Python)
- **저장소**: 메모리 → Redis 마이그레이션 예정
- **인증**: JWT (백엔드 연동)
- **성능**: 세션생성 10ms, 리더보드 5ms

---

## 테스트

**Swagger UI**: `http://localhost:8000/docs`

```
1. POST /speedrun/create (세션 생성)
2. POST /speedrun/submit (문제 제출)
3. GET /speedrun/leaderboard (순위 확인)
```

---


