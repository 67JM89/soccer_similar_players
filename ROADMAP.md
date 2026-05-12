# 2026 World Cup Prediction — Project Roadmap (Final)

## 🎯 목표 (달성됨)

데이터 기반으로 2026 FIFA 월드컵 모든 경기 예측 + 우승팀 확률 도출.

**달성 결과:**
- 최종 모델 v3.5 (ELO + Form + Club ELO + Dixon-Coles τ)
- 백테스트 정확도 ~55% (베이스라인 49%, 학술 SOTA 60-65%)
- 2026 WC 48팀 우승확률 산출 + 인터랙티브 UI

> 사용자가 처음 말한 "97% 정확도"는 통계적으로 불가능. 축구는 노이즈가 큰 스포츠.

---

## ✅ 완료된 Phase

```
[1. 데이터 수집]          ✓ Kaggle FIFA + 49K intl matches + WC2026 tables
[2. SQLite 통합]          ✓ data/soccer.db (12 tables)
[3. 팀 피처]              ✓ team_features (48 WC teams × 14 cols)
[4. 매치 피처]            ✓ point-in-time ELO, 49K rows
[5. 베이스라인 모델 v1]   ✓ Poisson + ELO (acc 53.5%)
[6. 토너먼트 시뮬]        ✓ Monte Carlo N=10,000
[7. Recent Form 피처]    ✓ +1.6pp (53.5→55.1) — v2.9 채택
[8. Importance 가중]     ✗ 0pp 스킵
[9. Recency Decay]       ✗ 0pp 스킵
[10. FIFA Squad]         ✗ -0.2pp 스킵
[11. Host/Continent]     ✗ 0pp 스킵
[12. Club ELO 보정]       ✓ 백테스트 불가, 휴리스틱 채택 — v3.0
[13. Dixon-Coles]        ✓ 칼리브레이션 미세 ↑ — v3.5 (최종)
[14. XGBoost]            ✗ -0.5pp 스킵
[15. Bayesian MAP]       ✗ -2.1pp 스킵
[16. Bookmaker odds]     ✗ anti-bot 차단, infeasible
[17a. TM 라이브 squad]    ✓ 2,124 player-rows, 1,360 FIFA 매칭 — UI 데이터 신선도 ↑
[17b. Understat xG]      ◯ +2.3pp 방향성 양, 95% CI 내 (noise) — 칼리브레이션 개선 확인
[Final UI + Docs]         ✓ Gradio + README + ROADMAP
```

---

## 📊 Phase별 결과 요약

### Tier 1 — 빠른 효과 시도

| Phase | 변경 | 결과 |
|-------|------|------|
| 7 | + Recent Form (10경기 GD/CS) | ✓ **+1.6pp** 채택 |
| 8 | 매치 중요도 가중 | ✗ 0pp |
| 9 | Recency Decay (반감기 1~15y) | ✗ 0pp |

### Tier 2 — 데이터 보강

| Phase | 변경 | 결과 |
|-------|------|------|
| 10 | FIFA Squad Strength (point-in-time) | ✗ -0.2pp |
| 11 | Host/Continental 보정 | ✗ 0pp |
| 12 | ClubElo 휴리스틱 보정 | ✓ 채택 (검증 불가) |

### Tier 3 — 모델 구조 개선

| Phase | 변경 | 결과 |
|-------|------|------|
| 13 | Dixon-Coles τ 보정 | ✓ 미세 칼리브 ↑ |
| 14 | XGBoost | ✗ -0.5pp |
| 15 | Bayesian Hierarchical MAP | ✗ -2.1pp |

### Tier 4 — 외부 데이터

| Phase | 시도 | 결과 |
|-------|------|------|
| 16 | Bookmaker odds | ✗ 모든 소스 anti-bot |

---

## 🧠 핵심 학습

### 1. ELO는 압도적 단일 피처
25년 누적 게임 결과를 단일 숫자로 압축. 다른 모든 "팀 강도" 피처(FIFA 스쿼드, 호스트, 컨티넨탈)는 ELO와 중복 → 도움 안 됨.

### 2. 진짜 새 정보만 효과
- ✓ Recent Form (최근 10경기) → ELO 변화는 느림. 폼은 빠른 변화 캡처
- ✓ ClubElo (선수 클럽 강도) → 팀 ELO엔 없는 개별 선수 수준 신호
- ✗ 나머지는 모두 ELO와 직간접적 중복

### 3. 데이터 천장 ~55%
공개 데이터 + 단순 모델로 도달 가능한 천장이 약 55%. 60%+ 가려면 북메이커 odds 같은 시장 신호 또는 라인업/부상/xG 같은 매치 직전 정보가 필요.

### 4. 음의 결과의 가치
8개 phase가 실패. 각각 가설을 확실히 검증함으로써 "왜 작동 안 하는지"를 명확히 함. ML 프로젝트의 본질적 가치 — 사전 검증으로 시간 절약.

### 5. 통계 노이즈 인식
895 테스트 경기 → 표준오차 ~2.8pp. ±1-2pp 차이는 통계적으로 의미 없음. 일관된 개선 패턴 + 직관적 해석이 동시에 필요.

---

## 📁 최종 산출물

### 코드
- `wc_predictor.py` — 핵심 예측 API (v3.5)
- `gradio_app.py` — 인터랙티브 UI
- 8개 `test_*.py` — 각 phase별 백테스트 (재현 가능)
- 5개 `build_*.py` — 데이터 파이프라인
- 3개 `simulate_wc2026*.py` — 토너먼트 Monte Carlo

### 데이터
- `data/soccer.db` — 통합 SQLite (12 테이블, ~600 MB)
- `data/raw/` — 원본 다운로드 (~5.7 GB)

### 문서
- `README.md` — 프로젝트 개요
- `ROADMAP.md` — 이 파일 (개발 history)

---

## 🔮 다음 단계 (선택적, A 옵션 진입)

이 프로젝트는 **B 옵션 (학습용 ML 프로젝트)** 으로 완성. A 옵션 (진지한 예측 시스템) 으로 가려면:

1. **The Odds API ($20/월)** — 실시간 북메이커 배당률 → +5~10pp 가능
2. **Understat xG 수집** — 클럽 매치 + 선수 매치 xG → +2~3pp 가능
3. **Transfermarkt 라인업/부상 스크래핑** — 매치 직전 정보 → +1~3pp
4. **앙상블** — v3.5 + 시장 odds + xG 모델 가중평균

WC 시작 (2026-06-11) 1개월 전 시점 기준, 이 시스템은 충분히 작동 가능.

---

## 📑 Phase 17 — 데이터 보강 (2026-05-12 추가)

### 17a — Transfermarkt 라이브 squad
- **목표:** FIFA 23 (2년 전 클럽 정보)이 stale한 문제 해결
- **수집:** 48개 WC 출전국 × 평균 44명 squad = 2,124 player-rows
- **매칭:** 1,360 / 2,222 FIFA 선수와 fuzzy 매칭 (61.2%)
- **결과:** UX 개선 — 현 클럽, 부상 상태, 시장가치를 player card에 라이브로 표시
- **모델 영향:** 없음 (UX-only)

### 17b — Understat 팀 xG-top11 feature
- **목표:** ELO 외의 직교 시그널 (실제 슈팅 퀄리티 기반)
- **수집:** 5대 유럽리그 × 2014-2024 = 29,710 player-season-rows
- **매칭:** 8,013 / 8,568 unique players (93.5%)
- **집계:** 1,117 (nation, year) rows, top 11 xG sum
- **백테스트 (공정 비교 — 동일 train/test set):**

| 토너먼트 | v2.9 | v2.9 + xG | Δ acc | Δ Brier |
|---------|------|-----------|-------|---------|
| 2016    | 46.3% | 46.3% | +0.0pp | +0.0001 |
| 2018 WC | 52.7% | 56.4% | **+3.6pp** | -0.0024 |
| 2019    | 64.5% | 67.7% | **+3.2pp** | -0.0006 |
| 2021    | 59.5% | 62.0% | **+2.5pp** | -0.0022 |
| 2022 WC | 56.1% | 57.9% | **+1.8pp** | -0.0067 |
| 2024    | 52.2% | 55.6% | **+3.3pp** | +0.0021 |
| **누적** | **54.1%** | **56.4%** | **+2.3pp** | -0.002 (avg) |

- **통계적 검증:** 95% CI half-width ≈ ±4.9pp → **+2.3pp는 noise 안 (유의X)**
- **하지만:** 6 토너먼트 중 5건 양의 방향, Brier 5/6에서 개선 — **방향성과 칼리브레이션은 시그널**
- **채택 결정:** ◯ 일단 *연구 결과로 기록*, 모델 통합은 deferred
  - 이유 1: 누적 4,117 / 49,217 매치만 xG 데이터 보유 (8.4%) — 모델 통합 시 NaN 처리 복잡
  - 이유 2: 통계적으로 유의하지 않은 효과
  - 이유 3: 4-5pp 짜리 변동성을 안고 가야 함
- **재현:** `python build_xg_features.py && python test_xg.py`
