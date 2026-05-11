# 2026 FIFA World Cup Predictor

데이터 기반 2026 월드컵 경기 예측 시스템. 49,000건의 국제경기 결과 + FIFA 선수 능력치 + ClubElo 데이터를 결합한 Poisson 회귀 모델.

**최종 모델 v3.5**: ELO + Recent Form + Club ELO 보정 + Dixon-Coles τ 보정
**백테스트 정확도**: ~55% (2014~2024 메이저 토너먼트 895경기)

---

## 🌐 라이브 대시보드 (Live Demo)

> 🎯 **인터랙티브 데모 바로 가기 →** **[🤗 Hugging Face Spaces에서 열기](https://huggingface.co/spaces/67JM89/wc2026-predictor)**

[![Open in HF Spaces](https://img.shields.io/badge/🤗%20Open%20in-Hugging%20Face%20Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces/67JM89/wc2026-predictor)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**대시보드 기능:**
- 📊 매치 예측 (승무패 확률 + 예상 스코어 + 점수 분포)
- 🏆 토너먼트 전망 (Top 시상대 + 단계별 진출 히트맵)
- 🌍 그룹 스테이지 (12개 조 진출 확률)
- 👤 선수 카드 (FIFA / FM 스타일)
- 🎲 브래킷 시뮬레이터 (단일 토너먼트 시각화)
- 🌐 **8개 언어 지원** — 한국어 / English / Español / Français / Italiano / Português / 日本語 / 中文
- 🎵 Spotify 미니 플레이어 (BGM)

---

## 🎯 예측 결과 미리보기

| 순위 | 팀 | 우승 확률 | 결승 | 4강 |
|------|-----|-----------|------|-----|
| 1 | 🇪🇸 Spain | 20.3% | 31.9% | 45.9% |
| 2 | 🇦🇷 Argentina | 17.9% | 28.4% | 42.3% |
| 3 | 🇫🇷 France | 14.7% | 24.2% | 38.1% |
| 4 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 9.3% | 17.7% | 31.2% |
| 5 | 🇳🇱 Netherlands | 6.6% | 13.1% | 25.1% |
| 6 | 🇩🇪 Germany | 4.7% | 10.4% | 21.1% |
| 7 | 🇧🇷 Brazil | 4.7% | 10.7% | 21.8% |

전체 48팀 결과는 Gradio UI 또는 `wc2026_predictions_v3` SQLite 테이블 참조.

---

## 🚀 사용법

대시보드는 위 [🤗 Hugging Face Spaces](https://huggingface.co/spaces/67JM89/wc2026-predictor) 링크에서 바로 사용 가능. 아래는 개발/재현 용도.

### Python 모듈로 사용
```python
from wc_predictor import Predictor

p = Predictor()
result = p.predict_match("Spain", "Brazil", neutral=True)
# {p_a_win, p_draw, p_b_win, expected_a_goals, expected_b_goals, most_likely_score, ...}
```

### Monte Carlo 토너먼트 시뮬레이션 재실행
```bash
python simulate_wc2026_v3.py
```

---

## 📁 프로젝트 구조

```
soccer_similar_players/
├── README.md                    # 이 파일
├── ROADMAP.md                   # 개발 로드맵 + 결과 정리
├── wc_predictor.py              # 핵심 예측 모듈 (v3.5)
├── gradio_app.py                # 인터랙티브 UI
│
├── 데이터 파이프라인:
│   ├── download_data.py         # 1회: Kaggle FIFA + intl results + Wikipedia
│   ├── build_database.py        # raw → SQLite + ELO 계산
│   ├── build_form_features.py   # 매치별 폼 피처
│   ├── build_match_features.py  # point-in-time ELO 피처
│   ├── build_team_features.py   # WC팀 능력치/폼 요약
│   ├── build_club_elo.py        # ClubElo 수집 + 클럽 ↔ 선수 매칭
│
├── 모델 + 검증:
│   ├── train_predict.py         # v1 (ELO만)
│   ├── train_predict_v2.py      # v2.9 (ELO + Form)
│   ├── test_form_robust.py      # 폼 피처 검증
│   ├── test_form_variants.py    # 폼 변형 비교
│   ├── test_importance_weights.py  # Phase 8: 매치 중요도 가중
│   ├── test_recency_decay.py    # Phase 9: 시간 가중치
│   ├── test_squad_strength.py   # Phase 10: FIFA 스쿼드
│   ├── test_host_continent.py   # Phase 11: 호스트/대륙
│   ├── test_dixon_coles.py      # Phase 13: DC τ 보정
│   ├── test_xgboost.py          # Phase 14: XGBoost
│   ├── test_bayesian.py         # Phase 15: Bayesian MAP
│
├── 시뮬레이션:
│   ├── simulate_wc2026.py       # v1 토너먼트 MC
│   ├── simulate_wc2026_v2.py    # v2.9 토너먼트 MC
│   ├── simulate_wc2026_v3.py    # v3.0 + 클럽 ELO 보정
│
└── data/
    ├── soccer.db                # 통합 SQLite (12 테이블)
    └── raw/                     # 원본 다운로드 (5.7GB)
```

---

## 🔬 모델 발전사

| 버전 | 피처 | 백테스트 acc | 채택 |
|------|------|--------------|------|
| v1.0 | ELO + home advantage | 53.5% | ✓ baseline |
| v2.9 | + form_gd_diff + form_cs_diff | **55.1%** | ✓ +1.6pp |
| v3.0 | + 클럽 ELO 휴리스틱 보정 | 검증 불가 | ✓ 직관 합리 |
| **v3.5** | + Dixon-Coles τ 보정 (ρ=-0.05) | 칼리브레이션 ↑ | ✓ 최종 |

## ✗ 실패한 시도 (음의 결과도 학습)

| Phase | 시도 | 결과 | 이유 |
|-------|------|------|------|
| 8 | 매치 중요도 가중 (친선 0.2) | 0pp | ELO K-factor가 이미 흡수 |
| 9 | Recency Decay (1~15년 반감기) | 0pp | ELO + Form이 이미 반영 |
| 10 | FIFA Squad Strength | -0.2pp | ELO와 강한 공선성 |
| 11 | Host/Continental 보정 | 0pp | 대부분 매치가 본국 개최 → 변동성 부족 |
| 14 | XGBoost | -0.5pp | 데이터 양 부족, 비선형 신호 없음 |
| 15 | Bayesian Hierarchical MAP | -2.1pp | 수렴 부족, ELO 누적 정보 손실 |
| 16 | Bookmaker odds | N/A | 모든 소스 anti-bot 차단 |

**핵심 인사이트:** ELO는 25년 누적 정보를 담은 압도적 단일 피처. 같은 정보를 다른 방식으로 재가공하는 시도는 모두 실패. **진짜 새로운 직교 정보**(폼, 클럽 ELO)만 유의미한 개선.

---

## 📊 데이터 소스

| 소스 | 용도 | 규모 |
|------|------|------|
| Kaggle `martj42/international-football-results-from-1872-to-2017` | 국제 경기 결과 49,215건 | 3.5 MB |
| Kaggle `stefanoleone992/fifa-23-complete-player-dataset` | FIFA 23 능력치 | 86 MB |
| Wikipedia `2026_FIFA_World_Cup` | 조 편성 + 일정 | 스크랩 |
| ClubElo (`soccerdata` 라이브러리) | 630개 클럽 ELO | 실시간 |

---

## 🧪 백테스트 방법론

- **학습 셋**: 2000-01-01 ~ 각 테스트 연도 이전까지의 모든 국제경기
- **테스트 셋**: 각 연도의 메이저 토너먼트 (FIFA WC, EURO, Copa América, AFCON, Asian Cup) 경기 단독
- **메트릭**: Accuracy (W/D/L), Brier score, log-loss, 무승부 예측 칼리브레이션
- **6년 walk-forward**: 2014, 2016, 2018, 2020, 2022, 2024
- **표준 오차**: 895경기 풀에서 약 ±2.8pp

---

## 🎓 사용 모델/기법

- **Poisson 회귀** (`sklearn.linear_model.PoissonRegressor`) — 기대 득점 모델링
- **Dixon-Coles τ 보정** — 저득점 결과 (0-0, 1-0, 0-1, 1-1) 칼리브레이션
- **World Football ELO** (자체 구현) — K-factor: WC=60, 컨티넨탈=50, 예선=40, 친선=20
- **포아송 곱 (joint distribution)** — 점수 분포 계산
- **Monte Carlo 시뮬레이션** — 토너먼트 진행 모델링 (N=10,000)
- **Fuzzy matching** (`fuzzywuzzy`) — 클럽/팀 이름 통합

---

## 💡 한계 + 다음 단계

이 프로젝트는 **B 옵션 (학습용 ML 프로젝트)** 완성. A 옵션 (진지한 예측 시스템) 으로 가려면:

1. **북메이커 배당률 데이터** — 가장 강력한 단일 피처, 유료 API 필요 (The Odds API 등)
2. **xG 데이터** — Understat/FBref에서 클럽 매치 단위 xG, 선수 매치 단위 xG
3. **라인업/부상 데이터** — Transfermarkt 등에서 매치 직전 선수 가용성
4. **딥러닝 모델** — LSTM/Transformer로 시퀀스 패턴 학습 (단, 데이터 양 보강 필수)

---

*2026-05-10 기준, FIFA WC 시작 1개월 전.*
