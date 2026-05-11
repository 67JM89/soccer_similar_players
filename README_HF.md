---
title: 2026 WC Predictor
emoji: 🏆
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: mit
short_description: 데이터 기반 2026 FIFA 월드컵 예측 대시보드 (v3.5)
---

# 🏆 2026 FIFA World Cup Predictor

데이터 기반 2026 월드컵 매치 예측 시스템. v3.5 모델 (ELO + Recent Form + ClubElo 보정 + Dixon-Coles τ).

## 모델 + 데이터
- **학습**: 49,215건 국제 경기 (1872~2026-03)
- **선수**: 18,533명 FIFA 23 능력치 (12개 파생 스탯)
- **백테스트**: ~55% 정확도 (895경기, 2014~2024 메이저 토너먼트)

## 6개 탭
1. **Match Predictor** — 두 팀 매치업 → 승무패 + 예상 스코어 + 점수 분포
2. **Tournament Outlook** — 우승 시상대 + Top N + 단계별 진출 히트맵
3. **Group Stage** — 12개 조 진출 확률 시각화
4. **Player Cards** — FIFA / FM 스타일 선수 카드 (FIFA 23 / 2022-23 시즌)
5. **Bracket Simulator** — 단일 토너먼트 시뮬레이션
6. **About** — 모델 발전 history

## GitHub
- Repo: https://github.com/67JM89/soccer_similar_players
