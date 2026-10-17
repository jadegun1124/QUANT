# QUANT — Upbit BTC/KRW Neural-Network Trading Bot (draft) / 업비트 BTC/KRW 신경망 자동매매 봇 (draft)

> ⚠️ **Status: experimental draft.** / **현재 상태: 실험용 초안.**
> The model weights and scaler committed here look like untrained placeholders, so the bot cannot produce meaningful signals as-is. See [Why no trades happened](#why-no-trades-happened--매매가-한-번도-안-된-이유) and [Improvements](#improvements--개선할-점).
> 이 저장소의 모델 가중치와 스케일러는 학습되지 않은 placeholder로 보여서, 지금 상태로는 의미 있는 신호가 나오지 않습니다. [매매가 한 번도 안 된 이유](#why-no-trades-happened--매매가-한-번도-안-된-이유)와 [개선할 점](#improvements--개선할-점)을 먼저 확인하세요.

---

## 1. Overview / 개요

**EN** — A live trading bot for the **BTC/KRW** market on Upbit. Every 5 minutes it reads hourly candles, builds 7 numeric features, feeds them to a small PyTorch neural network, and turns the output score into a **BUY / SELL / HOLD** signal that is sent to the exchange as an order.

**KO** — 업비트(Upbit) **BTC/KRW** 시장에서 동작하는 실시간 자동매매 봇입니다. 5분마다 1시간봉을 읽어 7개의 수치 피처를 만들고, 작은 PyTorch 신경망에 넣어 나온 점수를 **매수 / 매도 / 관망** 신호로 바꿔 주문을 실행합니다.

---

## 2. Pipeline / 전체 흐름

```
[Upbit API]  50 hourly candles (OHLCV)  /  1시간봉 50개
     │
     ▼
 7 features / 피처 7개
 (log return, volatility, MA diff, volume change, high-low spread, close, volume)
     │
     ▼
 StandardScaler  (quant_scaler.pkl)
     │
     ▼
 Neural net / 신경망:  7 → 16 → ReLU → Dropout(0.3) → 1   (quant_model.pth)
     │   output: one real-valued score / 출력: 실수 점수 1개
     ▼
 score > +0.005 → BUY │ score < -0.005 → SELL │ otherwise → HOLD
     │
     ▼
 place order → sleep 300 s → repeat  /  주문 실행 → 300초 대기 → 반복
```

---

## 3. Algorithm Details / 알고리즘 상세

### 3.1 Data collection / 데이터 수집

**EN** — Uses `ccxt` to call `fetch_ohlcv('BTC/KRW', timeframe='1h', limit=50)` and loads the latest 50 hourly candles (open, high, low, close, volume) into a DataFrame.

**KO** — `ccxt`로 `fetch_ohlcv('BTC/KRW', timeframe='1h', limit=50)`을 호출해 최근 1시간봉 50개(open, high, low, close, volume)를 DataFrame으로 불러옵니다.

### 3.2 Features (7) / 피처 (7개)

| # | Feature | Formula / 계산식 | Meaning (EN) | 의미 (KO) |
|---|---------|------------------|--------------|-----------|
| 1 | `log_return` | ln(close_t / close_{t-1}) | Log return vs. previous candle | 직전 봉 대비 로그 수익률 |
| 2 | `volatility` | 10-bar rolling std of `log_return` | Recent volatility | 최근 변동성 |
| 3 | `ma_diff` | MA(5) − MA(20) of close | Short vs. long trend gap (in KRW) | 단기·장기 추세 차이 (원화 단위) |
| 4 | `vol_change` | `pct_change` of volume | Volume growth rate | 거래량 증감률 |
| 5 | `hl_spread` | (high − low) / close | Intra-candle range | 봉 내 변동 폭 |
| 6 | `close` | raw close price | Price level | 가격 수준 |
| 7 | `volume` | raw volume | Trading size | 거래 규모 |

**EN** — Rows with NaN from rolling windows are dropped, and only the **last row** is used as the model input.

**KO** — 롤링 계산으로 생기는 NaN 행은 `dropna()`로 제거하고, **마지막 행 하나**만 모델 입력으로 사용합니다.

### 3.3 Scaling / 정규화

**EN** — `quant_scaler.pkl` (scikit-learn `StandardScaler`) applies `(x − mean) / scale`, then the result is converted to a `float32` tensor.

**KO** — `quant_scaler.pkl`(scikit-learn `StandardScaler`)로 `(x − mean) / scale` 변환 후 `float32` 텐서로 바꿉니다.

### 3.4 Model (`CryptoPredictorNN`) / 모델

```
Linear(7 → 16) → ReLU → Dropout(p=0.3) → Linear(16 → 1)
```

**EN** — A one-hidden-layer MLP regressor. The output is meant to be read as an expected-return-like score; the exact training target cannot be confirmed because no training code is in the repo. Inference runs with `model.eval()` and `torch.no_grad()` (Dropout off).

**KO** — 은닉층 1개짜리 MLP 회귀 모델입니다. 출력은 기대 수익률 비슷한 점수로 해석하도록 설계된 것으로 보이지만, 레포에 학습 코드가 없어 정확한 타깃은 확인할 수 없습니다. 추론은 `model.eval()` + `torch.no_grad()`(Dropout 꺼짐)로 수행합니다.

### 3.5 Signal rule (threshold ±0.005) / 신호 규칙

| Model output / 모델 출력 | Signal / 신호 |
|--------------------------|---------------|
| `> 0.005` | **BUY** (+1) / 매수 |
| `< -0.005` | **SELL** (−1) / 매도 |
| otherwise / 그 외 | **HOLD** (0) / 관망 |

### 3.6 Order execution (`execute_trade`) / 주문 실행

**EN**
- **BUY**: take the best ask (`asks[0]`) from the order book, compute the BTC amount for `TRADE_AMOUNT_KRW` (default 6,000 KRW), and place a **limit buy at that price**.
- **SELL**: fetch the free BTC balance and, **only if its value exceeds 5,000 KRW**, **market-sell the entire balance** (matches Upbit's minimum order value).
- **HOLD**: do nothing.

**KO**
- **BUY**: 호가창 최우선 매도호가(`asks[0]`)를 가져와 `TRADE_AMOUNT_KRW`(기본 6,000원)에 해당하는 BTC 수량을 계산한 뒤 **그 가격에 지정가 매수**합니다.
- **SELL**: BTC 가용 잔고를 조회해 **평가금액이 5,000원을 넘을 때만** 보유 BTC **전량을 시장가 매도**합니다(업비트 최소 주문 금액 대응).
- **HOLD**: 아무 것도 하지 않습니다.

### 3.7 Main loop / 메인 루프

**EN** — `while True` loop that repeats every **300 s (5 min)**. `KeyboardInterrupt` stops the bot cleanly; any other exception is printed and the bot waits 60 s before retrying.

**KO** — `while True` 루프가 **300초(5분)** 마다 반복됩니다. `KeyboardInterrupt`면 정상 종료하고, 그 외 예외는 오류를 출력한 뒤 60초 쉬고 재시도합니다.

### 3.8 Model-file auto-repair (`auto_fix_model_files`) / 모델 파일 자동 복구

**EN** — On startup, if `quant_model.pth` or `quant_scaler.pkl` is missing or fails to load, both are deleted and replaced with a **randomly initialized model and a dummy scaler**. `KMP_DUPLICATE_LIB_OK=True` is also set to bypass OpenMP Error #15.

**KO** — 시작할 때 `quant_model.pth` 또는 `quant_scaler.pkl`이 없거나 로드에 실패하면 두 파일을 지우고 **무작위 초기화 모델과 더미 스케일러**로 다시 만듭니다. OpenMP 중복 로드 오류(OMP Error #15)를 피하려고 `KMP_DUPLICATE_LIB_OK=True`도 설정합니다.

---

## 4. Files / 파일 구성

| File / 파일 | Description / 설명 |
|-------------|--------------------|
| `live_bot.py` | Live engine: data → features → inference → order → loop / 실시간 실행 엔진 |
| `quant_model.pth` | PyTorch `state_dict` of the network / 신경망 가중치 |
| `quant_scaler.pkl` | scikit-learn `StandardScaler` saved with joblib / 정규화 스케일러 |
| `test_trade.ipynb` | Notebook to test API auth and a small limit buy / API 인증 및 소액 지정가 매수 테스트 노트북 |

---

## 5. Install & Run / 설치 및 실행

```bash
pip install ccxt torch numpy pandas joblib scikit-learn
```

**EN**
1. **Do not hard-code API keys.** Read them from environment variables (change `UPBIT_ACCESS` / `UPBIT_SECRET` in `live_bot.py` to use `os.environ[...]`).
   ```bash
   export UPBIT_ACCESS="your_access_key"
   export UPBIT_SECRET="your_secret_key"
   ```
2. Verify authentication and the order flow with a small order in `test_trade.ipynb`.
3. Run `python live_bot.py`.

**KO**
1. **API 키를 코드에 직접 쓰지 마세요.** 환경변수로 읽도록 `live_bot.py`의 `UPBIT_ACCESS` / `UPBIT_SECRET`을 `os.environ[...]`으로 바꾸세요.
2. `test_trade.ipynb`에서 소액 주문으로 인증과 주문 흐름을 먼저 확인하세요.
3. `python live_bot.py`로 실행합니다.

Settings at the top of `live_bot.py` / 설정 값: `SYMBOL = 'BTC/KRW'`, `TRADE_AMOUNT_KRW = 6000.0`, `SLEEP_INTERVAL = 300`

---

## 6. Known Limitations / 알려진 한계

**1. The model and scaler appear untrained / 모델과 스케일러가 학습되지 않은 상태로 보임**
- **EN** — `quant_scaler.pkl` has mean = 0.5, scale = 0.5 and 2 training samples, identical to the dummy `fit([[0]*7, [1]*7])` built by `auto_fix_model_files()`. The weights of `quant_model.pth` lie exactly within PyTorch's default `nn.Linear` initialization ranges (±1/√7 ≈ 0.378 and ±0.25).
- **KO** — `quant_scaler.pkl`은 mean 0.5, scale 0.5, 샘플 수 2로, `auto_fix_model_files()`가 만드는 더미 스케일러와 같습니다. `quant_model.pth`의 가중치는 PyTorch `nn.Linear` 기본 초기화 범위(±0.378, ±0.25) 안에 그대로 들어 있습니다.

**2. No training / validation code / 학습·검증 코드 없음**
- **EN** — Data collection, labeling, training and backtesting are not in the repo, so performance cannot be verified.
- **KO** — 데이터 수집, 라벨링, 학습, 백테스트 코드가 없어 성능을 검증할 수 없습니다.

**3. Feature design / 피처 설계**
- **EN** — `close`, `volume` and `ma_diff` depend on the price level (non-stationary), so even a properly fitted scaler drifts out of distribution over time. The last candle is still **in progress**, so its values keep changing. Hourly signals are recomputed every 5 minutes, so the same candle can trigger repeated signals.
- **KO** — `close`, `volume`, `ma_diff`는 가격 수준에 의존(비정상)해서 스케일러를 제대로 학습시켜도 시간이 지나면 분포가 어긋납니다. 마지막 캔들은 아직 **진행 중**이라 값이 계속 변하고, 1시간봉 신호를 5분마다 다시 계산하므로 같은 봉에서 신호가 반복될 수 있습니다.

**4. Orders & risk management / 주문·리스크 관리**
- **EN** — The log says "market order" but BUY is actually a limit order, so it may never fill. Consecutive BUY signals keep stacking positions every 5 minutes, SELL always exits everything, and there is no stop-loss, take-profit, position cap, cooldown, or fee/slippage handling. If model files are corrupted, the bot silently overwrites them with a random model and keeps trading live.
- **KO** — 로그에는 "market order"라고 나오지만 실제 매수는 지정가라 체결되지 않을 수 있습니다. BUY가 연속되면 5분마다 포지션이 쌓이고 SELL은 항상 전량 청산이며, 손절·익절·최대 포지션·쿨다운·수수료/슬리피지 고려가 없습니다. 모델 파일이 손상되면 조용히 무작위 모델로 덮어쓰고 실거래를 계속합니다.

**5. Security / 보안**
- **EN** — Never commit API keys. A key that was ever pushed to a public repository must be **revoked and re-issued**; deleting it from the file is not enough. `torch.load(..., weights_only=False)` is pickle-based and unsafe for untrusted files; prefer `weights_only=True`.
- **KO** — API 키를 커밋하지 마세요. 한 번이라도 공개 저장소에 올라간 키는 파일에서 지우는 것만으로 부족하고 **폐기 후 재발급**해야 합니다. `torch.load(..., weights_only=False)`는 pickle 기반이라 신뢰할 수 없는 파일에는 위험하므로 `weights_only=True`를 권장합니다.

---

## Why no trades happened / 매매가 한 번도 안 된 이유

**EN** — Reading the code path, the most likely cause is a combination of two things. Which one you hit can be told from the log lines (see the table).

**KO** — 코드 경로를 따라가 보면, 가장 유력한 원인은 두 가지가 겹친 경우입니다. 어떤 경우인지는 로그 문구로 구분할 수 있습니다(아래 표).

### Cause A — The bot only ever produces SELL, and you hold no BTC / 원인 A — 항상 SELL만 나오는데 보유 BTC가 없음 (가장 유력)

1. **EN** — The committed scaler is the dummy one (`mean=0.5, scale=0.5`). Real inputs such as `close ≈ 100,000,000 KRW` become about `2×10⁸` after scaling, which pushes the untrained network's output to a huge negative number (about **−1.1×10⁷** in a simulation with synthetic BTC-like features).
   **KO** — 레포의 스케일러는 더미(`mean=0.5, scale=0.5`)입니다. `close ≈ 1억 원` 같은 실제 입력은 스케일링 후 약 `2×10⁸`이 되어, 학습되지 않은 신경망의 출력이 매우 큰 음수(비트코인과 비슷한 가상 피처로 시뮬레이션했을 때 약 **−1.1×10⁷**)로 쏠립니다.
2. **EN** — That is far below `-0.005`, so `signal = -1` (**SELL**) on every loop. BUY is never reached.
   **KO** — `-0.005`보다 훨씬 작아서 매 루프 `signal = -1`(**SELL**)이 되고, BUY 경로는 한 번도 실행되지 않습니다.
3. **EN** — The SELL branch only sells if `btc_balance × price > 5000`. Because the bot never buys, your BTC balance stays 0, so it prints `Insufficient BTC balance to execute sell` and does nothing. **No order is ever sent.**
   **KO** — SELL 분기는 `btc_balance × price > 5000`일 때만 매도합니다. 봇이 매수를 한 번도 하지 않으니 BTC 잔고가 0이라 `Insufficient BTC balance to execute sell`만 출력하고 아무 주문도 내지 않습니다. **주문 자체가 한 번도 나가지 않습니다.**

### Cause B — Authentication / permission failure / 원인 B — 인증·권한 문제

**EN** — Fetching candles and the order book is public, so those calls work without valid keys, but `fetch_balance` and order placement need authentication. `test_trade.ipynb` in the repo itself ended with `invalid_access_key`. Also check these on Upbit's API key page:
- the key is still active (not revoked or expired),
- the **IP address of the machine running the bot is on the key's allowed-IP list** (otherwise Upbit rejects requests, e.g. `no_authorization_ip`),
- the key has the **asset lookup** and **place order** permissions enabled.

**KO** — 캔들과 호가창 조회는 공개 API라 키가 틀려도 동작하지만, `fetch_balance`와 주문은 인증이 필요합니다. 레포의 `test_trade.ipynb`도 `invalid_access_key`로 끝났습니다. 업비트 API 키 관리 페이지에서 다음을 확인하세요.
- 키가 폐기·만료되지 않았는지,
- **봇을 실행하는 컴퓨터의 IP가 허용 IP 목록에 등록**되어 있는지(아니면 `no_authorization_ip` 등으로 거부됨),
- **자산조회**와 **주문하기** 권한이 켜져 있는지.

### How to tell which / 구분하는 방법

| Log line / 로그 | Meaning / 의미 |
|-----------------|----------------|
| `NN Output Score:` is a huge negative number such as `-11000000.0000` / 점수가 `-11000000.0000`처럼 매우 큰 음수 | Cause A (untrained model + dummy scaler) / 원인 A |
| `SIGNAL: SELL` then `Insufficient BTC balance to execute sell` | Cause A, no BTC held / 원인 A (BTC 미보유) |
| `API EXECUTION ERROR: ... invalid_access_key` or `no_authorization_ip` | Cause B (key / IP / permission) / 원인 B |
| `SIGNAL: HOLD` forever / 계속 HOLD | Score stays inside ±0.005 — see Improvement 4 / 개선 4번 참고 |

---

## Improvements / 개선할 점

**1. Train a real model and a real scaler / 실제 모델과 스케일러 학습**
- **EN** — Collect historical OHLCV, define a label (e.g. return N candles ahead), split train/validation by time, fit the `StandardScaler` **on the training set only**, and save both artifacts. Never ship the random-init files.
- **KO** — 과거 OHLCV를 모으고 라벨(예: N봉 뒤 수익률)을 정의한 뒤 시간순으로 학습/검증을 나누세요. `StandardScaler`는 **학습 데이터로만 `fit`** 하고 두 파일을 함께 저장합니다. 무작위 초기화 파일은 배포하지 마세요.

**2. Fail fast instead of auto-repairing / 자동 복구 대신 즉시 중단**
- **EN** — Remove the random-model fallback in `auto_fix_model_files()`. If the files are missing or invalid, raise an error and stop. Also add a sanity check at startup: if the scaler has `n_samples_seen_ < 100` or the model output on a typical feature vector is outside a plausible range (e.g. `|score| > 1`), refuse to trade.
- **KO** — `auto_fix_model_files()`의 무작위 모델 대체를 없애고, 파일이 없거나 잘못되면 에러를 내고 멈추세요. 시작 시 점검도 추가합니다. 스케일러의 `n_samples_seen_`가 너무 작거나(예: 100 미만) 일반적인 피처에서 모델 출력이 비현실적(예: `|score| > 1`)이면 거래를 거부하세요.

**3. Validate API keys at startup / 시작 시 API 키 검증**
- **EN** — Call `exchange.fetch_balance()` once before the loop; if it fails, print the exact exchange error (`invalid_access_key`, `no_authorization_ip`, permission errors) and exit. Register the machine's IP in Upbit's allowed-IP list and enable asset-lookup and place-order permissions.
- **KO** — 루프 전에 `exchange.fetch_balance()`를 한 번 호출해서 실패하면 거래소가 준 에러(`invalid_access_key`, `no_authorization_ip`, 권한 오류)를 그대로 출력하고 종료하세요. 업비트에서 실행 PC의 IP를 허용 목록에 등록하고 자산조회·주문하기 권한을 켜야 합니다.

**4. Calibrate thresholds from the score distribution / 점수 분포로 임계값 보정**
- **EN** — Instead of a fixed ±0.005, choose thresholds from the validation-set output distribution (e.g. top/bottom quantiles) and account for fees (Upbit's spot fee is about 0.05% per side) so signals are neither always on nor always off.
- **KO** — 고정값 ±0.005 대신 검증 데이터의 출력 분포(예: 상·하위 분위수)로 임계값을 정하고 수수료(업비트 현물 편도 약 0.05%)를 반영해서, 신호가 항상 켜지거나 항상 꺼지지 않게 하세요.

**5. Make the strategy position-aware (long-only) / 포지션을 고려한 전략**
- **EN** — Track whether you hold BTC. Treat BUY as "enter if flat" and SELL as "exit if long"; log "no position, ignoring SELL" explicitly so a silent no-op is easy to spot. Add a max position size and a cooldown after each order.
- **KO** — BTC 보유 여부를 추적하세요. BUY는 "무포지션일 때 진입", SELL은 "보유 중일 때 청산"으로 처리하고, 포지션이 없어서 SELL을 무시했다면 그 사실을 로그에 분명히 남겨서 조용한 무동작을 알아채기 쉽게 하세요. 최대 포지션 크기와 주문 후 쿨다운도 추가합니다.

**6. Use closed candles and one decision per candle / 마감된 캔들, 봉당 1회 판단**
- **EN** — Use `df.iloc[-2]` (the last closed candle) and only act when a new candle has closed. Replace raw `close`, `volume` and `ma_diff` with scale-free features (returns, ratios such as `close / MA20 − 1`, volume relative to its rolling mean).
- **KO** — 마지막으로 마감된 캔들(`df.iloc[-2]`)을 사용하고 새 봉이 마감됐을 때만 판단하세요. `close`, `volume`, `ma_diff` 원본 대신 수익률, `close / MA20 − 1` 같은 비율, 거래량 대비 이동평균 비율처럼 단위에 의존하지 않는 피처로 바꾸세요.

**7. Fix order handling / 주문 처리 개선**
- **EN** — Make the log message match the order type, and either use a market order (for KRW market buys with ccxt on Upbit, pass the KRW cost as required by the exchange) or cancel/replace unfilled limit orders after a timeout. Check order status after placing it and log fills.
- **KO** — 로그 문구를 실제 주문 유형과 맞추고, 시장가 주문을 쓰거나(업비트 KRW 시장가 매수는 거래소 규격에 맞게 KRW 금액으로 지정) 미체결 지정가 주문은 일정 시간 뒤 취소·재주문하세요. 주문 후 상태를 조회해 체결 여부를 로그로 남기세요.

**8. Add a dry-run mode and a backtest / 드라이런 모드와 백테스트**
- **EN** — Add a `DRY_RUN=True` flag that logs would-be orders without sending them, and backtest the same feature/signal code on historical data (with fees and slippage) before going live. Then paper-trade, then trade with a small amount.
- **KO** — 주문을 보내지 않고 로그만 남기는 `DRY_RUN=True` 옵션을 추가하고, 실거래 전에 같은 피처·신호 코드를 과거 데이터(수수료·슬리피지 포함)로 백테스트하세요. 그다음 페이퍼 트레이딩, 소액 실거래 순서로 진행하세요.

**9. Secrets & logging / 비밀 값과 로깅**
- **EN** — Keep keys in environment variables or a `.env` file listed in `.gitignore`; revoke and re-issue any key that was committed. Write logs (score, signal, balance, order result, errors) to a file with timestamps so failures like "no trades" can be diagnosed afterwards.
- **KO** — 키는 환경변수나 `.gitignore`에 넣은 `.env`로 관리하고, 커밋된 적 있는 키는 폐기 후 재발급하세요. 점수·신호·잔고·주문 결과·에러를 타임스탬프와 함께 파일로 기록해서 "매매가 안 된다" 같은 문제를 나중에도 추적할 수 있게 하세요.

---

> **EN** — This project is research/learning code, not investment advice. You are responsible for any trading losses.
> **KO** — 이 프로젝트는 연구·학습용 코드이며 투자 조언이 아닙니다. 실거래 손실에 대한 책임은 사용자에게 있습니다.
