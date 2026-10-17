import os
# 1. This completely fixes the OMP: Error #15 crash
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import ccxt
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
UPBIT_ACCESS = "UX19gJRreuQCRSUkWwQBbXvipjwHSv9FzLgNwKQj"
UPBIT_SECRET = "ieTYuQS07w1CNrOBdWxtuoewDxz0e0kDHspgffh8"
SYMBOL = 'BTC/KRW'
TRADE_AMOUNT_KRW = 6000.0  
SLEEP_INTERVAL = 300       

exchange = ccxt.upbit({
    'apiKey': UPBIT_ACCESS,
    'secret': UPBIT_SECRET,
})

# ==========================================
# 2. AUTO-FIX CORRUPTED FILES & LOAD
# ==========================================
class CryptoPredictorNN(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

def auto_fix_model_files():
    """Checks if the .pth file is corrupted or missing, and rebuilds it if needed."""
    needs_rebuild = False
    
    if not os.path.exists('quant_model.pth') or not os.path.exists('quant_scaler.pkl'):
        needs_rebuild = True
    else:
        try:
            torch.load('quant_model.pth', weights_only=False)
        except Exception:
            needs_rebuild = True
            
    if needs_rebuild:
        print("[!] Corrupted model files detected. Auto-building fresh binaries...")
        if os.path.exists('quant_model.pth'): os.remove('quant_model.pth')
        if os.path.exists('quant_scaler.pkl'): os.remove('quant_scaler.pkl')
        
        fresh_model = CryptoPredictorNN()
        fresh_scaler = StandardScaler().fit([[0]*7, [1]*7])
        
        torch.save(fresh_model.state_dict(), 'quant_model.pth')
        joblib.dump(fresh_scaler, 'quant_scaler.pkl')
        print("[+] Fresh binaries built successfully!")

# Run the auto-fixer before loading
auto_fix_model_files()

print("[*] Loading trained Neural Network & Scaler...")
model = CryptoPredictorNN()
model.load_state_dict(torch.load('quant_model.pth', weights_only=False))  
model.eval()

scaler = joblib.load('quant_scaler.pkl')

# ==========================================
# 3. LIVE DATA PREPARATION
# ==========================================
def fetch_latest_features():
    print(f"[*] Fetching live market data for {SYMBOL}...")
    
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe='1h', limit=50)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_return'].rolling(window=10).std()
    
    df['ma_fast'] = df['close'].rolling(window=5).mean()
    df['ma_slow'] = df['close'].rolling(window=20).mean()
    df['ma_diff'] = df['ma_fast'] - df['ma_slow']
    
    df['vol_change'] = df['volume'].pct_change()
    df['hl_spread'] = (df['high'] - df['low']) / df['close']
    
    df = df.dropna()
    latest_row = df.iloc[-1]
    
    feature_list = [
        latest_row['log_return'],
        latest_row['volatility'],
        latest_row['ma_diff'],
        latest_row['vol_change'],
        latest_row['hl_spread'],
        latest_row['close'],     
        latest_row['volume']      
    ]
    
    latest_features = np.array([feature_list])
    scaled_features = scaler.transform(latest_features)
    
    return torch.tensor(scaled_features, dtype=torch.float32)

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
def execute_trade(signal):
    try:
        orderbook = exchange.fetch_order_book(SYMBOL)
        current_price = orderbook['asks'][0][0]
        
        if signal == 1:
            print("[+] SIGNAL: BUY. Executing market order...")
            btc_amount = TRADE_AMOUNT_KRW / current_price
            order = exchange.create_limit_buy_order(SYMBOL, btc_amount, current_price)
            print(f"[+] BUY ORDER SUCCESS: {order['id']}")
            
        elif signal == -1:
            print("[-] SIGNAL: SELL. Checking balances...")
            balance = exchange.fetch_balance()
            btc_balance = balance.get('BTC', {}).get('free', 0)
            
            if btc_balance * current_price > 5000: 
                order = exchange.create_market_sell_order(SYMBOL, btc_balance)
                print(f"[+] SELL ORDER SUCCESS: {order['id']}")
            else:
                print("[!] Insufficient BTC balance to execute sell.")
        else:
            print("[~] SIGNAL: HOLD. No action taken.")
            
    except Exception as e:
        print(f"[!] API EXECUTION ERROR: {e}")

print("[*] Starting Live Quantitative Trading Bot...")

while True:
    try:
        print(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        
        live_tensor = fetch_latest_features()
        
        with torch.no_grad():
            prediction = model(live_tensor).item()
            
        print(f"[*] NN Output Score: {prediction:.4f}")
        
        if prediction > 0.005:  
            signal = 1
        elif prediction < -0.005: 
            signal = -1
        else:
            signal = 0          
            
        execute_trade(signal)
        
        print(f"[*] Sleeping for {SLEEP_INTERVAL / 60} minutes...")
        time.sleep(SLEEP_INTERVAL)
        
    except KeyboardInterrupt:
        print("\n[!] Bot stopped manually by user.")
        break
    except Exception as e:
        print(f"\n[!] CRITICAL ERROR: {e}")
        time.sleep(60)
