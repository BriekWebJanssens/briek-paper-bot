# bot.py — backtest met ALLE trades + dag-PnL + CSV-logging
import os, math, datetime as dt
import ccxt, pandas as pd, numpy as np
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator

# ---------- Config ----------
EXCHANGE = 'binance'
QUOTE = 'USDT'
TIMEFRAME = '4h'
LOOKBACK = 3000

FEE = 0.0006
SLIPPAGE = 0.003            # basis-slippage
RISK_PCT = 0.005
MAX_NOTIONAL_PCT = 0.05
MAX_OPEN_POS = 1
ATR_MULT = 3
BREAKOUT_N = 35
ADX_MIN = 20
RSI_MIN, RSI_MAX = 55, 68
DAILY_DD_STOP = -0.02
MAX_HOLD_BARS = 60          # ≈ 10 dagen op 4h

START_DATE = os.getenv('START_DATE', '2025-03-26')
END_DATE   = os.getenv('END_DATE',   '2025-07-31')

START_EQUITY = 1000.0
SYMBOLS_FILE = 'symbols.txt'

# ----------------------------

ex = getattr(ccxt, EXCHANGE)({
    'enableRateLimit': True,
    'timeout': 20000,
    'options': {'adjustForTimeDifference': True, 'defaultType': 'spot'}
})

# per-pair spread → adaptieve slippage
PAIR_SPREAD = {}

def fetch_symbols():
    print("Loading markets & tickers...")
    markets = ex.load_markets()
    tickers = ex.fetch_tickers()

    EX_BASE_BLACKLIST = {
        'USDT','USDC','TUSD','FDUSD','BUSD','DAI','EURI','EUR','TRY','BRL','GBP','NGN','UAH',
        'WBTC','WETH','WBETH','STETH','CBETH'
    }
    BLACKLIST = {'FTT/USDT','ERA/USDT','HOME/USDT','THE/USDT','APE/USDT'}

    MIN_VOL = 100_000      # ↑ strenger: min 300k USDT / 24h
    MAX_VOL = 3_000_000
    MAX_SPREAD = 0.003    # ↓ strakker: 0.25%

    def is_pegged(sym):
        base = markets[sym]['base'].upper()
        if base in EX_BASE_BLACKLIST:
            return True
        return any(k in base for k in ['USD','EUR','WBTC','WETH','WBETH','STETH','CBETH'])

    pre = []
    for s, m in markets.items():
        if not m.get('active'):
            continue
        if not s.endswith('/USDT'):
            continue
        if s in BLACKLIST:
            continue
        if is_pegged(s):
            continue

        t = tickers.get(m['symbol'])
        if not t:
            continue
        last = (t.get('last') or t.get('close') or 0) or 0
        baseVol = t.get('baseVolume') or 0
        bid = t.get('bid') or 0
        ask = t.get('ask') or 0
        spread = (ask - bid) / ask if ask else 1.0
        quoteVol = baseVol * last
        if MIN_VOL <= quoteVol <= MAX_VOL and spread <= MAX_SPREAD and last > 0:
            pre.append((s, spread))

    pre.sort(key=lambda x: x[1])
    symbols = [s for s, _ in pre][:30]
    print(f"Preselected {len(symbols)}; checking 30d volatility...")

    kept = []
    for s in symbols:
        o = ex.fetch_ohlcv(s, timeframe='1d', limit=35)
        if len(o) < 20:
            continue
        closes = pd.Series([r[4] for r in o], dtype='float64')
        vol = closes.pct_change().std()
        if pd.isna(vol) or vol < 0.02:
            continue
        kept.append(s)

    kept = kept[:20]
    print(f"Selected {len(kept)} symbols (clean small/mid caps)")
    return kept

def load_symbols():
    if os.path.exists(SYMBOLS_FILE):
        with open(SYMBOLS_FILE, 'r') as f:
            syms = [l.strip() for l in f if l.strip()]
        print(f"Loaded {len(syms)} symbols from {SYMBOLS_FILE}")
        return syms
    syms = fetch_symbols()
    with open(SYMBOLS_FILE, 'w') as f:
        f.write('\n'.join(syms))
    print(f"Wrote {len(syms)} symbols to {SYMBOLS_FILE}")
    return syms

def init_pair_spreads(symbols):
    """Initialiseer adaptieve slippage per pair: max(basis, halve spread, 0.06%)."""
    global PAIR_SPREAD
    tick = ex.fetch_tickers()
    for s in symbols:
        m = ex.market(s)
        t = tick.get(m['symbol'], {})
        bid = t.get('bid') or 0
        ask = t.get('ask') or 0
        sp = (ask - bid)/ask if ask else SLIPPAGE
        PAIR_SPREAD[s] = max(SLIPPAGE, sp * 0.5, 0.0006)

def apply_slip(side, px, symbol=None):
    s = PAIR_SPREAD.get(symbol, SLIPPAGE)
    return px * (1 + s) if side == 'buy' else px * (1 - s)

def ohlcv_df(symbol):
    o = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LOOKBACK)
    df = pd.DataFrame(o, columns=['ts','o','h','l','c','v'])
    for col in ['o','h','l','c','v']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('dt', inplace=True)
    df = df.tz_convert('Europe/Brussels')
    if START_DATE and END_DATE:
        df = df.loc[START_DATE:END_DATE]
    return df.dropna()

def add_indicators(df):
    # voldoende bars voor indicatoren
    MIN_IND_BARS = max(20, BREAKOUT_N + 1, 14 + 1)  # breakout & ADX
    if df is None or len(df) < MIN_IND_BARS:
        return pd.DataFrame()  # leeg -> wordt later geskipt

    df = df.copy()
    df['ema20'] = EMAIndicator(df['c'], 20).ema_indicator()
    df['ema50'] = EMAIndicator(df['c'], 50).ema_indicator()

    tr1 = (df['h'] - df['l']).abs()
    tr2 = (df['h'] - df['c'].shift()).abs()
    tr3 = (df['l'] - df['c'].shift()).abs()
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr14'] = df['tr'].rolling(14).mean()

    adx = ADXIndicator(high=df['h'], low=df['l'], close=df['c'], window=14)
    df['adx'] = adx.adx()

    df['rsi'] = RSIIndicator(df['c'], window=14).rsi()
    df['brk'] = df['h'].rolling(BREAKOUT_N).max().shift(1)
    return df

def signal_row(df, i):
    row, _ = df.iloc[i], df.iloc[i-1]
    trend = row.ema20 > row.ema50
    breakout = (row.c > row.brk) if not math.isnan(row.brk) else False
    adx_ok = row.adx >= ADX_MIN
    rsi_ok = (row.rsi >= RSI_MIN) and (row.rsi <= RSI_MAX)
    long_entry = trend and breakout and adx_ok and rsi_ok
    exit_trend = row.ema20 < row.ema50
    return long_entry, exit_trend

def position_size(equity, price, atr, min_notional):
    risk_usd = equity * RISK_PCT
    stop_dist = max(atr * ATR_MULT, price*0.01)
    qty = risk_usd / stop_dist
    notional = qty * price
    cap_notional = equity * MAX_NOTIONAL_PCT
    if notional > cap_notional:
        qty = cap_notional / price
    if qty * price < min_notional:
        return 0.0
    return max(0.0, float(qty))

def min_notional(symbol):
    m = ex.market(symbol)
    mn = m.get('limits', {}).get('cost', {}).get('min', 5.0)
    return mn or 5.0

def backtest(symbols):
    equity = START_EQUITY
    open_pos = {}  # sym -> dict(entry, stop, qty, bars)
    trades = []
    dfs = {}
    for s in symbols:
        try:
            raw = ohlcv_df(s)
            if raw is None or raw.empty:
                continue
            df = add_indicators(raw)
            print(f"{s}: {len(df)} bars")
            if df is None or df.empty or len(df) < 60:   # extra marge
                continue
            dfs[s] = df
        except Exception as e:
            print(f"Skip {s}: {e}")
            continue

    if not dfs:
        print("Geen geschikte symbols met voldoende data.")
        return START_EQUITY, pd.DataFrame(), pd.DataFrame()


    # gezamenlijke tijd-index
    time_index = sorted(set().union(*[df.index for df in dfs.values()]))
    current_day = None
    day_pnl_running = 0.0

    for t in time_index:
        day = t.date()
        if current_day is None:
            current_day = day

        # dagwissel -> dagresultaat loggen
        if day != current_day:
            print(f"[{current_day}] Dag-PnL: {day_pnl_running:+.2f} EUR")
            current_day = day
            day_pnl_running = 0.0

        # tijdsexit / bar-teller
        for s in list(open_pos.keys()):
            if s in dfs and t in dfs[s].index:
                open_pos[s]['bars'] += 1
                if open_pos[s]['bars'] >= MAX_HOLD_BARS:
                    price = dfs[s].loc[t].c
                    fill = apply_slip('sell', price, s)
                    pos = open_pos[s]
                    gross = (fill - pos['entry']) * pos['qty']
                    fees = (pos['entry'] * pos['qty']) * FEE + (fill * pos['qty']) * FEE
                    pnl = gross - fees
                    equity += pnl
                    day_pnl_running += pnl
                    trades.append({
                        'time': t, 'symbol': s, 'side': 'TIME_EXIT',
                        'price': float(fill), 'qty': float(pos['qty']),
                        'pnl': float(pnl), 'equity': float(equity)
                    })
                    print(f"{t}  {s:12} TIME_EXIT px={fill:.6f} qty={pos['qty']:.6f}  PnL={pnl:+.2f}  Eq={equity:.2f}")
                    del open_pos[s]

        # exits op signaal of stop
        for s in list(open_pos.keys()):
            df = dfs.get(s)
            if df is None or t not in df.index:
                continue
            i = df.index.get_loc(t)
            _, exit_sig = signal_row(df, i)
            price = df.iloc[i].c
            stop = open_pos[s]['stop']
            qty = open_pos[s]['qty']
            if exit_sig or price <= stop:
                fill = apply_slip('sell', price, s)
                entry = open_pos[s]['entry']
                gross = (fill - entry) * qty
                fees = (entry * qty) * FEE + (fill * qty) * FEE
                pnl = gross - fees
                equity += pnl
                day_pnl_running += pnl
                trades.append({
                    'time': t, 'symbol': s, 'side': 'EXIT',
                    'price': float(fill), 'qty': float(qty),
                    'pnl': float(pnl), 'equity': float(equity)
                })
                print(f"{t}  {s:12} EXIT  px={fill:.6f} qty={qty:.6f}  PnL={pnl:+.2f}  Eq={equity:.2f}")
                del open_pos[s]

        # entries met prioritering
        if len(open_pos) < MAX_OPEN_POS and day_pnl_running > DAILY_DD_STOP * equity:
            cands = []
            for s in symbols:
                if s in open_pos:
                    continue
                df = dfs.get(s)
                if df is None or t not in df.index:
                    continue
                i = df.index.get_loc(t)
                if i < 50:
                    continue
                go_long, _ = signal_row(df, i)
                if not go_long:
                    continue
                row = df.iloc[i]
                if row.atr14 <= 0 or math.isnan(row.atr14):
                    continue
                if row.c < 0.01:
                    continue  # microprice guard
                # score: breakout-overshoot + beetje ADX-gewicht
                brk_rel = (row.c / row.brk) - 1 if (row.brk and not math.isnan(row.brk) and row.brk > 0) else 0
                score = brk_rel + 0.1 * (row.adx / 50.0)
                cands.append((score, s, row))

            if cands:
                cands.sort(reverse=True, key=lambda x: x[0])
                _, s, row = cands[0]
                mn = min_notional(s)
                qty = position_size(equity, row.c, row.atr14, mn)
                if qty > 0:
                    fill = apply_slip('buy', row.c, s)
                    stop = fill - ATR_MULT * row.atr14
                    cost = fill * qty
                    fees = cost * FEE
                    equity -= fees  # instapfee
                    open_pos[s] = {'entry': fill, 'stop': stop, 'qty': qty, 'bars': 0}
                    trades.append({
                        'time': t, 'symbol': s, 'side': 'ENTRY',
                        'price': float(fill), 'qty': float(qty),
                        'pnl': float(-fees), 'equity': float(equity)
                    })
                    print(f"{t}  {s:12} ENTRY px={fill:.6f} qty={qty:.6f}  Fee={-fees:.2f}  Eq={equity:.2f}")

    # force exits op laatste bar
    for s, pos in list(open_pos.items()):
        df = dfs[s]
        t = df.index[-1]
        price = df.iloc[-1].c
        fill = apply_slip('sell', price, s)
        gross = (fill - pos['entry']) * pos['qty']
        fees = (pos['entry'] * pos['qty']) * FEE + (fill * pos['qty']) * FEE
        pnl = gross - fees
        equity += pnl
        trades.append({
            'time': t, 'symbol': s, 'side': 'FORCE_EXIT',
            'price': float(fill), 'qty': float(pos['qty']),
            'pnl': float(pnl), 'equity': float(equity)
        })
        print(f"{t}  {s:12} FORCE_EXIT px={fill:.6f} qty={pos['qty']:.6f}  PnL={pnl:+.2f}  Eq={equity:.2f}")

    # resultaten + metrics
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("Geen trades gegenereerd.")
        return equity, trades_df, pd.DataFrame()

    trades_df['date'] = trades_df['time'].dt.tz_convert('Europe/Brussels').dt.date
    day_pnl = trades_df.groupby('date', as_index=False)['pnl'].sum().rename(columns={'pnl': 'day_pnl'})

    ret = (equity / START_EQUITY) - 1
    curve = trades_df['equity'].tolist()
    peak, max_dd = START_EQUITY, 0.0
    for v in curve:
        peak = max(peak, v)
        dd = (v - peak) / peak
        max_dd = min(max_dd, dd)

    ex_rows = trades_df[trades_df['side'].isin(['EXIT', 'FORCE_EXIT', 'TIME_EXIT'])]
    wins = ex_rows[ex_rows['pnl'] > 0]
    losses = ex_rows[ex_rows['pnl'] <= 0]
    pf = (wins['pnl'].sum() / abs(losses['pnl'].sum())) if not losses.empty else float('inf')

    print("\n--- Samenvatting ---")
    print(f"Trades: {len(trades_df[trades_df.side=='ENTRY'])} entries")
    print(f"Equity start: {START_EQUITY:.2f} -> end: {equity:.2f} | Return: {ret*100:.2f}%")
    print(f"Max drawdown: {max_dd*100:.2f}% | Profit factor: {pf:.2f}")

    trades_df.to_csv('trades.csv', index=False)
    day_pnl.to_csv('daily_pnl.csv', index=False)
    print("CSV geschreven: trades.csv, daily_pnl.csv")

    if not day_pnl.empty:
        last_row = day_pnl.iloc[-1]
        print(f"Laatst berekende dag [{last_row['date']}]: Dag-PnL {last_row['day_pnl']:+.2f} EUR")

    return equity, trades_df, day_pnl

if __name__ == "__main__":
    print(f"Backtest window: {START_DATE} -> {END_DATE}")
    syms = load_symbols()
    print("Pairs:", syms)
    init_pair_spreads(syms)
    backtest(syms)

