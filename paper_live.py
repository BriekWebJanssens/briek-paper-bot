import os, time, math, pandas as pd, ccxt, datetime as dt, json, pathlib
from bot import load_symbols, init_pair_spreads, add_indicators, apply_slip, position_size, min_notional, signal_row, TIMEFRAME, START_EQUITY, FEE, ATR_MULT, MAX_OPEN_POS, DAILY_DD_STOP

EXCHANGE = 'binance'
ex = getattr(ccxt, EXCHANGE)({
    'enableRateLimit': True,
    'timeout': 30000,
    'options': {'defaultType': 'spot'}
})

EQUITY = START_EQUITY
open_pos = {}
log_trades = 'paper_trades.csv'
log_equity = 'paper_equity.csv'
STATE_PATH = pathlib.Path('paper_state.json')

def fetch_ohlcv(sym, since=None, limit=300):
    o = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=limit, since=since)
    df = pd.DataFrame(o, columns=['ts','o','h','l','c','v'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('dt', inplace=True)
    df.index = df.index.tz_convert('Europe/Brussels')
    return df[['o','h','l','c','v']].astype('float64')

def append_csv(path, rowdict):
    import csv, os
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rowdict.keys())
        if write_header: w.writeheader()
        w.writerow(rowdict)

def load_state():
    global EQUITY, open_pos
    if STATE_PATH.exists():
        try:
            st = json.loads(STATE_PATH.read_text(encoding='utf-8'))
            EQUITY = float(st.get('equity', START_EQUITY))
            open_pos = st.get('open_pos', {}) or {}
        except Exception as e:
            print(f"[state] load failed: {e}")
print(f"TIMEFRAME={TIMEFRAME}")
print(f"EQUITY_START={EQUITY}")
print("state_exists=", os.path.exists("paper_state.json"))

def save_state():
    try:
        STATE_PATH.write_text(json.dumps({
            'equity': float(EQUITY),
            'open_pos': open_pos
        }, ensure_ascii=False), encoding='utf-8')
    except Exception as e:
        print(f"[state] save failed: {e}")

def main():
    global EQUITY
    ONE_PASS = os.getenv("ONE_PASS", "0") == "1"
    load_state()

    syms = load_symbols()
    init_pair_spreads(syms)
    print("Paper start. Symbols:", syms)

    print("Warm-up fetch…")
    for s in syms[:3]:
        try:
            dfw = fetch_ohlcv(s, limit=20)
            print(f"  {s} ok, last bar: {dfw.index[-1]}")
        except Exception as e:
            print(f"  {s} warm-up FAILED: {e}")

    last_bar_time = {}
    loop_num = 0

    while True:
        loop_num += 1
        start_loop = dt.datetime.now().astimezone()
        print(f"[{start_loop:%Y-%m-%d %H:%M:%S}] Start loop #{loop_num}")
        day_pnl_running = 0.0

        for idx, s in enumerate(syms, start=1):
            print(f"  [{idx}/{len(syms)}] fetch {s}…", end='', flush=True)
            try:
                df = fetch_ohlcv(s, limit=120)
                df = add_indicators(df)
                if df.empty or len(df) < 60:
                    print(" skip (te weinig data)")
                    continue

                t = df.index[-2]  # gesloten bar
                if last_bar_time.get(s) == t:
                    print(" no new bar")
                    continue
                last_bar_time[s] = t
                i = df.index.get_loc(t)
                row = df.iloc[i]
                long_entry, exit_trend = signal_row(df, i)

                # exits
                if s in open_pos:
                    pos = open_pos[s]
                    price = row.c
                    if exit_trend or price <= pos['stop']:
                        fill = apply_slip('sell', price, s)
                        gross = (fill - pos['entry']) * pos['qty']
                        fees = (pos['entry']*pos['qty'] + fill*pos['qty']) * FEE
                        pnl = gross - fees
                        EQUITY += pnl
                        day_pnl_running += pnl
                        append_csv(log_trades, {
                            'time': str(t), 'symbol': s, 'side':'EXIT',
                            'price': float(fill), 'qty': float(pos['qty']),
                            'pnl': float(pnl), 'equity': float(EQUITY)
                        })
                        del open_pos[s]
                        print(" EXIT")
                        continue

                # entries
                entered = False
                if len(open_pos) < MAX_OPEN_POS and day_pnl_running > DAILY_DD_STOP * EQUITY:
                    if long_entry and s not in open_pos and row.atr14 > 0 and row.c >= 0.01:
                        mn = min_notional(s)
                        qty = position_size(EQUITY, row.c, row.atr14, mn)
                        if qty > 0:
                            fill = apply_slip('buy', row.c, s)
                            stop = fill - ATR_MULT * row.atr14
                            cost = fill * qty
                            fees = cost * FEE
                            EQUITY -= fees
                            open_pos[s] = {'entry': fill, 'stop': stop, 'qty': qty, 'bars': 0}
                            append_csv(log_trades, {
                                'time': str(t), 'symbol': s, 'side':'ENTRY',
                                'price': float(fill), 'qty': float(qty),
                                'pnl': float(-fees), 'equity': float(EQUITY)
                            })
                            entered = True
                print(" ENTRY" if entered else " ok")

            except ccxt.NetworkError as e:
                print(f" neterr: {e}")
            except Exception as e:
                print(f" err: {e}")

        append_csv(log_equity, {'time': str(start_loop), 'equity': float(EQUITY), 'open_positions': len(open_pos)})
        save_state()
        print(f"[{dt.datetime.now().astimezone():%H:%M:%S}] Loop #{loop_num} done. equity={EQUITY:.2f}, open={len(open_pos)}")

        if ONE_PASS:
            break
        time.sleep(900)

if __name__ == "__main__":
    main()
