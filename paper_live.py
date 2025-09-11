# paper_live.py
import os, time, math, json, pathlib, datetime as dt
import pandas as pd, ccxt
from bot import (
    load_symbols, init_pair_spreads, add_indicators, apply_slip,
    position_size, min_notional, signal_row,
    TIMEFRAME, START_EQUITY, FEE, ATR_MULT, MAX_OPEN_POS, DAILY_DD_STOP
)

EXCHANGE = 'binance'
ex = getattr(ccxt, EXCHANGE)({
    'enableRateLimit': True,
    'timeout': 30000,
    'options': {'defaultType': 'spot'}
})

EQUITY = START_EQUITY
open_pos = {}
LOG_TRADES = 'paper_trades.csv'
LOG_EQUITY = 'paper_equity.csv'
STATE_PATH = pathlib.Path('paper_state.json')
TZ = 'Europe/Brussels'


def fetch_ohlcv(sym, since=None, limit=300):
    o = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=limit, since=since)
    df = pd.DataFrame(o, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('dt', inplace=True)
    df.index = df.index.tz_convert(TZ)
    return df[['o', 'h', 'l', 'c', 'v']].astype('float64')


def append_csv(path, rowdict, header):
    import csv, os
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(rowdict)


def log_equity_row(ts, equity, n_open):
    append_csv(
        LOG_EQUITY,
        {'time': str(ts), 'equity': float(equity), 'open_positions': int(n_open)},
        ['time', 'equity', 'open_positions']
    )


def log_trade_row(t, s, side, price, qty, pnl, equity):
    append_csv(
        LOG_TRADES,
        {
            'time': str(t), 'symbol': s, 'side': side,
            'price': float(price), 'qty': float(qty),
            'pnl': float(pnl), 'equity': float(equity)
        },
        ['time', 'symbol', 'side', 'price', 'qty', 'pnl', 'equity']
    )


def load_state():
    global EQUITY, open_pos
    if STATE_PATH.exists():
        try:
            st = json.loads(STATE_PATH.read_text(encoding='utf-8'))
            EQUITY = float(st.get('equity', START_EQUITY))
            open_pos = st.get('open_pos', {}) or {}
        except Exception as e:
            print(f"[state] load failed: {e}")


def save_state():
    try:
        STATE_PATH.write_text(json.dumps({
            'equity': float(EQUITY),
            'open_pos': open_pos
        }, ensure_ascii=False), encoding='utf-8')
    except Exception as e:
        print(f"[state] save failed: {e}")


def now_tz():
    return dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(0))).astimezone()


def main():
    global EQUITY
    ONE_PASS = os.getenv("ONE_PASS", "0") == "1"

    load_state()
    print(f"TIMEFRAME={TIMEFRAME}")
    print(f"START_EQUITY_CODE={START_EQUITY}")
    print(f"STATE_FILE_EQUITY={EQUITY}")
    print(f"state_exists={STATE_PATH.exists()}")

    # Log één initiële equity-rij zodat CSV altijd bestaat
    start_loop = dt.datetime.now().astimezone()
    log_equity_row(start_loop, EQUITY, len(open_pos))

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
                        fees = (pos['entry'] * pos['qty'] + fill * pos['qty']) * FEE
                        pnl = gross - fees
                        EQUITY += pnl
                        day_pnl_running += pnl
                        log_trade_row(t, s, 'EXIT', fill, pos['qty'], pnl, EQUITY)
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
                            log_trade_row(t, s, 'ENTRY', fill, qty, -fees, EQUITY)
                            entered = True
                print(" ENTRY" if entered else " ok")

            except ccxt.NetworkError as e:
                print(f" neterr: {e}")
            except Exception as e:
                print(f" err: {e}")

        log_equity_row(start_loop, EQUITY, len(open_pos))
        save_state()
        print(f"[{dt.datetime.now().astimezone():%H:%M:%S}] Loop #{loop_num} done. equity={EQUITY:.2f}, open={len(open_pos)}")

        if ONE_PASS:
            break
        time.sleep(900)


if __name__ == "__main__":
    main()
