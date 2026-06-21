from flask import Flask, request, jsonify
import requests, json, datetime, traceback, os
import pandas as pd
import numpy as np
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import config

app = Flask(__name__)

trades_today = []
daily_pnl = 0.0
consecutive_losses = 0
system_halted = False
simulated_crypto_position = None

def discord_post(webhook_url, content=None, embed=None):
    if not webhook_url:
        return
    payload = {}
    if content:
        payload["content"] = content
    if embed:
        payload["embeds"] = [embed]
    try:
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e:
        print("Discord post failed: " + str(e))

def get_alpaca_creds():
    if config.TRADING_MODE == "live":
        return config.ALPACA_LIVE_KEY, config.ALPACA_LIVE_SECRET, config.ALPACA_LIVE_URL
    return config.ALPACA_PAPER_KEY, config.ALPACA_PAPER_SECRET, config.ALPACA_PAPER_URL

def calculate_indicators(df):
    df['ema9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['Close'].ewm(span=21, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['vwap'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['vol_avg'] = df['Volume'].rolling(20).mean()
    return df

def process_trade_signal(data):
    global trades_today, daily_pnl, consecutive_losses, system_halted

    if system_halted:
        discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "Signal received but system is HALTED. Trade skipped.")
        return

    action = data.get("action")
    symbol = data.get("symbol")
    payload_mode = data.get("mode")
    price = float(data.get("price", 0))
    stop_loss = float(data.get("stop_loss", 0))
    tp1 = float(data.get("take_profit_1", 0))

    if payload_mode != config.TRADING_MODE:
        discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "Mode mismatch detected. Trade skipped.")
        return
    if len(trades_today) >= config.MAX_TRADES_PER_DAY:
        discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "Daily trade limit reached. Skipping.")
        return

    key, secret, base_url = get_alpaca_creds()
    if not key or not secret:
        discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "Missing Alpaca credentials. Trade skipped.")
        return

    if config.TRADING_MODE == "live":
        discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "LIVE TRADING MODE ACTIVATED - real money is now being used")

    side = "buy" if action == "buy" else "sell"
    qty = 1

    order = {
        "symbol": symbol, "qty": qty, "side": side, "type": "market", "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": round(tp1, 2)},
        "stop_loss": {"stop_price": round(stop_loss, 2)}
    }
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    resp = requests.post(base_url + "/v2/orders", json=order, headers=headers, timeout=10)

    tag = "[LIVE]" if config.TRADING_MODE == "live" else "[PAPER]"
    timestamp = datetime.datetime.now().isoformat()
    trades_today.append({"symbol": symbol, "side": side, "price": price, "time": timestamp})

    embed_color = 0x3498db if config.TRADING_MODE == "paper" else 0x2ecc71
    embed = {
        "title": tag + " " + side.upper() + " " + symbol,
        "color": embed_color,
        "fields": [
            {"name": "Price", "value": str(price), "inline": True},
            {"name": "Stop Loss", "value": str(stop_loss), "inline": True},
            {"name": "Take Profit 1", "value": str(tp1), "inline": True}
        ]
    }
    target_channel = config.DISCORD_WEBHOOK_LIVE_TRADES if config.TRADING_MODE == "live" else config.DISCORD_WEBHOOK_PAPER_TRADES
    discord_post(target_channel, embed=embed)
    discord_post(config.DISCORD_WEBHOOK_TRADE_ALERTS, embed=embed)

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
        process_trade_signal(data)
        return jsonify({"status": "processed"}), 200
    except Exception as e:
        traceback.print_exc()
        discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "ERROR in webhook: " + str(e))
        return jsonify({"status": "error", "detail": str(e)}), 500

@app.route("/halt", methods=["POST"])
def halt():
    global system_halted
    system_halted = True
    discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "KILL SWITCH ACTIVATED - trading halted.")
    return jsonify({"status": "halted"}), 200

@app.route("/resume", methods=["POST"])
def resume():
    global system_halted
    password = request.json.get("password", "")
    if password != config.MODE_SWITCH_PASSWORD:
        return jsonify({"status": "wrong_password"}), 403
    system_halted = False
    discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "Trading resumed.")
    return jsonify({"status": "resumed"}), 200

@app.route("/set-mode", methods=["POST"])
def set_mode():
    password = request.json.get("password", "")
    new_mode = request.json.get("mode", "")
    confirm = request.json.get("confirm_phrase", "")

    if password != config.MODE_SWITCH_PASSWORD:
        return jsonify({"status": "wrong_password"}), 403
    if new_mode not in ["paper", "live"]:
        return jsonify({"status": "invalid_mode"}), 400
    if new_mode == "live" and confirm != config.LIVE_CONFIRM_PHRASE:
        return jsonify({"status": "live_requires_confirm_phrase"}), 403

    config.TRADING_MODE = new_mode
    discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "Mode switched to " + new_mode.upper())
    return jsonify({"status": "mode_set", "mode": new_mode}), 200

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "mode": config.TRADING_MODE,
        "halted": system_halted,
        "trades_today": len(trades_today),
        "daily_pnl": daily_pnl
    })

def run_strategy_check():
    global system_halted, trades_today
    try:
        ny_tz = pytz.timezone("America/New_York")
        now = datetime.datetime.now(ny_tz)

        if now.weekday() >= 5:
            return
        if now.hour < 9 or (now.hour == 9 and now.minute < 35):
            return
        if now.hour >= 16 or (now.hour == 15 and now.minute >= 45):
            return
        if system_halted:
            return
        if len(trades_today) >= config.MAX_TRADES_PER_DAY:
            return

        df5 = yf.download("SPY", period="5d", interval="5m", progress=False)
        df15 = yf.download("SPY", period="5d", interval="15m", progress=False)
        df60 = yf.download("SPY", period="10d", interval="60m", progress=False)

        if len(df5) < 30 or len(df15) < 30 or len(df60) < 30:
            return

        df5 = calculate_indicators(df5)
        df15 = calculate_indicators(df15)
        df60 = calculate_indicators(df60)

        last = df5.iloc[-1]
        htf_bullish = df15.iloc[-1]['ema9'] > df15.iloc[-1]['ema21'] and df60.iloc[-1]['ema9'] > df60.iloc[-1]['ema21']
        htf_bearish = df15.iloc[-1]['ema9'] < df15.iloc[-1]['ema21'] and df60.iloc[-1]['ema9'] < df60.iloc[-1]['ema21']

        today_session = df5[df5.index.date == now.date()]
        or_window = today_session.between_time("09:30", "10:00")
        if len(or_window) == 0:
            return
        or_high = or_window['High'].max()
        or_low = or_window['Low'].min()

        vol_spike = last['Volume'] > last['vol_avg'] * 1.5
        close = last['Close']

        long_cond = (last['ema9'] > last['ema21'] and close > last['vwap'] and
                     45 < last['rsi'] < 65 and vol_spike and close > or_high and htf_bullish)
        short_cond = (last['ema9'] < last['ema21'] and close < last['vwap'] and
                      35 < last['rsi'] < 55 and vol_spike and close < or_low and htf_bearish)

        if not long_cond and not short_cond:
            return

        atr = last['atr']
        action = "buy" if long_cond else "sell"
        stop_loss = close - atr * 1.5 if action == "buy" else close + atr * 1.5
        tp1 = close + atr if action == "buy" else close - atr

        fake_request_data = {
            "symbol": "SPY",
            "action": action,
            "price": float(close),
            "stop_loss": float(stop_loss),
            "take_profit_1": float(tp1),
            "mode": config.TRADING_MODE
        }
        process_trade_signal(fake_request_data)

    except Exception as e:
        traceback.print_exc()
        discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "ERROR in strategy check: " + str(e))

def run_crypto_strategy_check():
    global system_halted, trades_today, simulated_crypto_position
    try:
        if system_halted:
            return

        df5 =
