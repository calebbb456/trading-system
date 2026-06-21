from flask import Flask, request, jsonify
import requests, datetime, traceback, os
import pandas as pd
import numpy as np
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import config
import discord
from discord.ext import commands
import threading

app = Flask(__name__)

trades_today = []
daily_pnl = 0.0
consecutive_losses = 0
system_halted = False
simulated_crypto_position = None

# ====== DISCORD BOT SETUP ======
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

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
    df = df.copy()
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

def get_spy_condition_details():
    try:
        df5 = yf.download("SPY", period="5d", interval="5m", progress=False)
        df15 = yf.download("SPY", period="5d", interval="15m", progress=False)
        df60 = yf.download("SPY", period="10d", interval="60m", progress=False)
        if len(df5) < 30 or len(df15) < 30 or len(df60) < 30:
            return None
        df5 = calculate_indicators(df5)
        df15 = calculate_indicators(df15)
        df60 = calculate_indicators(df60)
        last = df5.iloc[-1]
        ny_tz = pytz.timezone("America/New_York")
        now = datetime.datetime.now(ny_tz)
        today_session = df5[df5.index.date == now.date()]
        or_window = today_session.between_time("09:30", "10:00")
        or_high = float(or_window['High'].max()) if len(or_window) > 0 else None
        or_low = float(or_window['Low'].min()) if len(or_window) > 0 else None
        close = float(last['Close'])
        ema9 = float(last['ema9'])
        ema21 = float(last['ema21'])
        rsi = float(last['rsi'])
        vwap = float(last['vwap'])
        volume = float(last['Volume'])
        vol_avg = float(last['vol_avg'])
        htf_bull = float(df15.iloc[-1]['ema9']) > float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) > float(df60.iloc[-1]['ema21'])
        htf_bear = float(df15.iloc[-1]['ema9']) < float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) < float(df60.iloc[-1]['ema21'])
        return {
            "close": close, "ema9": ema9, "ema21": ema21, "rsi": rsi,
            "vwap": vwap, "volume": volume, "vol_avg": vol_avg,
            "or_high": or_high, "or_low": or_low,
            "htf_bull": htf_bull, "htf_bear": htf_bear
        }
    except Exception as e:
        print("Error getting SPY conditions: " + str(e))
        return None

def get_btc_condition_details():
    try:
        df5 = yf.download("BTC-USD", period="5d", interval="5m", progress=False)
        df15 = yf.download("BTC-USD", period="5d", interval="15m", progress=False)
        df60 = yf.download("BTC-USD", period="10d", interval="60m", progress=False)
        if len(df5) < 30 or len(df15) < 30 or len(df60) < 30:
            return None
        df5 = calculate_indicators(df5)
        df15 = calculate_indicators(df15)
        df60 = calculate_indicators(df60)
        last = df5.iloc[-1]
        close = float(last['Close'])
        ema9 = float(last['ema9'])
        ema21 = float(last['ema21'])
        rsi = float(last['rsi'])
        vwap = float(last['vwap'])
        volume = float(last['Volume'])
        vol_avg = float(last['vol_avg'])
        htf_bull = float(df15.iloc[-1]['ema9']) > float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) > float(df60.iloc[-1]['ema21'])
        htf_bear = float(df15.iloc[-1]['ema9']) < float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) < float(df60.iloc[-1]['ema21'])
        return {
            "close": close, "ema9": ema9, "ema21": ema21, "rsi": rsi,
            "vwap": vwap, "volume": volume, "vol_avg": vol_avg,
            "htf_bull": htf_bull, "htf_bear": htf_bear
        }
    except Exception as e:
        print("Error getting BTC conditions: " + str(e))
        return None

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

# ====== DISCORD BOT COMMANDS ======
@bot.command(name="status")
async def cmd_status(ctx):
    halted_text = "HALTED" if system_halted else "Running"
    msg = (
        "**System Status**\n"
        "Mode: " + config.TRADING_MODE.upper() + "\n"
        "Status: " + halted_text + "\n"
        "Trades today: " + str(len(trades_today)) + " / " + str(config.MAX_TRADES_PER_DAY) + "\n"
        "Simulated P&L today: $" + str(round(daily_pnl, 2))
    )
    await ctx.send(msg)

@bot.command(name="check")
async def cmd_check(ctx):
    await ctx.send("Checking SPY conditions now, give me a few seconds...")
    d = get_spy_condition_details()
    if d is None:
        await ctx.send("Could not fetch SPY data right now. Try again in a minute.")
        return
    def yn(val): return "YES" if val else "NO"
    ema_bull = d['ema9'] > d['ema21']
    ema_bear = d['ema9'] < d['ema21']
    above_vwap = d['close'] > d['vwap']
    below_vwap = d['close'] < d['vwap']
    rsi_bull = 45 < d['rsi'] < 65
    rsi_bear = 35 < d['rsi'] < 55
    vol_ok = d['volume'] > d['vol_avg'] * 1.5
    or_bull = d['or_high'] is not None and d['close'] > d['or_high']
    or_bear = d['or_low'] is not None and d['close'] < d['or_low']
    long_score = sum([ema_bull, above_vwap, rsi_bull, vol_ok, or_bull, d['htf_bull']])
    short_score = sum([ema_bear, below_vwap, rsi_bear, vol_ok, or_bear, d['htf_bear']])
    msg = (
        "**SPY Signal Check** (price: $" + str(round(d['close'], 2)) + ")\n\n"
        "**LONG conditions (" + str(long_score) + "/6 passing):**\n"
        "EMA9 > EMA21: " + yn(ema_bull) + "\n"
        "Price > VWAP: " + yn(above_vwap) + "\n"
        "RSI 45-65: " + yn(rsi_bull) + " (RSI = " + str(round(d['rsi'], 1)) + ")\n"
        "Volume spike: " + yn(vol_ok) + "\n"
        "Above opening range high: " + yn(or_bull) + "\n"
        "HTF bias bullish: " + yn(d['htf_bull']) + "\n\n"
        "**SHORT conditions (" + str(short_score) + "/6 passing):**\n"
        "EMA9 < EMA21: " + yn(ema_bear) + "\n"
        "Price < VWAP: " + yn(below_vwap) + "\n"
        "RSI 35-55: " + yn(rsi_bear) + " (RSI = " + str(round(d['rsi'], 1)) + ")\n"
        "Volume spike: " + yn(vol_ok) + "\n"
        "Below opening range low: " + yn(or_bear) + "\n"
        "HTF bias bearish: " + yn(d['htf_bear']) + "\n\n"
        "**Need all 6 to trigger a trade.**"
    )
    await ctx.send(msg)

@bot.command(name="btccheck")
async def cmd_btccheck(ctx):
    await ctx.send("Checking BTC-USD conditions now, give me a few seconds...")
    d = get_btc_condition_details()
    if d is None:
        await ctx.send("Could not fetch BTC data right now. Try again in a minute.")
        return
    def yn(val): return "YES" if val else "NO"
    ema_bull = d['ema9'] > d['ema21']
    ema_bear = d['ema9'] < d['ema21']
    above_vwap = d['close'] > d['vwap']
    below_vwap = d['close'] < d['vwap']
    rsi_bull = 45 < d['rsi'] < 65
    rsi_bear = 35 < d['rsi'] < 55
    vol_ok = d['volume'] > d['vol_avg'] * 1.5
    long_score = sum([ema_bull, above_vwap, rsi_bull, vol_ok, d['htf_bull']])
    short_score = sum([ema_bear, below_vwap, rsi_bear, vol_ok, d['htf_bear']])
    msg = (
        "**BTC-USD Signal Check** (price: $" + str(round(d['close'], 2)) + ")\n\n"
        "**LONG conditions (" + str(long_score) + "/5 passing):**\n"
        "EMA9 > EMA21: " + yn(ema_bull) + "\n"
        "Price > VWAP: " + yn(above_vwap) + "\n"
        "RSI 45-65: " + yn(rsi_bull) + " (RSI = " + str(round(d['rsi'], 1)) + ")\n"
        "Volume spike: " + yn(vol_ok) + "\n"
        "HTF bias bullish: " + yn(d['htf_bull']) + "\n\n"
        "**SHORT conditions (" + str(short_score) + "/5 passing):**\n"
        "EMA9 < EMA21: " + yn(ema_bear) + "\n"
        "Price < VWAP: " + yn(below_vwap) + "\n"
        "RSI 35-55: " + yn(rsi_bear) + " (RSI = " + str(round(d['rsi'], 1)) + ")\n"
        "Volume spike: " + yn(vol_ok) + "\n"
        "HTF bias bearish: " + yn(d['htf_bear']) + "\n\n"
        "**Need all 5 to trigger a sandbox signal.**"
    )
    await ctx.send(msg)

@bot.command(name="trades")
async def cmd_trades(ctx):
    if len(trades_today) == 0:
        await ctx.send("No trades placed today yet.")
        return
    msg = "**Trades Today:**\n"
    for t in trades_today:
        msg += t['side'].upper() + " " + t['symbol'] + " @ $" + str(round(t['price'], 2)) + " at " + t['time'][:19] + "\n"
    await ctx.send(msg)

@bot.command(name="positions")
async def cmd_positions(ctx):
    try:
        key, secret, base_url = get_alpaca_creds()
        headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        resp = requests.get(base_url + "/v2/positions", headers=headers, timeout=10)
        positions = resp.json()
        if len(positions) == 0:
            await ctx.send("No open positions right now.")
            return
        msg = "**Open Positions:**\n"
        for p in positions:
            msg += (p['symbol'] + " | " + p['side'].upper() + " | qty: " + str(p['qty']) +
                    " | entry: $" + str(round(float(p['avg_entry_price']), 2)) +
                    " | current: $" + str(round(float(p['current_price']), 2)) +
                    " | P&L: $" + str(round(float(p['unrealized_pl']), 2)) + "\n")
        await ctx.send(msg)
    except Exception as e:
        await ctx.send("Error fetching positions: " + str(e))

@bot.command(name="pnl")
async def cmd_pnl(ctx):
    try:
        key, secret, base_url = get_alpaca_creds()
        headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        resp = requests.get(base_url + "/v2/account", headers=headers, timeout=10)
        acct = resp.json()
        equity = round(float(acct['equity']), 2)
        last_eq = round(float(acct['last_equity']), 2)
        day_pnl = round(equity - last_eq, 2)
        pct = round((day_pnl / last_eq) * 100, 2)
        msg = (
            "**P&L Summary**\n"
            "Account equity: $" + str(equity) + "\n"
            "Today's P&L: $" + str(day_pnl) + " (" + str(pct) + "%)\n"
            "Trades placed today: " + str(len(trades_today))
        )
        await ctx.send(msg)
    except Exception as e:
        await ctx.send("Error fetching P&L: " + str(e))

@bot.command(name="halt")
async def cmd_halt(ctx):
    global system_halted
    system_halted = True
    discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "KILL SWITCH ACTIVATED via Discord command - trading halted.")
    await ctx.send("Trading HALTED. Use /resume to restart.")

@bot.command(name="resume")
async def cmd_resume(ctx, password: str = ""):
    global system_halted
    if password != config.MODE_SWITCH_PASSWORD:
        await ctx.send("Wrong password. Use: /resume yourpassword")
        return
    system_halted = False
    discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "Trading RESUMED via Discord command.")
    await ctx.send("Trading resumed.")

@bot.event
async def on_ready():
    print("Discord bot online: " + str(bot.user))
    discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "Bot online. Commands ready: /status /check /btccheck /trades /positions /pnl /halt /resume")

# ====== FLASK ROUTES ======
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

# ====== SCHEDULED JOBS ======
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
        htf_bullish = float(df15.iloc[-1]['ema9']) > float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) > float(df60.iloc[-1]['ema21'])
        htf_bearish = float(df15.iloc[-1]['ema9']) < float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) < float(df60.iloc[-1]['ema21'])
        today_session = df5[df5.index.date == now.date()]
        or_window = today_session.between_time("09:30", "10:00")
        if len(or_window) == 0:
            return
        or_high = float(or_window['High'].max())
        or_low = float(or_window['Low'].min())
        vol_spike = float(last['Volume']) > float(last['vol_avg']) * 1.5
        close = float(last['Close'])
        long_cond = (float(last['ema9']) > float(last['ema21']) and close > float(last['vwap']) and
                     45 < float(last['rsi']) < 65 and vol_spike and close > or_high and htf_bullish)
        short_cond = (float(last['ema9']) < float(last['ema21']) and close < float(last['vwap']) and
                      35 < float(last['rsi']) < 55 and vol_spike and close < or_low and htf_bearish)
        if not long_cond and not short_cond:
            return
        atr = float(last['atr'])
        action = "buy" if long_cond else "sell"
        stop_loss = close - atr * 1.5 if action == "buy" else close + atr * 1.5
        tp1 = close + atr if action == "buy" else close - atr
        process_trade_signal({
            "symbol": "SPY", "action": action, "price": close,
            "stop_loss": stop_loss, "take_profit_1": tp1, "mode": config.TRADING_MODE
        })
    except Exception as e:
        traceback.print_exc()
        discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "ERROR in strategy check: " + str(e))

def run_crypto_strategy_check():
    global system_halted, simulated_crypto_position
    try:
        if system_halted:
            return
        df5 = yf.download("BTC-USD", period="5d", interval="5m", progress=False)
        df15 = yf.download("BTC-USD", period="5d", interval="15m", progress=False)
        df60 = yf.download("BTC-USD", period="10d", interval="60m", progress=False)
        if len(df5) < 30 or len(df15) < 30 or len(df60) < 30:
            return
        df5 = calculate_indicators(df5)
        df15 = calculate_indicators(df15)
        df60 = calculate_indicators(df60)
        last = df5.iloc[-1]
        current_price = float(last['Close'])
        if simulated_crypto_position is not None:
            pos = simulated_crypto_position
            hit_target = False
            hit_stop = False
            if pos['direction'] == "buy":
                if current_price >= pos['take_profit']:
                    hit_target = True
                elif current_price <= pos['stop_loss']:
                    hit_stop = True
            else:
                if current_price <= pos['take_profit']:
                    hit_target = True
                elif current_price >= pos['stop_loss']:
                    hit_stop = True
            if hit_target or hit_stop:
                result = "WIN" if hit_target else "LOSS"
                pnl = (current_price - pos['entry']) if pos['direction'] == "buy" else (pos['entry'] - current_price)
                pnl_pct = (pnl / pos['entry']) * 100
                discord_post(config.DISCORD_WEBHOOK_PAPER_TRADES,
                    content="[CRYPTO-SANDBOX RESULT] " + result + " - " + pos['direction'].upper() +
                            " BTC-USD entered @ $" + str(round(pos['entry'], 2)) +
                            ", closed @ $" + str(round(current_price, 2)) +
                            " | Simulated P&L: $" + str(round(pnl, 2)) + " (" + str(round(pnl_pct, 2)) + "%)")
                simulated_crypto_position = None
            return
        htf_bullish = float(df15.iloc[-1]['ema9']) > float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) > float(df60.iloc[-1]['ema21'])
        htf_bearish = float(df15.iloc[-1]['ema9']) < float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) < float(df60.iloc[-1]['ema21'])
        vol_spike = float(last['Volume']) > float(last['vol_avg']) * 1.5
        long_cond = (float(last['ema9']) > float(last['ema21']) and current_price > float(last['vwap']) and
                     45 < float(last['rsi']) < 65 and vol_spike and htf_bullish)
        short_cond = (float(last['ema9']) < float(last['ema21']) and current_price < float(last['vwap']) and
                      35 < float(last['rsi']) < 55 and vol_spike and htf_bearish)
        if not long_cond and not short_cond:
            return
        atr = float(last['atr'])
        direction = "buy" if long_cond else "sell"
        stop_loss = current_price - atr * 1.5 if direction == "buy" else current_price + atr * 1.5
        take_profit = current_price + atr if direction == "buy" else current_price - atr
        simulated_crypto_position = {"direction": direction, "entry": current_price, "stop_loss": stop_loss, "take_profit": take_profit}
        discord_post(config.DISCORD_WEBHOOK_PAPER_TRADES,
            content="[CRYPTO-SANDBOX SIGNAL] " + direction.upper() + " BTC-USD @ $" + str(round(current_price, 2)) +
                    " | Stop: $" + str(round(stop_loss, 2)) + " | Target: $" + str(round(take_profit, 2)) +
                    " (tracking hypothetical outcome...)")
    except Exception as e:
        traceback.print_exc()
        discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "ERROR in crypto check: " + str(e))

def market_open_alert():
    discord_post(config.DISCORD_WEBHOOK_MARKET_HOURS, "Market is now OPEN")

def market_close_alert():
    discord_post(config.DISCORD_WEBHOOK_MARKET_HOURS, "Market is now CLOSED")

scheduler = BackgroundScheduler(timezone=pytz.timezone("America/New_York"))
scheduler.add_job(run_strategy_check, "cron", day_of_week="mon-fri", minute="*/5", hour="9-15")
scheduler.add_job(run_crypto_strategy_check, "cron", day_of_week="sat,sun", minute="*/5")
scheduler.add_job(market_open_alert, "cron", day_of_week="mon-fri", hour=9, minute=30)
scheduler.add_job(market_close_alert, "cron", day_of_week="mon-fri", hour=16, minute=0)
scheduler.start()

def run_bot():
    if config.DISCORD_BOT_TOKEN:
        bot.run(config.DISCORD_BOT_TOKEN)

bot_thread = threading.Thread(target=run_bot, daemon=True)
bot_thread.start()

if __name__ == "__main__":
    discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "Webhook server started.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
