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
manual_position = None

# Tracks the timestamp of the last CLOSED candle that already triggered a
# signal, so the lookback window below never fires twice on the same bar.
last_spy_signal_time = None
last_btc_signal_time = None

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
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
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

def get_current_price_and_atr(symbol):
    # Used by manual /buy /sell /close /position — these want the live price
    # at the moment of execution, so the forming candle is fine to use here.
    try:
        df = yf.download(symbol, period="5d", interval="5m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) < 20:
            return None, None
        df = calculate_indicators(df)
        last = df.iloc[-1]
        return float(last['Close']), float(last['atr'])
    except Exception as e:
        print("Error getting price/atr: " + str(e))
        return None, None

def get_spy_condition_details():
    # Powers /check. Reads the last FULLY CLOSED candle so what the user sees
    # always matches exactly what the automated checker is acting on.
    try:
        df5 = yf.download("SPY", period="5d", interval="5m", progress=False)
        df15 = yf.download("SPY", period="5d", interval="15m", progress=False)
        df60 = yf.download("SPY", period="10d", interval="60m", progress=False)
        if len(df5) < 31 or len(df15) < 30 or len(df60) < 30:
            return None
        df5 = calculate_indicators(df5)
        df15 = calculate_indicators(df15)
        df60 = calculate_indicators(df60)
        closed5 = df5.iloc[:-1]
        last = closed5.iloc[-1]
        ny_tz = pytz.timezone("America/New_York")
        now = datetime.datetime.now(ny_tz)
        today_session = closed5[closed5.index.date == now.date()]
        or_window = today_session.between_time("09:30", "10:00")
        or_high = float(or_window['High'].max()) if len(or_window) > 0 else None
        or_low = float(or_window['Low'].min()) if len(or_window) > 0 else None
        return {
            "close": float(last['Close']), "ema9": float(last['ema9']),
            "ema21": float(last['ema21']), "rsi": float(last['rsi']),
            "vwap": float(last['vwap']), "volume": float(last['Volume']),
            "vol_avg": float(last['vol_avg']), "atr": float(last['atr']),
            "or_high": or_high, "or_low": or_low,
            "candle_time": last.name,
            "htf_bull": float(df15.iloc[-1]['ema9']) > float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) > float(df60.iloc[-1]['ema21']),
            "htf_bear": float(df15.iloc[-1]['ema9']) < float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) < float(df60.iloc[-1]['ema21'])
        }
    except Exception as e:
        print("Error getting SPY conditions: " + str(e))
        return None

def get_btc_condition_details():
    try:
        df5 = yf.download("BTC-USD", period="5d", interval="5m", progress=False)
        df15 = yf.download("BTC-USD", period="5d", interval="15m", progress=False)
        df60 = yf.download("BTC-USD", period="10d", interval="60m", progress=False)
        if len(df5) < 31 or len(df15) < 30 or len(df60) < 30:
            return None
        df5 = calculate_indicators(df5)
        df15 = calculate_indicators(df15)
        df60 = calculate_indicators(df60)
        closed5 = df5.iloc[:-1]
        last = closed5.iloc[-1]
        return {
            "close": float(last['Close']), "ema9": float(last['ema9']),
            "ema21": float(last['ema21']), "rsi": float(last['rsi']),
            "vwap": float(last['vwap']), "volume": float(last['Volume']),
            "vol_avg": float(last['vol_avg']), "atr": float(last['atr']),
            "candle_time": last.name,
            "htf_bull": float(df15.iloc[-1]['ema9']) > float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) > float(df60.iloc[-1]['ema21']),
            "htf_bear": float(df15.iloc[-1]['ema9']) < float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) < float(df60.iloc[-1]['ema21'])
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
    side = "buy" if action == "buy" else "sell"
    qty = 1
    order = {
        "symbol": symbol, "qty": qty, "side": side, "type": "market", "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": round(tp1, 2)},
        "stop_loss": {"stop_price": round(stop_loss, 2)}
    }
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    requests.post(base_url + "/v2/orders", json=order, headers=headers, timeout=10)
    tag = "[LIVE]" if config.TRADING_MODE == "live" else "[PAPER]"
    timestamp = datetime.datetime.now().isoformat()
    trades_today.append({"symbol": symbol, "side": side, "price": price, "time": timestamp})
    color = 0x2ecc71 if side == "buy" else 0xe74c3c
    embed = {
        "title": ("🟢 " if side == "buy" else "🔴 ") + tag + " " + side.upper() + " " + symbol,
        "color": color,
        "fields": [
            {"name": "💰 Entry Price", "value": "$" + str(round(price, 2)), "inline": True},
            {"name": "🛑 Stop Loss", "value": "$" + str(round(stop_loss, 2)), "inline": True},
            {"name": "🎯 Take Profit", "value": "$" + str(round(tp1, 2)), "inline": True},
            {"name": "📋 Mode", "value": config.TRADING_MODE.upper(), "inline": True},
            {"name": "🕐 Time", "value": timestamp[:19], "inline": True}
        ],
        "footer": {"text": "Momentum Confluence Scalper"}
    }
    target_channel = config.DISCORD_WEBHOOK_LIVE_TRADES if config.TRADING_MODE == "live" else config.DISCORD_WEBHOOK_PAPER_TRADES
    discord_post(target_channel, embed=embed)
    discord_post(config.DISCORD_WEBHOOK_TRADE_ALERTS, embed=embed)

async def send_embed(ctx, title, color, fields, footer="Momentum Confluence Scalper"):
    embed = discord.Embed(title=title, color=color, timestamp=datetime.datetime.utcnow())
    for name, value, inline in fields:
        embed.add_field(name=name, value=value, inline=inline)
    embed.set_footer(text=footer)
    await ctx.send(embed=embed)

# ====== DISCORD COMMANDS ======

@bot.command(name="status")
async def cmd_status(ctx):
    halted_text = "🔴 HALTED" if system_halted else "🟢 Running"
    mode_color = 0xe74c3c if config.TRADING_MODE == "live" else 0x3498db
    await send_embed(ctx, "⚙️ System Status", mode_color, [
        ("🔁 Mode", config.TRADING_MODE.upper(), True),
        ("📡 Status", halted_text, True),
        ("📊 Trades Today", str(len(trades_today)) + " / " + str(config.MAX_TRADES_PER_DAY), True),
        ("💵 Simulated P&L", "$" + str(round(daily_pnl, 2)), True),
        ("🤖 Bot", "Online", True),
        ("⏰ Checked At", datetime.datetime.now(pytz.timezone("America/New_York")).strftime("%I:%M %p ET"), True)
    ])

@bot.command(name="check")
async def cmd_check(ctx):
    checking = discord.Embed(title="🔍 Checking SPY...", color=0xf39c12, description="Fetching the last closed candle, give me a few seconds...")
    await ctx.send(embed=checking)
    d = get_spy_condition_details()
    if d is None:
        await send_embed(ctx, "❌ SPY Data Unavailable", 0xe74c3c, [("Error", "Could not fetch SPY data. Market may be closed or try again in a minute.", False)])
        return
    def yn(val): return "✅ YES" if val else "❌ NO"
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
    long_color = 0x2ecc71 if long_score == 6 else 0xf39c12 if long_score >= 4 else 0xe74c3c
    await send_embed(ctx, "📈 SPY Signal Check — $" + str(round(d['close'], 2)), long_color, [
        ("🟢 LONG Score", str(long_score) + "/6 passing", False),
        ("EMA9 > EMA21", yn(ema_bull), True),
        ("Price > VWAP", yn(above_vwap), True),
        ("RSI 45-65", yn(rsi_bull) + " (" + str(round(d['rsi'], 1)) + ")", True),
        ("Volume Spike", yn(vol_ok), True),
        ("Above OR High", yn(or_bull), True),
        ("HTF Bullish", yn(d['htf_bull']), True),
        ("🔴 SHORT Score", str(short_score) + "/6 passing", False),
        ("EMA9 < EMA21", yn(ema_bear), True),
        ("Price < VWAP", yn(below_vwap), True),
        ("RSI 35-55", yn(rsi_bear) + " (" + str(round(d['rsi'], 1)) + ")", True),
        ("Volume Spike", yn(vol_ok), True),
        ("Below OR Low", yn(or_bear), True),
        ("HTF Bearish", yn(d['htf_bear']), True),
        ("📍 Candle Used", str(d['candle_time'])[:16] + " (last closed bar)", False),
        ("⚡ Trigger", "Need ALL 6 to fire a trade", False)
    ])

@bot.command(name="btccheck")
async def cmd_btccheck(ctx):
    checking = discord.Embed(title="🔍 Checking BTC-USD...", color=0xf39c12, description="Fetching the last closed candle, give me a few seconds...")
    await ctx.send(embed=checking)
    d = get_btc_condition_details()
    if d is None:
        await send_embed(ctx, "❌ BTC Data Unavailable", 0xe74c3c, [("Error", "Could not fetch BTC data. Try again in a minute.", False)])
        return
    def yn(val): return "✅ YES" if val else "❌ NO"
    ema_bull = d['ema9'] > d['ema21']
    ema_bear = d['ema9'] < d['ema21']
    above_vwap = d['close'] > d['vwap']
    below_vwap = d['close'] < d['vwap']
    rsi_bull = 45 < d['rsi'] < 65
    rsi_bear = 35 < d['rsi'] < 55
    vol_ok = d['volume'] > d['vol_avg'] * 1.5
    long_score = sum([ema_bull, above_vwap, rsi_bull, vol_ok, d['htf_bull']])
    short_score = sum([ema_bear, below_vwap, rsi_bear, vol_ok, d['htf_bear']])
    btc_color = 0x2ecc71 if long_score == 5 or short_score == 5 else 0xf39c12 if long_score >= 3 or short_score >= 3 else 0xe74c3c
    await send_embed(ctx, "₿ BTC-USD Signal Check — $" + str(round(d['close'], 2)), btc_color, [
        ("🟢 LONG Score", str(long_score) + "/5 passing", False),
        ("EMA9 > EMA21", yn(ema_bull), True),
        ("Price > VWAP", yn(above_vwap), True),
        ("RSI 45-65", yn(rsi_bull) + " (" + str(round(d['rsi'], 1)) + ")", True),
        ("Volume Spike", yn(vol_ok), True),
        ("HTF Bullish", yn(d['htf_bull']), True),
        ("🔴 SHORT Score", str(short_score) + "/5 passing", False),
        ("EMA9 < EMA21", yn(ema_bear), True),
        ("Price < VWAP", yn(below_vwap), True),
        ("RSI 35-55", yn(rsi_bear) + " (" + str(round(d['rsi'], 1)) + ")", True),
        ("Volume Spike", yn(vol_ok), True),
        ("HTF Bearish", yn(d['htf_bear']), True),
        ("📍 Candle Used", str(d['candle_time'])[:16] + " (last closed bar)", False),
        ("⚡ Trigger", "Need ALL 5 to fire a sandbox signal", False)
    ])

@bot.command(name="buy")
async def cmd_buy(ctx):
    global manual_position
    if manual_position is not None:
        await send_embed(ctx, "⚠️ Position Already Open", 0xe74c3c, [
            ("Active Position", manual_position['direction'].upper() + " " + manual_position['symbol'], True),
            ("Entry", "$" + str(round(manual_position['entry'], 2)), True),
            ("Tip", "Use /close to close it first", False)
        ])
        return
    ny_tz = pytz.timezone("America/New_York")
    now = datetime.datetime.now(ny_tz)
    symbol = "BTC-USD" if now.weekday() >= 5 else "SPY"
    await ctx.send(embed=discord.Embed(title="⏳ Opening BUY...", color=0x2ecc71, description="Fetching current price and ATR for " + symbol + "..."))
    price, atr = get_current_price_and_atr(symbol)
    if price is None:
        await send_embed(ctx, "❌ Failed to Open BUY", 0xe74c3c, [("Error", "Could not fetch price data. Try again.", False)])
        return
    stop_loss = price - atr * 1.5
    take_profit = price + atr * 3.0
    manual_position = {
        "symbol": symbol, "direction": "buy",
        "entry": price, "stop_loss": stop_loss,
        "take_profit": take_profit, "time": now.isoformat()
    }
    await send_embed(ctx, "🟢 MANUAL BUY OPENED — " + symbol, 0x2ecc71, [
        ("💰 Entry Price", "$" + str(round(price, 2)), True),
        ("🛑 Stop Loss", "$" + str(round(stop_loss, 2)), True),
        ("🎯 Take Profit", "$" + str(round(take_profit, 2)), True),
        ("📉 Risk", "$" + str(round(price - stop_loss, 2)) + " per unit", True),
        ("📈 Reward", "$" + str(round(take_profit - price, 2)) + " per unit", True),
        ("⚖️ R:R Ratio", "1:2", True),
        ("📋 Mode", "SANDBOX — not sent to Alpaca", False),
        ("ℹ️ Info", "This is tracked separately from the auto-bot. Use /close to exit manually — it does not auto-close.", False)
    ])
    discord_post(config.DISCORD_WEBHOOK_PAPER_TRADES, embed={
        "title": "🟢 MANUAL BUY — " + symbol,
        "color": 0x2ecc71,
        "fields": [
            {"name": "💰 Entry", "value": "$" + str(round(price, 2)), "inline": True},
            {"name": "🛑 Stop", "value": "$" + str(round(stop_loss, 2)), "inline": True},
            {"name": "🎯 Target", "value": "$" + str(round(take_profit, 2)), "inline": True}
        ],
        "footer": {"text": "Manual trade — sandbox tracking"}
    })

@bot.command(name="sell")
async def cmd_sell(ctx):
    global manual_position
    if manual_position is not None:
        await send_embed(ctx, "⚠️ Position Already Open", 0xe74c3c, [
            ("Active Position", manual_position['direction'].upper() + " " + manual_position['symbol'], True),
            ("Entry", "$" + str(round(manual_position['entry'], 2)), True),
            ("Tip", "Use /close to close it first", False)
        ])
        return
    ny_tz = pytz.timezone("America/New_York")
    now = datetime.datetime.now(ny_tz)
    symbol = "BTC-USD" if now.weekday() >= 5 else "SPY"
    await ctx.send(embed=discord.Embed(title="⏳ Opening SELL...", color=0xe74c3c, description="Fetching current price and ATR for " + symbol + "..."))
    price, atr = get_current_price_and_atr(symbol)
    if price is None:
        await send_embed(ctx, "❌ Failed to Open SELL", 0xe74c3c, [("Error", "Could not fetch price data. Try again.", False)])
        return
    stop_loss = price + atr * 1.5
    take_profit = price - atr * 3.0
    manual_position = {
        "symbol": symbol, "direction": "sell",
        "entry": price, "stop_loss": stop_loss,
        "take_profit": take_profit, "time": now.isoformat()
    }
    await send_embed(ctx, "🔴 MANUAL SELL OPENED — " + symbol, 0xe74c3c, [
        ("💰 Entry Price", "$" + str(round(price, 2)), True),
        ("🛑 Stop Loss", "$" + str(round(stop_loss, 2)), True),
        ("🎯 Take Profit", "$" + str(round(take_profit, 2)), True),
        ("📉 Risk", "$" + str(round(stop_loss - price, 2)) + " per unit", True),
        ("📈 Reward", "$" + str(round(price - take_profit, 2)) + " per unit", True),
        ("⚖️ R:R Ratio", "1:2", True),
        ("📋 Mode", "SANDBOX — not sent to Alpaca", False),
        ("ℹ️ Info", "This is tracked separately from the auto-bot. Use /close to exit manually — it does not auto-close.", False)
    ])
    discord_post(config.DISCORD_WEBHOOK_PAPER_TRADES, embed={
        "title": "🔴 MANUAL SELL — " + symbol,
        "color": 0xe74c3c,
        "fields": [
            {"name": "💰 Entry", "value": "$" + str(round(price, 2)), "inline": True},
            {"name": "🛑 Stop", "value": "$" + str(round(stop_loss, 2)), "inline": True},
            {"name": "🎯 Target", "value": "$" + str(round(take_profit, 2)), "inline": True}
        ],
        "footer": {"text": "Manual trade — sandbox tracking"}
    })

@bot.command(name="close")
async def cmd_close(ctx):
    global manual_position
    if manual_position is None:
        await send_embed(ctx, "⚠️ No Open Position", 0xf39c12, [("Info", "You don't have a manual position open right now.", False)])
        return
    pos = manual_position
    price, _ = get_current_price_and_atr(pos['symbol'])
    if price is None:
        await send_embed(ctx, "❌ Could Not Fetch Price", 0xe74c3c, [("Error", "Try again in a moment.", False)])
        return
    if pos['direction'] == "buy":
        pnl = price - pos['entry']
    else:
        pnl = pos['entry'] - price
    pnl_pct = (pnl / pos['entry']) * 100
    result = "WIN 🏆" if pnl > 0 else "LOSS 📉"
    color = 0x2ecc71 if pnl > 0 else 0xe74c3c
    manual_position = None
    await send_embed(ctx, "🔒 Position Closed Manually — " + result, color, [
        ("📊 Symbol", pos['symbol'], True),
        ("📋 Direction", pos['direction'].upper(), True),
        ("💰 Entry", "$" + str(round(pos['entry'], 2)), True),
        ("🏁 Exit Price", "$" + str(round(price, 2)), True),
        ("💵 P&L", ("+" if pnl > 0 else "") + "$" + str(round(pnl, 2)), True),
        ("📈 Return", ("+" if pnl > 0 else "") + str(round(pnl_pct, 2)) + "%", True)
    ])
    discord_post(config.DISCORD_WEBHOOK_PAPER_TRADES, embed={
        "title": "🔒 MANUAL CLOSE — " + result,
        "color": color,
        "fields": [
            {"name": "Symbol", "value": pos['symbol'], "inline": True},
            {"name": "Direction", "value": pos['direction'].upper(), "inline": True},
            {"name": "P&L", "value": ("+" if pnl > 0 else "") + "$" + str(round(pnl, 2)), "inline": True}
        ],
        "footer": {"text": "Manual close"}
    })

@bot.command(name="position")
async def cmd_position(ctx):
    global manual_position
    if manual_position is None:
        await send_embed(ctx, "📭 No Open Position", 0x95a5a6, [("Info", "No manual position is currently open. Use /buy or /sell to open one.", False)])
        return
    pos = manual_position
    price, _ = get_current_price_and_atr(pos['symbol'])
    if price is None:
        await send_embed(ctx, "⚠️ Position Open — Price Unavailable", 0xf39c12, [
            ("Symbol", pos['symbol'], True),
            ("Direction", pos['direction'].upper(), True),
            ("Entry", "$" + str(round(pos['entry'], 2)), True)
        ])
        return
    if pos['direction'] == "buy":
        pnl = price - pos['entry']
    else:
        pnl = pos['entry'] - price
    pnl_pct = (pnl / pos['entry']) * 100
    color = 0x2ecc71 if pnl > 0 else 0xe74c3c
    await send_embed(ctx, "📊 Open Position — " + pos['symbol'], color, [
        ("📋 Direction", pos['direction'].upper(), True),
        ("💰 Entry", "$" + str(round(pos['entry'], 2)), True),
        ("📡 Current Price", "$" + str(round(price, 2)), True),
        ("🛑 Stop Loss", "$" + str(round(pos['stop_loss'], 2)), True),
        ("🎯 Take Profit", "$" + str(round(pos['take_profit'], 2)), True),
        ("💵 Unrealized P&L", ("+" if pnl > 0 else "") + "$" + str(round(pnl, 2)) + " (" + str(round(pnl_pct, 2)) + "%)", True),
        ("🕐 Opened At", pos['time'][:19], False)
    ])

@bot.command(name="trades")
async def cmd_trades(ctx):
    if len(trades_today) == 0:
        await send_embed(ctx, "📭 No Trades Today", 0x95a5a6, [("Info", "No automated trades have been placed today yet.", False)])
        return
    fields = []
    for i, t in enumerate(trades_today):
        fields.append(("Trade " + str(i+1), t['side'].upper() + " " + t['symbol'] + " @ $" + str(round(t['price'], 2)), True))
        fields.append(("Time", t['time'][:19], True))
        fields.append(("\u200b", "\u200b", True))
    await send_embed(ctx, "📋 Trades Today — " + str(len(trades_today)) + "/" + str(config.MAX_TRADES_PER_DAY), 0x3498db, fields)

@bot.command(name="positions")
async def cmd_positions(ctx):
    try:
        key, secret, base_url = get_alpaca_creds()
        headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        resp = requests.get(base_url + "/v2/positions", headers=headers, timeout=10)
        positions = resp.json()
        if not isinstance(positions, list) or len(positions) == 0:
            await send_embed(ctx, "📭 No Open Alpaca Positions", 0x95a5a6, [("Info", "No positions currently open in Alpaca.", False)])
            return
        fields = []
        for p in positions:
            pnl = float(p['unrealized_pl'])
            fields.append(("Symbol", p['symbol'], True))
            fields.append(("Side", p['side'].upper(), True))
            fields.append(("P&L", ("+" if pnl > 0 else "") + "$" + str(round(pnl, 2)), True))
        color = 0x2ecc71 if sum(float(p['unrealized_pl']) for p in positions) > 0 else 0xe74c3c
        await send_embed(ctx, "📊 Alpaca Open Positions", color, fields)
    except Exception as e:
        await send_embed(ctx, "❌ Error Fetching Positions", 0xe74c3c, [("Error", str(e), False)])

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
        color = 0x2ecc71 if day_pnl >= 0 else 0xe74c3c
        await send_embed(ctx, "💵 P&L Summary", color, [
            ("💼 Account Equity", "$" + str(equity), True),
            ("📈 Today's P&L", ("+" if day_pnl > 0 else "") + "$" + str(day_pnl) + " (" + str(pct) + "%)", True),
            ("📊 Trades Today", str(len(trades_today)), True),
            ("📋 Mode", config.TRADING_MODE.upper(), True)
        ])
    except Exception as e:
        await send_embed(ctx, "❌ Error Fetching P&L", 0xe74c3c, [("Error", str(e), False)])

@bot.command(name="halt")
async def cmd_halt(ctx):
    global system_halted
    system_halted = True
    discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, None, {
        "title": "🛑 KILL SWITCH ACTIVATED",
        "color": 0xe74c3c,
        "description": "All trading has been halted via Discord command.",
        "footer": {"text": "Use /resume [password] to restart"}
    })
    await send_embed(ctx, "🛑 Trading HALTED", 0xe74c3c, [("Status", "All automated trading is now paused.", True), ("Resume", "Use /resume [password] to restart", False)])

@bot.command(name="resume")
async def cmd_resume(ctx, password: str = ""):
    global system_halted
    if password != config.MODE_SWITCH_PASSWORD:
        await send_embed(ctx, "❌ Wrong Password", 0xe74c3c, [("Tip", "Use: /resume yourpassword", False)])
        return
    system_halted = False
    discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, None, {
        "title": "✅ Trading Resumed",
        "color": 0x2ecc71,
        "description": "System is back online and monitoring for signals.",
        "footer": {"text": "Momentum Confluence Scalper"}
    })
    await send_embed(ctx, "✅ Trading Resumed", 0x2ecc71, [("Status", "Bot is back online and watching for signals.", False)])

@bot.event
async def on_ready():
    print("Discord bot online: " + str(bot.user))
    discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, None, {
        "title": "🤖 Bot Online",
        "color": 0x2ecc71,
        "description": "All systems running. Commands ready.",
        "fields": [
            {"name": "📊 Commands", "value": "/status /check /btccheck /buy /sell /close /position /trades /positions /pnl /halt /resume", "inline": False}
        ],
        "footer": {"text": "Momentum Confluence Scalper"}
    })

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
    global system_halted, trades_today, last_spy_signal_time
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
        if len(df5) < 31 or len(df15) < 30 or len(df60) < 30:
            return
        df5 = calculate_indicators(df5)
        df15 = calculate_indicators(df15)
        df60 = calculate_indicators(df60)

        # Only ever act on candles that have fully closed.
        closed = df5.iloc[:-1]

        htf_bullish = float(df15.iloc[-1]['ema9']) > float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) > float(df60.iloc[-1]['ema21'])
        htf_bearish = float(df15.iloc[-1]['ema9']) < float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) < float(df60.iloc[-1]['ema21'])

        today_session = closed[closed.index.date == now.date()]
        or_window = today_session.between_time("09:30", "10:00")
        if len(or_window) == 0:
            return
        or_high = float(or_window['High'].max())
        or_low = float(or_window['Low'].min())

        # Look back across the last 3 closed candles so a setup that fires
        # between two scheduler ticks doesn't get silently skipped.
        lookback = closed.tail(3)
        for ts, row in lookback.iterrows():
            if last_spy_signal_time is not None and ts <= last_spy_signal_time:
                continue
            close = float(row['Close'])
            vol_spike = float(row['Volume']) > float(row['vol_avg']) * 1.5
            long_cond = (float(row['ema9']) > float(row['ema21']) and close > float(row['vwap']) and
                         45 < float(row['rsi']) < 65 and vol_spike and close > or_high and htf_bullish)
            short_cond = (float(row['ema9']) < float(row['ema21']) and close < float(row['vwap']) and
                          35 < float(row['rsi']) < 55 and vol_spike and close < or_low and htf_bearish)
            if not long_cond and not short_cond:
                continue
            atr = float(row['atr'])
            action = "buy" if long_cond else "sell"
            stop_loss = close - atr * 1.5 if action == "buy" else close + atr * 1.5
            tp1 = close + atr if action == "buy" else close - atr
            last_spy_signal_time = ts
            process_trade_signal({
                "symbol": "SPY", "action": action, "price": close,
                "stop_loss": stop_loss, "take_profit_1": tp1, "mode": config.TRADING_MODE
            })
            break
    except Exception as e:
        traceback.print_exc()
        discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "ERROR in strategy check: " + str(e))

def run_crypto_strategy_check():
    global system_halted, simulated_crypto_position, last_btc_signal_time
    try:
        if system_halted:
            return
        df5 = yf.download("BTC-USD", period="5d", interval="5m", progress=False)
        df15 = yf.download("BTC-USD", period="5d", interval="15m", progress=False)
        df60 = yf.download("BTC-USD", period="10d", interval="60m", progress=False)
        if len(df5) < 31 or len(df15) < 30 or len(df60) < 30:
            return
        df5 = calculate_indicators(df5)
        df15 = calculate_indicators(df15)
        df60 = calculate_indicators(df60)
        closed = df5.iloc[:-1]

        # Manage an open simulated position first, using High/Low so a stop
        # or target touched intra-candle isn't missed just because price
        # drifted back by the time the next check runs.
        if simulated_crypto_position is not None:
            pos = simulated_crypto_position
            since_entry = closed[closed.index > pos['opened_at']]
            for ts, row in since_entry.iterrows():
                high = float(row['High'])
                low = float(row['Low'])
                hit_target = False
                hit_stop = False
                if pos['direction'] == "buy":
                    if high >= pos['take_profit']:
                        hit_target = True
                    elif low <= pos['stop_loss']:
                        hit_stop = True
                else:
                    if low <= pos['take_profit']:
                        hit_target = True
                    elif high >= pos['stop_loss']:
                        hit_stop = True
                if hit_target or hit_stop:
                    result = "WIN 🏆" if hit_target else "LOSS 📉"
                    exit_price = pos['take_profit'] if hit_target else pos['stop_loss']
                    pnl = (exit_price - pos['entry']) if pos['direction'] == "buy" else (pos['entry'] - exit_price)
                    pnl_pct = (pnl / pos['entry']) * 100
                    color = 0x2ecc71 if hit_target else 0xe74c3c
                    discord_post(config.DISCORD_WEBHOOK_PAPER_TRADES, embed={
                        "title": "🏁 CRYPTO SANDBOX RESULT — " + result,
                        "color": color,
                        "fields": [
                            {"name": "Direction", "value": pos['direction'].upper(), "inline": True},
                            {"name": "Entry", "value": "$" + str(round(pos['entry'], 2)), "inline": True},
                            {"name": "Exit", "value": "$" + str(round(exit_price, 2)), "inline": True},
                            {"name": "P&L", "value": ("+" if pnl > 0 else "") + "$" + str(round(pnl, 2)) + " (" + str(round(pnl_pct, 2)) + "%)", "inline": True}
                        ],
                        "footer": {"text": "Sandbox only — not a real trade"}
                    })
                    simulated_crypto_position = None
                    break
            return

        htf_bullish = float(df15.iloc[-1]['ema9']) > float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) > float(df60.iloc[-1]['ema21'])
        htf_bearish = float(df15.iloc[-1]['ema9']) < float(df15.iloc[-1]['ema21']) and float(df60.iloc[-1]['ema9']) < float(df60.iloc[-1]['ema21'])

        lookback = closed.tail(3)
        for ts, row in lookback.iterrows():
            if last_btc_signal_time is not None and ts <= last_btc_signal_time:
                continue
            current_price = float(row['Close'])
            vol_spike = float(row['Volume']) > float(row['vol_avg']) * 1.5
            long_cond = (float(row['ema9']) > float(row['ema21']) and current_price > float(row['vwap']) and
                         45 < float(row['rsi']) < 65 and vol_spike and htf_bullish)
            short_cond = (float(row['ema9']) < float(row['ema21']) and current_price < float(row['vwap']) and
                          35 < float(row['rsi']) < 55 and vol_spike and htf_bearish)
            if not long_cond and not short_cond:
                continue
            atr = float(row['atr'])
            direction = "buy" if long_cond else "sell"
            stop_loss = current_price - atr * 1.5 if direction == "buy" else current_price + atr * 1.5
            take_profit = current_price + atr if direction == "buy" else current_price - atr
            last_btc_signal_time = ts
            simulated_crypto_position = {
                "direction": direction, "entry": current_price, "stop_loss": stop_loss,
                "take_profit": take_profit, "symbol": "BTC-USD", "opened_at": ts
            }
            discord_post(config.DISCORD_WEBHOOK_PAPER_TRADES, embed={
                "title": ("🟢" if direction == "buy" else "🔴") + " CRYPTO SANDBOX SIGNAL — " + direction.upper() + " BTC-USD",
                "color": 0x2ecc71 if direction == "buy" else 0xe74c3c,
                "fields": [
                    {"name": "💰 Entry", "value": "$" + str(round(current_price, 2)), "inline": True},
                    {"name": "🛑 Stop", "value": "$" + str(round(stop_loss, 2)), "inline": True},
                    {"name": "🎯 Target", "value": "$" + str(round(take_profit, 2)), "inline": True},
                    {"name": "ℹ️ Info", "value": "Tracking hypothetical outcome (candle: " + str(ts)[:16] + ")", "inline": False}
                ],
                "footer": {"text": "Sandbox only — not a real trade"}
            })
            break
    except Exception as e:
        traceback.print_exc()
        discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "ERROR in crypto check: " + str(e))

def market_open_alert():
    discord_post(config.DISCORD_WEBHOOK_MARKET_HOURS, embed={
        "title": "🟢 Market is OPEN",
        "color": 0x2ecc71,
        "description": "SPY trading hours have begun. Bot is now monitoring for signals.",
        "footer": {"text": "Market Hours — 9:35 AM to 3:45 PM ET"}
    })

def market_close_alert():
    discord_post(config.DISCORD_WEBHOOK_MARKET_HOURS, embed={
        "title": "🔴 Market is CLOSED",
        "color": 0xe74c3c,
        "description": "Trading hours have ended. See you tomorrow.",
        "footer": {"text": "Momentum Confluence Scalper"}
    })

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
