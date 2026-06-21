from flask import Flask, request, jsonify
import requests, json, datetime, traceback, os
import config

app = Flask(__name__)

trades_today = []
daily_pnl = 0.0
consecutive_losses = 0
system_halted = False

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
        print(f"Discord post failed: {e}")

def get_alpaca_creds():
    if config.TRADING_MODE == "live":
        return config.ALPACA_LIVE_KEY, config.ALPACA_LIVE_SECRET, config.ALPACA_LIVE_URL
    return config.ALPACA_PAPER_KEY, config.ALPACA_PAPER_SECRET, config.ALPACA_PAPER_URL

@app.route("/webhook", methods=["POST"])
def webhook():
    global trades_today, daily_pnl, consecutive_losses, system_halted

    if system_halted:
        discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS,
                      "Webhook received but system is HALTED. Trade skipped.")
        return jsonify({"status": "halted"}), 200

    try:
        data = request.get_json(force=True)
        action = data.get("action")
        symbol = data.get("symbol")
        payload_mode = data.get("mode")
        price = float(data.get("price", 0))
        stop_loss = float(data.get("stop_loss", 0))
        tp1 = float(data.get("take_profit_1", 0))

        if payload_mode != config.TRADING_MODE:
            discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS,
                f"Mode