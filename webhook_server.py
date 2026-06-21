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
        print("Discord post failed: " + str(e))

def get_alpaca_creds():
    if config.TRADING_MODE == "live":
        return config.ALPACA_LIVE_KEY, config.ALPACA_LIVE_SECRET, config.ALPACA_LIVE_URL
    return config.ALPACA_PAPER_KEY, config.ALPACA_PAPER_SECRET, config.ALPACA_PAPER_URL

@app.route("/webhook", methods=["POST"])
def webhook():
    global trades_today, daily_pnl, consecutive_losses, system_halted

    if system_halted:
        discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "Webhook received but system is HALTED. Trade skipped.")
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
            discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "Mode mismatch detected. Trade skipped.")
            return jsonify({"status": "mode_mismatch_skipped"}), 200

        if len(trades_today) >= config.MAX_TRADES_PER_DAY:
            discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "Daily trade limit reached. Skipping.")
            return jsonify({"status": "max_trades_reached"}), 200

        key, secret, base_url = get_alpaca_creds()
        if not key or not secret:
            discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "Missing Alpaca credentials. Trade skipped.")
            return jsonify({"status": "missing_creds"}), 200

        if config.TRADING_MODE == "live":
            discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "LIVE TRADING MODE ACTIVATED - real money is now being used")

        side = "buy" if action == "buy" else "sell"
        qty = 1

        order = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "order_class": "bracket",
            "take_profit": {"limit_price": round(tp1, 2)},
            "stop_loss": {"stop_price": round(stop_loss, 2)}
        }
        headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        resp = requests.post(base_url + "/v2/orders", json=order, headers=headers, timeout=10)

        tag = "[LIVE]" if config.TRADING_MODE == "live" else "[PAPER]"
        timestamp = datetime.datetime.now().isoformat()
        print(tag + " " + timestamp + " " + side.upper() + " " + symbol + " @ " + str(price) + " -> " + str(resp.status_code))

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

        return jsonify({"status": "order_placed"}), 200

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

if __name__ == "__main__":
    discord_post(config.DISCORD_WEBHOOK_SYSTEM_STATUS, "Webhook server started.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
