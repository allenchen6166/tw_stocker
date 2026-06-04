"""
台股預測平台 - 本機網頁伺服器
執行: python3 app.py
開啟: http://localhost:5001
"""

import os
import shutil
import subprocess
import threading
from flask import Flask, render_template_string, request, send_file, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiYWxsZW5jaGVuNjE2NiIsImVtYWlsIjoiYWxsZW5jaGVuNjE2NkBnbWFpbC5jb20iLCJ0b2tlbl92ZXJzaW9uIjowfQ.vZcZbZVkEUpnWJ1yPct6utMzGyEkOGRXn0FNt-bGz3w"

run_status = {
    "running": False,
    "message": "",
    "log": "",
    "report_type": "market"
}


def run_analysis(cmd, label, report_type="market"):
    run_status["running"] = True
    run_status["message"] = label
    run_status["log"] = ""
    run_status["report_type"] = report_type
    env = os.environ.copy()
    env["FINMIND_TOKEN"] = FINMIND_TOKEN

    try:
        proc = subprocess.Popen(
            cmd, cwd=BASE_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in proc.stdout:
            run_status["log"] += line
        proc.wait()

        if proc.returncode == 0:
            src = os.path.join(BASE_DIR, "stock_report.html")
            if report_type == "single":
                dst = os.path.join(BASE_DIR, "single_report.html")
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                # 還原全市場報表
                bak = os.path.join(BASE_DIR, "stock_report_backup.html")
                if os.path.exists(bak):
                    shutil.copy2(bak, src)
            else:
                # 備份全市場報表
                bak = os.path.join(BASE_DIR, "stock_report_backup.html")
                if os.path.exists(src):
                    shutil.copy2(src, bak)
            run_status["message"] = "完成"
        else:
            run_status["message"] = "失敗"
    except Exception as e:
        run_status["message"] = f"錯誤: {e}"
    finally:
        run_status["running"] = False


PORTAL_HTML = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>台股預測平台</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{background:#0d1117;color:#e6edf3;font-family:-apple-system,sans-serif}
    .header{background:#161b22;border-bottom:1px solid #30363d;padding:1rem 2rem;display:flex;align-items:center;gap:1rem}
    .header h1{font-size:1.3rem;color:#58a6ff}
    .container{max-width:960px;margin:1.5rem auto;padding:0 1rem}
    .card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:1.25rem;margin-bottom:1.25rem}
    .card h2{font-size:0.85rem;color:#8b949e;margin-bottom:0.75rem;text-transform:uppercase;letter-spacing:.05em}
    .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
    input[type=text]{flex:1;min-width:200px;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;padding:9px 12px;font-size:0.95rem}
    input[type=text]:focus{outline:none;border-color:#58a6ff}
    .btn{border:none;border-radius:6px;padding:9px 18px;font-size:0.9rem;cursor:pointer;color:#fff;font-weight:500}
    .btn:disabled{opacity:.4;cursor:not-allowed}
    .btn-green{background:#238636}.btn-green:hover:not(:disabled){background:#2ea043}
    .btn-blue{background:#1f6feb}.btn-blue:hover:not(:disabled){background:#388bfd}
    .btn-purple{background:#6e40c9}.btn-purple:hover:not(:disabled){background:#8250df}
    .btn-gray{background:#21262d;border:1px solid #30363d}.btn-gray:hover:not(:disabled){background:#30363d}
    .hint{color:#8b949e;font-size:0.8rem;margin-top:6px}
    .chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
    .chip{background:#21262d;border:1px solid #30363d;color:#cdd9e5;border-radius:20px;padding:4px 12px;font-size:0.8rem;cursor:pointer}
    .chip:hover{border-color:#58a6ff;color:#58a6ff}
    #status{padding:10px 14px;border-radius:6px;font-size:0.88rem;margin-bottom:8px}
    .idle{background:#1c2128;border:1px solid #30363d;color:#8b949e}
    .running{background:#0d2137;border:1px solid #1f6feb;color:#58a6ff}
    .done{background:#0d2c1c;border:1px solid #238636;color:#3fb950}
    .err{background:#2c0d0d;border:1px solid #b91c1c;color:#f85149}
    #log{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px 10px;height:160px;overflow-y:auto;font-family:monospace;font-size:0.76rem;color:#8b949e;display:none;margin-bottom:8px}
    .tabs{display:flex;gap:6px;margin-bottom:8px}
    .tab{padding:7px 16px;border-radius:6px;font-size:0.85rem;cursor:pointer;border:1px solid #30363d;color:#8b949e;background:#21262d}
    .tab.active{border-color:#58a6ff;color:#58a6ff;background:#0d2137}
    iframe{width:100%;height:78vh;border:none;border-radius:8px;border:1px solid #30363d}
    .spin{display:inline-block;width:11px;height:11px;border:2px solid #58a6ff;border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;margin-right:5px;vertical-align:middle}
    @keyframes spin{to{transform:rotate(360deg)}}
  </style>
</head>
<body>
<div class="header">
  <h1>🎯 台股預測平台</h1>
  <span style="color:#8b949e;font-size:0.82rem">本機版</span>
</div>
<div class="container">

  <div class="card">
    <h2>📊 全市場每日選股</h2>
    <div class="row">
      <button class="btn btn-blue" id="btn-full" onclick="runFull()">更新今日報表</button>
    </div>
    <p class="hint">從全市場篩選成交額 Top-50，選出評分最高的 Top-15</p>
  </div>

  <div class="card">
    <h2>🔍 指定股票分析</h2>
    <div class="row">
      <input type="text" id="tickers" placeholder="輸入股票代號，多檔用空格分隔，例：2330 2454 2317">
      <button class="btn btn-green" id="btn-single" onclick="runSingle()">開始分析</button>
    </div>
    <div class="chips">
      <span class="chip" onclick="pick('2330')">2330 台積電</span>
      <span class="chip" onclick="pick('2454')">2454 聯發科</span>
      <span class="chip" onclick="pick('2317')">2317 鴻海</span>
      <span class="chip" onclick="pick('2382')">2382 廣達</span>
      <span class="chip" onclick="pick('2881')">2881 富邦金</span>
      <span class="chip" onclick="pick('2882')">2882 國泰金</span>
      <span class="chip" onclick="pick('3481')">3481 群創</span>
      <span class="chip" onclick="pick('2308')">2308 台達電</span>
      <span class="chip" onclick="pick('2603')">2603 長榮</span>
      <span class="chip" onclick="pick('6770')">6770 力積電</span>
    </div>
  </div>

  <div id="status" class="idle">⏸ 待機中</div>
  <div id="log"></div>

  <div class="card">
    <div class="tabs">
      <div class="tab active" id="tab-market" onclick="switchTab('market')">📊 全市場報表</div>
      <div class="tab" id="tab-single" onclick="switchTab('single')">🔍 單股報表</div>
      <button class="btn btn-gray" style="margin-left:auto" onclick="reload()">🔄 重新載入</button>
    </div>
    <iframe id="frame" src="/report"></iframe>
  </div>

</div>
<script>
var curTab = 'market';
var polling = false;

function pick(code) {
  document.getElementById('tickers').value = code;
}

function switchTab(tab) {
  curTab = tab;
  document.getElementById('tab-market').className = 'tab' + (tab==='market'?' active':'');
  document.getElementById('tab-single').className = 'tab' + (tab==='single'?' active':'');
  reload();
}

function reload() {
  document.getElementById('frame').src = (curTab==='single' ? '/single_report' : '/report') + '?t=' + Date.now();
}

function setStatus(html, cls) {
  var el = document.getElementById('status');
  el.className = cls;
  el.innerHTML = html;
}

function setBtns(dis) {
  document.getElementById('btn-full').disabled = dis;
  document.getElementById('btn-single').disabled = dis;
}

function startPolling() {
  if (polling) return;
  polling = true;
  poll();
}

function poll() {
  fetch('/status')
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (d.running) {
        setStatus('<span class="spin"></span>執行中... ' + d.message, 'running');
        if (d.log) {
          var log = document.getElementById('log');
          log.style.display = 'block';
          log.innerHTML = d.log.replace(/\\n/g,'<br>');
          log.scrollTop = log.scrollHeight;
        }
        setTimeout(poll, 2000);
      } else if (d.message === '完成') {
        setStatus('✅ 完成！報表已更新', 'done');
        setBtns(false);
        polling = false;
        if (d.report_type === 'single') {
          switchTab('single');
        } else {
          switchTab('market');
        }
      } else if (d.message.indexOf('失敗') >= 0 || d.message.indexOf('錯誤') >= 0) {
        setStatus('❌ ' + d.message, 'err');
        setBtns(false);
        polling = false;
      } else {
        // 剛啟動還沒跑，稍後再查
        setTimeout(poll, 800);
      }
    })
    .catch(function(){ setTimeout(poll, 2000); });
}

function runFull() {
  setBtns(true);
  document.getElementById('log').innerHTML = '';
  document.getElementById('log').style.display = 'none';
  setStatus('<span class="spin"></span>送出請求...', 'running');
  fetch('/run/full', {method:'POST'})
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (d.ok) startPolling();
      else { setStatus('❌ ' + (d.error||'失敗'), 'err'); setBtns(false); }
    });
}

function runSingle() {
  var t = document.getElementById('tickers').value.trim();
  if (!t) { alert('請輸入股票代號'); return; }
  setBtns(true);
  document.getElementById('log').innerHTML = '';
  document.getElementById('log').style.display = 'none';
  setStatus('<span class="spin"></span>送出請求...', 'running');
  fetch('/run/single', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({tickers: t})
  })
    .then(function(r){ return r.json(); })
    .then(function(d){
      if (d.ok) startPolling();
      else { setStatus('❌ ' + (d.error||'失敗'), 'err'); setBtns(false); }
    });
}
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(PORTAL_HTML)


@app.route('/report')
def report():
    p = os.path.join(BASE_DIR, 'stock_report.html')
    return send_file(p) if os.path.exists(p) else ("<p style='color:#888;padding:2rem'>尚無全市場報表，請先執行分析。</p>", 200)


@app.route('/single_report')
def single_report():
    p = os.path.join(BASE_DIR, 'single_report.html')
    return send_file(p) if os.path.exists(p) else ("<p style='color:#888;padding:2rem'>尚無單股報表，請先輸入股票代號分析。</p>", 200)


@app.route('/status')
def status():
    return jsonify(run_status)


@app.route('/run/full', methods=['POST'])
def api_run_full():
    if run_status["running"]:
        return jsonify({"error": "已在執行中"}), 400
    cmd = [
        "python3", "ai_report.py",
        "--regime-filter", "--regime-graduated", "--breadth-regime",
        "--regime-floor", "0.10", "--tp-atr", "4.0", "--sl-atr", "3.0",
        "--hold-days", "20", "--top-k", "15", "--gap-filter", "1.5",
        "--days", "365", "--inst-flow", "1.0"
    ]
    threading.Thread(target=run_analysis, args=(cmd, "全市場選股分析中...", "market"), daemon=True).start()
    return jsonify({"ok": True})


@app.route('/run/single', methods=['POST'])
def api_run_single():
    if run_status["running"]:
        return jsonify({"error": "已在執行中"}), 400
    data = request.get_json()
    tickers = data.get("tickers", "").strip().split()
    if not tickers:
        return jsonify({"error": "未提供股票代號"}), 400
    cmd = [
        "python3", "ai_report.py",
        "--tickers", *tickers, "--static-pool",
        "--regime-filter", "--regime-graduated", "--breadth-regime",
        "--regime-floor", "0.10", "--tp-atr", "4.0", "--sl-atr", "3.0",
        "--hold-days", "20", "--top-k", "99", "--gap-filter", "1.5",
        "--days", "365", "--inst-flow", "1.0"
    ]
    label = f"分析 {' '.join(tickers)} 中..."
    threading.Thread(target=run_analysis, args=(cmd, label, "single"), daemon=True).start()
    return jsonify({"ok": True})


if __name__ == '__main__':
    print("🚀 台股預測平台啟動中...")
    print("📊 請開啟瀏覽器前往：http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
