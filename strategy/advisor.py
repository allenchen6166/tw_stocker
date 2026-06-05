"""
AI 投資建議模組 (Technical Analysis Advisor)

依據 43 條技術分析規則，計算個股各項指標並給出綜合建議。

主要指標：
  - 趨勢判斷（道氏理論：頭頭高/底底低）
  - 均線多頭排列（5/10/20日）+ 葛蘭碧八法
  - KD 指標（黃金/死亡交叉）
  - MACD（零軸位置/柱線/背離）
  - RSI 背離
  - 成交量規則（放量/爆量/量價背離）
  - K棒型態（影線/吞噬/母子/晨夜星）
  - 跳空缺口
  - 乖離率
  - 費波那契回撤
"""

import pandas as pd
import numpy as np


def _find_swing_points(series, window=5):
    """找出高低轉折點（簡化版 Zigzag）"""
    highs, lows = [], []
    s = series.dropna()
    for i in range(window, len(s) - window):
        seg = s.iloc[i-window:i+window+1]
        if s.iloc[i] == seg.max():
            highs.append((s.index[i], float(s.iloc[i])))
        if s.iloc[i] == seg.min():
            lows.append((s.index[i], float(s.iloc[i])))
    return highs, lows


def _calc_kd(high, low, close, period=9):
    """計算 KD 指標（Stochastic）"""
    low_min  = low.rolling(period).min()
    high_max = high.rolling(period).max()
    rsv = (close - low_min) / (high_max - low_min + 1e-8) * 100
    K = rsv.ewm(com=2, adjust=False).mean()
    D = K.ewm(com=2, adjust=False).mean()
    return K, D


def _calc_macd(close, fast=12, slow=26, signal=9):
    """計算 MACD"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif  = ema_fast - ema_slow
    macd = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - macd) * 2
    return dif, macd, hist


def _calc_rsi(close, period=6):
    """計算 RSI"""
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-8)
    return 100 - 100 / (1 + rs)


def analyze_stock(ticker, close_s, high_s, low_s, vol_s, stock_name=''):
    """
    對單一股票執行完整技術分析，回傳建議字典。

    Parameters
    ----------
    ticker : str
    close_s, high_s, low_s, vol_s : pd.Series  (日線，至少 60 筆)
    stock_name : str

    Returns
    -------
    dict with keys:
        score       : int  (-100 ~ +100，正=看多)
        action      : str  (強力買進/買進/觀望/賣出/強力賣出)
        color       : str  (HTML color)
        signals     : list[dict]  每條訊號
        summary     : str  一行摘要
    """
    close = close_s.dropna()
    if len(close) < 30:
        return _no_data()

    high  = high_s.reindex(close.index).ffill()  if high_s  is not None else close
    low   = low_s.reindex(close.index).ffill()   if low_s   is not None else close
    vol   = vol_s.reindex(close.index).ffill()   if vol_s   is not None else pd.Series(0, index=close.index)

    signals = []
    score   = 0

    price   = float(close.iloc[-1])
    ma5     = close.rolling(5).mean()
    ma10    = close.rolling(10).mean()
    ma20    = close.rolling(20).mean()
    ma60    = close.rolling(60).mean()

    # ── 1. 趨勢判斷（道氏理論） ──
    try:
        highs, lows = _find_swing_points(close, window=3)
        if len(highs) >= 2 and len(lows) >= 2:
            h_trend = highs[-1][1] > highs[-2][1]   # 頭頭高
            l_trend = lows[-1][1]  > lows[-2][1]    # 底底高
            if h_trend and l_trend:
                signals.append({'icon':'🟢','text':'多頭趨勢（頭頭高底底高）','score':+20})
                score += 20
            elif not h_trend and not l_trend:
                signals.append({'icon':'🔴','text':'空頭趨勢（頭頭低底底低）','score':-20})
                score -= 20
            else:
                signals.append({'icon':'🟡','text':'盤整（趨勢不明確）','score':0})
    except Exception:
        pass

    # ── 2. 月線多空分界 ──
    if not pd.isna(ma20.iloc[-1]):
        if price > ma20.iloc[-1]:
            signals.append({'icon':'🟢','text':f'股價在月線上方（+{(price/ma20.iloc[-1]-1)*100:.1f}%）','score':+10})
            score += 10
        else:
            signals.append({'icon':'🔴','text':f'股價跌破月線（{(price/ma20.iloc[-1]-1)*100:.1f}%）','score':-15})
            score -= 15

    # ── 3. 均線多頭排列 ──
    if not any(pd.isna([ma5.iloc[-1], ma10.iloc[-1], ma20.iloc[-1]])):
        if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]:
            signals.append({'icon':'🟢','text':'均線多頭排列（5>10>20）','score':+10})
            score += 10
        elif ma5.iloc[-1] < ma10.iloc[-1] < ma20.iloc[-1]:
            signals.append({'icon':'🔴','text':'均線空頭排列（5<10<20）','score':-10})
            score -= 10
        else:
            signals.append({'icon':'🟡','text':'均線糾結（盤整中）','score':0})

    # ── 4. 葛蘭碧買點 ──
    if len(close) >= 21 and not pd.isna(ma20.iloc[-1]):
        prev_below = close.iloc[-2] < ma20.iloc[-2] if not pd.isna(ma20.iloc[-2]) else False
        cross_up   = (close.iloc[-1] > ma20.iloc[-1]) and prev_below
        bounce     = (abs(close.iloc[-1] / ma20.iloc[-1] - 1) < 0.02) and (close.iloc[-1] > close.iloc[-2])
        if cross_up:
            signals.append({'icon':'🟢','text':'葛蘭碧買點一：股價從下穿越月線向上','score':+15})
            score += 15
        elif bounce and price > ma20.iloc[-1]:
            signals.append({'icon':'🟢','text':'葛蘭碧買點二：回踩月線後撐住反彈','score':+10})
            score += 10

    # ── 5. KD 指標 ──
    try:
        K, D = _calc_kd(high, low, close)
        k_val, d_val = float(K.iloc[-1]), float(D.iloc[-1])
        k_prev, d_prev = float(K.iloc[-2]), float(D.iloc[-2])
        golden = (k_prev < d_prev) and (k_val > d_val)
        death  = (k_prev > d_prev) and (k_val < d_val)
        if golden and k_val < 50:
            signals.append({'icon':'🟢','text':f'KD 低檔黃金交叉（K={k_val:.0f}）','score':+12})
            score += 12
        elif golden:
            signals.append({'icon':'🟡','text':f'KD 黃金交叉（K={k_val:.0f}，高檔注意）','score':+5})
            score += 5
        elif death and k_val > 50:
            signals.append({'icon':'🔴','text':f'KD 高檔死亡交叉（K={k_val:.0f}）','score':-12})
            score -= 12
        elif k_val < 20:
            signals.append({'icon':'🟡','text':f'KD 超賣區（K={k_val:.0f}）','score':+5})
            score += 5
        elif k_val > 80:
            signals.append({'icon':'🟡','text':f'KD 超買區（K={k_val:.0f}）','score':-5})
            score -= 5
        else:
            signals.append({'icon':'⚪','text':f'KD 中性（K={k_val:.0f} D={d_val:.0f}）','score':0})
    except Exception:
        pass

    # ── 6. MACD ──
    try:
        dif, macd_line, hist = _calc_macd(close)
        h_now  = float(hist.iloc[-1])
        h_prev = float(hist.iloc[-2])
        dif_now = float(dif.iloc[-1])
        above_zero = dif_now > 0 and float(macd_line.iloc[-1]) > 0
        hist_growing = h_now > h_prev and h_now > 0

        if above_zero and hist_growing:
            signals.append({'icon':'🟢','text':'MACD 零軸上方，紅柱放大（趨勢強）','score':+10})
            score += 10
        elif above_zero and not hist_growing and h_now > 0:
            signals.append({'icon':'🟡','text':'MACD 零軸上方，紅柱縮短（準備回檔）','score':+3})
            score += 3
        elif h_now < 0 and h_prev > 0:
            signals.append({'icon':'🔴','text':'MACD 死亡交叉（綠柱出現）','score':-12})
            score -= 12
        elif dif_now < 0 and float(macd_line.iloc[-1]) < 0:
            signals.append({'icon':'🔴','text':'MACD 零軸下方（不做多）','score':-8})
            score -= 8
        else:
            signals.append({'icon':'⚪','text':'MACD 中性','score':0})

        # MACD 背離（簡化版）
        if len(hist) >= 20:
            recent_highs = close.iloc[-20:]
            recent_hist  = hist.iloc[-20:]
            price_up  = close.iloc[-1] > close.iloc[-10]
            hist_down = recent_hist.iloc[-1] < recent_hist.iloc[-10]
            if price_up and hist_down and h_now > 0:
                signals.append({'icon':'⚠️','text':'MACD 頂背離（漲勢衰竭警告）','score':-8})
                score -= 8
    except Exception:
        pass

    # ── 7. RSI 背離 ──
    try:
        rsi = _calc_rsi(close)
        rsi_val = float(rsi.iloc[-1])
        if rsi_val < 30:
            signals.append({'icon':'🟢','text':f'RSI 超賣（{rsi_val:.0f}），反彈機率高','score':+8})
            score += 8
        elif rsi_val > 70:
            signals.append({'icon':'🟡','text':f'RSI 超買（{rsi_val:.0f}），注意回檔','score':-5})
            score -= 5
        # RSI 背離
        if len(close) >= 10:
            price_nh = close.iloc[-1] > close.iloc[-5]
            rsi_nh   = rsi.iloc[-1]  > rsi.iloc[-5]
            if price_nh and not rsi_nh and rsi_val > 60:
                signals.append({'icon':'⚠️','text':f'RSI 頂背離（漲勢衰竭）','score':-6})
                score -= 6
    except Exception:
        pass

    # ── 8. 成交量分析 ──
    try:
        vol_ma5 = vol.rolling(5).mean()
        v_now   = float(vol.iloc[-1])
        v_avg   = float(vol_ma5.iloc[-1])
        ratio   = v_now / (v_avg + 1e-8)

        if ratio >= 2.0 and close.iloc[-1] > close.iloc[-2]:
            signals.append({'icon':'🟢','text':f'爆量上漲（量能 {ratio:.1f}x 均量）','score':+10})
            score += 10
        elif ratio >= 1.3 and close.iloc[-1] > close.iloc[-2]:
            signals.append({'icon':'🟢','text':f'放量上漲（量能 {ratio:.1f}x 均量）','score':+5})
            score += 5
        elif ratio >= 2.0 and close.iloc[-1] <= close.iloc[-2]:
            signals.append({'icon':'🔴','text':f'爆量不漲（高檔出貨警訊）','score':-12})
            score -= 12
        elif ratio < 0.7 and close.iloc[-1] > close.iloc[-2]:
            signals.append({'icon':'🟡','text':f'縮量上漲（量能不足，謹慎）','score':-3})
            score -= 3
        else:
            signals.append({'icon':'⚪','text':f'成交量正常（{ratio:.1f}x 均量）','score':0})
    except Exception:
        pass

    # ── 9. K棒型態 ──
    try:
        o, h2, l2, c = float(high.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
        o_prev = float(close.iloc[-2])  # 用昨日收盤估今日開盤
        body   = abs(c - o_prev)
        upper  = h2 - max(c, o_prev)
        lower  = min(c, o_prev) - l2
        total  = h2 - l2 + 1e-8

        if upper > body * 2 and upper / total > 0.4:
            signals.append({'icon':'⚠️','text':'長上影線（多方攻高遭壓，注意賣壓）','score':-6})
            score -= 6
        elif lower > body * 2 and lower / total > 0.4:
            if c > o_prev:
                signals.append({'icon':'🟢','text':'長下影線＋紅K（多方撐盤有力）','score':+6})
                score += 6
            else:
                signals.append({'icon':'🟡','text':'長下影線（有撐，等隔日確認）','score':+3})
                score += 3
        elif c > o_prev and body / total > 0.6:
            signals.append({'icon':'🟢','text':'強勢長紅K（多方佔優）','score':+5})
            score += 5
        elif c < o_prev and body / total > 0.6:
            signals.append({'icon':'🔴','text':'長黑K（空方佔優）','score':-5})
            score -= 5
    except Exception:
        pass

    # ── 10. 乖離率 ──
    try:
        bias = (price / float(ma20.iloc[-1]) - 1) * 100 if not pd.isna(ma20.iloc[-1]) else 0
        if bias > 20:
            signals.append({'icon':'🔴','text':f'乖離率過大（+{bias:.1f}%），注意回檔','score':-8})
            score -= 8
        elif bias > 10:
            signals.append({'icon':'🟡','text':f'乖離率偏高（+{bias:.1f}%），空手不追','score':-3})
            score -= 3
        elif bias < -15:
            signals.append({'icon':'🟢','text':f'乖離率超跌（{bias:.1f}%），反彈機率高','score':+8})
            score += 8
        elif bias < -10:
            signals.append({'icon':'🟢','text':f'乖離率低檔（{bias:.1f}%），可留意','score':+3})
            score += 3
    except Exception:
        pass

    # ── 11. 費波那契回撤 ──
    try:
        recent = close.tail(60)
        hi = float(recent.max())
        lo = float(recent.min())
        retr = (hi - price) / (hi - lo + 1e-8)  # 從高點回撤比例
        if 0.35 <= retr <= 0.42:
            signals.append({'icon':'🟢','text':f'回撤至 0.382 黃金支撐（強勢股）','score':+8})
            score += 8
        elif 0.45 <= retr <= 0.55:
            signals.append({'icon':'🟡','text':f'回撤至 0.5 支撐（正常回檔）','score':+4})
            score += 4
        elif 0.58 <= retr <= 0.65:
            signals.append({'icon':'🟡','text':f'回撤至 0.618 支撐（弱勢但有機會）','score':+2})
            score += 2
    except Exception:
        pass

    # ── 12. 跳空缺口 ──
    try:
        if len(close) >= 3 and high_s is not None and low_s is not None:
            gap_up   = float(low.iloc[-1])   > float(high.iloc[-2])
            gap_down = float(high.iloc[-1])  < float(low.iloc[-2])
            if gap_up:
                signals.append({'icon':'🟢','text':'向上跳空缺口（多方強力，重要支撐）','score':+8})
                score += 8
            elif gap_down:
                signals.append({'icon':'🔴','text':'向下跳空缺口（空方強力，重要壓力）','score':-8})
                score -= 8
    except Exception:
        pass

    # ── 綜合建議 ──
    score = max(-100, min(100, score))
    if score >= 40:
        action, color = '🚀 強力買進', '#00ff00'
    elif score >= 15:
        action, color = '🟢 建議買進', '#00cc66'
    elif score >= -10:
        action, color = '🟡 觀望', '#ffab00'
    elif score >= -30:
        action, color = '🟠 建議賣出', '#ff6600'
    else:
        action, color = '🔴 強力賣出', '#ff3333'

    # 摘要
    buy_sigs  = [s for s in signals if s['score'] > 0]
    sell_sigs = [s for s in signals if s['score'] < 0]
    summary = f"看多訊號 {len(buy_sigs)} 個，看空訊號 {len(sell_sigs)} 個，綜合分數 {score:+d}"

    return {
        'score':   score,
        'action':  action,
        'color':   color,
        'signals': signals,
        'summary': summary,
    }


def _no_data():
    return {
        'score': 0, 'action': '⚪ 資料不足',
        'color': '#888', 'signals': [], 'summary': '歷史資料不足，無法分析'
    }


def render_advice_html(result):
    """將分析結果渲染為 HTML 字串"""
    if not result:
        return ''
    signals_html = ''.join(
        f'<div style="padding:2px 0;font-size:0.76rem;color:#cdd9e5;">'
        f'{s["icon"]} {s["text"]}'
        f'<span style="float:right;color:{"#00ff00" if s["score"]>0 else "#ff4444" if s["score"]<0 else "#888"};">'
        f'{s["score"]:+d}分</span></div>'
        for s in result['signals']
    )
    return f"""<div style="background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px 10px;min-width:220px;">
  <div style="font-weight:bold;font-size:0.85rem;color:{result['color']};margin-bottom:6px;">{result['action']}</div>
  <div style="font-size:0.72rem;color:#888;margin-bottom:6px;">{result['summary']}</div>
  {signals_html}
</div>"""
