"""
AI 多維度 Rank 橫向排名策略 (AI Ensemble Cross-Sectional Ranking) — v8.3

Production scoring: rank_momentum × 3 + rank_trend × 1
(可選: + rank_liq × 0.3 when liq_stability=True)

已驗證無效 (Phase 1-4, 43 configs):
- ml_weights, residual_momentum, trend_quality: 有害
- breakeven, trailing, confidence-k, mid-hold-review: 有害或零效果
"""

STRATEGY_VERSION = "v8.5"

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 資料快取
try:
    from strategy.data_cache import DataCache, load_all_cache, update_all_cache
    HAS_CACHE = True
except ImportError:
    try:
        from data_cache import DataCache, load_all_cache, update_all_cache
        HAS_CACHE = True
    except ImportError:
        HAS_CACHE = False

# FinMind 優先，fallback 到 yfinance
try:
    from FinMind.data import DataLoader as _FMLoader
    _FM_TOKEN = os.environ.get('FINMIND_TOKEN', '')
    _fm = _FMLoader()
    if _FM_TOKEN:
        _fm.login_by_token(api_token=_FM_TOKEN)
    HAS_FINMIND = True
except ImportError:
    HAS_FINMIND = False

import yfinance as yf


def fetch_panel_data(tickers, days=365, start_date=None, end_date=None):
    """
    批次下載多檔台股的 OHLCV 日線資料（優先使用快取）。

    快取邏輯：
    - 首次執行：完整下載 days 天並存入快取
    - 後續執行：只下載快取結束日後的新資料，合併快取後回傳

    Parameters
    ----------
    tickers : list[str]
        台股代號列表
    days : int
        回溯天數（首次建快取時使用，預設 365 天）
    start_date : str or datetime, optional
        明確指定起始日期
    end_date : str or datetime, optional
        明確指定結束日期（預設為今天）

    Returns
    -------
    close_df, open_df, high_df, low_df, vol_df : tuple[pd.DataFrame]
    """
    if end_date is not None:
        end_dt = pd.Timestamp(end_date)
    else:
        end_dt = pd.Timestamp(datetime.today())

    end_str = end_dt.strftime('%Y-%m-%d')

    if start_date is not None:
        start_dt = pd.Timestamp(start_date)
    else:
        start_dt = end_dt - timedelta(days=days)

    start_str = start_dt.strftime('%Y-%m-%d')

    # ===== 快取邏輯 =====
    if HAS_CACHE:
        cache = DataCache()
        fetch_start, fetch_end, needs_full, missing_tickers = cache.get_missing_range(
            tickers, end_str, full_days=days)

        # 判斷是否需要下載
        need_download = needs_full or (fetch_start <= fetch_end) or bool(missing_tickers)

        if need_download:
            # 首次或有新資料/新股票需要下載
            dl_tickers = tickers if needs_full else list(set(tickers + missing_tickers))
            dl_start = start_str if needs_full else fetch_start

            print(f"📥 下載新資料：{dl_start} → {fetch_end}，{len(dl_tickers)} 檔...")
            if HAS_FINMIND:
                new_close, new_open, new_high, new_low, new_vol = _fetch_panel_finmind(
                    dl_tickers, pd.Timestamp(dl_start), pd.Timestamp(fetch_end))
            else:
                new_close, new_open, new_high, new_low, new_vol = _fetch_panel_yfinance(
                    dl_tickers, pd.Timestamp(dl_start), pd.Timestamp(fetch_end))

            # 更新快取
            print("💾 更新資料快取...")
            update_all_cache(new_close, new_open, new_high, new_low, new_vol, cache)
        else:
            print(f"✅ 快取已是最新，跳過下載")

        # 從快取讀取完整資料
        close_df, open_df, high_df, low_df, vol_df = load_all_cache(
            tickers, start_str, end_str, cache)

        if close_df is not None and not close_df.empty:
            print(f"📦 從快取載入：{close_df.index[0].strftime('%Y-%m-%d')}"
                  f" → {close_df.index[-1].strftime('%Y-%m-%d')}，{len(close_df.columns)} 檔")
            return close_df, open_df, high_df, low_df, vol_df

        print("⚠️ 快取讀取失敗，改為直接下載...")

    # ===== 無快取：直接下載 =====
    print(f"📥 直接下載 {len(tickers)} 檔，{start_str} → {end_str}...")
    if HAS_FINMIND:
        return _fetch_panel_finmind(tickers, start_dt, end_dt)
    else:
        return _fetch_panel_yfinance(tickers, start_dt, end_dt)


def _fetch_panel_finmind(tickers, start_dt, end_dt):
    """用 FinMind 批次下載台股 OHLCV（速度快、不限速）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    start_str = start_dt.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')

    col_map = {'open': 'Open', 'max': 'High', 'min': 'Low',
               'close': 'Close', 'Trading_Volume': 'Volume'}

    def _fetch_one(ticker):
        try:
            df = _fm.taiwan_stock_daily(
                stock_id=ticker, start_date=start_str, end_date=end_str)
            if df is None or df.empty:
                return ticker, None
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            df = df.rename(columns=col_map)
            return ticker, df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception:
            return ticker, None

    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            ticker, df = future.result()
            done += 1
            if df is not None:
                results[ticker] = df
            if done % 20 == 0:
                print(f"   📦 已下載 {done}/{len(tickers)} 檔...")

    if not results:
        raise RuntimeError("FinMind 無法下載任何資料")

    # 建立共同日期索引
    all_dates = sorted(set().union(*[set(df.index) for df in results.values()]))
    idx = pd.DatetimeIndex(all_dates)

    def _build(col):
        d = {t: results[t][col].reindex(idx) for t in results if col in results[t].columns}
        df = pd.DataFrame(d)
        return df.ffill(limit=1) if col == 'Close' else df

    close_df = _build('Close')
    open_df  = _build('Open')
    high_df  = _build('High')
    low_df   = _build('Low')
    vol_df   = _build('Volume')

    print(f"   ✅ FinMind 下載完成，{close_df.index[0].strftime('%Y-%m-%d')}"
          f" → {close_df.index[-1].strftime('%Y-%m-%d')}，共 {len(close_df.columns)} 檔")
    return close_df, open_df, high_df, low_df, vol_df


def _fetch_panel_yfinance(tickers, start_dt, end_dt):
    """yfinance fallback（原有邏輯）。"""
    def _extract_field(raw_df, field):
        if raw_df.empty:
            return pd.DataFrame()
        if isinstance(raw_df.columns, pd.MultiIndex):
            try:
                extracted = raw_df.xs(field, level=0, axis=1)
            except KeyError:
                return pd.DataFrame(index=raw_df.index)
        elif field in raw_df.columns:
            extracted = raw_df[[field]]
        else:
            return pd.DataFrame(index=raw_df.index)
        extracted = extracted.copy()
        extracted.columns = [
            str(c).replace('.TW', '').replace('.TWO', '')
            for c in extracted.columns
        ]
        if extracted.columns.duplicated().any():
            extracted = extracted.T.groupby(level=0).first().T
        return extracted

    def _download_symbols(symbols):
        import yfinance as yf
        downloaded = []
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            batch_df = yf.download(batch, start=start_dt, end=end_dt, progress=False)
            if not batch_df.empty:
                downloaded.append(batch_df)
        return downloaded

    tw_tickers = [f"{t}.TW" for t in tickers]
    all_dfs = _download_symbols(tw_tickers)
    if not all_dfs:
        raise RuntimeError("yfinance 無法下載任何資料")
    df = all_dfs[0] if len(all_dfs) == 1 else pd.concat(all_dfs, axis=1)

    close_probe = _extract_field(df, 'Close')
    missing = [t for t in tickers if t not in close_probe.columns or close_probe[t].dropna().empty]
    if missing:
        two_dfs = _download_symbols([f"{t}.TWO" for t in missing])
        if two_dfs:
            df = pd.concat([df] + two_dfs, axis=1)

    data = {}
    for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
        temp_df = _extract_field(df, col)
        if not temp_df.empty:
            data[col] = temp_df.ffill(limit=1) if col == 'Close' else temp_df

    print(f"   ✅ yfinance 下載完成，{data['Close'].index[0].strftime('%Y-%m-%d')}"
          f" → {data['Close'].index[-1].strftime('%Y-%m-%d')}，共 {len(data['Close'].columns)} 檔")
    return data['Close'], data['Open'], data['High'], data['Low'], data['Volume']


def build_liquid_universe(close_df, vol_df, top_n=50, lookback=20):
    """
    建立動態流動性 Universe。

    每日取「過去 lookback 日平均成交額 Top-N」作為當日可投資池。

    Parameters
    ----------
    close_df : pd.DataFrame
        收盤價矩陣
    vol_df : pd.DataFrame
        成交量矩陣
    top_n : int
        每日 universe 大小
    lookback : int
        成交額均值回溯期

    Returns
    -------
    universe_mask : pd.DataFrame (bool)
        (日期 x 股票) 的布林矩陣，True 代表當日在 universe 中
    """
    print(f"🌐 建立動態流動性 Universe (Top-{top_n}, 回溯 {lookback} 日)...")

    # 平均成交額 = 收盤價 × 成交量 的 rolling mean
    turnover = (close_df * vol_df).rolling(lookback).mean()

    # 每日取 top_n
    universe_mask = turnover.rank(axis=1, ascending=False) <= top_n

    # 確保 NaN 的位置不被選入
    universe_mask = universe_mask & close_df.notna() & (close_df > 0)

    avg_size = universe_mask.sum(axis=1).mean()
    print(f"   ✅ 動態 Universe 建立完成，平均每日 {avg_size:.0f} 檔")
    return universe_mask


def fetch_dynamic_top_tickers(top_n=50, days=5, verbose=True):
    """
    用 FinMind 單日全市場成交資料（1 個請求）快速選出成交額 Top-N 檔。

    流程：
    1. 抓最近交易日全市場所有股票的成交資料（1 個 API 請求）
    2. 計算成交額 = 收盤價 × 成交量
    3. 排序取 Top-N，過濾掉非普通股（ETF、特別股等）

    Parameters
    ----------
    top_n : int
        要選出幾檔（預設 50）
    days : int
        往前找最近幾天的交易日（預設 5，確保能抓到最近交易日）

    Returns
    -------
    list[str]
        股票代號列表（不含 .TW 後綴）
    """
    if not HAS_FINMIND:
        print("⚠️ FinMind 未安裝，使用預設股池")
        return None

    if verbose:
        print(f"🔍 動態篩選：單日全市場成交額 Top-{top_n}（1 個請求）...")

    end_dt = pd.Timestamp(datetime.today())
    start_dt = end_dt - timedelta(days=days + 3)
    start_str = start_dt.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')

    try:
        # 一次抓所有股票的近幾日成交資料
        df = _fm.taiwan_stock_daily(start_date=start_str, end_date=end_str)
        if df is None or df.empty:
            print("   ⚠️ FinMind 無資料，使用預設股池")
            return None

        # 只保留最近一個交易日
        latest_date = df['date'].max()
        df_latest = df[df['date'] == latest_date].copy()

        # 只取 4 碼普通股（排除 ETF、特別股等）
        df_latest = df_latest[df_latest['stock_id'].str.match(r'^[0-9]{4}$')]

        # 計算成交額
        df_latest['turnover'] = df_latest['close'] * df_latest['Trading_Volume']
        df_latest = df_latest[df_latest['turnover'] > 0]

        # 取 Top-N
        top_df = df_latest.sort_values('turnover', ascending=False).head(top_n)
        top_tickers = top_df['stock_id'].tolist()

        if verbose:
            print(f"   ✅ 最近交易日：{latest_date}，選出 {len(top_tickers)} 檔")
            print(f"   📌 前 10 名：{top_tickers[:10]}")

        return top_tickers

    except Exception as e:
        print(f"   ⚠️ 動態篩選失敗: {e}，使用預設股池")
        return None


def engineer_features(close_df, vol_df, universe_mask=None,
                      ma_period=60, short_ma_period=20, multi_ma=False,
                      ml_weights=False, inst_flow_weight=0.0,
                      inst_flow_df=None,
                      residual_momentum=False,
                      trend_quality=False,
                      liq_stability=False,
                      liq_mode='raw',
                      market_close=None,
                      rsi_weight=0.0,
                      breakout_weight=0.0,
                      value_weight=0.0,
                      rev_momentum_weight=0.0,
                      us_signals=None,
                      rs_weight=1.5):
    """
    計算 AI 多維度特徵並做橫向百分位排名。

    Parameters
    ----------
    close_df : pd.DataFrame
        收盤價矩陣 (日期 x 股票代號)
    vol_df : pd.DataFrame
        成交量矩陣 (日期 x 股票代號)
    universe_mask : pd.DataFrame (bool), optional
        動態 Universe 遮罩。若提供，只在當日 universe 中做排名。
    ma_period : int
        主趨勢均線天數（預設 60）
    short_ma_period : int
        短期均線天數（用於多均線確認，預設 20）
    multi_ma : bool
        啟用多均線確認（short_ma > long_ma 才允許進場）
    ml_weights : bool
        啟用 ML 因子加權（LightGBM 取代等權加總）

    Returns
    -------
    total_score : pd.DataFrame
        各股票的 AI 綜合評分，日期 x 股票
    ma_long : pd.DataFrame
        主趨勢均線矩陣，用於進場信號過濾
    atr_df : pd.DataFrame
        20 日 ATR 矩陣，用於自適應 TP/SL 與 position sizing
    short_ma : pd.DataFrame or None
        短期均線矩陣（multi_ma=True 時有效）
    """
    print("🧠 正在計算多維度弱特徵與 Rank 排名...")

    # === 動態評分窗口（根據大盤 macro_regime 調整）===
    # macro_regime 範圍：0.0（極熊）～ 1.0（極牛）
    # 牛市（≥0.7）：拉長動能窗口至 40 日，捕捉趨勢延伸
    # 熊市（≤0.3）：縮短動能窗口至 10 日，快速反應反彈
    # 中性（0.3~0.7）：維持預設 20 日
    if us_signals is not None and 'macro_regime' in us_signals.columns:
        # 取最新一天的 regime 值作為當前市場狀態
        latest_regime = float(us_signals['macro_regime'].dropna().iloc[-1]) if not us_signals['macro_regime'].dropna().empty else 0.5
        if latest_regime >= 0.7:
            mom_window = 40
            regime_label = f"牛市 (regime={latest_regime:.2f})"
        elif latest_regime <= 0.3:
            mom_window = 10
            regime_label = f"熊市 (regime={latest_regime:.2f})"
        else:
            mom_window = 20
            regime_label = f"中性 (regime={latest_regime:.2f})"
        print(f"   📊 動態動能窗口：{mom_window} 日 [{regime_label}]")
    else:
        mom_window = 20
        print(f"   📊 動能窗口：{mom_window} 日（固定，無大盤信號）")

    # === 原始指標計算 ===
    # 1. 動態動能：今天收盤 / N 天前收盤（N 由 macro_regime 決定）
    mom_20 = close_df / close_df.shift(mom_window)

    # 2. 加強版趨勢計算（三維度合成）
    ma_long = close_df.rolling(ma_period).mean()
    ma_20   = close_df.rolling(20).mean()

    # 2a. MA60 乖離率（原有）：價格在均線上方多少
    trend_bias = close_df / ma_long

    # 2b. MA60 斜率：均線本身是否持續上升（用 10 日前後比較）
    #     斜率 > 0 = 均線上升中（趨勢健康）
    #     斜率 < 0 = 均線下降中（趨勢轉弱）
    #     ⚠️ 加入截尾(clip)避免極端值扭曲排名（±3% per 10 days）
    ma60_slope = ((ma_long - ma_long.shift(10)) / (ma_long.shift(10) + 1e-8)).clip(-0.03, 0.03)

    # 2c. 均線多頭排列分數：MA20 > MA60 加分，額外確認 close > MA20
    #     完整多頭排列：close > MA20 > MA60 → 1.0
    #     部分：close > MA60 但 MA20 < MA60 → 0.5
    #     空頭排列：close < MA60 → 0.0
    ma_align = (
        ((close_df > ma_20) & (ma_20 > ma_long)).astype(float) * 1.0 +
        ((close_df > ma_long) & ~((close_df > ma_20) & (ma_20 > ma_long))).astype(float) * 0.5
    )

    # 2d. 過熱懲罰：乖離超過 20% 時開始扣分（避免追高）
    #     乖離 20% = 係數 1.0（無懲罰）
    #     乖離 30% = 係數 0.7
    #     乖離 40%+ = 係數 0.4
    overheat = (trend_bias - 1.0).clip(lower=0)  # 只算上方乖離
    overheat_penalty = 1.0 - (overheat - 0.20).clip(lower=0, upper=0.30) / 0.30 * 0.60

    # 2e. 合成加強版趨勢分數
    #     乖離率 × 斜率加成 × 均線排列加成 × 過熱懲罰
    enhanced_trend = (
        trend_bias * 0.5          # 基礎：MA60 乖離率（原有）
        + ma60_slope * 5.0        # 斜率加成（放大讓排名有區分度）
        + ma_align * 0.3          # 均線排列加成
    ) * overheat_penalty          # 過熱懲罰

    print(f"   📐 加強版趨勢已計算（乖離+斜率+排列+過熱懲罰）")

    # 3. 量能爆發比：5 日均量 / 20 日均量
    vol_surge = vol_df.rolling(5).mean() / (vol_df.rolling(20).mean() + 1e-8)

    # 3b. 均線突破 + 量能確認因子
    #   條件一：價格剛突破 MA5 或 MA10（前一日低於，今日高於）
    #   條件二：突破當日量能 > 20日均量的 1.5 倍
    #   分數：雙確認突破 = 1.0，單條件 = 0.5，未突破 = 0.0
    ma_5  = close_df.rolling(5).mean()
    ma_10 = close_df.rolling(10).mean()
    vol_20avg = vol_df.rolling(20).mean()

    # 是否剛突破（今日 > MA，昨日 <= MA）
    break_ma5  = (close_df > ma_5)  & (close_df.shift(1) <= ma_5.shift(1))
    break_ma10 = (close_df > ma_10) & (close_df.shift(1) <= ma_10.shift(1))
    any_break  = break_ma5 | break_ma10

    # 量能是否放大（今日量 > 20日均量 × 1.5）
    vol_confirm = vol_df > (vol_20avg * 1.5)

    # 突破評分
    breakout_score = (
        (any_break & vol_confirm).astype(float) * 1.0   # 突破 + 量能：滿分
        + (any_break & ~vol_confirm).astype(float) * 0.5 # 突破但量能不足：半分
    )
    # 突破訊號持續 3 日（讓信號不只一天）
    breakout_score = breakout_score.rolling(3).max().fillna(0)
    print(f"   🚀 均線突破因子已計算 (MA5/MA10 + 量能確認)")

    # 4. 穩定度：波動率的倒數（越穩定越好）
    volatility = close_df.pct_change().rolling(20).std()
    stability = 1 / (volatility + 1e-8)

    # 短期均線（多均線確認用）
    short_ma = close_df.rolling(short_ma_period).mean() if multi_ma else None
    # ma_20 已在加強版趨勢計算中定義

    # === ATR 計算 (用於 TP/SL 與 sizing) ===
    atr_df = close_df.pct_change().abs().rolling(20).mean() * close_df

    # === 殘差動量：扣除市場 beta ===
    residual_mom = None
    if residual_momentum and market_close is not None:
        try:
            stock_ret = close_df.pct_change()
            mkt_ret = market_close.pct_change()
            mkt_ret_aligned = mkt_ret.reindex(stock_ret.index, method='ffill')
            stock_cum_20 = stock_ret.rolling(20).sum()
            mkt_cum_20 = mkt_ret_aligned.rolling(20).sum()
            residual_mom = stock_cum_20.sub(mkt_cum_20, axis=0)
            print("   \U0001f52c 殘差動量已計算 (market-beta adjusted)")
        except Exception as e:
            print(f"   ⚠️ 殘差動量計算失敗: {e}")

    # === 相對強度（RS Rating）===
    # 參考 IBD RS Rating：多窗口加權，衡量個股相對大盤的強弱
    # 公式：RS = 個股累計報酬 / 大盤累計報酬（多時間窗口加權）
    # 窗口：63日(×2) + 126日(×1) + 252日(×1)，近期更重要
    rs_score = None
    if market_close is not None:
        try:
            # 用 close / close.shift(n) 計算累計報酬，速度快、NaN 少
            mkt_close = market_close.reindex(close_df.index, method='ffill')

            def _period_ret_stk(n):
                return close_df / close_df.shift(n) - 1

            def _period_ret_mkt(n):
                return mkt_close / mkt_close.shift(n) - 1

            # 個股超額報酬 = 個股累計報酬 - 大盤累計報酬
            # 窗口：63日(×2) + 126日(×1)，不用 252 日避免暖機太長
            rs_63  = _period_ret_stk(63).sub(_period_ret_mkt(63),   axis=0)
            rs_126 = _period_ret_stk(126).sub(_period_ret_mkt(126), axis=0)

            # 加權合成：近期(63日)權重更高
            rs_score = rs_63 * 2 + rs_126 * 1
            print(f"   📈 相對強度(RS)已計算 (63日×2 + 126日×1，快速版)")
        except Exception as e:
            print(f"   ⚠️ RS 計算失敗: {e}")

    # === 趨勢品質 ===
    tq_score = None
    if trend_quality:
        try:
            ma60_slope = (ma_long - ma_long.shift(5)) / (ma_long.shift(5) + 1e-8)
            ma_alignment = ((close_df > ma_20) & (ma_20 > ma_long)).astype(float)
            overheat = (close_df / ma_20 - 1).clip(lower=0)
            overheat_penalty = 1 - overheat.clip(upper=0.15) / 0.15
            tq_score = ma60_slope * 100 + ma_alignment * 0.5 + overheat_penalty * 0.3
            print("   \U0001f4d0 趨勢品質已計算")
        except Exception as e:
            print(f"   ⚠️ 趨勢品質計算失敗: {e}")

    # === 流動性穩定度 ===
    liq_stab = None
    if liq_stability:
        try:
            turnover = close_df * vol_df
            raw_liq = turnover.rolling(20).mean() / (turnover.rolling(20).std() + 1e-8)

            if liq_mode == 'demeaned':
                # 殘差 liq: 扣除橫截面平均，保留個股相對穩定度
                cross_mean = raw_liq.mean(axis=1)
                liq_stab = raw_liq.sub(cross_mean, axis=0)
                print("   \U0001f4a7 流動性穩定度已計算 (demeaned)")
            elif liq_mode == 'sector':
                # 行業中性: 電子 vs 非電子分開計算再合併
                elec_prefixes = ('23','24','30','33','34','35','36','37',
                                 '49','61','63','64','65','66','67','68','69')
                elec_cols = [c for c in close_df.columns if str(c).startswith(elec_prefixes)]
                non_elec_cols = [c for c in close_df.columns if c not in elec_cols]
                liq_stab = raw_liq.copy()
                if elec_cols:
                    elec_mean = raw_liq[elec_cols].mean(axis=1)
                    liq_stab[elec_cols] = raw_liq[elec_cols].sub(elec_mean, axis=0)
                if non_elec_cols:
                    ne_mean = raw_liq[non_elec_cols].mean(axis=1)
                    liq_stab[non_elec_cols] = raw_liq[non_elec_cols].sub(ne_mean, axis=0)
                print("   \U0001f4a7 流動性穩定度已計算 (sector-neutral)")
            else:
                liq_stab = raw_liq
                print("   \U0001f4a7 流動性穩定度已計算 (raw)")
        except Exception:
            pass

    # === 橫向百分位排名 ===
    def _rank(df):
        if universe_mask is not None:
            return df.where(universe_mask).rank(axis=1, pct=True)
        return df.rank(axis=1, pct=True)

    rank_mom = _rank(mom_20)
    rank_trend = _rank(enhanced_trend)   # 使用加強版趨勢
    rank_rs = _rank(rs_score) if rs_score is not None else None
    rank_res_mom = _rank(residual_mom) if residual_mom is not None else None
    rank_tq = _rank(tq_score) if tq_score is not None else None
    rank_liq = _rank(liq_stab) if liq_stab is not None else None

    # === 籌碼因子排名 ===
    rank_inst = None
    if inst_flow_weight > 0 and inst_flow_df is not None:
        inst_aligned = inst_flow_df.reindex(
            index=close_df.index, columns=close_df.columns
        )
        rank_inst = _rank(inst_aligned)
        print(f"   \U0001f3db\ufe0f 籌碼因子已載入 (weight={inst_flow_weight})")

    # === FinLab 啟發因子（可選） ===
    rank_rsi = None
    rank_breakout = None
    rank_value = None
    rank_rev_mom = None

    if rsi_weight > 0:
        try:
            from strategy.finlab_factors import compute_rsi_rank
            rank_rsi = compute_rsi_rank(close_df, period=20, universe_mask=universe_mask)
            print(f"   📈 RSI-20 因子已計算 (weight={rsi_weight})")
        except Exception as e:
            print(f"   ⚠️ RSI 因子計算失敗: {e}")

    if breakout_weight > 0:
        try:
            from strategy.finlab_factors import compute_breakout_rank
            rank_breakout = compute_breakout_rank(close_df, window=300, universe_mask=universe_mask)
            print(f"   🏔️ Breakout-300 因子已計算 (weight={breakout_weight})")
        except Exception as e:
            print(f"   ⚠️ Breakout 因子計算失敗: {e}")

    if value_weight > 0:
        try:
            from strategy.finlab_factors import compute_value_rank
            rank_value = compute_value_rank(
                close_df, universe_mask=universe_mask,
                tickers=list(close_df.columns),
            )
            print(f"   💰 Value (PB+PE) 因子已計算 (weight={value_weight})")
        except Exception as e:
            print(f"   ⚠️ Value 因子計算失敗: {e}")

    if rev_momentum_weight > 0:
        try:
            from strategy.finlab_factors import compute_revenue_momentum
            rank_rev_mom = compute_revenue_momentum(close_df, period=60, universe_mask=universe_mask)
            print(f"   📊 RevMomentum-60 因子已計算 (weight={rev_momentum_weight})")
        except Exception as e:
            print(f"   ⚠️ RevMomentum 因子計算失敗: {e}")

    # === 因子加權 ===
    if ml_weights:
        # NOTE: ml_weights 已驗證無效 (Sharpe -55%), 保留但加警告
        import warnings as _w
        _w.warn('ml_weights 已驗證無效 (Sharpe -55%), 不建議使用', stacklevel=2)
        rank_vol = _rank(vol_surge)
        rank_stab = _rank(stability)
        total_score = _ml_factor_score(
            close_df, rank_mom, rank_trend, rank_vol, rank_stab, universe_mask
        )
    else:
        mom_factor = rank_res_mom if rank_res_mom is not None else rank_mom
        trend_factor = rank_tq if rank_tq is not None else rank_trend

        # === 修正版評分公式 ===
        # 原問題：動能×3 + RS×1.5 + 趨勢×1 → 三個因子都在衡量漲幅，重疊嚴重
        # 修正：降低動能權重，加入量能因子，讓選股更多元
        #
        # 新公式：動能×2 + 趨勢×1.5 + RS×1 + 量能×0.5 + 法人×1
        rank_vol_factor  = _rank(vol_surge)
        rank_breakout_sig = _rank(breakout_score)   # 突破因子排名
        total_score = mom_factor * 2 + trend_factor * 1.5
        total_score = total_score + rank_vol_factor * 0.5    # 量能因子
        total_score = total_score + rank_breakout_sig * 1.0  # 突破因子（×1）
        print(f"   📊 評分公式：動能×2 + 趨勢×1.5 + 量能×0.5 + 突破×1")

        if rank_liq is not None:
            total_score = total_score + rank_liq * 0.3

        # 相對強度因子（RS Rating）：找跑贏大盤的股票（權重從1.5降至1.0）
        if rank_rs is not None and rs_weight > 0:
            rs_w = min(rs_weight, 1.0)   # 最高 1.0，避免與動能過度重疊
            total_score = total_score + rank_rs * rs_w
            print(f"   📈 RS 因子已加入評分 (weight={rs_w})")

        # FinLab 因子加權（opt-in，預設全部為 0 不影響 baseline）
        if rank_rsi is not None and rsi_weight > 0:
            total_score = total_score + rank_rsi * rsi_weight
        if rank_breakout is not None and breakout_weight > 0:
            total_score = total_score + rank_breakout * breakout_weight
        if rank_value is not None and value_weight > 0:
            total_score = total_score + rank_value * value_weight
        if rank_rev_mom is not None and rev_momentum_weight > 0:
            total_score = total_score + rank_rev_mom * rev_momentum_weight

    # 籌碼因子加權（opt-in）
    if rank_inst is not None and inst_flow_weight > 0:
        total_score = total_score + rank_inst * inst_flow_weight

    print("   ✅ 特徵計算完成")

    # 回傳各因子 rank 矩陣，供報表顯示用（與實際評分一致）
    try:
        _rvf = rank_vol_factor
    except NameError:
        _rvf = _rank(vol_surge)
    try:
        _rbs = rank_breakout_sig
    except NameError:
        _rbs = None

    factor_ranks = {
        'mom':      (rank_res_mom if rank_res_mom is not None else rank_mom),
        'trend':    (rank_tq if rank_tq is not None else rank_trend),
        'vol':      _rvf,
        'rs':       rank_rs,
        'inst':     rank_inst,
        'breakout': _rbs,
        # 突破原始信號（用於報表顯示突破清單）
        'breakout_raw':   breakout_score,
        'break_ma5_raw':  break_ma5,
        'break_ma10_raw': break_ma10,
        'vol_confirm_raw': vol_confirm,
    }
    return total_score, ma_long, atr_df, short_ma, factor_ranks


def _ml_factor_score(close_df, rank_mom, rank_trend, rank_vol, rank_stab,
                     universe_mask=None, train_window=120, forward_days=10):
    """
    使用 LightGBM 進行因子加權。
    滾動訓練：用過去 train_window 天的因子 → 未來 forward_days 天報酬的關係，
    產出每日因子加權分數。

    若 LightGBM 未安裝，自動 fallback 到等權加總。
    """
    try:
        import lightgbm as lgb
        print("   🤖 使用 LightGBM 因子加權模式...")
    except ImportError:
        print("   ⚠️ lightgbm 未安裝，fallback 到等權加總")
        return rank_mom + rank_trend + rank_vol + rank_stab

    # 未來 N 天報酬（作為 label）
    fwd_ret = close_df.shift(-forward_days) / close_df - 1

    total_score = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns)
    dates = close_df.index

    # 每 20 天重新訓練一次模型（避免每天都訓練太慢）
    retrain_interval = 20
    model = None
    last_train_idx = -retrain_interval

    for i in range(train_window + forward_days, len(dates)):
        # 訓練（每 retrain_interval 天更新）
        if i - last_train_idx >= retrain_interval:
            train_start = max(0, i - train_window - forward_days)
            train_end = i - forward_days  # 確保 label 可用

            # 收集訓練資料
            X_list, y_list = [], []
            for t in range(train_start, train_end):
                for col in close_df.columns:
                    if universe_mask is not None:
                        if not universe_mask[col].iloc[t]:
                            continue
                    feats = [
                        rank_mom[col].iloc[t],
                        rank_trend[col].iloc[t],
                        rank_vol[col].iloc[t],
                        rank_stab[col].iloc[t],
                    ]
                    label = fwd_ret[col].iloc[t]
                    if any(pd.isna(f) for f in feats) or pd.isna(label):
                        continue
                    X_list.append(feats)
                    y_list.append(label)

            if len(X_list) >= 50:
                X_train = np.array(X_list)
                y_train = np.array(y_list)
                model = lgb.LGBMRegressor(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    min_child_samples=10, subsample=0.8,
                    verbosity=-1, n_jobs=-1
                )
                model.fit(X_train, y_train)
                last_train_idx = i

        # 預測
        if model is not None:
            for col in close_df.columns:
                feats = [
                    rank_mom[col].iloc[i],
                    rank_trend[col].iloc[i],
                    rank_vol[col].iloc[i],
                    rank_stab[col].iloc[i],
                ]
                if any(pd.isna(f) for f in feats):
                    continue
                pred = model.predict([feats])[0]
                total_score[col].iloc[i] = pred
        else:
            # 模型還沒訓練好，用等權 fallback
            total_score.iloc[i] = (rank_mom.iloc[i] + rank_trend.iloc[i]
                                   + rank_vol.iloc[i] + rank_stab.iloc[i])

    # 轉為橫向排名（讓分數可比較）
    total_score = total_score.rank(axis=1, pct=True) * 4

    print(f"   ✅ ML 因子加權完成 (模型訓練 {(len(dates) - train_window) // retrain_interval} 次)")
    return total_score






