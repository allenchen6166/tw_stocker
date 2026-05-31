"""
三大法人籌碼因子模組 (Institutional Flow Factor)

從 voidful/tw-institutional-stocker GitHub Pages 抓取三大法人持股時序數據，
計算橫向排名作為策略因子。

數據來源:
- https://voidful.github.io/tw-institutional-stocker/data/timeseries/{code}.json
- https://voidful.github.io/tw-institutional-stocker/data/top_three_inst_change_{w}_up.json

欄位:
- foreign_ratio: 外資持股比重 (%)
- trust_ratio: 投信持股比重 (%)
- dealer_ratio: 自營商持股比重 (%)
- three_inst_ratio: 三大法人合計 (%)
- three_inst_ratio_change_20: 近 20 日持股變化 (%)
"""

import json
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://voidful.github.io/tw-institutional-stocker/data"
TIMEOUT = 15


def _fetch_json(url):
    """從 URL 抓 JSON，失敗回 None。"""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'tw_stocker/1.0'})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"   ⚠️ 抓取失敗 {url}: {e}")
        return None


def fetch_inst_timeseries(ticker):
    """
    抓取單檔股票的三大法人持股時序。

    Parameters
    ----------
    ticker : str
        股票代號，例如 '2330'

    Returns
    -------
    list[dict] or None
        時序資料列表，每筆含 date, foreign_ratio, trust_ratio,
        dealer_ratio, three_inst_ratio, three_inst_ratio_change_20
    """
    url = f"{BASE_URL}/timeseries/{ticker}.json"
    return _fetch_json(url)


def fetch_inst_rankings(window=20, direction='up'):
    """
    抓取三大法人持股變化排名表。

    Parameters
    ----------
    window : int
        變化視窗天數 (5, 20, 60, 120)
    direction : str
        'up' 或 'down'

    Returns
    -------
    list[dict] or None
        排名列表，每筆含 code, name, market, three_inst_ratio, change
    """
    url = f"{BASE_URL}/top_three_inst_change_{window}_{direction}.json"
    return _fetch_json(url)


def build_inst_flow_df(tickers, close_df, verbose=True):
    """
    批次抓取多檔股票的籌碼時序，構建 DataFrame 對齊到 close_df。

    Parameters
    ----------
    tickers : list[str]
        股票代號列表
    close_df : pd.DataFrame
        收盤價矩陣 (date × ticker)，用於日期對齊
    verbose : bool
        是否印出進度

    Returns
    -------
    inst_flow_df : pd.DataFrame
        三大法人 20 日持股變化矩陣 (date × ticker)
    inst_ratio_df : pd.DataFrame
        三大法人持股比重矩陣 (date × ticker)
    """
    if verbose:
        print(f"🏛️ 正在平行抓取 {len(tickers)} 檔股票的三大法人數據（最多 20 執行緒）...")

    flow_data = {}
    ratio_data = {}
    success = 0
    failed = 0

    def _fetch_one(ticker):
        series = fetch_inst_timeseries(ticker)
        return ticker, series

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tickers}
        done_count = 0
        for future in as_completed(futures):
            ticker, series = future.result()
            done_count += 1
            if series is None or len(series) == 0:
                continue
            for record in series:
                dt = record.get('date')
                if dt is None:
                    continue
                try:
                    date_idx = pd.Timestamp(dt)
                except Exception:
                    continue
                change_20 = record.get('three_inst_ratio_change_20', 0.0)
                ratio = record.get('three_inst_ratio', 0.0)
                if date_idx not in flow_data:
                    flow_data[date_idx] = {}
                    ratio_data[date_idx] = {}
                flow_data[date_idx][ticker] = change_20
                ratio_data[date_idx][ticker] = ratio

            if verbose and done_count % 20 == 0:
                print(f"   📦 已處理 {done_count}/{len(tickers)} 檔...")

    success = sum(1 for d in flow_data.values() for _ in [d])
    if verbose:
        print(f"   ✅ 籌碼數據抓取完成，共 {len(flow_data)} 個交易日有資料")

    if not flow_data:
        # 回傳空 DataFrame（與 close_df 同形狀，全 NaN）
        empty = pd.DataFrame(np.nan, index=close_df.index,
                             columns=close_df.columns)
        return empty, empty.copy()

    inst_flow_df = pd.DataFrame.from_dict(flow_data, orient='index')
    inst_ratio_df = pd.DataFrame.from_dict(ratio_data, orient='index')

    # 對齊到 close_df 的日期索引
    inst_flow_df = inst_flow_df.reindex(index=close_df.index,
                                         columns=close_df.columns)
    inst_ratio_df = inst_ratio_df.reindex(index=close_df.index,
                                           columns=close_df.columns)

    return inst_flow_df, inst_ratio_df


def build_inst_score_from_rankings(tickers, close_df, window=20, verbose=True):
    """
    用排名 JSON（兩個請求）快速建立法人買超評分矩陣。
    比逐檔抓時序快 100 倍，適合每日自動排程使用。

    買超名單中的股票給正分，賣超給負分，其餘為 0。
    分數以 change 幅度為準（變化越大分數越高/低）。

    Parameters
    ----------
    tickers : list[str]
    close_df : pd.DataFrame
        用於對齊日期索引
    window : int
        排名視窗天數 (5, 20, 60, 120)

    Returns
    -------
    inst_flow_df : pd.DataFrame
        (date × ticker) 法人買超分數矩陣，最新一列有值，其餘填 0
    """
    if verbose:
        print(f"🏛️ 抓取三大法人排名（輕量模式，僅 2 個請求）...")

    up_list = fetch_inst_rankings(window, 'up') or []
    down_list = fetch_inst_rankings(window, 'down') or []

    score_map = {}
    for item in up_list:
        code = item.get('code', '')
        score_map[code] = abs(item.get('change', 0.0))
    for item in down_list:
        code = item.get('code', '')
        score_map[code] = -abs(item.get('change', 0.0))

    # 建立與 close_df 同形狀的矩陣，全部填 0
    inst_flow_df = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)

    # 只在最新一列填入排名分數（代表當下法人籌碼狀態）
    latest_idx = close_df.index[-1]
    for ticker in tickers:
        if ticker in score_map:
            inst_flow_df.loc[latest_idx, ticker] = score_map[ticker]

    matched = sum(1 for t in tickers if t in score_map)
    if verbose:
        print(f"   ✅ 法人排名抓取完成，{matched}/{len(tickers)} 檔有法人資料")

    return inst_flow_df


def get_inst_flow_for_signals(tickers, window=20):
    """
    為即時信號取得三大法人籌碼標注。
    用排名表做快速查詢（不需抓全部時序）。

    Parameters
    ----------
    tickers : list[str]
        候選股票代號
    window : int
        變化視窗 (default 20)

    Returns
    -------
    dict
        {ticker: {'change': float, 'ratio': float, 'label': str}}
    """
    up_list = fetch_inst_rankings(window, 'up') or []
    down_list = fetch_inst_rankings(window, 'down') or []

    # 建立查找表
    lookup = {}
    for item in up_list:
        lookup[item['code']] = {
            'change': item.get('change', 0.0),
            'ratio': item.get('three_inst_ratio', 0.0),
        }
    for item in down_list:
        lookup[item['code']] = {
            'change': -abs(item.get('change', 0.0)),
            'ratio': item.get('three_inst_ratio', 0.0),
        }

    result = {}
    for t in tickers:
        if t in lookup:
            info = lookup[t]
            change = info['change']
            if change > 2.0:
                label = '🟢 大買'
            elif change > 0.5:
                label = '🟡 小買'
            elif change < -2.0:
                label = '🔴 大賣'
            elif change < -0.5:
                label = '🟠 小賣'
            else:
                label = '⚪ 中性'
            result[t] = {
                'change': change,
                'ratio': info['ratio'],
                'label': label,
            }
        else:
            result[t] = {
                'change': 0.0,
                'ratio': 0.0,
                'label': '⚪ 無資料',
            }

    return result


