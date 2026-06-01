"""
資料快取模組 (Data Cache)

將 OHLCV 歷史資料存為 parquet 格式在 data/ 資料夾，
每次執行只補抓最新幾天，大幅縮短下載時間。

快取檔案：
  data/cache_close.parquet
  data/cache_open.parquet
  data/cache_high.parquet
  data/cache_low.parquet
  data/cache_volume.parquet

使用方式：
  cache = DataCache()
  close_df = cache.load('close')         # 讀取快取
  cache.update('close', new_close_df)    # 更新快取
  missing_days = cache.missing_days(tickers, start, end)  # 找出缺漏日期
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / 'data'
CACHE_COLS = ['close', 'open', 'high', 'low', 'volume']


class DataCache:
    """
    OHLCV 資料快取管理器。
    """

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, col):
        return self.cache_dir / f'cache_{col}.parquet'

    def exists(self, col):
        return self._path(col).exists()

    def load(self, col):
        """
        讀取快取。

        Returns
        -------
        pd.DataFrame or None
        """
        path = self._path(col)
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"   ⚠️ 快取讀取失敗 ({col}): {e}")
            return None

    def save(self, col, df):
        """
        儲存快取。
        """
        if df is None or df.empty:
            return
        try:
            df.index = pd.to_datetime(df.index)
            df.to_parquet(self._path(col), compression='snappy')
        except Exception as e:
            print(f"   ⚠️ 快取寫入失敗 ({col}): {e}")

    def update(self, col, new_df):
        """
        合併新資料到現有快取（新資料優先覆蓋舊資料）。

        Parameters
        ----------
        col : str
            欄位名稱（'close', 'open', 'high', 'low', 'volume'）
        new_df : pd.DataFrame
            新下載的資料
        """
        if new_df is None or new_df.empty:
            return
        existing = self.load(col)
        if existing is None:
            self.save(col, new_df)
            return

        # 合併：以新資料為優先，補齊舊快取沒有的欄位
        merged = pd.concat([existing, new_df])
        merged = merged[~merged.index.duplicated(keep='last')]
        merged = merged.sort_index()

        # 補齊所有股票欄位（新加入的股票）
        all_cols = merged.columns.union(existing.columns)
        merged = merged.reindex(columns=all_cols)

        self.save(col, merged)

    def get_cache_info(self):
        """
        回傳快取狀態摘要。
        """
        info = {}
        for col in CACHE_COLS:
            df = self.load(col)
            if df is not None:
                info[col] = {
                    'start': df.index.min().strftime('%Y-%m-%d'),
                    'end': df.index.max().strftime('%Y-%m-%d'),
                    'tickers': len(df.columns),
                    'rows': len(df),
                }
            else:
                info[col] = None
        return info

    def needs_update(self, buffer_days=1):
        """
        判斷快取是否需要更新（距今超過 buffer_days 天）。
        """
        close = self.load('close')
        if close is None:
            return True
        last_date = close.index.max()
        today = pd.Timestamp(datetime.today().date())
        return (today - last_date).days > buffer_days

    def get_missing_range(self, tickers, end_date=None, full_days=365):
        """
        計算需要下載的日期範圍。

        - 若無快取：下載完整 full_days 天
        - 若有快取：只下載快取結束日之後的資料（最多 10 天）

        Returns
        -------
        start_date, end_date : str
            需要下載的起訖日期
        needs_full : bool
            是否需要完整下載（首次建立快取）
        missing_tickers : list
            快取中缺少的股票
        """
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        close = self.load('close')

        if close is None:
            # 無快取，需要完整下載
            start_date = (pd.Timestamp(end_date) - timedelta(days=full_days)).strftime('%Y-%m-%d')
            return start_date, end_date, True, tickers

        # 有快取，只補最新資料
        cache_end = close.index.max()
        start_date = (cache_end + timedelta(days=1)).strftime('%Y-%m-%d')

        # 找出快取中缺少的股票
        cached_tickers = set(close.columns.tolist())
        missing_tickers = [t for t in tickers if t not in cached_tickers]

        # 如果起始日 >= 結束日，代表快取已是最新
        if start_date >= end_date:
            return start_date, end_date, False, missing_tickers

        return start_date, end_date, False, missing_tickers


def load_all_cache(tickers, start_date, end_date, cache=None):
    """
    從快取讀取 OHLCV，裁切到指定日期範圍。

    Returns
    -------
    close_df, open_df, high_df, low_df, vol_df : pd.DataFrame or None
        若快取不存在回傳 None
    """
    if cache is None:
        cache = DataCache()

    close = cache.load('close')
    if close is None:
        return None, None, None, None, None

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    def _slice(df, col_name):
        if df is None:
            return None
        # 只保留在 tickers 中的欄位
        available = [t for t in tickers if t in df.columns]
        df = df[available].copy()
        # 裁切日期
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        return df.loc[mask]

    close_df = _slice(close, 'close')
    open_df  = _slice(cache.load('open'), 'open')
    high_df  = _slice(cache.load('high'), 'high')
    low_df   = _slice(cache.load('low'), 'low')
    vol_df   = _slice(cache.load('volume'), 'volume')

    if close_df is None or close_df.empty:
        return None, None, None, None, None

    return close_df, open_df, high_df, low_df, vol_df


def update_all_cache(close_df, open_df, high_df, low_df, vol_df, cache=None):
    """
    一次更新所有 OHLCV 快取。
    """
    if cache is None:
        cache = DataCache()

    cache.update('close', close_df)
    cache.update('open', open_df)
    cache.update('high', high_df)
    cache.update('low', low_df)
    cache.update('volume', vol_df)
    print(f"   ✅ 快取已更新")
