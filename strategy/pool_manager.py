"""
動態股票池管理模組 (Dynamic Pool Manager)

每週自動從 FinMind 取得全市場股票清單，
按近期成交額排名，維護 Top-150 的動態股池。

快取檔案：data/pool_cache.csv
格式：stock_id, avg_turnover, rank, updated_at

執行策略：
  - 每週一執行一次池更新（耗時約 30 秒）
  - 每日執行時直接讀取 pool_cache.csv
  - 若 pool_cache.csv 不存在或超過 7 天，自動重建
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

POOL_CACHE_PATH = Path(__file__).parent.parent / 'data' / 'pool_cache.csv'
POOL_SIZE = 150        # 動態池大小
POOL_MAX_AGE_DAYS = 7  # 超過幾天自動更新


def _get_finmind():
    """取得已登入的 FinMind DataLoader，失敗回 None"""
    try:
        from FinMind.data import DataLoader
        fm = DataLoader()
        token = os.environ.get('FINMIND_TOKEN', '')
        if token:
            fm.login_by_token(api_token=token)
        return fm
    except Exception:
        return None


def pool_needs_update():
    """判斷股池是否需要更新"""
    if not POOL_CACHE_PATH.exists():
        return True
    try:
        df = pd.read_csv(POOL_CACHE_PATH)
        if 'updated_at' not in df.columns or df.empty:
            return True
        last_update = pd.Timestamp(df['updated_at'].iloc[0])
        age_days = (pd.Timestamp.now() - last_update).days
        return age_days >= POOL_MAX_AGE_DAYS
    except Exception:
        return True


def update_pool(verbose=True):
    """
    從 FinMind 取得全市場成交額，更新動態股池。

    流程：
    1. 取得全市場近 5 個交易日成交資料（1 個請求）
    2. 計算平均日成交額
    3. 過濾普通股（4 碼代號）
    4. 排序取 Top-POOL_SIZE
    5. 存入 pool_cache.csv

    Returns
    -------
    list[str] or None
    """
    fm = _get_finmind()
    if fm is None:
        if verbose:
            print("   ⚠️ FinMind 未安裝，無法更新股池")
        return None

    end_dt = pd.Timestamp.today()
    start_dt = end_dt - timedelta(days=10)

    if verbose:
        print(f"🔄 更新動態股池（全市場成交額 Top-{POOL_SIZE}）...")

    try:
        df = fm.taiwan_stock_daily(
            start_date=start_dt.strftime('%Y-%m-%d'),
            end_date=end_dt.strftime('%Y-%m-%d')
        )
        if df is None or df.empty:
            if verbose:
                print("   ⚠️ FinMind 無法取得資料")
            return None

        # 只保留 4 碼普通股
        df = df[df['stock_id'].str.match(r'^\d{4}$')].copy()

        # 計算成交額
        df['turnover'] = df['close'] * df['Trading_Volume']

        # 取最近一個交易日的數據
        latest_date = df['date'].max()
        df_latest = df[df['date'] == latest_date].copy()
        df_latest = df_latest[df_latest['turnover'] > 0]

        # 排序取 Top-N
        df_top = df_latest.sort_values('turnover', ascending=False).head(POOL_SIZE)

        # 建立快取 DataFrame
        pool_df = pd.DataFrame({
            'stock_id': df_top['stock_id'].values,
            'avg_turnover': df_top['turnover'].values,
            'rank': range(1, len(df_top) + 1),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        # 儲存
        POOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pool_df.to_csv(POOL_CACHE_PATH, index=False)

        tickers = pool_df['stock_id'].tolist()
        if verbose:
            print(f"   ✅ 股池更新完成：{len(tickers)} 檔（基準日：{latest_date}）")
            print(f"   📌 前 10 名：{tickers[:10]}")
        return tickers

    except Exception as e:
        if verbose:
            print(f"   ⚠️ 股池更新失敗: {e}")
        return None


def load_pool(verbose=True):
    """
    讀取動態股池。若快取過期則自動更新。

    Returns
    -------
    list[str]
        股票代號列表，失敗時回傳 None
    """
    if pool_needs_update():
        if verbose:
            print("📋 股池快取過期，重新取得...")
        result = update_pool(verbose=verbose)
        if result:
            return result
        # 更新失敗，若有舊快取就用舊的
        if POOL_CACHE_PATH.exists():
            if verbose:
                print("   ⚠️ 使用舊版股池快取")
        else:
            return None

    try:
        df = pd.read_csv(POOL_CACHE_PATH)
        tickers = df['stock_id'].astype(str).tolist()
        updated_at = df['updated_at'].iloc[0] if 'updated_at' in df.columns else '未知'
        if verbose:
            print(f"   📋 載入動態股池：{len(tickers)} 檔（更新時間：{updated_at[:10]}）")
        return tickers
    except Exception as e:
        if verbose:
            print(f"   ⚠️ 讀取股池失敗: {e}")
        return None


def get_pool_info():
    """回傳股池狀態摘要"""
    if not POOL_CACHE_PATH.exists():
        return {'status': '無快取', 'size': 0, 'updated_at': None, 'needs_update': True}
    try:
        df = pd.read_csv(POOL_CACHE_PATH)
        updated_at = df['updated_at'].iloc[0] if 'updated_at' in df.columns else None
        age_days = (pd.Timestamp.now() - pd.Timestamp(updated_at)).days if updated_at else 999
        return {
            'status': '正常' if age_days < POOL_MAX_AGE_DAYS else '需更新',
            'size': len(df),
            'updated_at': updated_at,
            'age_days': age_days,
            'needs_update': age_days >= POOL_MAX_AGE_DAYS
        }
    except Exception:
        return {'status': '錯誤', 'size': 0, 'updated_at': None, 'needs_update': True}
