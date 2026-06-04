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


def _update_pool_from_twse(verbose=True):
    """
    使用 TWSE 官方 API 取得全市場成交額排名（免費、無需帳號）。
    抓取當日或最近交易日的「成交量前 20 名」及全市場個股資料。
    """
    import urllib.request, json, ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    if verbose:
        print(f"🔄 從 TWSE 官方 API 更新動態股池（全市場成交額 Top-{POOL_SIZE}）...")

    try:
        # 抓取 TWSE 上市股票當日成交資訊（含成交額）
        from datetime import date
        today = date.today().strftime('%Y%m%d')
        url = f"https://www.twse.com.tw/exchangeReport/MI_INDEX?response=json&date={today}&type=ALLBUT0999"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        if data.get('stat') != 'OK' or 'data9' not in data:
            if verbose:
                print("   ⚠️ TWSE API 無資料（可能非交易日），改用快取資料建池")
            return None

        rows = data['data9']
        records = []
        for row in rows:
            try:
                code = row[0].strip()
                if not (len(code) == 4 and code.isdigit()):
                    continue
                turnover_str = row[4].replace(',', '') if len(row) > 4 else '0'
                turnover = float(turnover_str) if turnover_str else 0
                records.append({'stock_id': code, 'avg_turnover': turnover})
            except Exception:
                continue

        if not records:
            return None

        df = pd.DataFrame(records)
        df = df[df['avg_turnover'] > 0].sort_values('avg_turnover', ascending=False).head(POOL_SIZE)
        tickers = df['stock_id'].tolist()

        pool_df = pd.DataFrame({
            'stock_id': df['stock_id'].values,
            'avg_turnover': df['avg_turnover'].values,
            'rank': range(1, len(df) + 1),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        POOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pool_df.to_csv(POOL_CACHE_PATH, index=False)

        if verbose:
            print(f"   ✅ TWSE 股池更新完成：{len(tickers)} 檔")
            print(f"   📌 前 10 名：{tickers[:10]}")
        return tickers

    except Exception as e:
        if verbose:
            print(f"   ⚠️ TWSE API 失敗: {e}")
        return None


def _update_pool_from_cache(verbose=True):
    """
    從現有 OHLCV 快取計算成交額，建立股池（離線 fallback）。
    """
    try:
        cache_dir = POOL_CACHE_PATH.parent
        close_path = cache_dir / 'cache_close.parquet'
        vol_path   = cache_dir / 'cache_volume.parquet'
        if not close_path.exists() or not vol_path.exists():
            return None

        close = pd.read_parquet(close_path)
        vol   = pd.read_parquet(vol_path)

        # 最近 20 日平均成交額
        turnover = (close * vol).rolling(20).mean().iloc[-1].dropna()
        turnover = turnover[turnover > 0].sort_values(ascending=False)

        tickers = turnover.head(POOL_SIZE).index.tolist()
        pool_df = pd.DataFrame({
            'stock_id': tickers,
            'avg_turnover': turnover.head(POOL_SIZE).values,
            'rank': range(1, len(tickers) + 1),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        POOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pool_df.to_csv(POOL_CACHE_PATH, index=False)

        if verbose:
            print(f"   ✅ 從快取建立股池：{len(tickers)} 檔")
        return tickers
    except Exception as e:
        if verbose:
            print(f"   ⚠️ 快取建池失敗: {e}")
        return None


def update_pool(verbose=True):
    """
    更新動態股池。優先順序：TWSE 官方 API → FinMind → 現有快取

    Returns
    -------
    list[str] or None
    """
    # 方法一：TWSE 官方 API（免費、最即時）
    result = _update_pool_from_twse(verbose=verbose)
    if result:
        return result

    # 方法二：FinMind（需帳號）
    fm = _get_finmind()
    if fm is not None:
        try:
            if verbose:
                print(f"🔄 嘗試 FinMind 方式更新股池...")
            end_dt = pd.Timestamp.today()
            start_dt = end_dt - timedelta(days=5)
            df = fm.taiwan_stock_daily(
                start_date=start_dt.strftime('%Y-%m-%d'),
                end_date=end_dt.strftime('%Y-%m-%d')
            )
            if df is not None and not df.empty:
                df = df[df['stock_id'].str.match(r'^\d{4}$')].copy()
                df['turnover'] = df['close'] * df['Trading_Volume']
                latest = df['date'].max()
                df_l = df[df['date'] == latest][df['turnover'] > 0]
                df_top = df_l.sort_values('turnover', ascending=False).head(POOL_SIZE)
                tickers = df_top['stock_id'].tolist()
                pool_df = pd.DataFrame({
                    'stock_id': tickers,
                    'avg_turnover': df_top['turnover'].values,
                    'rank': range(1, len(tickers)+1),
                    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                POOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                pool_df.to_csv(POOL_CACHE_PATH, index=False)
                if verbose:
                    print(f"   ✅ FinMind 股池更新完成：{len(tickers)} 檔")
                return tickers
        except Exception as e:
            if verbose:
                print(f"   ⚠️ FinMind 失敗: {e}")

    # 方法三：從現有 OHLCV 快取計算
    if verbose:
        print("   📦 改用本地快取建立股池...")
    return _update_pool_from_cache(verbose=verbose)


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
