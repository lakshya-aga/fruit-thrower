

# ---- agentic tool request 20260322T094114Z-e0c4204e ----
def calculate_rolling_zscore_v2(series, window=20, min_periods=None, ddof=0):
 import pandas as pd
 min_periods = window if min_periods is None else min_periods
 mean = series.rolling(window=window, min_periods=min_periods).mean()
 std = series.rolling(window=window, min_periods=min_periods).std(ddof=ddof)
 return (series - mean) / std


# ---- agentic tool request 20260322T094912Z-52c626cd ----
def calculate_rolling_zscore_v2(series, window=20, min_periods=None, ddof=0):
 import pandas as pd
 min_periods = window if min_periods is None else min_periods
 mean = series.rolling(window=window, min_periods=min_periods).mean()
 std = series.rolling(window=window, min_periods=min_periods).std(ddof=ddof)
 return (series - mean) / std
