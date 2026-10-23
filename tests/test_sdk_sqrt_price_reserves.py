"""load_sdk_ticks: reserves rescaled to the SDK sqrt price before the
2026-06-25 runtime change, untouched after (2026-10-22)."""

import math
import warnings

import pandas as pd
import pytest

from bt_trading_tools.data.ticks import (SDK_SQRT_PRICE_LAST_BLOCK, load_sdk_ticks)

PRE_TS = "2026-06-25 20:32:24+00:00"
POST_TS = "2026-06-25 20:47:24+00:00"


def _csv(tmp_path, with_block=True):
    rows = [
        # pre-switch: price (sqrtP^2) 0.86 x reserve ratio, like sn103
        dict(timestamp=PRE_TS, block=SDK_SQRT_PRICE_LAST_BLOCK, netuid=103,
             price=0.0086, tao_in=1000.0, alpha_in=100_000.0, alpha_out=1e6),
        dict(timestamp=PRE_TS, block=SDK_SQRT_PRICE_LAST_BLOCK, netuid=5,
             price=0.01, tao_in=1000.0, alpha_in=100_000.0, alpha_out=1e6),
        # post-switch: a price that disagrees must NOT be used for reserves
        dict(timestamp=POST_TS, block=SDK_SQRT_PRICE_LAST_BLOCK + 75, netuid=103,
             price=0.0090, tao_in=1000.0, alpha_in=100_000.0, alpha_out=1e6),
        dict(timestamp=POST_TS, block=SDK_SQRT_PRICE_LAST_BLOCK + 75, netuid=5,
             price=0.01, tao_in=1000.0, alpha_in=100_000.0, alpha_out=1e6),
    ]
    df = pd.DataFrame(rows)
    if not with_block:
        df = df.drop(columns="block")
    p = tmp_path / "sdk.csv"
    df.to_csv(p, index=False)
    return p


@pytest.mark.parametrize("with_block", [True, False])
def test_pre_switch_rescaled_post_switch_untouched(tmp_path, with_block):
    p = _csv(tmp_path, with_block)
    with pytest.warns(UserWarning, match="rescaled reserves to the SDK price") as rec:
        ticks = load_sdk_ticks("2026-06-25", "2026-06-26", csv_path=p)
    msg = str(rec[0].message)
    assert "on 2 rows" in msg and ">1% on 1 rows (1 subnets)" in msg
    pre, post = ticks
    st = pre.subnets[103]
    assert st.tao_pool / st.alpha_pool == pytest.approx(0.0086)
    assert st.tao_pool * st.alpha_pool == pytest.approx(1000.0 * 100_000.0)   # depth kept
    assert pre.subnets[5].tao_pool == pytest.approx(1000.0)                    # agreed: no-op
    st = post.subnets[103]
    assert (st.tao_pool, st.alpha_pool) == (1000.0, 100_000.0)


def test_flag_off_reproduces_raw_reserves(tmp_path):
    p = _csv(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ticks = load_sdk_ticks("2026-06-25", "2026-06-26", csv_path=p, sqrt_price_reserves=False)
    st = ticks[0].subnets[103]
    assert (st.tao_pool, st.alpha_pool) == (1000.0, 100_000.0)
    assert st.price == pytest.approx(0.0086)


def test_post_switch_window_has_no_warning(tmp_path):
    p = _csv(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ticks = load_sdk_ticks("2026-06-25 20:40", "2026-06-26", csv_path=p)
    assert len(ticks) == 1 and ticks[0].subnets[103].tao_pool == 1000.0
