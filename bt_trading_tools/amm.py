"""
AMM Math — constant product: k = tao_in * alpha_in

Pure functions for Bittensor AMM buy/sell calculations.
All functions are stateless with no external dependencies.
"""

from decimal import Decimal


def amm_buy(tao_spend: float, tao_pool: float, alpha_pool: float
            ) -> tuple[float, float, float]:
    """Buy alpha by spending TAO.

    Returns (alpha_received, new_tao_pool, new_alpha_pool).
    """
    if tao_spend <= 0:
        return 0.0, tao_pool, alpha_pool
    k = tao_pool * alpha_pool
    new_tao = tao_pool + tao_spend
    new_alpha = k / new_tao
    return alpha_pool - new_alpha, new_tao, new_alpha


def amm_sell(alpha_sell: float, tao_pool: float, alpha_pool: float
             ) -> tuple[float, float, float]:
    """Sell alpha for TAO.

    Returns (tao_received, new_tao_pool, new_alpha_pool).
    """
    if alpha_sell <= 0:
        return 0.0, tao_pool, alpha_pool
    k = tao_pool * alpha_pool
    new_alpha = alpha_pool + alpha_sell
    new_tao = k / new_alpha
    return tao_pool - new_tao, new_tao, new_alpha


def spot_price(tao_pool: float, alpha_pool: float) -> float:
    """Current spot price: TAO per alpha token."""
    if alpha_pool <= 0:
        return 0.0
    return tao_pool / alpha_pool


def slippage_pct(trade_tao: float, tao_pool: float) -> float:
    """Exact price impact for a buy of ``trade_tao`` on a constant-product AMM.

    Formula: trade_tao / (tao_pool + trade_tao)
    This is the fraction of the pool you're consuming, which equals the
    percentage your effective price exceeds the spot price.

    Returns a float in [0, 1). Multiply by 100 for percent.
    """
    if tao_pool <= 0:
        return 1.0
    if trade_tao <= 0:
        return 0.0
    return trade_tao / (tao_pool + trade_tao)


def slippage_pct_decimal(trade_amount: float, pool_depth: float) -> Decimal:
    """Slippage using Decimal arithmetic for precision-sensitive contexts.

    Avoids float rounding errors (e.g. 3.0000000004% instead of 3%).
    Returns percentage (e.g. Decimal('3.0') for 3%).
    """
    if pool_depth <= 0:
        return Decimal("100")
    amt = Decimal(str(trade_amount))
    pool = Decimal(str(pool_depth))
    return (amt / (pool + amt)) * Decimal("100")


def max_trade_for_slippage(tao_pool: float, max_slippage: float) -> float:
    """Maximum TAO trade that stays within ``max_slippage`` price impact.

    Args:
        tao_pool: Current TAO in the pool.
        max_slippage: Maximum acceptable slippage as a fraction (e.g. 0.05 for 5%).

    Returns:
        Maximum TAO that can be traded. Returns 0 if pool is empty or
        max_slippage is non-positive.
    """
    if tao_pool <= 0 or max_slippage <= 0:
        return 0.0
    if max_slippage >= 1.0:
        return float("inf")
    # From slippage = trade / (pool + trade), solve for trade:
    # trade = pool * slippage / (1 - slippage)
    return tao_pool * max_slippage / (1.0 - max_slippage)


def effective_price(tao_spend: float, tao_pool: float, alpha_pool: float
                    ) -> float:
    """Effective price paid per alpha token for a given TAO spend.

    Returns TAO per alpha. Returns 0 if no alpha would be received.
    """
    alpha_out, _, _ = amm_buy(tao_spend, tao_pool, alpha_pool)
    if alpha_out <= 0:
        return 0.0
    return tao_spend / alpha_out


def rate_tolerance(trade_tao: float, tao_pool: float,
                   buffer_pct: float = 2.0) -> float:
    """Compute on-chain rate_tolerance for a trade.

    Returns the slippage percentage plus a buffer, suitable for passing
    to ``add_stake(rate_tolerance=...)`` or ``unstake(rate_tolerance=...)``.
    Result is a fraction (e.g. 0.05 for 5%).

    Args:
        trade_tao: TAO amount to trade.
        tao_pool: Current TAO in pool.
        buffer_pct: Extra buffer in percentage points (default 2%).
    """
    slip = slippage_pct(trade_tao, tao_pool)
    return slip + buffer_pct / 100.0
