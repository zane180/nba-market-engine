"""Persistent storage for the historical dataset.

SQLAlchemy Core (no ORM) over SQLite in dev and Postgres in prod — every
statement here is dialect-generic, and upserts go through the per-dialect
``on_conflict_do_update`` so re-running any backfill is idempotent by
construction.

Timestamps are stored as integer epoch seconds. SQLite has no timezone-aware
column type, and a naive datetime silently round-tripping through the database
is exactly the class of bug the domain models exist to prevent; epoch integers
are unambiguous, sortable, and converted back to aware-UTC at this boundary and
nowhere else.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

import sqlalchemy as sa

if TYPE_CHECKING:
    import numpy as np
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from engine.data.models import Candle, Game, GameStatus, LiveGameState, MarketInfo, MarketResult

SCHEMA_VERSION = 1

metadata = sa.MetaData()

games_table = sa.Table(
    "games",
    metadata,
    sa.Column("game_id", sa.String, primary_key=True),
    sa.Column("home_team", sa.String, nullable=False),
    sa.Column("away_team", sa.String, nullable=False),
    sa.Column("start_time", sa.BigInteger, nullable=False, index=True),
    sa.Column("status", sa.String, nullable=False),
    sa.Column("home_score", sa.Integer, nullable=False),
    sa.Column("away_score", sa.Integer, nullable=False),
    sa.Column("season_type", sa.Integer, nullable=True),
)

snapshots_table = sa.Table(
    "game_snapshots",
    metadata,
    sa.Column("game_id", sa.String, primary_key=True),
    sa.Column("play_id", sa.String, primary_key=True),
    sa.Column("as_of", sa.BigInteger, nullable=False, index=True),
    sa.Column("period", sa.Integer, nullable=False),
    sa.Column("seconds_remaining_in_period", sa.Float, nullable=False),
    sa.Column("home_score", sa.Integer, nullable=False),
    sa.Column("away_score", sa.Integer, nullable=False),
)

markets_table = sa.Table(
    "markets",
    metadata,
    sa.Column("ticker", sa.String, primary_key=True),
    sa.Column("event_ticker", sa.String, nullable=False, index=True),
    sa.Column("game_id", sa.String, nullable=True, index=True),
    sa.Column("yes_team", sa.String, nullable=False),
    sa.Column("result", sa.String, nullable=True),
    sa.Column("open_time", sa.BigInteger, nullable=False),
    sa.Column("close_time", sa.BigInteger, nullable=False),
    sa.Column("volume", sa.Float, nullable=False),
)

candles_table = sa.Table(
    "market_candles",
    metadata,
    sa.Column("ticker", sa.String, primary_key=True),
    sa.Column("end_time", sa.BigInteger, primary_key=True),
    sa.Column("period_seconds", sa.Integer, primary_key=True),
    sa.Column("yes_bid_close", sa.Integer, nullable=True),
    sa.Column("yes_ask_close", sa.Integer, nullable=True),
    sa.Column("trade_close", sa.Integer, nullable=True),
    sa.Column("volume", sa.Float, nullable=False),
    sa.Column("open_interest", sa.Float, nullable=False),
)

paper_trades_table = sa.Table(
    "paper_trades",
    metadata,
    sa.Column("trade_id", sa.String, primary_key=True),  # deterministic: ticker + entry epoch
    sa.Column("ticker", sa.String, nullable=False, index=True),
    sa.Column("game_id", sa.String, nullable=False, index=True),
    sa.Column("yes_team", sa.String, nullable=False),
    sa.Column("entered_at", sa.BigInteger, nullable=False),
    sa.Column("side", sa.String, nullable=False),
    sa.Column("contracts", sa.Float, nullable=False),
    sa.Column("price_cents", sa.Integer, nullable=False),
    sa.Column("fee", sa.Float, nullable=False),
    sa.Column("cost", sa.Float, nullable=False),
    sa.Column("model_prob", sa.Float, nullable=False),
    sa.Column("market_prob", sa.Float, nullable=False),
    sa.Column("settled_at", sa.BigInteger, nullable=True),
    sa.Column("payout", sa.Float, nullable=True),
)

book_snapshots_table = sa.Table(
    "book_snapshots",
    metadata,
    sa.Column("ticker", sa.String, primary_key=True),
    sa.Column("as_of", sa.BigInteger, primary_key=True),
    sa.Column("side", sa.String, primary_key=True),
    sa.Column("level", sa.Integer, primary_key=True),  # 0 = best
    sa.Column("price_cents", sa.Integer, nullable=False),
    sa.Column("quantity", sa.Float, nullable=False),
)

schema_version_table = sa.Table(
    "schema_version",
    metadata,
    sa.Column("version", sa.Integer, primary_key=True),
    sa.Column("applied_at", sa.BigInteger, nullable=False),
)


def _epoch(ts: datetime) -> int:
    if ts.tzinfo is None:
        raise ValueError("naive datetime reached the store boundary")
    return int(ts.timestamp())


def _utc(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch, tz=UTC)


class SchemaVersionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PaperTradeRecord:
    """One would-be trade from the paper loop, persisted for later evaluation."""

    trade_id: str
    ticker: str
    game_id: str
    yes_team: str
    entered_at: datetime
    side: str  # "yes" | "no"
    contracts: float
    price_cents: int
    fee: float
    cost: float
    model_prob: float  # P(YES) at entry
    market_prob: float  # de-vigged market P(YES) at entry
    settled_at: datetime | None = None
    payout: float | None = None

    @property
    def is_open(self) -> bool:
        return self.settled_at is None


class OrderBookLike(Protocol):
    """Structural view of engine.data.models.OrderBook (avoids a hard import
    cycle risk and keeps the store decoupled from parser types)."""

    @property
    def ticker(self) -> str: ...
    @property
    def as_of(self) -> datetime: ...
    @property
    def yes_bids(self) -> Sequence[Any]: ...
    @property
    def no_bids(self) -> Sequence[Any]: ...


@dataclass(frozen=True)
class SnapshotColumns:
    """Column-oriented view of game_snapshots for bulk model training."""

    game_id: list[str]
    as_of_epoch: np.ndarray[Any, np.dtype[np.int64]]
    period: np.ndarray[Any, np.dtype[np.int64]]
    seconds_remaining_in_period: np.ndarray[Any, np.dtype[np.float64]]
    home_score: np.ndarray[Any, np.dtype[np.int64]]
    away_score: np.ndarray[Any, np.dtype[np.int64]]

    def __len__(self) -> int:
        return len(self.game_id)


class Store:
    def __init__(self, database_url: str) -> None:
        self._engine = sa.create_engine(database_url)

    def init_schema(self) -> None:
        """Create tables if missing and stamp/verify the schema version.

        A real multi-version migration chain arrives when the schema first
        changes incompatibly; until then, refusing to run against an unknown
        version is the load-bearing part.
        """
        metadata.create_all(self._engine)
        with self._engine.begin() as conn:
            row = conn.execute(
                sa.select(schema_version_table.c.version).order_by(
                    schema_version_table.c.version.desc()
                )
            ).first()
            if row is None:
                conn.execute(
                    schema_version_table.insert().values(
                        version=SCHEMA_VERSION, applied_at=_epoch(datetime.now(UTC))
                    )
                )
            elif row.version != SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"database is schema v{row.version}, code expects v{SCHEMA_VERSION}"
                )

    # ---------------------------------------------------------------- upserts

    # SQLite caps bound variables (~32k); chunk conservatively by row count so
    # any table up to ~16 columns stays under the limit in one statement.
    _UPSERT_CHUNK_ROWS = 1_000

    def _upsert(
        self, table: sa.Table, rows: Sequence[dict[str, Any]], *, update_cols: Sequence[str]
    ) -> int:
        if not rows:
            return 0
        dialect = self._engine.dialect.name
        pk = [c.name for c in table.primary_key.columns]
        with self._engine.begin() as conn:
            for i in range(0, len(rows), self._UPSERT_CHUNK_ROWS):
                chunk = rows[i : i + self._UPSERT_CHUNK_ROWS]
                if dialect == "sqlite":
                    stmt = sqlite_insert(table).values(chunk)
                elif dialect == "postgresql":
                    stmt = pg_insert(table).values(chunk)  # type: ignore[assignment]
                else:
                    raise NotImplementedError(f"unsupported dialect {dialect!r}")
                conn.execute(
                    stmt.on_conflict_do_update(
                        index_elements=pk, set_={c: stmt.excluded[c] for c in update_cols}
                    )
                )
        return len(rows)

    def upsert_games(self, games: Iterable[Game]) -> int:
        rows = [
            {
                "game_id": g.game_id,
                "home_team": g.home_team,
                "away_team": g.away_team,
                "start_time": _epoch(g.start_time),
                "status": g.status.value,
                "home_score": g.home_score,
                "away_score": g.away_score,
                "season_type": g.season_type,
            }
            for g in games
        ]
        return self._upsert(
            games_table,
            rows,
            update_cols=["status", "home_score", "away_score", "start_time", "season_type"],
        )

    def upsert_snapshots(self, snapshots: Iterable[LiveGameState]) -> int:
        rows = [
            {
                "game_id": s.game_id,
                "play_id": s.source_play_id,
                "as_of": _epoch(s.as_of),
                "period": s.period,
                "seconds_remaining_in_period": s.seconds_remaining_in_period,
                "home_score": s.home_score,
                "away_score": s.away_score,
            }
            for s in snapshots
            if s.source_play_id is not None  # no stable key -> cannot dedupe -> don't store
        ]
        return self._upsert(
            snapshots_table,
            rows,
            update_cols=[
                "as_of",
                "period",
                "seconds_remaining_in_period",
                "home_score",
                "away_score",
            ],
        )

    def upsert_markets(self, markets: Iterable[MarketInfo]) -> int:
        rows = [
            {
                "ticker": m.ticker,
                "event_ticker": m.event_ticker,
                "game_id": m.game_id,
                "yes_team": m.yes_team,
                "result": m.result.value if m.result else None,
                "open_time": _epoch(m.open_time),
                "close_time": _epoch(m.close_time),
                "volume": m.volume,
            }
            for m in markets
        ]
        return self._upsert(
            markets_table, rows, update_cols=["game_id", "result", "close_time", "volume"]
        )

    def upsert_candles(self, candles: Iterable[Candle]) -> int:
        rows = [
            {
                "ticker": c.ticker,
                "end_time": _epoch(c.end_time),
                "period_seconds": c.period_seconds,
                "yes_bid_close": c.yes_bid_close,
                "yes_ask_close": c.yes_ask_close,
                "trade_close": c.trade_close,
                "volume": c.volume,
                "open_interest": c.open_interest,
            }
            for c in candles
        ]
        return self._upsert(
            candles_table,
            rows,
            update_cols=[
                "yes_bid_close",
                "yes_ask_close",
                "trade_close",
                "volume",
                "open_interest",
            ],
        )

    # ----------------------------------------------------------------- reads

    def games(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        statuses: Sequence[GameStatus] | None = None,
    ) -> list[Game]:
        stmt = sa.select(games_table).order_by(games_table.c.start_time)
        if start is not None:
            stmt = stmt.where(games_table.c.start_time >= _epoch(start))
        if end is not None:
            stmt = stmt.where(games_table.c.start_time < _epoch(end))
        if statuses is not None:
            stmt = stmt.where(games_table.c.status.in_([s.value for s in statuses]))
        with self._engine.connect() as conn:
            return [self._row_to_game(row) for row in conn.execute(stmt)]

    def game(self, game_id: str) -> Game | None:
        stmt = sa.select(games_table).where(games_table.c.game_id == game_id)
        with self._engine.connect() as conn:
            row = conn.execute(stmt).first()
        return self._row_to_game(row) if row else None

    @staticmethod
    def _row_to_game(row: sa.Row[Any]) -> Game:
        return Game(
            game_id=row.game_id,
            home_team=row.home_team,
            away_team=row.away_team,
            start_time=_utc(row.start_time),
            status=GameStatus(row.status),
            home_score=row.home_score,
            away_score=row.away_score,
            season_type=row.season_type,
        )

    def markets(self, *, with_game_only: bool = False) -> list[MarketInfo]:
        stmt = sa.select(markets_table).order_by(markets_table.c.close_time)
        if with_game_only:
            stmt = stmt.where(markets_table.c.game_id.is_not(None))
        with self._engine.connect() as conn:
            return [
                MarketInfo(
                    ticker=row.ticker,
                    event_ticker=row.event_ticker,
                    game_id=row.game_id,
                    yes_team=row.yes_team,
                    result=MarketResult(row.result) if row.result else None,
                    open_time=_utc(row.open_time),
                    close_time=_utc(row.close_time),
                    volume=row.volume,
                )
                for row in conn.execute(stmt)
            ]

    def candles(self, ticker: str) -> list[Candle]:
        stmt = (
            sa.select(candles_table)
            .where(candles_table.c.ticker == ticker)
            .order_by(candles_table.c.end_time)
        )
        with self._engine.connect() as conn:
            return [
                Candle(
                    ticker=row.ticker,
                    end_time=_utc(row.end_time),
                    period_seconds=row.period_seconds,
                    yes_bid_close=row.yes_bid_close,
                    yes_ask_close=row.yes_ask_close,
                    trade_close=row.trade_close,
                    volume=row.volume,
                    open_interest=row.open_interest,
                )
                for row in conn.execute(stmt)
            ]

    def snapshots(self, game_id: str) -> list[LiveGameState]:
        stmt = (
            sa.select(snapshots_table)
            .where(snapshots_table.c.game_id == game_id)
            .order_by(snapshots_table.c.as_of)
        )
        with self._engine.connect() as conn:
            return [
                LiveGameState(
                    game_id=row.game_id,
                    as_of=_utc(row.as_of),
                    period=row.period,
                    seconds_remaining_in_period=row.seconds_remaining_in_period,
                    home_score=row.home_score,
                    away_score=row.away_score,
                    source_play_id=row.play_id,
                )
                for row in conn.execute(stmt)
            ]

    def counts(self) -> dict[str, int]:
        with self._engine.connect() as conn:
            return {
                table.name: conn.execute(sa.select(sa.func.count()).select_from(table)).scalar_one()
                for table in (games_table, snapshots_table, markets_table, candles_table)
            }

    # ------------------------------------------------------- paper trading

    def insert_paper_trade(self, trade: PaperTradeRecord) -> None:
        self._upsert(
            paper_trades_table,
            [
                {
                    "trade_id": trade.trade_id,
                    "ticker": trade.ticker,
                    "game_id": trade.game_id,
                    "yes_team": trade.yes_team,
                    "entered_at": _epoch(trade.entered_at),
                    "side": trade.side,
                    "contracts": trade.contracts,
                    "price_cents": trade.price_cents,
                    "fee": trade.fee,
                    "cost": trade.cost,
                    "model_prob": trade.model_prob,
                    "market_prob": trade.market_prob,
                    "settled_at": _epoch(trade.settled_at) if trade.settled_at else None,
                    "payout": trade.payout,
                }
            ],
            update_cols=["settled_at", "payout"],
        )

    def settle_paper_trade(self, trade_id: str, *, settled_at: datetime, payout: float) -> None:
        stmt = (
            paper_trades_table.update()
            .where(paper_trades_table.c.trade_id == trade_id)
            .values(settled_at=_epoch(settled_at), payout=payout)
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)

    def paper_trades(self, *, open_only: bool = False) -> list[PaperTradeRecord]:
        stmt = sa.select(paper_trades_table).order_by(paper_trades_table.c.entered_at)
        if open_only:
            stmt = stmt.where(paper_trades_table.c.settled_at.is_(None))
        with self._engine.connect() as conn:
            return [
                PaperTradeRecord(
                    trade_id=row.trade_id,
                    ticker=row.ticker,
                    game_id=row.game_id,
                    yes_team=row.yes_team,
                    entered_at=_utc(row.entered_at),
                    side=row.side,
                    contracts=row.contracts,
                    price_cents=row.price_cents,
                    fee=row.fee,
                    cost=row.cost,
                    model_prob=row.model_prob,
                    market_prob=row.market_prob,
                    settled_at=_utc(row.settled_at) if row.settled_at is not None else None,
                    payout=row.payout,
                )
                for row in conn.execute(stmt)
            ]

    def paper_bankroll(self, initial: float) -> float:
        """Cash bankroll: initial - all costs + settled payouts. Open positions
        are not marked to market (consistent with the backtest's at-cost view)."""
        trades = self.paper_trades()
        return (
            initial
            - sum(t.cost for t in trades)
            + sum(t.payout for t in trades if t.payout is not None)
        )

    def insert_book_snapshot(self, book: OrderBookLike) -> int:
        rows = []
        for side, levels in (("yes", book.yes_bids), ("no", book.no_bids)):
            for level, entry in enumerate(levels):
                rows.append(
                    {
                        "ticker": book.ticker,
                        "as_of": _epoch(book.as_of),
                        "side": side,
                        "level": level,
                        "price_cents": entry.price,
                        "quantity": entry.quantity,
                    }
                )
        return self._upsert(book_snapshots_table, rows, update_cols=["price_cents", "quantity"])

    def book_snapshot_count(self) -> int:
        with self._engine.connect() as conn:
            return conn.execute(
                sa.select(sa.func.count()).select_from(book_snapshots_table)
            ).scalar_one()

    def game_ids_with_snapshots(self) -> set[str]:
        stmt = sa.select(snapshots_table.c.game_id).distinct()
        with self._engine.connect() as conn:
            return {row.game_id for row in conn.execute(stmt)}

    def snapshot_columns(self) -> SnapshotColumns:
        """All snapshots as parallel column arrays, streamed in chunks.

        The training set is millions of rows; materializing pydantic models for
        each would cost gigabytes. This is the one read path that trades the
        typed-model boundary for arrays — the columns still come from the same
        validated writes.
        """
        import numpy as np

        stmt = sa.select(
            snapshots_table.c.game_id,
            snapshots_table.c.as_of,
            snapshots_table.c.period,
            snapshots_table.c.seconds_remaining_in_period,
            snapshots_table.c.home_score,
            snapshots_table.c.away_score,
        ).order_by(snapshots_table.c.game_id, snapshots_table.c.as_of)
        game_ids: list[str] = []
        as_of: list[int] = []
        period: list[int] = []
        seconds: list[float] = []
        home: list[int] = []
        away: list[int] = []
        with self._engine.connect() as conn:
            for row in conn.execution_options(yield_per=50_000).execute(stmt):
                game_ids.append(row.game_id)
                as_of.append(row.as_of)
                period.append(row.period)
                seconds.append(row.seconds_remaining_in_period)
                home.append(row.home_score)
                away.append(row.away_score)
        return SnapshotColumns(
            game_id=game_ids,
            as_of_epoch=np.array(as_of, dtype=np.int64),
            period=np.array(period, dtype=np.int64),
            seconds_remaining_in_period=np.array(seconds, dtype=np.float64),
            home_score=np.array(home, dtype=np.int64),
            away_score=np.array(away, dtype=np.int64),
        )

    def candle_count_by_ticker(self) -> dict[str, int]:
        stmt = sa.select(candles_table.c.ticker, sa.func.count()).group_by(candles_table.c.ticker)
        with self._engine.connect() as conn:
            return dict(conn.execute(stmt).all())  # type: ignore[arg-type]
