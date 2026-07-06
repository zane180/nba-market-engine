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
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
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

    def _upsert(
        self, table: sa.Table, rows: Sequence[dict[str, Any]], *, update_cols: Sequence[str]
    ) -> int:
        if not rows:
            return 0
        dialect = self._engine.dialect.name
        if dialect == "sqlite":
            stmt = sqlite_insert(table).values(rows)
        elif dialect == "postgresql":
            stmt = pg_insert(table).values(rows)  # type: ignore[assignment]
        else:
            raise NotImplementedError(f"unsupported dialect {dialect!r}")
        pk = [c.name for c in table.primary_key.columns]
        stmt = stmt.on_conflict_do_update(
            index_elements=pk, set_={c: stmt.excluded[c] for c in update_cols}
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)
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

    def candle_count_by_ticker(self) -> dict[str, int]:
        stmt = sa.select(candles_table.c.ticker, sa.func.count()).group_by(candles_table.c.ticker)
        with self._engine.connect() as conn:
            return dict(conn.execute(stmt).all())  # type: ignore[arg-type]
