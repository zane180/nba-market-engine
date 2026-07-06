# Test fixtures

Real API payloads captured 2026-07-06, trimmed for size but structurally intact.
Parsers are tested against these, so a schema-drift failure in tests means the
fixture (and parser) need re-verification against the live API.

| file | source | notes |
|---|---|---|
| `espn_scoreboard.json` | `GET site.api.espn.com/.../nba/scoreboard?dates=20260115` | 3 of 9 events kept; bulky non-parsed keys stripped |
| `espn_summary.json` | `GET .../nba/summary?event=401810433` | header + 5 of 491 plays (spread across periods, incl. first/last) |
| `kalshi_markets.json` | `GET /trade-api/v2/markets?series_ticker=KXNBAGAME` | 2 settled playoff markets (2026 Finals game 5) |
| `kalshi_event.json` | `GET /trade-api/v2/events/KXNBAGAME-26JUN13NYKSAS` | event + its 2 markets |
| `kalshi_candlesticks.json` | `GET /trade-api/v2/series/KXNBAGAME/markets/.../candlesticks` | 3 one-minute candles |
| `kalshi_orderbook.json` | `GET /trade-api/v2/markets/{t}/orderbook?depth=5` | from a liquid open MLB game market (NBA offseason at capture time); levels ascending, fractional quantities |
| `kalshi_trades.json` | `GET /trade-api/v2/markets/trades?ticker=...` | 3 trades |
