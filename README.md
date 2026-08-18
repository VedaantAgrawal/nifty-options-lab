# NiftyCorridor

A NIFTY50 options backtester — data and research tooling for NIFTY50 index
options, wrapped in a self-hosted Streamlit UI.

## Status

Early stage but end-to-end functional. In place so far:

- `data/` — NSE bhavcopy download + caching, an expiry-date calendar
  covering NIFTY's historical expiry-day regime changes, and a swappable
  `OptionsChainProvider` interface.
- `models/` — Pydantic schemas for the backtest engine (legs, trades,
  strategy config, parameter sweeps, results).
- `engine/position_builder.py` — computes strikes and looks up entry
  premiums for a strategy config on a given date, with nearest-strike
  fallback for illiquid/missing strikes.
- `engine/simulator.py` — runs one full trade cycle: opens a position
  (via `position_builder`), walks it forward day by day marking it to
  market, exits on a stop-loss or at expiry, and returns a closed `Trade`
  (or `None`, logged, if required margin exceeds capital). Lot size is
  resolved per `entry_date` via `data/lot_size_calendar.py`.
- `engine/metrics.py` — `compute_metrics(trades)` aggregates a list of
  closed `Trade`s into a `MetricsResult`: total return, win rate, max
  drawdown, annualized Sharpe (risk-free rate defaults to 0), avg P&L per
  trade, trade counts, and the equity curve. `num_skipped_insufficient_margin`
  must be tracked and passed in by the caller, since skipped attempts are
  `None`, never `Trade` instances.
- `engine/robustness.py` — two-tier anti-overfitting safeguard for
  parameter sweeps: cheap Deflated Sharpe Ratio (DSR, Bailey & Lopez de
  Prado 2014) computed per config via `compute_sweep_dsr`, and expensive
  Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
  Cross-Validation computed only on the DSR shortlist via `compute_pbo`
  (delegates CSCV to the `purgedcv` package). Wired into `engine/sweep.py`.
- `engine/sweep.py` — `run_sweep(grid, train_range, validation_range) ->
  (df, PBOResult)`: expands a `ParameterGrid`'s Cartesian product into
  `StrategyConfig`s (skipping cross-field-invalid combos), runs each
  through the same reentry loop as `scripts/validate_sample.py`
  (`run_trade_cycle_loop`, the canonical shared implementation — the
  script imports it rather than keeping its own copy) in parallel via
  `multiprocessing`, ranks by a configurable metric on the train range,
  and re-runs only the top N on the validation range so in-sample/
  out-of-sample performance are visible side by side. Every config also
  gets `engine/robustness.py`'s DSR computed on train performance
  (`dsr`, `robustness_flag`, `recommended` columns — dsr<0.90
  deprioritizes a config from `recommended` without dropping its row),
  and PBO/CSCV runs once on the DSR shortlist, returned separately as a
  `PBOResult` (a property of the sweep/selection process, not of any one
  config). The per-config return is a flat `pandas.DataFrame`, not
  `SweepResult` instances (the model has no train-vs-validation shape).
- `scripts/validate_sample.py` — a real, non-optimized validation run: loads
  the last 3 months of NIFTY data, runs the entry/exit/reentry loop for one
  fixed `short_strangle` config, and writes a plain per-trade CSV plus a
  printed summary of margin-skip and missing-strike events. Run via
  `python scripts/validate_sample.py`.
- `app.py` — the Streamlit UI ("NiftyCorridor"). Single-run mode (fill in
  one `StrategyConfig` and run it directly) and sweep mode (fill in a
  `ParameterGrid` and run `run_sweep`), with a Summary / Equity Curve /
  Trade Log / Leaderboard results area. The leaderboard shows train vs.
  validation metrics side by side plus the DSR/robustness columns, a
  sweep-level PBO reliability banner, and deprioritizes (greys out, keeps
  visible) any `robustness_flag="red"` config rather than hiding it. No
  engine/data logic lives here — see [Running locally](#running-locally).

No portfolio-level aggregation beyond what the sweep leaderboard shows.

## Layout

- `data/` — options chain data ingestion, caching, and expiry-date logic.
  See [data/README.md](data/README.md) for details.
- `models/` — Pydantic schemas shared across the backtest engine.
- `engine/` — backtest engine logic (position construction, simulation,
  metrics, robustness/DSR/PBO, parameter sweeps).
- `scripts/` — standalone runnable scripts (not imported by anything else).
- `tests/` — unit tests (pytest).
- `app.py` — the Streamlit UI.

## Running locally

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```
python -m pytest
```

## Deployment

NiftyCorridor is meant to be deployed self-hosted (**not** Streamlit
Community Cloud) at **niftycorridor.vedaantagrawal.com**, on
infrastructure you control. These steps assume a Linux server with
Docker, Docker Compose, and nginx already installed, and DNS you can
edit. `Dockerfile`, `docker-compose.yml`, and `nginx.conf.example` in the
repo root are the artifacts these steps use.

> **Not validated end-to-end**: the actual `docker build` / container run
> could not be tested in the environment these files were written in (no
> Docker available there). The Dockerfile and compose file were written
> carefully and the app itself is tested, but do a real build-and-run
> before trusting this in production — see step 2.

### 1. Point DNS at the server (Cloudflare, proxied)

This deployment sits behind Cloudflare with the proxy **on** (orange
cloud), not plain DNS. In the Cloudflare dashboard, under
`vedaantagrawal.com` → DNS → Records → Add record:

- Type: `A`
- Name: `niftycorridor`
- IPv4 address: `<your-server-ip>`
- Proxy status: **Proxied (orange cloud)**

Then, under SSL/TLS → Overview, set the encryption mode to **Full** or
**Full (strict)** — never **Flexible**. Flexible has Cloudflare talk plain
HTTP to this origin while nginx (once certbot is set up in step 5) forces
an HTTP→HTTPS redirect, and those two fight each other into an infinite
redirect loop. `nginx.conf.example` is written assuming this proxied
setup, and already restores the real visitor IP from Cloudflare's
`CF-Connecting-IP` header (see the comment block at the top of that file
for how to keep Cloudflare's IP ranges current).

Wait for the record to resolve (`dig niftycorridor.vedaantagrawal.com` /
`nslookup`) before continuing.

### 2. Build and run the container

```bash
git clone https://github.com/VedaantAgrawal/niftycorridor.git
cd niftycorridor
docker compose up -d --build
```

This builds the image from `Dockerfile` and starts the container bound to
`127.0.0.1:8501` only — nginx is the sole public entry point, so the app
port is never directly reachable from the internet (bypassing Basic Auth
would otherwise be as easy as hitting `server-ip:8501` directly). It also
creates a named volume (`niftycorridor_cache`) mounted at
`/app/data/cache` inside the container, so the downloaded NIFTY bhavcopy
parquet cache survives container restarts and re-deploys.

Verify:

```bash
docker compose ps
docker compose logs -f
curl http://127.0.0.1:8501/_stcore/health   # should print "ok"
```

### 3. Install the nginx reverse proxy config

```bash
sudo cp nginx.conf.example /etc/nginx/sites-available/niftycorridor
sudo ln -s /etc/nginx/sites-available/niftycorridor /etc/nginx/sites-enabled/
sudo nginx -t                      # validate syntax before reloading
sudo systemctl reload nginx
```

### 4. Set up HTTP Basic Auth

**This is a minimal access gate, not real authentication.** It's a single
shared password for everyone who needs it — enough to stop casual
drive-by access to a real (if personal-scale) trading strategy tool at a
public domain, not enough to stop a determined attacker, and there's no
way to tell users apart or revoke one person's access without changing
the password for everyone. If this ever needs to hold real credentials,
feed real trading decisions for multiple people, or distinguish between
users, that's separate work (a real auth provider in front of nginx, or
auth built into the app).

```bash
sudo apt-get install -y apache2-utils      # provides htpasswd
sudo htpasswd -c /etc/nginx/.htpasswd <username>   # -c creates/overwrites the file
# to add another user to an existing file, drop -c:
#   sudo htpasswd /etc/nginx/.htpasswd <another-username>
sudo systemctl reload nginx
```

### 5. HTTPS via Certbot / Let's Encrypt

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d niftycorridor.vedaantagrawal.com
```

Certbot's HTTP-01 challenge passes through Cloudflare's proxy to this
origin without extra configuration (Cloudflare proxies port 80 too), so
this works the same proxied or not. Certbot edits the installed nginx
config in place to add the TLS directives and an HTTP→HTTPS redirect —
see the commented-out block at the bottom of `nginx.conf.example` for
roughly what it adds. Follow the prompts; certbot also sets up automatic
renewal (systemd timer or cron). Verify renewal works without waiting for
it to matter:

```bash
sudo certbot renew --dry-run
```

### 6. Verify

Visit `https://niftycorridor.vedaantagrawal.com` — you should be prompted
for the Basic Auth username/password, then see the NiftyCorridor UI over
HTTPS with a valid certificate.

### Re-deploying after a code change

```bash
git pull
docker compose up -d --build
```

The `niftycorridor_cache` volume is untouched by this, so re-deploys
don't force re-downloading years of bhavcopy data.

### Disk space, before your first full sweep

Bhavcopy data across the full 2021–present range, especially once you're
sweeping across many strike/stop-loss combinations, adds up. Check the
server has real headroom (`df -h`) before running a large sweep for the
first time — particularly if the disk/partition backing the
`niftycorridor_cache` volume is size-constrained. A disk that fills up
mid-sweep is a worse failure mode than checking first.
