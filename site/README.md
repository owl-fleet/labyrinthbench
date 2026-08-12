# labyrinthbench.ai — static site

Eleventy 3 site: leaderboard (reads `../entries/*.json`), methodology (renders `../METHODOLOGY.md`), and the `/data` annex.

## Local preview

```bash
bash dev.sh          # containerized build + serve on http://<host>:8123, rebuilds on edit
bash dev.sh stop
PORT=9000 bash dev.sh
```

Needs only Docker. With Node installed you can instead run `npm ci && npx eleventy --serve` here.

## Production build (CF Pages)

```bash
npm ci && npm run build   # output: _site/
```

## Regenerating entries

```bash
python3 ../cli/seed_entries.py   # from labyrinthbench/; writes ../entries/*.json
```

Preview tip: `?curves=N` on the leaderboard overrides the distribution-curve n-gate (default 15; `?curves=0` always draws).
