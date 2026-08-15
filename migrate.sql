-- LabyrinthBench TimescaleDB schema
-- Run once against your TimescaleDB/Postgres instance.
-- One run row per session.

CREATE TABLE IF NOT EXISTS labyrinth_runs (
    ts               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    session_id       UUID            NOT NULL,
    model            TEXT            NOT NULL,
    deg_id           TEXT            NOT NULL,
    found_exit       BOOLEAN         NOT NULL,
    steps_to_exit    INT,                      -- NULL if DNF
    step_budget      INT             NOT NULL,
    optimal_commits  INT             NOT NULL,
    normalized_efficiency FLOAT,              -- optimal / actual; NULL if DNF
    gate_accuracy    FLOAT,                   -- NULL if no gates encountered
    path_correctness FLOAT,
    recovery_rate    FLOAT,
    chain_gate_count INT,                     -- dependent (chain) gates declared in this DEG
    chain_accuracy   FLOAT,                   -- first-attempt correctness on ATTEMPTED chain gates
    knowledge_state_consistency FLOAT,        -- chain answers derivable from model's OWN prior answer (did it execute the program?)
    note_used        BOOLEAN         NOT NULL DEFAULT FALSE,
    elapsed_seconds  FLOAT,
    turns            INT,                     -- total model turns (including observe/inspect)
    run_label        TEXT,                    -- e.g. "baseline-20260520"
    representation   TEXT            NOT NULL DEFAULT 'abstract'
);

SELECT create_hypertable('labyrinth_runs', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS lb_model_deg_ts  ON labyrinth_runs (model, deg_id, ts DESC);
CREATE INDEX IF NOT EXISTS lb_session       ON labyrinth_runs (session_id);
CREATE INDEX IF NOT EXISTS lb_label_ts      ON labyrinth_runs (run_label, ts DESC);

-- Phase 1 re-instrumentation (2026-06-02): chain-reasoning metrics. Additive + idempotent,
-- safe to re-run against an existing labyrinth_runs table.
ALTER TABLE labyrinth_runs ADD COLUMN IF NOT EXISTS chain_gate_count INT;
ALTER TABLE labyrinth_runs ADD COLUMN IF NOT EXISTS chain_accuracy FLOAT;
ALTER TABLE labyrinth_runs ADD COLUMN IF NOT EXISTS knowledge_state_consistency FLOAT;

-- lb-post-release chunk 02 (2026-07-15): provenance columns — the standing gate. base_url is
-- always known; n_ctx_slot is the operator-supplied journal-verified context window (never the
-- CLI --num-ctx flag, which ollama's /v1 endpoint silently drops — the effective-context confound
-- has burned two result sets already).
ALTER TABLE labyrinth_runs ADD COLUMN IF NOT EXISTS base_url TEXT;
ALTER TABLE labyrinth_runs ADD COLUMN IF NOT EXISTS n_ctx_slot INT;

-- Serving-stack provenance (2026-08-15): the rest of the identity tuple — engine, engine build,
-- weights digest, quantization, CHAT TEMPLATE hash, renderer/parser, sampling, n_ctx. Added after
-- an engine upgrade moved a benchmark score 1.000 -> 0.188 -> 1.000 on a FIXED model digest and
-- the records could not see why: they stored `model` and little else. JSONB rather than a column
-- per field because the tuple grows as engines expose more, and a run record is read whole.
ALTER TABLE labyrinth_runs ADD COLUMN IF NOT EXISTS prov JSONB;
