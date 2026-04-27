CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    input_text TEXT NOT NULL,
    label INTEGER NOT NULL,
    proba DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_ts_idx ON predictions(ts DESC);
