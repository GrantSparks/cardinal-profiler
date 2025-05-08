PRAGMA foreign_keys = OFF;
BEGIN;

-- guard table to mark schema version
CREATE TABLE IF NOT EXISTS meta(
    version INTEGER NOT NULL
);
INSERT OR IGNORE INTO meta(version) VALUES (2);

-- recreate cache table for v2
DROP TABLE IF EXISTS cache;
CREATE TABLE IF NOT EXISTS cache (
    prompt_hash TEXT NOT NULL,
    layer       INTEGER NOT NULL,
    response    TEXT NOT NULL,
    PRIMARY KEY (prompt_hash, layer)
);

COMMIT;
