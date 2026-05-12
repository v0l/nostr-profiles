-- Destructured labels table for fast exact label search.
-- Populated whenever a classification is saved.

CREATE TABLE classification_labels (
    pubkey TEXT NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (pubkey, label)
);

CREATE INDEX idx_classification_labels_label ON classification_labels (label);

-- Backfill existing classifications into the new table
INSERT INTO classification_labels (pubkey, label, score)
    SELECT pubkey, key, value FROM classifications, json_each(scores);
