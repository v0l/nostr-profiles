-- Add lud16, lud06, and website columns to the profiles table
ALTER TABLE profiles ADD COLUMN lud16 TEXT;
ALTER TABLE profiles ADD COLUMN lud06 TEXT;
ALTER TABLE profiles ADD COLUMN website TEXT;

-- Recreate FTS table with display_name, lud16, lud06, website columns
-- FTS5 doesn't support ALTER TABLE, so we drop and recreate.
-- Data will be rebuilt on startup if FTS is empty.
DROP TABLE IF EXISTS classifications_fts;
CREATE VIRTUAL TABLE classifications_fts USING fts5(
    name,
    display_name,
    about,
    nip05,
    lud16,
    lud06,
    website,
    labels,
    scores,
    bio,
    pubkey UNINDEXED
);