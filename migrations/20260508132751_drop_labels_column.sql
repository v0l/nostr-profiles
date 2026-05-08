-- Remove redundant labels column (now derived from scores on read)
ALTER TABLE classifications DROP COLUMN labels;
