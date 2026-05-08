-- Add kind_breakdown JSON column to store the event kind breakdown at classification time
ALTER TABLE classifications ADD COLUMN kind_breakdown TEXT NOT NULL DEFAULT '[]';
