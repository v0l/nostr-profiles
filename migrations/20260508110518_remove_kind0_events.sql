-- Remove metadata (kind 0) events from the events table.
-- Metadata is already stored in the profiles table; including kind 0 in classification
-- inflates event counts and provides no classification value.
DELETE FROM events WHERE kind = 0;
