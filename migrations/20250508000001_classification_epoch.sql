-- Store the classification epoch alongside each classification.
-- This lets us determine which profiles need re-classification
-- without clobbering is_classified on the profiles table.
ALTER TABLE classifications ADD COLUMN classification_epoch INTEGER NOT NULL DEFAULT 0;
