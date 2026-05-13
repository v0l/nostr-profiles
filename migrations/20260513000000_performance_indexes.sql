-- Performance indexes for the stats query and other common access patterns.
--
-- idx_classifications_epoch_pubkey: the stats query now drives from classifications,
--   so SQLite scans the classification table (~5K rows) instead of classification_labels
--   (~75K rows). This index covers the WHERE classification_epoch >= ? predicate.
--
-- idx_classification_labels_score: lets SQLite skip below-threshold labels during
--   the join without touching those rows at all.
CREATE INDEX idx_classifications_epoch_pubkey ON classifications (classification_epoch, pubkey);
CREATE INDEX idx_classification_labels_score ON classification_labels (score);
