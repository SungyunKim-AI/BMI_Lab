-- =============================================================================
-- Grant SELECT, INSERT, UPDATE, DELETE on MI-CDM Pathology tables to haramyeom
-- Dialect: PostgreSQL
-- Prerequisite: 'haramyeom' role must exist (CREATE ROLE haramyeom LOGIN ...).
-- =============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE ON specimen_pathology            TO haramyeom;
GRANT SELECT, INSERT, UPDATE, DELETE ON image_occurrence_pathology    TO haramyeom;
GRANT SELECT, INSERT, UPDATE, DELETE ON image_feature_pathology       TO haramyeom;
GRANT SELECT, INSERT, UPDATE, DELETE ON fact_relationship_pathology   TO haramyeom;
GRANT SELECT, INSERT, UPDATE, DELETE ON observation_pathology         TO haramyeom;
GRANT SELECT, INSERT, UPDATE, DELETE ON measurement_pathology         TO haramyeom;
GRANT SELECT, INSERT, UPDATE, DELETE ON note_pathology                TO haramyeom;
GRANT SELECT, INSERT, UPDATE, DELETE ON note_nlp_pathology            TO haramyeom;
GRANT SELECT, INSERT, UPDATE, DELETE ON visit_occurrence_pathology    TO haramyeom;
GRANT SELECT, INSERT, UPDATE, DELETE ON procedure_occurrence_pathology TO haramyeom;
