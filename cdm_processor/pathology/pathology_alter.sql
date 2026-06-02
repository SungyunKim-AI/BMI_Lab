-- =============================================================================
-- MI-CDM Pathology Schema Migration
-- 기존에 생성된 테이블에 대한 수정 사항 적용 (PostgreSQL)
-- =============================================================================

-- 1) image_occurrence_pathology.anatomic_site_concept_id 의 FK 제거
ALTER TABLE image_occurrence_pathology
    DROP CONSTRAINT fk_image_occurrence_pathology_specimen;


-- 2) specimen_pathology.specimen_source_value: VARCHAR(50) -> VARCHAR(200)
ALTER TABLE specimen_pathology
    ALTER COLUMN specimen_source_value TYPE VARCHAR(200);


-- 3) measurement_pathology.value_source_value: VARCHAR(50) -> VARCHAR(150)
ALTER TABLE measurement_pathology
    ALTER COLUMN value_source_value TYPE VARCHAR(150);


-- 4) observation_pathology.observation_source_concept_id: INTEGER -> BIGINT
--    SNOMED CT id 가 INTEGER 범위(약 21억) 초과(예: 15931221000119100)
ALTER TABLE observation_pathology
    ALTER COLUMN observation_source_concept_id TYPE BIGINT;


-- 5) observation_pathology.value_as_string, value_source_value: VARCHAR -> VARCHAR(200)
ALTER TABLE observation_pathology
    ALTER COLUMN value_as_string    TYPE VARCHAR(200);

ALTER TABLE observation_pathology
    ALTER COLUMN value_source_value TYPE VARCHAR(200);
