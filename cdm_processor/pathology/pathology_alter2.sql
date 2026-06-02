-- =============================================================================
-- MI-CDM Pathology Schema Migration 2
-- 기존에 생성된 테이블에 대한 수정 사항 적용 (PostgreSQL)
-- =============================================================================

-- -----------------------------------------------------------------------
-- 1) DATATYPE: INTEGER -> BIGINT(INT8)
-- -----------------------------------------------------------------------

-- specimen_pathology.specimen_concept_source_id
ALTER TABLE specimen_pathology
    ALTER COLUMN specimen_concept_source_id TYPE BIGINT;

-- specimen_pathology.anatomic_site_source_id
ALTER TABLE specimen_pathology
    ALTER COLUMN anatomic_site_source_id TYPE BIGINT;

-- image_occurrence_pathology.anatomic_site_source
ALTER TABLE image_occurrence_pathology
    ALTER COLUMN anatomic_site_source TYPE BIGINT;


-- -----------------------------------------------------------------------
-- 2) FK 제약 삭제
-- -----------------------------------------------------------------------

-- image_occurrence_pathology.procedure_occurrence_id
ALTER TABLE image_occurrence_pathology
    DROP CONSTRAINT fk_image_occurrence_pathology_procedure_occurrence;

-- image_feature_pathology.anatomic_site_concept_id
ALTER TABLE image_feature_pathology
    DROP CONSTRAINT fk_image_feature_pathology_specimen;

-- image_feature_pathology.image_occurrence_id
ALTER TABLE image_feature_pathology
    DROP CONSTRAINT fk_image_feature_pathology_image_occurrence;


-- -----------------------------------------------------------------------
-- 3) NULL 허용 (NOT NULL -> NULL)
-- -----------------------------------------------------------------------

-- image_occurrence_pathology.procedure_occurrence_id
ALTER TABLE image_occurrence_pathology
    ALTER COLUMN procedure_occurrence_id DROP NOT NULL;

-- image_occurrence_pathology.modality_concept_id
ALTER TABLE image_occurrence_pathology
    ALTER COLUMN modality_concept_id DROP NOT NULL;

-- image_feature_pathology.image_occurrence_id
ALTER TABLE image_feature_pathology
    ALTER COLUMN image_occurrence_id DROP NOT NULL;

-- image_feature_pathology.image_feature_type_concept_id
ALTER TABLE image_feature_pathology
    ALTER COLUMN image_feature_type_concept_id DROP NOT NULL;

-- image_feature_pathology.image_finding_concept_id  (기존에 이미 NULL 이므로 skip 가능, 명시적 실행 무해)
ALTER TABLE image_feature_pathology
    ALTER COLUMN image_finding_concept_id DROP NOT NULL;

-- image_feature_pathology.image_finding_id  (기존에 이미 NULL 이므로 skip 가능, 명시적 실행 무해)
ALTER TABLE image_feature_pathology
    ALTER COLUMN image_finding_id DROP NOT NULL;
