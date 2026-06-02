-- =============================================================================
-- MI-CDM Pathology Schema DDL
-- Source: MI-CDM_Pathology_schema.xlsx
-- Dialect: PostgreSQL
-- Note: 'ㅇ' 표시된 컬럼은 SNUH 확장 필드입니다.
--       FK는 외부 CDM 표준 테이블(PERSON, CONCEPT, VISIT_OCCURRENCE 등)을
--       참조하므로 컬럼 옆 주석으로 표기하고, 동일 스키마 내 테이블 간
--       FK만 ALTER TABLE 로 정의합니다.
-- =============================================================================


-- =============================================================================
-- TABLE: specimen_pathology
-- Description: 병리 검체(Specimen) 정보
-- =============================================================================
CREATE TABLE specimen_pathology (
    specimen_id                         INTEGER             NOT NULL,                   -- 검체 고유 ID (66으로 시작/10자리)
    person_id                           INTEGER             NOT NULL,                   -- FK -> PERSON.person_id
    specimen_concept_id                 INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (OMOP specimen concept)
    specimen_concept_source_id          BIGINT                  NULL,                   -- [SNUH 확장] SNOMED CT specimen concept 매칭 ID
    specimen_type_concept_id            INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (데이터 소스 예: 32817 EHR)
    specimen_concept_source_type        VARCHAR(50)             NULL,                   -- [SNUH 확장] 매핑 정보 출처 [gross/heading/slidekey/request]
    specimen_concept_exact_type         VARCHAR(50)             NULL,                   -- [SNUH 확장] 매핑 type [exact/broadly]
    specimen_date                       DATE                NOT NULL,                   -- 채취일 (현재는 병리 접수일로 저장)
    specimen_datetime                   TIMESTAMP               NULL,                   -- 채취 시간
    quantity                            DOUBLE PRECISION        NULL,                   -- 검체 양 (볼륨/크기)
    unit_concept_id                     INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id (검체 단위)
    anatomic_site_concept_id            INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id (채취 장기)
    anatomic_site_source_id             BIGINT                  NULL,                   -- [SNUH 확장] SNOMED CT organ 매칭 ID
    anatomic_site_concept_exact_type    VARCHAR(50)             NULL,                   -- [SNUH 확장] 채취 부위 매핑 type [exact/broadly]
    disease_status_concept_id           INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id
    specimen_info_source_value          VARCHAR(50)             NULL,                   -- [SNUH 확장] 검체정보 출처 [gross/request/slidekey]
    specimen_source_id                  VARCHAR(50)             NULL,                   -- 원검체 인식번호 (예: S 260000010001)
    specimen_source_value               VARCHAR(200)            NULL,                   -- 검체 이름 원자료
    unit_source_value                   VARCHAR(50)             NULL,                   -- 검체 단위 원자료
    anatomic_site_source_value          VARCHAR(50)             NULL,                   -- 채취 부위 원자료
    disease_status_source_value         VARCHAR(50)             NULL,
    CONSTRAINT pk_specimen_pathology PRIMARY KEY (specimen_id)
);


-- =============================================================================
-- TABLE: image_occurrence_pathology
-- Description: 병리 이미지(WSI) 발생 기록
-- =============================================================================
CREATE TABLE image_occurrence_pathology (
    image_occurrence_id                 INTEGER             NOT NULL,                   -- 이미지 연구 기록 고유 키
    person_id                           INTEGER             NOT NULL,                   -- FK -> PERSON.person_id
    procedure_occurrence_id             INTEGER                 NULL,                   -- 병리검사 오더 (FK 제거됨)
    visit_occurrence_id                 INTEGER                 NULL,                   -- FK -> VISIT_OCCURRENCE.visit_occurrence_id
    anatomic_site_concept_id            INTEGER             NOT NULL,                   -- 슬라이드 채취 장기 (FK 제거됨)
    anatomic_site_source                BIGINT                  NULL,                   -- [SNUH 확장] SNOMED CT 용어
    anatomic_site_source_type           VARCHAR(50)             NULL,                   -- [SNUH 확장] 용어 매핑 type (exact / 상위개념)
    wadors_uri                          TEXT                    NULL,                   -- 웹상 이미지 접속 위치
    local_path                          TEXT                NOT NULL,                   -- 이미지 저장 경로
    image_occurrence_date               DATE                NOT NULL,                   -- 스캔 획득일자 (현재 접수일과 동일)
    image_study_uid                     VARCHAR(250)        NOT NULL,                   -- DICOM Study UID (병리번호 기준 accession number)
    image_series_uid                    VARCHAR(250)        NOT NULL,                   -- DICOM container identifier (슬라이드 단위)
    modality_concept_id                 INTEGER                 NULL,                   -- DICOM modality (병리는 VS->SM 통일)
    modality_concept_source             VARCHAR(50)             NULL,                   -- [SNUH 확장 / SNUH_Required=yes] SM concept_id 미존재로 string 저장
    CONSTRAINT pk_image_occurrence_pathology PRIMARY KEY (image_occurrence_id)
);


-- =============================================================================
-- TABLE: image_feature_pathology
-- Description: 이미지에서 추출된 feature 및 임상 소견 매핑
-- =============================================================================
CREATE TABLE image_feature_pathology (
    image_feature_id                        INTEGER         NOT NULL,                   -- 이미지 feature 고유 ID (55로 시작/10자리)
    person_id                               INTEGER         NOT NULL,                   -- FK -> PERSON.person_id
    image_occurrence_id                     INTEGER             NULL,                   -- 이미지 연구 기록 ID (FK 제거됨)
    image_feature_event_field_concept_id    INTEGER         NOT NULL,                   -- FK -> CONCEPT.concept_id (Measurement / Observation 등 도메인)
    image_feature_event_id                  INTEGER         NOT NULL,                   -- FK -> 도메인 테이블 PK (예: measurement_pathology.measurement_id)
    image_feature_concept_id                INTEGER         NOT NULL,                   -- 관찰값 concept_id (예: 병리결과 상세항목)
    image_feature_type_concept_id           INTEGER             NULL,                   -- feature 출처 (DICOM SR / 알고리즘 등)
    image_finding_concept_id                INTEGER             NULL,                   -- 그룹화된 이미지 소견 concept_id (RadLex 등)
    image_finding_id                        INTEGER             NULL,                   -- 동일 feature 다중(병변 2개 이상) 시 index 번호
    anatomic_site_concept_id                INTEGER             NULL,                   -- 검체 장기명 (FK 제거됨)
    alg_system                              TEXT                NULL,                   -- feature 추출 알고리즘 URI/버전
    alg_datetime                            TIMESTAMP           NULL,                   -- 알고리즘 처리 일시
    CONSTRAINT pk_image_feature_pathology PRIMARY KEY (image_feature_id)
);


-- =============================================================================
-- TABLE: fact_relationship_pathology
-- Description: 두 도메인 fact 간의 관계 정의
-- =============================================================================
CREATE TABLE fact_relationship_pathology (
    domain_concept_id_1                 INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (도메인 1 ID)
    domain_concept_source_1             VARCHAR(50)             NULL,                   -- [SNUH 확장] athena 미존재 시 원본 테이블명
    fact_id_1                           INTEGER             NOT NULL,                   -- 테이블 1 PK
    domain_concept_id_2                 INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (도메인 2 ID)
    domain_concept_source_2             VARCHAR(50)             NULL,                   -- [SNUH 확장] athena 미존재 시 원본 테이블명
    fact_id_2                           INTEGER             NOT NULL,                   -- 테이블 2 PK
    relationship_concept_id             INTEGER             NOT NULL                    -- FK -> CONCEPT.concept_id (관계 ID)
);


-- =============================================================================
-- TABLE: observation_pathology
-- Description: 병리 관측값 (예: 병리학적 진단)
-- =============================================================================
CREATE TABLE observation_pathology (
    observation_id                      INTEGER             NOT NULL,                   -- 관측값 고유 ID
    person_id                           INTEGER             NOT NULL,                   -- FK -> PERSON.person_id
    observation_concept_id              INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (예: 병리학적 진단)
    observation_date                    DATE                NOT NULL,                   -- 관측일 / 병리검사 보고일
    observation_datetime                TIMESTAMP               NULL,
    observation_type_concept_id         INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (정보 소스 예: HIS)
    value_as_number                     DOUBLE PRECISION        NULL,                   -- 숫자형 결과값
    value_as_string                     VARCHAR(200)            NULL,                   -- 문자형 결과값
    value_as_concept_id                 INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id (그룹형 결과값)
    qualifier_concept_id                INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id
    unit_concept_id                     INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id (단위)
    provider_id                         INTEGER                 NULL,                   -- FK -> PROVIDER.provider_id (시행자)
    visit_occurrence_id                 INTEGER                 NULL,                   -- FK -> VISIT_OCCURRENCE.visit_occurrence_id
    visit_detail_id                     INTEGER                 NULL,                   -- FK -> VISIT_DETAIL.visit_detail_id
    observation_source_value            VARCHAR(50)             NULL,                   -- 실제 결과값 (예: gross type)
    observation_source_concept_id       BIGINT                  NULL,                   -- FK -> CONCEPT.concept_id (SNOMED CT id 가 INT 범위 초과)
    unit_source_value                   VARCHAR(50)             NULL,                   -- 단위 원자료
    qualifier_source_value              VARCHAR(50)             NULL,
    value_source_value                  VARCHAR(200)            NULL,                   -- 실제 결과값 (예: solid)
    observation_event_id                INTEGER                 NULL,
    obs_event_field_concept_id          INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id (원자료 위치 컨셉트, 예: pathology report)
    CONSTRAINT pk_observation_pathology PRIMARY KEY (observation_id)
);


-- =============================================================================
-- TABLE: measurement_pathology
-- Description: 병리 검사 결과치 (수치/단위 가능 항목)
-- =============================================================================
CREATE TABLE measurement_pathology (
    measurement_id                      INTEGER             NOT NULL,                   -- 검사 결과치 고유 ID
    person_id                           INTEGER             NOT NULL,                   -- FK -> PERSON.person_id
    measurement_concept_id              INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (예: 종양크기)
    measurement_date                    DATE                NOT NULL,                   -- 측정일 / 병리검사 보고일
    measurement_datetime                TIMESTAMP           NOT NULL,                   -- 측정 일시
    measurement_type_concept_id         INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (정보 소스 예: HIS)
    operator_concept_id                 INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id
    value_as_number                     DOUBLE PRECISION        NULL,                   -- 수치 결과값 (예: 5)
    value_as_concept_id                 INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id (그룹형 결과값)
    unit_concept_id                     INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id (단위)
    range_low                           DOUBLE PRECISION        NULL,                   -- 정상 하한
    range_high                          DOUBLE PRECISION        NULL,                   -- 정상 상한
    provider_id                         INTEGER                 NULL,                   -- FK -> PROVIDER.provider_id (시행자)
    visit_occurrence_id                 INTEGER                 NULL,                   -- FK -> VISIT_OCCURRENCE.visit_occurrence_id
    visit_detail_id                     INTEGER                 NULL,                   -- FK -> VISIT_DETAIL.visit_detail_id
    measurement_source_value            VARCHAR(50)             NULL,                   -- 원자료 (예: size of tumor)
    measurement_source_concept_id       INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id
    unit_source_value                   VARCHAR(50)             NULL,                   -- 단위 원자료
    unit_source_concept_id              INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id
    value_source_value                  VARCHAR(150)            NULL,                   -- 실제 결과값 (예: 5 x 4 x 3cm)
    measurement_event_id                INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id
    meas_event_field_concept_id         INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id (원자료 위치 컨셉트, 예: pathology report)
    CONSTRAINT pk_measurement_pathology PRIMARY KEY (measurement_id)
);


-- =============================================================================
-- TABLE: note_pathology
-- Description: 병리 보고서 원문(Note)
-- =============================================================================
CREATE TABLE note_pathology (
    note_id                             INTEGER             NOT NULL,                   -- Note 고유 ID
    person_id                           INTEGER             NOT NULL,                   -- FK -> PERSON.person_id
    note_date                           DATE                NOT NULL,                   -- 결과 보고일 (현재 SNUH는 검사 시행일)
    note_datetime                       TIMESTAMP               NULL,
    note_type_concept_id                INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (note 생성 출처)
    note_class_concept_id               INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (note 유형)
    note_title                          VARCHAR(250)            NULL,
    encoding_concept_id                 INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (인코딩)
    language_concept_id                 INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (언어)
    provider_id                         INTEGER                 NULL,                   -- FK -> PROVIDER.provider_id
    visit_occurrence_id                 INTEGER                 NULL,                   -- FK -> VISIT_OCCURRENCE.visit_occurrence_id
    visit_detail_id                     INTEGER                 NULL,                   -- FK -> VISIT_DETAIL.visit_detail_id
    note_source_value                   VARCHAR(50)             NULL,                   -- 병리번호 (자번호)
    note_source_value2                  VARCHAR(50)             NULL,                   -- [SNUH 확장] 병리번호 (모번호)
    note_event_id                       INTEGER                 NULL,
    note_event_field_concept_id         INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id
    note_text                           TEXT                NOT NULL,                   -- 보고서 본문
    CONSTRAINT pk_note_pathology PRIMARY KEY (note_id)
);


-- =============================================================================
-- TABLE: note_nlp_pathology
-- Description: Note 본문의 NLP 추출 결과
-- =============================================================================
CREATE TABLE note_nlp_pathology (
    note_nlp_id                         INTEGER             NOT NULL,                   -- NLP 결과 고유 ID
    note_id                             INTEGER             NOT NULL,                   -- FK -> note_pathology.note_id
    section_concept_id                  INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id (섹션)
    snippet                             VARCHAR(250)            NULL,                   -- 추출된 단락
    "offset"                            VARCHAR(50)             NULL,                   -- 텍스트 내 위치(PostgreSQL 예약어로 큰따옴표 처리)
    lexical_variant                     VARCHAR(250)        NOT NULL,                   -- 표현 형태
    note_nlp_concept_id                 INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id
    note_nlp_source_concept_id          INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id
    nlp_system                          VARCHAR(250)            NULL,                   -- 사용된 NLP 시스템
    nlp_date                            DATE                NOT NULL,                   -- NLP 처리일
    nlp_datetime                        TIMESTAMP               NULL,
    term_exists                         VARCHAR(1)              NULL,                   -- 용어 존재 여부 (Y/N)
    term_temporal                       VARCHAR(50)             NULL,
    term_modifiers                      VARCHAR(2000)           NULL,
    CONSTRAINT pk_note_nlp_pathology PRIMARY KEY (note_nlp_id)
);


-- =============================================================================
-- TABLE: visit_occurrence_pathology
-- Description: 기관 방문 기록
-- Note: 시트에 필드명만 정의되어 datatype/제약은 OMOP CDM v5.4 표준을 따름.
-- =============================================================================
CREATE TABLE visit_occurrence_pathology (
    visit_occurrence_id                 INTEGER             NOT NULL,                   -- 방문 고유 ID
    person_id                           INTEGER             NOT NULL,                   -- FK -> PERSON.person_id
    visit_concept_id                    INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (방문 유형)
    visit_start_date                    DATE                NOT NULL,                   -- 방문 시작일
    visit_start_datetime                TIMESTAMP               NULL,                   -- 방문 시작 일시
    visit_end_date                      DATE                NOT NULL,                   -- 방문 종료일
    visit_end_datetime                  TIMESTAMP               NULL,                   -- 방문 종료 일시
    visit_type_concept_id               INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (정보 소스)
    provider_id                         INTEGER                 NULL,                   -- FK -> PROVIDER.provider_id
    care_site_id                        INTEGER                 NULL,                   -- FK -> CARE_SITE.care_site_id
    visit_source_value                  VARCHAR(50)             NULL,                   -- 방문 원자료
    visit_source_concept_id             INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id
    admitted_from_concept_id            INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id (입원 경로)
    admitted_from_source_value          VARCHAR(50)             NULL,
    discharged_to_concept_id            INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id (퇴원 경로)
    discharged_to_source_value          VARCHAR(50)             NULL,
    preceding_visit_occurrence_id       INTEGER                 NULL,                   -- FK -> visit_occurrence_pathology.visit_occurrence_id (직전 방문)
    CONSTRAINT pk_visit_occurrence_pathology PRIMARY KEY (visit_occurrence_id)
);


-- =============================================================================
-- TABLE: procedure_occurrence_pathology
-- Description: 검사/처치(병리 검사 오더 등) 시행 기록
-- Note: 시트의 'precedure_concept_id' 는 'procedure_concept_id' 오타로 보아 정정.
--       'Person_id' 대소문자도 소문자로 정규화. 'ext_ord_nm' 의 datatype 미정('-')
--       이라 VARCHAR(255) 로 기본 지정.
-- =============================================================================
CREATE TABLE procedure_occurrence_pathology (
    procedure_occurrence_id             INTEGER             NOT NULL,                   -- 검사/처치 고유 ID
    person_id                           INTEGER             NOT NULL,                   -- FK -> PERSON.person_id
    procedure_concept_id                INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (병리검사 오더명, SNOMED code 매핑)
    procedure_date                      DATE                NOT NULL,                   -- 검사 시행일 (병리검사 접수일)
    procedure_datetime                  TIMESTAMP           NOT NULL,                   -- 검사 시행 일시
    procedure_end_date                  DATE                    NULL,                   -- 검사 종료일 (병리검사 보고일)
    procedure_end_datetime              TIMESTAMP               NULL,                   -- 검사 종료 일시
    procedure_type_concept_id           INTEGER             NOT NULL,                   -- FK -> CONCEPT.concept_id (정보 소스 예: HIS)
    modifier_concept_id                 INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id (추가 정보)
    quantity                            INTEGER                 NULL,                   -- 횟수
    provider_id                         INTEGER                 NULL,                   -- FK -> PROVIDER.provider_id (시행자)
    visit_occurrence_id                 INTEGER                 NULL,                   -- FK -> visit_occurrence_pathology.visit_occurrence_id
    visit_detail_id                     INTEGER                 NULL,                   -- FK -> VISIT_DETAIL.visit_detail_id
    procedure_source_value              VARCHAR(50)             NULL,                   -- 검사/처치 구분 기준 시스템 (예: EDI code)
    procedure_source_concept_id         INTEGER                 NULL,                   -- FK -> CONCEPT.concept_id (SNOMED CT id)
    modifier_source_value               VARCHAR(50)             NULL,
    ext_ord_nm                          VARCHAR(255)            NULL,                   -- [SNUH 확장] 서울대병원 CDM 컬럼 - 검사명
    CONSTRAINT pk_procedure_occurrence_pathology PRIMARY KEY (procedure_occurrence_id)
);


-- =============================================================================
-- Foreign Keys (스키마 내부 참조)
-- 외부 CDM 표준 테이블(PERSON, CONCEPT, PROVIDER, CARE_SITE, VISIT_DETAIL)
-- 참조 FK는 컬럼 주석으로만 명시하였습니다.
-- =============================================================================
ALTER TABLE image_occurrence_pathology
    ADD CONSTRAINT fk_image_occurrence_pathology_visit_occurrence
        FOREIGN KEY (visit_occurrence_id) REFERENCES visit_occurrence_pathology (visit_occurrence_id);

ALTER TABLE observation_pathology
    ADD CONSTRAINT fk_observation_pathology_visit_occurrence
        FOREIGN KEY (visit_occurrence_id) REFERENCES visit_occurrence_pathology (visit_occurrence_id);

ALTER TABLE measurement_pathology
    ADD CONSTRAINT fk_measurement_pathology_visit_occurrence
        FOREIGN KEY (visit_occurrence_id) REFERENCES visit_occurrence_pathology (visit_occurrence_id);

ALTER TABLE note_pathology
    ADD CONSTRAINT fk_note_pathology_visit_occurrence
        FOREIGN KEY (visit_occurrence_id) REFERENCES visit_occurrence_pathology (visit_occurrence_id);

ALTER TABLE note_nlp_pathology
    ADD CONSTRAINT fk_note_nlp_pathology_note
        FOREIGN KEY (note_id) REFERENCES note_pathology (note_id);

ALTER TABLE procedure_occurrence_pathology
    ADD CONSTRAINT fk_procedure_occurrence_pathology_visit_occurrence
        FOREIGN KEY (visit_occurrence_id) REFERENCES visit_occurrence_pathology (visit_occurrence_id);

ALTER TABLE visit_occurrence_pathology
    ADD CONSTRAINT fk_visit_occurrence_pathology_preceding
        FOREIGN KEY (preceding_visit_occurrence_id) REFERENCES visit_occurrence_pathology (visit_occurrence_id);
