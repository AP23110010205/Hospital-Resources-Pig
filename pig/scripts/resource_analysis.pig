
-- ==========================================
-- HOSPITAL RESOURCES OPTIMIZATION
-- Apache Pig Data Analysis
-- ==========================================

-- 1. Load all CSV fields as text

raw_data = LOAD 'data/raw/hospital_resources.csv'
USING PigStorage(',')
AS (
    record_date:chararray,
    department:chararray,
    admissions_raw:chararray,
    available_beds_raw:chararray,
    occupied_beds_raw:chararray,
    staff_available_raw:chararray,
    staff_required_raw:chararray,
    age_group:chararray,
    gender:chararray
);

-- 2. Remove CSV header

clean_data = FILTER raw_data
BY record_date != 'date';

-- 3. Convert numeric fields

typed_data = FOREACH clean_data GENERATE
    record_date,
    department,
    (int)admissions_raw AS admissions,
    (int)available_beds_raw AS available_beds,
    (int)occupied_beds_raw AS occupied_beds,
    (int)staff_available_raw AS staff_available,
    (int)staff_required_raw AS staff_required,
    age_group,
    gender;

-- 4. Calculate hospital resource metrics

resource_metrics = FOREACH typed_data GENERATE
    record_date,
    department,
    admissions,
    available_beds,
    occupied_beds,
    ((double)occupied_beds / (double)available_beds) * 100
        AS occupancy_percentage,
    (occupied_beds > available_beds ?
        occupied_beds - available_beds : 0)
        AS bed_shortage,
    (staff_required > staff_available ?
        staff_required - staff_available : 0)
        AS staff_shortage,
    age_group,
    gender;

-- 5. Display analyzed data

DUMP resource_metrics;

-- 6. Store processed results

STORE resource_metrics
INTO 'data/processed/resource_metrics'
USING PigStorage(',');
