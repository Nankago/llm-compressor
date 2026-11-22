-- Apply to the SQLite export of the idle H100 trace.
WITH first_search AS (
    SELECT n.start, n.end, n.globalTid
    FROM NVTX_EVENTS AS n LEFT JOIN StringIds AS s ON n.textId = s.id
    WHERE coalesce(n.text, s.value) = 'GPTQ/scale_search' AND n.end IS NOT NULL
    ORDER BY n.start LIMIT 1
), longest_api AS (
    SELECT f.start AS scale_start, f.end AS scale_end, o.start, o.end,
           o.globalTid, s.value AS api
    FROM first_search AS f
    JOIN OSRT_API AS o ON o.globalTid = f.globalTid
        AND o.start >= f.start AND o.end <= f.end
    JOIN StringIds AS s ON s.id = o.nameId
    ORDER BY o.end - o.start DESC LIMIT 1
), switches AS (
    SELECT start, lead(start) OVER (ORDER BY start) AS next_start, isSchedIn,
           lead(isSchedIn) OVER (ORDER BY start) AS next_sched_in
    FROM SCHED_EVENTS WHERE globalTid = (SELECT globalTid FROM longest_api)
), device_ops AS (
    SELECT start, end, globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL
    UNION ALL SELECT start, end, globalPid FROM CUPTI_ACTIVITY_KIND_MEMCPY
    UNION ALL SELECT start, end, globalPid FROM CUPTI_ACTIVITY_KIND_MEMSET
)
SELECT (a.scale_end-a.scale_start)/1e9 AS first_search_s, a.api,
       a.start AS api_start_ns, a.end AS api_end_ns, (a.end-a.start)/1e9 AS api_wall_s,
       (SELECT sum(min(w.next_start,a.end)-max(w.start,a.start))/1e9
        FROM switches AS w WHERE w.isSchedIn=0 AND w.next_sched_in<>0
        AND w.start<a.end AND w.next_start>a.start) AS api_off_cpu_s,
       (SELECT count(*) FROM device_ops AS d WHERE d.globalPid=(a.globalTid & ~16777215)
        AND d.start<a.end AND d.end>a.start) AS overlapping_gpu_records
FROM longest_api AS a;
