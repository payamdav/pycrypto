/* ------------------------------------------------------------------ *
 *  Configuration
 *
 *  All deployment-specific constants live here so the app can be
 *  re-pointed at a different bucket / prefix without touching logic.
 * ------------------------------------------------------------------ */

window.APP_CONFIG = {
  // Public GCS bucket that holds the observation reports.
  BUCKET_NAME: "payamdpycryptoreports",

  // Public object base URL for fetching individual report JSON files.
  // Pattern: {PUBLIC_BASE}/{path}
  get PUBLIC_BASE() {
    return `https://storage.googleapis.com/${this.BUCKET_NAME}`;
  },

  // XML listing endpoint (list-type=2 = S3-compatible list objects v2).
  // We add &prefix=... at call time when filtering to one observation set.
  get LIST_ENDPOINT() {
    return `https://storage.googleapis.com/${this.BUCKET_NAME}`;
  },

  // Prefixes under which report JSON files live. The listing walks each
  // prefix; leave one empty string to list the whole bucket.
  // The python sweep writes to "{OBSERVATION_SET_NAME}/..." so we list all.
  REPORT_PREFIXES: [""],

  // Fallback manifest (a JSON array of report object paths) used if the
  // bucket XML listing fails (e.g. listing not enabled). Resolved relative
  // to the bucket public base.
  MANIFEST_PATH: "manifest.json",

  // Only object keys ending in this suffix are treated as reports.
  REPORT_SUFFIX: ".json",

  // Keys to ignore even if they end with the report suffix.
  IGNORE_KEYS: ["manifest.json"],
};
