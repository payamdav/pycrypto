/* ------------------------------------------------------------------ *
 *  Configuration
 *
 *  The viewer is deployed per-study at
 *    https://storage.googleapis.com/{bucket}/{studyPrefix}/app/index.html
 *  and is SELF-LOCATING: it derives its own bucket and study folder from
 *  window.location, then lists only that study's reports under
 *  "{studyPrefix}/reports/".
 *
 *  Manual overrides (for local / file:// testing) take precedence:
 *    - ?bucket=... / ?study=... query params, OR
 *    - the hardcoded OVERRIDE_* constants below.
 * ------------------------------------------------------------------ */

(function () {
  // ---- manual override hooks (highest precedence) ---------------------
  // Set these to force a bucket / study when not served from GCS (e.g.
  // opening index.html from disk). Leave null to disable.
  const OVERRIDE_BUCKET = null;
  const OVERRIDE_STUDY = null;

  // ---- fallbacks (lowest precedence) ----------------------------------
  const DEFAULT_BUCKET = "payamdpycryptoreports";
  const DEFAULT_STUDY = "lookback_lookahead_lstm_singlehead_quantile_mse";

  /**
   * Derive {bucket, studyPrefix} from the current location.
   *
   * When served from GCS the pathname looks like:
   *   /{bucket}/{...studyPath...}/app/index.html
   * so the segments are: [bucket, ...studyPath, "app", "index.html"].
   * The app directory is the segment just before the final file segment;
   * everything between the bucket and that "app" dir is the study prefix
   * (supports multi-segment study paths).
   *
   * Query params (?bucket=, ?study=) and the OVERRIDE_* constants win over
   * anything derived from the path.
   */
  function deriveContext() {
    let bucket = null;
    let studyPrefix = null;

    try {
      const loc = window.location;

      if (loc.hostname === "storage.googleapis.com") {
        const segs = loc.pathname.split("/").filter((s) => s.length > 0);
        // segs = [bucket, ...studyPath..., "app", "index.html"]
        if (segs.length >= 1) bucket = segs[0];
        // locate the app directory: the segment right before the final
        // file segment (index.html). Prefer an explicit "app" segment.
        let appDirIndex = segs.lastIndexOf("app");
        if (appDirIndex < 1) {
          // no explicit "app" dir; assume it sits just before the file.
          appDirIndex = segs.length - 1;
        }
        if (appDirIndex >= 1) {
          studyPrefix = segs.slice(1, appDirIndex).join("/");
        }
      }

      // query-param overrides (allow local/file:// testing)
      const qs = new URLSearchParams(loc.search || "");
      if (qs.get("bucket")) bucket = qs.get("bucket");
      if (qs.get("study")) studyPrefix = qs.get("study");
    } catch (e) {
      // window.location unavailable — fall back below.
    }

    // hardcoded overrides take ultimate precedence
    if (OVERRIDE_BUCKET) bucket = OVERRIDE_BUCKET;
    if (OVERRIDE_STUDY) studyPrefix = OVERRIDE_STUDY;

    // final fallbacks
    if (!bucket) bucket = DEFAULT_BUCKET;
    if (studyPrefix === null || studyPrefix === undefined) {
      studyPrefix = DEFAULT_STUDY;
    }
    studyPrefix = String(studyPrefix).replace(/^\/+|\/+$/g, "");

    return { bucket, studyPrefix };
  }

  const ctx = deriveContext();
  const reportsPrefix = ctx.studyPrefix
    ? `${ctx.studyPrefix}/reports/`
    : "reports/";

  window.APP_CONFIG = {
    // Public GCS bucket that holds the observation reports (self-derived).
    BUCKET_NAME: ctx.bucket,

    // The study folder this viewer belongs to (for display + scoping).
    STUDY_PREFIX: ctx.studyPrefix,

    // Public object base URL for fetching individual report JSON files.
    // Pattern: {PUBLIC_BASE}/{path}
    get PUBLIC_BASE() {
      return `https://storage.googleapis.com/${this.BUCKET_NAME}`;
    },

    // XML listing endpoint (list-type=2 = S3-compatible list objects v2).
    // We add &prefix=... at call time when filtering to one study's reports.
    get LIST_ENDPOINT() {
      return `https://storage.googleapis.com/${this.BUCKET_NAME}`;
    },

    // Prefixes under which report JSON files live. Scoped to THIS study's
    // reports/ folder so the listing returns only this study's reports and
    // naturally ignores the sibling app/ files.
    REPORT_PREFIXES: [reportsPrefix],

    // Fallback manifest (a JSON array of report object paths) used if the
    // bucket XML listing fails (e.g. listing not enabled). Resolved under
    // the study's reports/ prefix.
    MANIFEST_PATH: `${reportsPrefix}manifest.json`,

    // Only object keys ending in this suffix are treated as reports.
    REPORT_SUFFIX: ".json",

    // Keys to ignore even if they end with the report suffix.
    IGNORE_KEYS: ["manifest.json"],
  };
})();
