/* ------------------------------------------------------------------ *
 *  GCS access layer
 *
 *  Public GCS buckets expose an S3-compatible XML listing endpoint at
 *  https://storage.googleapis.com/{bucket}?list-type=2[&prefix=...].
 *  Objects are fetchable directly at
 *  https://storage.googleapis.com/{bucket}/{key}.
 *
 *  This module discovers report keys (with a manifest fallback) and
 *  fetches report JSON payloads.
 * ------------------------------------------------------------------ */

(function () {
  const cfg = window.APP_CONFIG;

  /**
   * Fetch and parse one page of the bucket XML listing.
   * Returns { keys: string[], nextToken: string|null }.
   */
  async function listPage(prefix, continuationToken) {
    const params = new URLSearchParams({ "list-type": "2" });
    if (prefix) params.set("prefix", prefix);
    if (continuationToken) params.set("continuation-token", continuationToken);

    const url = `${cfg.LIST_ENDPOINT}?${params.toString()}`;
    const resp = await fetch(url, { method: "GET", cache: "no-store" });
    if (!resp.ok) {
      throw new Error(`Listing failed: HTTP ${resp.status}`);
    }
    const text = await resp.text();
    const xml = new DOMParser().parseFromString(text, "application/xml");

    if (xml.querySelector("parsererror")) {
      throw new Error("Listing response was not valid XML");
    }

    const keys = Array.from(xml.getElementsByTagName("Key")).map(
      (n) => n.textContent
    );

    const truncated =
      (xml.getElementsByTagName("IsTruncated")[0] || {}).textContent === "true";
    const tokenNode = xml.getElementsByTagName("NextContinuationToken")[0];
    const nextToken = truncated && tokenNode ? tokenNode.textContent : null;

    return { keys, nextToken };
  }

  /**
   * List all report keys across the configured prefixes via the XML
   * listing endpoint. Handles pagination via continuation tokens.
   */
  async function listViaXml() {
    const all = new Set();
    for (const prefix of cfg.REPORT_PREFIXES) {
      let token = null;
      let guard = 0;
      do {
        const { keys, nextToken } = await listPage(prefix, token);
        keys.forEach((k) => all.add(k));
        token = nextToken;
        guard += 1;
      } while (token && guard < 100);
    }
    return filterReportKeys(Array.from(all));
  }

  /**
   * Fallback: load a manifest.json from the bucket root. The manifest is
   * a JSON array of report object paths (relative to the bucket).
   */
  async function listViaManifest() {
    const url = `${cfg.PUBLIC_BASE}/${cfg.MANIFEST_PATH}`;
    const resp = await fetch(url, { method: "GET", cache: "no-store" });
    if (!resp.ok) {
      throw new Error(`Manifest fetch failed: HTTP ${resp.status}`);
    }
    const data = await resp.json();
    if (!Array.isArray(data)) {
      throw new Error("Manifest must be a JSON array of report paths");
    }
    return filterReportKeys(data);
  }

  /** Keep only .json report keys, drop ignored ones, sort. */
  function filterReportKeys(keys) {
    const ignore = new Set(cfg.IGNORE_KEYS);
    return keys
      .filter((k) => typeof k === "string")
      .filter((k) => k.endsWith(cfg.REPORT_SUFFIX))
      .filter((k) => !ignore.has(k.split("/").pop()))
      .sort();
  }

  /**
   * Discover report keys. Tries the XML listing first; on failure falls
   * back to the manifest. Returns { keys, source }.
   */
  async function discoverReports() {
    try {
      const keys = await listViaXml();
      if (keys.length) return { keys, source: "bucket listing" };
      // Empty listing — still try the manifest as a secondary source.
      throw new Error("Listing returned no reports");
    } catch (xmlErr) {
      try {
        const keys = await listViaManifest();
        return { keys, source: "manifest.json" };
      } catch (manifestErr) {
        const e = new Error(
          `Could not discover reports. Listing: ${xmlErr.message}; ` +
            `Manifest: ${manifestErr.message}`
        );
        e.both = true;
        throw e;
      }
    }
  }

  /** Fetch and parse a single report by its bucket-relative key. */
  async function fetchReportByKey(key) {
    const url = `${cfg.PUBLIC_BASE}/${key}`;
    return fetchReportByUrl(url);
  }

  /** Fetch and parse a single report by an absolute URL. */
  async function fetchReportByUrl(url) {
    const resp = await fetch(url, { method: "GET", cache: "no-store" });
    if (!resp.ok) {
      throw new Error(`Report fetch failed: HTTP ${resp.status}`);
    }
    return resp.json();
  }

  window.GCS = {
    discoverReports,
    fetchReportByKey,
    fetchReportByUrl,
  };
})();
