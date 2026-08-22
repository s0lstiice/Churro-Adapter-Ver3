# Audited LOC inventory

This directory contains a dated, fail-closed audit of the
[Charles S. Hamlin Papers](https://www.loc.gov/collections/charles-s-hamlin-papers/).

The live API audit ran from 2026-08-20 21:49:16 UTC through 23:08:18 UTC. It
queried 204 collection items and accepted 157 image-only items whose item JSON
did not advertise online text, full text, transcript fields, word coordinates,
text-service URLs, or textual resource files. It produced 14,637 page records.

## Citation and provenance fields

- `inventory_summary.json` cites the LOC collection and records the policy and
  aggregate counts.
- `accepted_items.jsonl` has one canonical `item_url` for every accepted item.
- `rejected_items.jsonl` preserves every rejected item URL and exact signal or
  request failure.
- `verified_remote_pages.jsonl` has one `loc_url`, `item_url`, and LOC IIIF
  `image_url` for every candidate page.

The original source should be cited using the page and item URLs in each row.
The Library of Congress states that its digital scans of the Charles S. Hamlin
Papers are public domain and supplies this credit line:

> Library of Congress, Manuscript Division, Charles S. Hamlin Papers.

See an example [LOC Rights & Access statement](https://www.loc.gov/item/mss246610001/#rights-and-access).

The audit is evidence about the API response during a fixed time window, not a
permanent guarantee that a page remains untranscribed. Re-audit before
publication or before selecting new inference pages.
