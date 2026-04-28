// JS half of the cross-language feature test. Run after Python regenerates
// the fixture. Exits non-zero if features don't match within 1e-5.
//
//   node tests/test_features_js.mjs
//
// (Optionally exposed as `npm test` — see package.json)

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = path.resolve(
  __dirname,
  "../web_control_demo/modules/features.fixture.json",
);
const FEATURES = path.resolve(
  __dirname,
  "../web_control_demo/modules/features.js",
);

const fixture = JSON.parse(await readFile(FIXTURE, "utf-8"));
const { landmarksToFeature } = await import(FEATURES);

const lm = fixture.landmarks.map((p) => ({ x: p[0], y: p[1], z: p[2] }));
const featRight = landmarksToFeature(lm, "Right");
const featLeft = landmarksToFeature(lm, "Left");

let maxDiff = 0;
for (let i = 0; i < featRight.length; i += 1) {
  maxDiff = Math.max(
    maxDiff,
    Math.abs(featRight[i] - fixture.feature_right[i]),
    Math.abs(featLeft[i] - fixture.feature_left[i]),
  );
}

if (maxDiff > 1e-5) {
  console.error(`CROSS-LANG MISMATCH max diff=${maxDiff}`);
  process.exit(1);
}
console.log(`CROSS-LANG OK max diff=${maxDiff.toExponential(2)}`);
