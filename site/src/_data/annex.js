const fs = require("fs");
const path = require("path");

const read = (f) =>
  fs.readFileSync(path.join(__dirname, "..", "..", "..", "docs", "annex", f), "utf8");

module.exports = () => ({
  paired: read("e1a-table1-paired.md"),
  control: read("e1a-table1-control.md"),
  prereg: read("prereg-cohort-campaign.md"),
  a3: read("a3-ceiling-arm.md"),
  lookgate: read("look-gate-result.md"),
  mcv: read("marginal-context-value.md"),
});
