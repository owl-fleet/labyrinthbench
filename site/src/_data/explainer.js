const fs = require("fs");
const path = require("path");

// The explainer is authored once, in docs/, and rendered here as /what-it-is/.
module.exports = () =>
  fs.readFileSync(
    path.join(__dirname, "..", "..", "..", "docs", "what-labyrinthbench-is.md"),
    "utf8"
  );
