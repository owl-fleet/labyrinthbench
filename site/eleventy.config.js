const markdownIt = require("markdown-it");

module.exports = function (eleventyConfig) {
  const md = markdownIt({ html: true, linkify: true });
  eleventyConfig.addFilter("markdownify", (content) => md.render(content || ""));

  // JSON safe to inline in a <script> block: "</script>" in any string field
  // would otherwise terminate the element at the HTML-parser level.
  eleventyConfig.addFilter("jsonScript", (v) =>
    JSON.stringify(v)
      .replace(/</g, "\\u003c")
      .replace(/\u2028/g, "\\u2028")
      .replace(/\u2029/g, "\\u2029")
  );

  eleventyConfig.addPassthroughCopy({ "src/assets": "assets" });
  eleventyConfig.addPassthroughCopy({
    "node_modules/@observablehq/plot/dist/plot.umd.min.js": "assets/vendor/plot.umd.min.js",
    "node_modules/d3/dist/d3.min.js": "assets/vendor/d3.min.js",
  });
  // entries are served as-is so tools (and people) can fetch the raw data
  eleventyConfig.addPassthroughCopy({ "../entries": "entries" });
  // the explainer's figure, served beside /what-it-is/ so the doc's relative image path
  // resolves the same way on GitHub and on the site
  eleventyConfig.addPassthroughCopy({ "../docs/assets/first-labyrinth.svg": "what-it-is/assets/first-labyrinth.svg" });

  return {
    dir: { input: "src", includes: "_includes", output: "_site" },
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
  };
};
