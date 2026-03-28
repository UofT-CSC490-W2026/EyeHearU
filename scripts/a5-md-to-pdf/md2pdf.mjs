#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";
import puppeteer from "puppeteer";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "../..");

marked.use({ gfm: true, breaks: false });

const defaultIn = path.join(REPO_ROOT, "docs/a5_writeup.md");
const defaultOut = path.join(REPO_ROOT, "docs/a5_writeup.pdf");

const input = path.resolve(process.argv[2] || defaultIn);
const output = path.resolve(process.argv[3] || defaultOut);

const css = `
  @page { margin: 14mm 16mm; }
  * { box-sizing: border-box; }
  html {
    font-family: "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #1a1a1a;
  }
  body { max-width: 100%; margin: 0; }
  h1 { font-size: 1.55rem; border-bottom: 1px solid #ccc; padding-bottom: 0.25em; margin-top: 0; }
  h2 { font-size: 1.2rem; margin-top: 1.25em; page-break-after: avoid; }
  h3 { font-size: 1.05rem; margin-top: 1em; page-break-after: avoid; }
  h4 { font-size: 1rem; }
  p { margin: 0.55em 0; }
  a { color: #0969da; text-decoration: none; }
  code {
    font-family: ui-monospace, "Cascadia Mono", "SF Mono", Menlo, monospace;
    font-size: 0.88em;
    background: #f3f4f6;
    padding: 0.12em 0.35em;
    border-radius: 3px;
  }
  pre {
    font-family: ui-monospace, "Cascadia Mono", "SF Mono", Menlo, monospace;
    font-size: 0.82em;
    background: #f6f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 12px 14px;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    page-break-inside: avoid;
  }
  pre code { background: none; padding: 0; font-size: inherit; }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.75em 0;
    font-size: 0.82em;
    page-break-inside: auto;
  }
  th, td {
    border: 1px solid #d0d7de;
    padding: 5px 7px;
    vertical-align: top;
    text-align: left;
    word-break: break-word;
  }
  th { background: #f3f4f6; font-weight: 600; }
  tr { page-break-inside: avoid; }
  hr { border: none; border-top: 1px solid #ddd; margin: 1.25em 0; }
  ul, ol { margin: 0.5em 0; padding-left: 1.35em; }
  blockquote {
    margin: 0.75em 0;
    padding-left: 1em;
    border-left: 3px solid #d0d7de;
    color: #444;
  }
`;

async function main() {
  const md = await fs.readFile(input, "utf8");
  const body = await marked.parse(md);
  const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><style>${css}</style></head><body>${body}</body></html>`;

  const browser = await puppeteer.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: "domcontentloaded" });
    await fs.mkdir(path.dirname(output), { recursive: true });
    await page.pdf({
      path: output,
      format: "A4",
      printBackground: true,
      preferCSSPageSize: true,
      margin: { top: "12mm", bottom: "14mm", left: "14mm", right: "14mm" },
    });
  } finally {
    await browser.close();
  }

  console.error(`Wrote ${output}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
