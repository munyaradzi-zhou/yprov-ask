"""The widget shows an answer's Markdown as headings, lists and paragraphs, not raw "#" and
"-", and never reads the answer as HTML. Runs the widget's own code under Node (skipped when
Node is not installed), with a minimal stand-in for the DOM."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

WIDGET = Path(__file__).parents[1] / "yprov_ask" / "static" / "widget" / "yprov-ask.js"

HARNESS = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[1], "utf8");
const code = src.slice(src.indexOf("  /** An element with optional class"),
                       src.indexOf("  function describeArgs"));
class Element {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.className = ""; }
  append(...nodes) { this.children.push(...nodes); }
  get lastElementChild() { return this.children.filter((c) => typeof c !== "string").at(-1); }
  set textContent(t) { this.children = [String(t)]; }
}
const document = { createElement: (tag) => new Element(tag) };
const show = (n) => typeof n === "string" ? n.replace(/&/g, "&amp;").replace(/</g, "&lt;")
  : `<${n.tagName.toLowerCase()}${n.start ? ` start="${n.start}"` : ""}>` +
    n.children.map(show).join("") + `</${n.tagName.toLowerCase()}>`;
const answerText = new Function("document", code + "; return answerText;")(document);
let text = "";
process.stdin.on("data", (d) => (text += d));
process.stdin.on("end", () =>
  process.stdout.write(show(answerText(document.createElement("div"), text))));
"""


def render(text: str) -> str:
    out = subprocess.run(
        ["node", "-e", HARNESS, str(WIDGET)],
        input=text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return out.stdout.removeprefix("<div>").removesuffix("</div>")


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")


def test_headings_lists_and_paragraphs():
    html = render(
        "It contains:\n\n- Entities: 39\n- Agents: 1\n\n### Key Attributes:\n"
        "- **Parameters**: the `optimizer`\n  - nested\n\n## Workflow\n1. Tasks: 9\n2) Time\n"
        "\n---\nFirst line.\nSecond line."
    )
    assert html == (
        "<p>It contains:</p>"
        "<ul><li>Entities: 39</li><li>Agents: 1</li></ul>"
        "<h4>Key Attributes</h4>"
        "<ul><li><strong>Parameters</strong>: the <code>optimizer</code>"
        "<ul><li>nested</li></ul></li></ul>"
        "<h3>Workflow</h3>"
        "<ol><li>Tasks: 9</li><li>Time</li></ol>"
        "<hr></hr>"
        "<p>First line.<br></br>Second line.</p>"
    )


def test_an_answer_is_never_read_as_html():
    html = render("# <img src=x onerror=alert(1)>\n<script>alert(1)</script> and 2 * 3 * 4")
    assert "<img" not in html and "<script" not in html
    assert html == (
        "<h3>&lt;img src=x onerror=alert(1)></h3>"
        "<p>&lt;script>alert(1)&lt;/script> and 2 * 3 * 4</p>"
    )


def test_numbered_list_keeps_its_first_number_and_names_keep_underscores():
    assert render("3. third\n4. fourth") == '<ol start="3"><li>third</li><li>fourth</li></ol>'
    assert render("run example_GR0_0") == "<p>run example_GR0_0</p>"
