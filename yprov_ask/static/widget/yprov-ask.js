/*
 * <yprov-ask>: a chat panel for asking questions about provenance, and an MCP client.
 *
 * Drop it into any page:
 *
 *   <script src="https://host/widget/yprov-ask.js"></script>
 *   <yprov-ask endpoint="https://host/mcp" style="height: 600px"></yprov-ask>
 *
 * Attributes
 *   endpoint     the MCP server's streamable HTTP URL     (default: /mcp on this page's origin)
 *   max-turns    questions one chat holds; then the pill   (default: 10)
 *                asks for a new chat
 *   suggestions  starter questions, separated by "|"      (default: three general ones)
 *   file-picker  show a paper clip in the pill to open a   (default: off; the host page
 *                PROV-JSON file from the computer          hands the file over)
 *   theme        "light" or "dark"                        (default: follow the system)
 *   file-url     the store URL of the file being viewed   (default: the page's ?file=, which
 *                                                          is where the yProv Explorer keeps it)
 *   require-file "false" lets it ask with no file open,    (default: a file is required)
 *                over everything the server holds
 *
 * One file at a time. Questions are about the file the user has open, and only that file:
 * the server pins the agent's tools to it. The host page hands the file over in one of two ways:
 *
 *   widget.openFile({ content, name })   a file from the user's computer (text, a parsed
 *                                         object, or a File/Blob); uploaded once, to /files
 *   ?file=<url> in the page address, or the file-url attribute
 *                                         opened once by URL (the server fetches it)
 *
 * Opening a different file starts a new chat.
 *
 * Laid out like a search page: the name, the question pill under it, then the chat, user
 * questions in blue pills on the right and the system's answers in white cards on the left.
 * Before the first question the name and pill sit in the middle with starter questions; after
 * it they move to the top.
 *
 * The element fills the box it is given; its styles are in a Shadow DOM, so the host page's
 * CSS doesn't reach in and ours doesn't leak out. Each question is one call to the server's
 * `ask` tool over MCP; the answer, its grounding verdict and the tools used come back from
 * the server. The server issues a conversation id with the first answer; the widget sends it
 * back with each follow-up, so the agent sees the earlier turns. Turns and the id are kept
 * for the browser tab (sessionStorage), so a reload keeps them.
 */
(() => {
  "use strict";

  const VERSION = "0.1.0";
  const PROTOCOL_VERSION = "2025-06-18";
  const DEFAULT_MAX_TURNS = 10;
  const DEFAULT_SUGGESTIONS = [
    "What does this file record?",
    "Which step took the longest?",
    "Did anything fail?",
  ];

  /* ------------------------------------------------------------------ MCP client */

  class SessionExpired extends Error {}

  /** The one JSON-RPC message in a response that answers request `id`. Notifications that
   * come before it (progress) go to `onNotification`. */
  async function readReply(response, id, onNotification) {
    const type = response.headers.get("content-type") || "";
    if (!type.includes("text/event-stream")) {
      const body = await response.json();
      const messages = Array.isArray(body) ? body : [body];
      const reply = messages.find((m) => m.id === id);
      if (!reply) throw new Error("The server's reply did not answer the request.");
      return reply;
    }
    // Server-sent events: the reply may follow notifications (progress, logs), so read
    // events until the one carrying our id arrives.
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n?/g, "\n");
      let end;
      while ((end = buffer.indexOf("\n\n")) >= 0) {
        const event = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        const data = event
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).replace(/^ /, ""))
          .join("\n");
        if (!data) continue;
        const message = JSON.parse(data);
        if (message.id === id) {
          reader.cancel().catch(() => {});
          return message;
        }
        if (message.method && onNotification) onNotification(message);
      }
    }
    throw new Error("The server closed the connection without answering.");
  }

  /** A minimal MCP client over streamable HTTP: initialize once, then call tools. */
  class McpSession {
    constructor(endpoint) {
      this.endpoint = endpoint;
      this.reset();
    }

    reset() {
      this.sessionId = null;
      this.protocolVersion = null;
      this.nextId = 1;
      this.opening = null;
    }

    headers() {
      const headers = {
        "Content-Type": "application/json",
        Accept: "application/json, text/event-stream",
      };
      if (this.sessionId) headers["Mcp-Session-Id"] = this.sessionId;
      if (this.protocolVersion) headers["MCP-Protocol-Version"] = this.protocolVersion;
      return headers;
    }

    async post(message) {
      let response;
      try {
        response = await fetch(this.endpoint, {
          method: "POST",
          headers: this.headers(),
          body: JSON.stringify(message),
        });
      } catch {
        throw new Error(`Can't reach the provenance server at ${this.endpoint}.`);
      }
      // A server that has forgotten our session answers 404; start a new one.
      if (response.status === 404 && this.sessionId) throw new SessionExpired();
      if (!response.ok) throw new Error(`The provenance server answered HTTP ${response.status}.`);
      const sessionId = response.headers.get("mcp-session-id");
      if (sessionId) this.sessionId = sessionId;
      return response;
    }

    async request(method, params, onNotification) {
      const id = this.nextId++;
      const response = await this.post({ jsonrpc: "2.0", id, method, params });
      const reply = await readReply(response, id, onNotification);
      if (reply.error) throw new Error(reply.error.message || "The server returned an error.");
      return reply.result;
    }

    open() {
      this.opening ??= (async () => {
        const result = await this.request("initialize", {
          protocolVersion: PROTOCOL_VERSION,
          capabilities: {},
          clientInfo: { name: "yprov-ask", version: VERSION },
        });
        this.protocolVersion = result.protocolVersion;
        await this.post({ jsonrpc: "2.0", method: "notifications/initialized" });
      })().catch((error) => {
        this.reset();
        throw error;
      });
      return this.opening;
    }

    /** The tool's result: its structured content, else its text parsed as JSON.
     * `onProgress` gets the message of each progress notification the server sends. */
    async callTool(name, args, onProgress) {
      await this.open();
      const token = `p${this.nextId}`;
      const params = { name, arguments: args, _meta: { progressToken: token } };
      const onNotification = (message) => {
        const p = message.params || {};
        if (message.method === "notifications/progress" && p.progressToken === token) {
          if (onProgress && p.message) onProgress(p.message);
        }
      };
      let result;
      try {
        result = await this.request("tools/call", params, onNotification);
      } catch (error) {
        if (!(error instanceof SessionExpired)) throw error;
        this.reset();
        await this.open();
        result = await this.request("tools/call", params, onNotification);
      }
      const text = (result.content || []).map((c) => c.text || "").join("");
      if (result.isError) {
        throw new Error(text.replace(/^Error executing tool \w+: /, "") || "The tool failed.");
      }
      if (result.structuredContent) return result.structuredContent;
      try {
        return JSON.parse(text);
      } catch {
        return { answer: text };
      }
    }
  }

  /* ------------------------------------------------------------------ view */

  /* A search-page layout: the name, the question pill under it, then the chat. Before the
   * first question the name and pill sit in the middle, as on a search home page; after it
   * they move to the top and the turns follow below. The background is the Explorer's warm
   * off-white, lightened; the user's pills are blue, the system's are white cards. */
  const STYLE = `
    :host {
      --ya-bg: #f1efe9; --ya-surface: #ffffff; --ya-ink: #23262b; --ya-muted: #66645d;
      --ya-line: #dcd8ce; --ya-hover: #f7f6f2;
      --ya-user: #dae5f6; --ya-user-ink: #142640;
      --ya-accent: #2f5fa7; --ya-accent-ink: #ffffff; --ya-code: #f0eee8;
      --ya-shadow: 0 1px 2px rgba(60, 52, 36, .08), 0 2px 8px rgba(60, 52, 36, .08);
      --ya-shadow-hi: 0 1px 3px rgba(60, 52, 36, .12), 0 4px 16px rgba(60, 52, 36, .12);
      --ya-ok: #1e6b37; --ya-ok-bg: #e2f1e6; --ya-warn: #8a4b00; --ya-warn-bg: #fbeed6;
      --ya-err: #b3261e; --ya-err-bg: #fbe7e4;
      color-scheme: light; display: block; height: 100%; min-height: 360px;
      container-type: inline-size;
      font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
      color: var(--ya-ink);
    }
    @media (prefers-color-scheme: dark) {
      :host(:not([theme="light"])) {
        color-scheme: dark;
        --ya-bg: #1d1c1a; --ya-surface: #292825; --ya-ink: #e8e6e1; --ya-muted: #aaa69c;
        --ya-line: #3a3833; --ya-hover: #32302c;
        --ya-user: #2b3b55; --ya-user-ink: #e3ecfb;
        --ya-accent: #a9c6f5; --ya-accent-ink: #10213c; --ya-code: #33312d;
        --ya-shadow: 0 1px 2px rgba(0, 0, 0, .4); --ya-shadow-hi: 0 2px 10px rgba(0, 0, 0, .5);
        --ya-ok: #8fd6a2; --ya-ok-bg: #1d3526; --ya-warn: #f4c06a; --ya-warn-bg: #3a2e14;
        --ya-err: #f2b8b5; --ya-err-bg: #3e1d1b;
      }
    }
    :host([theme="dark"]) {
      color-scheme: dark;
      --ya-bg: #1d1c1a; --ya-surface: #292825; --ya-ink: #e8e6e1; --ya-muted: #aaa69c;
      --ya-line: #3a3833; --ya-hover: #32302c;
      --ya-user: #2b3b55; --ya-user-ink: #e3ecfb;
      --ya-accent: #a9c6f5; --ya-accent-ink: #10213c; --ya-code: #33312d;
      --ya-shadow: 0 1px 2px rgba(0, 0, 0, .4); --ya-shadow-hi: 0 2px 10px rgba(0, 0, 0, .5);
      --ya-ok: #8fd6a2; --ya-ok-bg: #1d3526; --ya-warn: #f4c06a; --ya-warn-bg: #3a2e14;
      --ya-err: #f2b8b5; --ya-err-bg: #3e1d1b;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    .panel { height: 100%; display: flex; flex-direction: column; background: var(--ya-bg);
      border-radius: inherit; overflow: hidden; }

    /* The name and the pill. */
    .top { width: 100%; max-width: 720px; margin: 0 auto; padding: 0 16px;
      display: flex; flex-direction: column; align-items: center; }
    .home .top { margin-block: auto; padding-bottom: 6vh; }
    .chat .top { padding-top: 14px; }
    .brand { text-align: center; margin-bottom: 24px; }
    .chat .brand { margin-bottom: 10px; }
    h1 { margin: 0; font-weight: 500; letter-spacing: -.02em; line-height: 1.1;
      font-size: clamp(34px, 11cqi, 64px); }
    .chat h1 { font-size: 22px; letter-spacing: -.01em; }
    h1 span { color: var(--ya-accent); }
    .tagline { margin: 10px 0 0; color: var(--ya-muted); font-size: 14px; }
    .chat .tagline { display: none; }

    form { width: 100%; margin: 0; }
    .field { display: flex; align-items: flex-end; gap: 4px; background: var(--ya-surface);
      border: 1px solid var(--ya-line); border-radius: 26px; padding: 5px 6px 5px 8px;
      box-shadow: var(--ya-shadow); transition: box-shadow .15s, border-color .15s; }
    .field:hover, .field:focus-within { box-shadow: var(--ya-shadow-hi); border-color: transparent; }
    .field.off { box-shadow: none; background: var(--ya-hover); }
    textarea { flex: 1; min-width: 0; border: 0; outline: 0; resize: none; background: transparent;
      color: var(--ya-ink); font: inherit; line-height: 22px; padding: 8px 4px 8px 8px;
      max-height: 132px; }
    textarea::placeholder { color: var(--ya-muted); }
    .icon { flex: none; width: 36px; height: 36px; border: 0; border-radius: 50%; padding: 0;
      background: transparent; color: var(--ya-muted); cursor: pointer;
      display: grid; place-items: center; }
    .icon:hover:not(:disabled) { background: var(--ya-hover); color: var(--ya-ink); }
    .icon svg { width: 20px; height: 20px; }
    .send, .send:hover:not(:disabled) { background: var(--ya-accent); color: var(--ya-accent-ink); }
    .send:disabled { background: transparent; color: var(--ya-line); cursor: default; }

    .bar { width: 100%; display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px;
      margin-top: 10px; padding: 0 6px; font-size: 13px; color: var(--ya-muted); min-height: 30px; }
    .home .bar { justify-content: center; }
    .file { display: inline-flex; align-items: center; gap: 6px; max-width: 100%; min-width: 0;
      border: 1px solid var(--ya-line); border-radius: 999px; padding: 3px 12px 3px 10px;
      background: var(--ya-surface); }
    .file .dot { flex: none; width: 7px; height: 7px; border-radius: 50%; background: var(--ya-ok); }
    .file .name { color: var(--ya-ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .file .content-id { font: 11px ui-monospace, Consolas, monospace; white-space: nowrap; }
    .file.bad .dot { background: var(--ya-err); }
    .problem { flex-basis: 100%; text-align: center; color: var(--ya-err); }
    .chat .problem { text-align: left; }
    .count { font-variant-numeric: tabular-nums; }
    .chat .count { margin-left: auto; }
    .ghost { border: 1px solid var(--ya-line); background: transparent; color: var(--ya-ink);
      font: inherit; font-size: 13px; padding: 4px 14px; border-radius: 999px; cursor: pointer; }
    .ghost:hover:not(:disabled) { background: var(--ya-surface); }
    .ghost:disabled { opacity: .45; cursor: default; }

    .chips { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-top: 18px; }
    .chat .chips { display: none; }
    .chip { border: 1px solid var(--ya-line); background: var(--ya-surface); color: var(--ya-ink);
      font: inherit; font-size: 13px; padding: 7px 14px; border-radius: 999px; cursor: pointer; }
    .chip:hover:not(:disabled) { box-shadow: var(--ya-shadow); border-color: transparent; }
    .chip:disabled { opacity: .5; cursor: default; }

    /* The chat. */
    .thread { position: relative; flex: 1; min-height: 0; overflow-y: auto;
      scrollbar-width: thin; }
    .home .thread { display: none; }
    .chat .thread { border-top: 1px solid var(--ya-line); margin-top: 12px; }
    .turns { max-width: 760px; margin: 0 auto; padding: 18px 16px 8px; }
    .note { text-align: center; color: var(--ya-muted); font-size: 12px; margin: 0 0 14px; }

    .turn { margin-bottom: 22px; }
    .q { display: flex; justify-content: flex-end; margin-bottom: 12px; }
    .q div { background: var(--ya-user); color: var(--ya-user-ink); padding: 9px 16px;
      border-radius: 20px 20px 6px 20px; max-width: min(80%, 560px); white-space: pre-wrap;
      overflow-wrap: anywhere; }
    .r { display: flex; align-items: flex-start; gap: 10px; }
    .avatar { flex: none; width: 30px; height: 30px; border-radius: 50%; margin-top: 2px;
      display: grid; place-items: center; font-size: 11px; font-weight: 700; letter-spacing: -.02em;
      background: var(--ya-surface); color: var(--ya-accent); border: 1px solid var(--ya-line); }
    .bubble { min-width: 0; max-width: min(100%, 640px); background: var(--ya-surface);
      border: 1px solid var(--ya-line); border-radius: 6px 20px 20px 20px; padding: 11px 16px 12px; }
    .bubble.error { background: var(--ya-err-bg); border-color: transparent; }
    @container (max-width: 420px) { .avatar { display: none; } .q div { max-width: 88%; } }

    .a { overflow-wrap: anywhere; }
    .a.error { color: var(--ya-err); white-space: pre-wrap; }
    .a > :first-child { margin-top: 0; }
    .a > :last-child { margin-bottom: 0; }
    .a p { margin: 0 0 8px; }
    .a h3, .a h4 { margin: 14px 0 4px; font-weight: 600; line-height: 1.3; }
    .a h3 { font-size: 16px; }
    .a h4 { font-size: 15px; }
    .a ul, .a ol { margin: 0 0 8px; padding-left: 22px; }
    .a li { margin: 2px 0; }
    .a li > ul, .a li > ol { margin: 2px 0 0; }
    .a li::marker { color: var(--ya-muted); }
    .a hr { border: 0; border-top: 1px solid var(--ya-line); margin: 12px 0; }
    .a code { background: var(--ya-code); padding: 1px 5px; border-radius: 4px; }

    .meta { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 10px; margin-top: 10px;
      color: var(--ya-muted); font-size: 12px; }
    .badge { font-weight: 600; padding: 2px 9px; border-radius: 999px; }
    .badge.ACCEPT { color: var(--ya-ok); background: var(--ya-ok-bg); }
    .badge.REVISE { color: var(--ya-warn); background: var(--ya-warn-bg); }
    .badge.plain, .badge.OUT_OF_SCOPE { color: var(--ya-muted); background: var(--ya-code); }
    .why { color: var(--ya-warn); }
    details { margin-top: 8px; font-size: 12px; color: var(--ya-muted); }
    summary { cursor: pointer; width: max-content; border-radius: 6px; }
    details ol { margin: 6px 0 0; padding-left: 20px; }
    details li { margin: 2px 0; overflow-wrap: anywhere; }
    code { font: 12.5px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .retry { border: 1px solid var(--ya-err); background: transparent; color: var(--ya-err);
      font: inherit; font-size: 12px; padding: 3px 12px; border-radius: 999px; cursor: pointer; }

    .thinking { display: flex; align-items: center; gap: 10px; color: var(--ya-muted); }
    .dots { display: inline-flex; gap: 4px; }
    .dots i { width: 6px; height: 6px; border-radius: 50%; background: var(--ya-accent);
      animation: pulse 1.2s infinite ease-in-out; }
    .dots i:nth-child(2) { animation-delay: .15s; } .dots i:nth-child(3) { animation-delay: .3s; }
    @keyframes pulse { 0%, 80%, 100% { opacity: .25; transform: scale(.8); }
      40% { opacity: 1; transform: scale(1); } }
    @media (prefers-reduced-motion: reduce) { .dots i { animation: none; opacity: .6; }
      .field { transition: none; } }

    .limit { text-align: center; margin: 4px 0 18px; color: var(--ya-muted); font-size: 13px; }
    .limit .ghost { margin-left: 8px; }
    .fine { text-align: center; color: var(--ya-muted); font-size: 11px; padding: 6px 16px 10px; }

    button:focus-visible, summary:focus-visible { outline: 2px solid var(--ya-accent);
      outline-offset: 2px; }
  `;

  const SEND_ICON =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" ' +
    'd="M12 4l-1.4 1.4 5.6 5.6H4v2h12.2l-5.6 5.6L12 20l8-8z" transform="rotate(-90 12 12)"/></svg>';
  const ATTACH_ICON =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M16.5 6v11.5a4 4 ' +
    '0 0 1-8 0V5a2.5 2.5 0 0 1 5 0v10.5a1 1 0 0 1-2 0V6H10v9.5a2.5 2.5 0 0 0 5 0V5a4 4 0 0 0-8 ' +
    '0v12.5a5.5 5.5 0 0 0 11 0V6z"/></svg>';

  /** An element with optional class and text; text is set as text, never as HTML. */
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  /** **bold**, *italic* and `code` in one line of an answer; the rest stays text. */
  function inline(node, text) {
    text.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`|\*[^*\s][^*\n]*\*)/).forEach((part) => {
      if (/^\*\*[^*\n]+\*\*$/.test(part)) node.append(el("strong", "", part.slice(2, -2)));
      else if (/^`[^`\n]+`$/.test(part)) node.append(el("code", "", part.slice(1, -1)));
      else if (/^\*[^*\s][^*\n]*\*$/.test(part)) node.append(el("em", "", part.slice(1, -1)));
      else if (part) node.append(part);
    });
    return node;
  }

  /** The answer's Markdown as elements: headings, bullet and numbered lists (nested by
   * indent), rules and paragraphs. Built from text nodes only, never parsed as HTML, so an
   * answer can't inject markup. Anything else is shown as written. */
  function answerText(node, text) {
    let para = null; // the paragraph that a following line continues
    let lists = []; // open lists, outermost first: { indent, node }
    const close = () => {
      para = null;
      lists = [];
    };
    for (const line of text.replace(/\r\n?/g, "\n").split("\n")) {
      let m;
      if (!line.trim()) {
        para = null;
      } else if ((m = /^ {0,3}(#{1,6})\s+(.*?)[\s#]*$/.exec(line))) {
        close();
        node.append(inline(el(m[1].length <= 2 ? "h3" : "h4"), m[2].replace(/:$/, "")));
      } else if (/^ {0,3}([-*_])(\s*\1){2,}\s*$/.test(line)) {
        close();
        node.append(el("hr"));
      } else if ((m = /^(\s*)(?:[-*+•]|(\d{1,3})[.)])\s+(.*)$/.exec(line))) {
        const indent = m[1].replace(/\t/g, "    ").length;
        const tag = m[2] ? "ol" : "ul";
        while (lists.length && lists.at(-1).indent > indent) lists.pop();
        let top = lists.at(-1);
        if (top && top.indent === indent && top.node.tagName.toLowerCase() !== tag) {
          lists.pop();
          top = lists.at(-1);
        }
        if (!top || top.indent < indent) {
          const list = el(tag);
          if (m[2] && m[2] !== "1") list.start = Number(m[2]);
          (top ? top.node.lastElementChild : node).append(list);
          lists.push((top = { indent, node: list }));
        }
        para = null;
        top.node.append(inline(el("li"), m[3]));
      } else if (lists.length && /^\s/.test(line) && !para) {
        // An indented line under a list item continues that item.
        const item = lists.at(-1).node.lastElementChild;
        item.append(el("br"));
        inline(item, line.trim());
      } else {
        lists = [];
        if (para) para.append(el("br"));
        else node.append((para = el("p")));
        inline(para, line.trim());
      }
    }
    return node;
  }

  function describeArgs(args) {
    const entries = Object.entries(args || {});
    return entries.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
  }

  /** A short fingerprint of a text (FNV-1a, 32 bits), to tell one file from another. */
  function digest(text) {
    let h = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return (h >>> 0).toString(16);
  }

  /** A short name for a file URL: its last path segment, else its host. */
  function fileName(url) {
    try {
      const u = new URL(url, document.baseURI);
      const last = decodeURIComponent(u.pathname.replace(/\/+$/, "").split("/").pop() || "");
      return last || u.host;
    } catch {
      return url;
    }
  }

  /** The badge for a verdict. It claims only what the check established: figures found in
   * the records, not that the wording is right. */
  function verdictBadge(verdict, g, declined) {
    const n = g.figures_checked ?? 0;
    let badge;
    if (verdict === "ACCEPT" && n > 0) {
      badge = el("span", "badge ACCEPT", `✓ Figures verified (${n})`);
      badge.title =
        `Each of the ${n} figures was found in what the lookups returned.` +
        " The check does not judge the wording.";
    } else if (verdict === "ACCEPT" && (g.small_numbers_unchecked ?? 0) > 0) {
      badge = el("span", "badge plain", "From the records · small numbers not checked");
      badge.title =
        "Answered from lookups in the records. Numbers below 10 are checked only when they count" +
        " something (\"3 tasks\"); the ones here were not, so they were not verified.";
    } else if (verdict === "ACCEPT") {
      badge = el("span", "badge plain", "From the records · no figures to check");
      badge.title = "Answered from lookups in the records; it states no figure the check could verify.";
    } else if (verdict === "REVISE") {
      badge = el("span", "badge REVISE", "! Not verified");
      badge.title = (g.reasons || []).join("; ");
    } else if (verdict === "OUT_OF_SCOPE") {
      badge = el("span", "badge OUT_OF_SCOPE", "Outside the records");
      badge.title = declined ? `The agent's reason: ${declined}` : "";
    }
    return badge || null;
  }

  function recordName(path) {
    return path ? path.split(/[\\/]/).pop().replace(/\.json$/, "") : null;
  }

  class YprovAsk extends HTMLElement {
    static observedAttributes = ["endpoint", "file-picker"];

    constructor() {
      super();
      this.turns = [];
      this.dropped = 0;
      this.conversationId = null;
      this.busy = false;
      this.given = null; // a file the host handed over with openFile()
      this.file = null; // the open file: { key, name, url?, content?, fileId? }
      this.root = this.attachShadow({ mode: "open" });
      this.root.innerHTML = `
        <style>${STYLE}</style>
        <section class="panel home" aria-label="Ask about provenance">
          <div class="top">
            <div class="brand">
              <h1>yProv <span>Ask</span></h1>
              <p class="tagline">Plain questions about a provenance file.</p>
            </div>
            <form>
              <div class="field">
                <button class="icon attach" type="button" aria-label="Open a provenance file" title="Open a provenance file" hidden>${ATTACH_ICON}</button>
                <textarea rows="1" placeholder="Ask about this file" aria-label="Your question"></textarea>
                <button class="icon send" type="submit" aria-label="Send" disabled>${SEND_ICON}</button>
              </div>
              <input class="picker" type="file" accept=".json,application/json" hidden>
            </form>
            <div class="bar">
              <span class="file" hidden><span class="dot"></span><span class="name"></span><span class="content-id"></span></span>
              <span class="problem" role="alert" hidden></span>
              <span class="count" hidden></span>
              <button class="ghost new" type="button" title="Start a new conversation" hidden>New chat</button>
            </div>
            <div class="chips"></div>
          </div>
          <div class="thread" role="log" aria-live="polite"><div class="turns"></div></div>
          <div class="fine">© 2026 yProv Ask</div>
        </section>`;
      this.panel = this.root.querySelector(".panel");
      this.scroller = this.root.querySelector(".thread");
      this.thread = this.root.querySelector(".turns");
      this.chips = this.root.querySelector(".chips");
      this.input = this.root.querySelector("textarea");
      this.field = this.root.querySelector(".field");
      this.sendButton = this.root.querySelector(".send");
      this.newButton = this.root.querySelector(".new");
      this.counter = this.root.querySelector(".count");
      this.context = this.root.querySelector(".file");
      this.attachButton = this.root.querySelector(".attach");
      this.picker = this.root.querySelector(".picker");
      this.attachButton.addEventListener("click", () => this.picker.click());
      this.picker.addEventListener("change", async () => {
        const chosen = this.picker.files[0];
        this.picker.value = "";
        if (chosen) {
          try {
            await this.openFile({ content: chosen, name: chosen.name });
          } catch (error) {
            this.picked = `${chosen.name}: ${error.message || error}`;
            this.showFile();
          }
        }
        this.input.focus();
      });

      this.root.querySelector("form").addEventListener("submit", (event) => {
        event.preventDefault();
        this.send(this.input.value);
      });
      this.input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
          event.preventDefault();
          this.send(this.input.value);
        }
      });
      this.input.addEventListener("input", () => this.fitInput());
      // The Explorer changes ?file= with history.replaceState, which fires no event, so look
      // again whenever the user comes back to the widget.
      this.input.addEventListener("focus", () => this.refreshFile());
      this.addEventListener("pointerenter", () => this.refreshFile());
      window.addEventListener("popstate", () => this.refreshFile());
      this.newButton.addEventListener("click", () => this.clear());
    }

    connectedCallback() {
      this.restore();
      this.refreshFile();
      this.render();
    }

    attributeChangedCallback(name) {
      if (name === "endpoint") this.session = null; // a new endpoint needs a new MCP session
      else this.showFile();
    }

    get endpoint() {
      return new URL(this.getAttribute("endpoint") || "/mcp", document.baseURI).href;
    }

    get maxTurns() {
      const n = parseInt(this.getAttribute("max-turns"), 10);
      return n > 0 ? n : DEFAULT_MAX_TURNS;
    }

    get suggestions() {
      const raw = this.getAttribute("suggestions");
      return raw ? raw.split("|").map((s) => s.trim()).filter(Boolean) : DEFAULT_SUGGESTIONS;
    }

    get fileUrl() {
      const own = this.getAttribute("file-url");
      if (own) return own;
      try {
        return new URLSearchParams(window.location.search).get("file") || null;
      } catch {
        return null;
      }
    }

    get requireFile() {
      return this.getAttribute("require-file") !== "false";
    }

    get filePicker() {
      const own = this.getAttribute("file-picker");
      return own !== null && own !== "false";
    }

    /** The chat holds a set number of questions; then it asks for a new one. */
    get full() {
      return this.turns.length >= this.maxTurns;
    }

    get uploadUrl() {
      return new URL("files", this.endpoint).href;
    }

    /** Hand the widget the file the user opened from their computer. */
    async openFile({ content, name } = {}) {
      let text = content;
      if (content instanceof Blob) text = await content.text();
      else if (content && typeof content === "object") text = JSON.stringify(content);
      if (typeof text !== "string" || !text.trim()) throw new Error("openFile needs the file's content");
      const label = name || (content instanceof File ? content.name : "file.json");
      this.given = { key: `upload:${label}:${text.length}:${digest(text)}`, name: label, content: text };
      this.refreshFile();
      this.render();
      this.ensureFile().catch(() => {}); // a refusal is shown under the pill
    }

    get storageKey() {
      return `yprov-ask:${this.endpoint}`;
    }

    /* -------------------------------------------------------------- conversation */

    async send(raw) {
      const question = raw.trim();
      if (!question || this.busy || this.full) return;
      const turn = { question, status: "pending" };
      this.turns.push(turn);
      this.input.value = "";
      this.fitInput();
      await this.answer(turn);
    }

    async answer(turn) {
      this.busy = true;
      turn.status = "pending";
      turn.error = null;
      this.render();
      try {
        this.session ??= new McpSession(this.endpoint);
        const ask = async () => {
          const args = { question: turn.question };
          if (this.conversationId) args.conversation_id = this.conversationId;
          const fileId = await this.ensureFile();
          if (fileId) args.file_id = fileId;
          return this.session.callTool("ask", args, (message) => {
            const label = this.thread.querySelector(".turn:last-child .thinking .label");
            if (label) label.textContent = `${message}…`;
          });
        };
        try {
          turn.result = await ask();
        } catch (error) {
          // The server forgets an idle file, and every file on restart: open it again, once.
          if (!this.file || !/unknown file id/i.test(error.message || "")) throw error;
          if (this.file.content === undefined && !this.file.url) {
            throw new Error("The server no longer has this file. Open it again to keep asking.");
          }
          this.file.fileId = null;
          turn.result = await ask();
        }
        // The server forgets idle conversations and loses them on restart; it then starts
        // a new one, and this answer no longer knows the turns above it.
        turn.restarted = Boolean(turn.result.conversation_restarted);
        this.conversationId = turn.result.conversation_id || null;
        turn.status = "done";
      } catch (error) {
        turn.status = "error";
        turn.error = error.message || String(error);
      }
      this.busy = false;
      this.save();
      this.render();
      this.input.focus();
    }

    /** The file id for the open file, opening it on the server first if need be. */
    async ensureFile() {
      const file = this.file;
      if (!file) {
        if (this.requireFile) throw new Error("Open a provenance file to ask about it.");
        return null;
      }
      if (file.fileId) return file.fileId;
      file.opening ??= this.openOnServer(file).then(
        (id) => {
          file.problem = null;
          return id;
        },
        (error) => {
          file.problem = error.message || String(error);
          throw error;
        },
      ).finally(() => {
        file.opening = null;
        if (file === this.file) this.showFile();
      });
      return file.opening;
    }

    async openOnServer(file) {
      if (file.content !== undefined) {
        let response;
        try {
          response = await fetch(`${this.uploadUrl}?name=${encodeURIComponent(file.name)}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: file.content,
          });
        } catch {
          throw new Error(`Can't reach the provenance server at ${this.uploadUrl}.`);
        }
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.error || `Upload failed (HTTP ${response.status}).`);
        file.fileId = body.file_id;
        file.contentId = body.content_id;
      } else {
        this.session ??= new McpSession(this.endpoint);
        const opened = await this.session.callTool("open_file", { url: file.url });
        file.fileId = opened.file_id;
        file.contentId = opened.content_id;
      }
      this.save();
      return file.fileId;
    }

    /** Look again for the open file; a different file than before starts a new chat. */
    refreshFile() {
      const url = this.fileUrl;
      let now = this.given || (url ? { key: `url:${url}`, name: fileName(url), url } : null);
      // After a reload an uploaded file's content is gone, but the server may still hold it
      // under its id: keep it, and the chat with it, until another file is opened.
      if (!now && this.file && this.file.key.startsWith("upload:") && this.file.fileId) now = this.file;
      if (now && this.file && now.key === this.file.key && now.content && !this.file.content) {
        this.file.content = now.content; // the same upload again: its content is back
      }
      if ((now && now.key) !== (this.file && this.file.key)) {
        const before = this.file;
        this.file = now ? { ...now, fileId: null } : null;
        if (before && !this.busy) {
          this.turns = [];
          this.dropped = 0;
          this.conversationId = null;
        }
        this.save();
        this.render();
      }
      this.showFile();
    }

    clear() {
      if (this.busy) return;
      this.turns = [];
      this.dropped = 0;
      this.conversationId = null;
      this.save();
      this.render();
      this.input.focus();
    }

    save() {
      try {
        const turns = this.turns.filter((t) => t.status !== "pending");
        // The file's content is not kept, only what names it and the server's id for it.
        const file = this.file && { key: this.file.key, name: this.file.name, url: this.file.url,
          fileId: this.file.fileId, contentId: this.file.contentId };
        const state = { turns, dropped: this.dropped, conversationId: this.conversationId, file };
        sessionStorage.setItem(this.storageKey, JSON.stringify(state));
      } catch {
        /* storage blocked: the conversation just won't survive a reload */
      }
    }

    restore() {
      try {
        const saved = JSON.parse(sessionStorage.getItem(this.storageKey) || "null");
        if (saved && Array.isArray(saved.turns)) {
          this.turns = saved.turns.slice(-this.maxTurns);
          this.dropped = (saved.dropped || 0) + saved.turns.length - this.turns.length;
          this.conversationId = saved.conversationId || null;
          // Kept even for an upload, whose content is gone: so that a different file opened
          // after a reload is seen as different, and the old chat is not shown under it.
          if (saved.file) this.file = saved.file;
        }
      } catch {
        this.turns = [];
      }
    }

    /* -------------------------------------------------------------- rendering */

    showFile() {
      const file = this.file;
      const blocked = !file && this.requireFile;
      this.attachButton.hidden = !this.filePicker;
      this.input.disabled = blocked || this.full;
      this.field.classList.toggle("off", this.input.disabled);
      if (this.full) {
        this.input.placeholder = `This chat has its ${this.maxTurns} questions; start a new chat`;
      } else if (blocked) {
        this.input.placeholder = "Open a provenance file to ask about it";
      } else {
        this.input.placeholder = "Ask about this file";
      }
      const problem = this.root.querySelector(".problem");
      const why = (file && file.problem) || (!file && this.picked) || "";
      problem.hidden = !why;
      problem.textContent = why ? `Can't open this file: ${why}` : "";
      this.context.classList.toggle("bad", Boolean(file && file.problem));
      this.context.hidden = !file;
      if (!file) return;
      this.picked = null;
      const name = this.context.querySelector(".name");
      name.textContent = file.name;
      this.context.title = file.url || file.name;
      // The content's fingerprint: two files with the same name and different content (a
      // renamed copy) show different ids; the answer records name the same id.
      const id = this.context.querySelector(".content-id");
      id.textContent = file.contentId ? `· ${file.contentId}` : "";
      id.title = file.contentId
        ? "Content fingerprint: the same file always has the same one"
        : "";
    }

    fitInput() {
      this.input.style.height = "auto";
      this.input.style.height = `${Math.min(this.input.scrollHeight, 132)}px`;
      this.sendButton.disabled = this.busy || this.full || !this.input.value.trim();
    }

    render() {
      const home = this.turns.length === 0;
      this.panel.classList.toggle("home", home);
      this.panel.classList.toggle("chat", !home);
      this.newButton.hidden = home;
      this.newButton.disabled = this.busy;
      this.counter.hidden = home;
      this.counter.textContent = `${this.turns.length} of ${this.maxTurns} questions`;
      this.showFile();
      this.fitInput();
      this.renderChips();

      this.thread.replaceChildren();
      if (home) return;
      if (this.dropped > 0) {
        this.thread.append(el("div", "note", "Earlier questions from this chat are hidden."));
      }
      this.turns.forEach((turn) => this.thread.append(this.renderTurn(turn)));
      if (this.full && !this.busy) {
        const limit = el("div", "limit", `That's the ${this.maxTurns} questions this chat holds.`);
        const again = el("button", "ghost", "New chat");
        again.type = "button";
        again.addEventListener("click", () => this.clear());
        limit.append(again);
        this.thread.append(limit);
      }
      // To the newest turn's question, so a long answer is read from its start: at once on
      // load, smoothly as the chat goes on.
      const behavior = this.shown && !matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "smooth" : "instant";
      this.shown = true;
      requestAnimationFrame(() => {
        const last = this.thread.querySelector(".turn:last-of-type");
        const bottom = this.scroller.scrollHeight - this.scroller.clientHeight;
        const top = last ? Math.min(last.offsetTop - 12, bottom) : bottom;
        this.scroller.scrollTo({ top, behavior });
      });
    }

    /** Starter questions under the pill, before the first question. */
    renderChips() {
      this.chips.replaceChildren();
      if (this.turns.length) return;
      if (!this.file && this.requireFile) {
        if (!this.filePicker) {
          this.chips.append(el("div", "tagline", "Load a provenance file in the Explorer, then ask about it here."));
        }
        return;
      }
      for (const suggestion of this.suggestions) {
        const chip = el("button", "chip", suggestion);
        chip.type = "button";
        chip.disabled = this.busy;
        chip.addEventListener("click", () => this.send(suggestion));
        this.chips.append(chip);
      }
    }

    renderTurn(turn) {
      const box = el("article", "turn");
      const q = el("div", "q");
      q.append(el("div", "", turn.question));
      box.append(q);
      const r = turn.result || {};
      if (turn.status === "done" && turn.restarted) {
        box.append(el("div", "note", "The server started a new conversation; earlier questions are not remembered."));
      }
      // The system's side: a white card, apart from the user's blue pill.
      const reply = el("div", "r");
      const avatar = el("div", "avatar", "yP");
      avatar.setAttribute("aria-hidden", "true");
      const bubble = el("div", "bubble");
      reply.append(avatar, bubble);
      box.append(reply);

      if (turn.status === "pending") {
        const thinking = el("div", "thinking");
        const dots = el("span", "dots");
        dots.append(el("i"), el("i"), el("i"));
        thinking.append(dots, el("span", "label", "Searching the records…"));
        bubble.append(thinking);
        return box;
      }
      if (turn.status === "error") {
        bubble.classList.add("error");
        bubble.append(el("div", "a error", turn.error));
        const meta = el("div", "meta");
        const retry = el("button", "retry", "Try again");
        retry.type = "button";
        retry.disabled = this.busy;
        retry.addEventListener("click", () => this.answer(turn));
        meta.append(retry);
        bubble.append(meta);
        return box;
      }

      bubble.append(answerText(el("div", "a"), String(r.answer ?? "")));

      const meta = el("div", "meta");
      const g = r.grounding || {};
      const badge = verdictBadge(r.verdict, g, r.declined);
      if (badge) meta.append(badge);
      const calls = r.tool_calls || [];
      const lookups = calls.filter((c) => c.tool !== "decline").length;
      const facts = [`${lookups} lookup${lookups === 1 ? "" : "s"}`];
      if (typeof r.seconds === "number") facts.push(`${r.seconds.toFixed(1)} s`);
      meta.append(el("span", "", facts.join(" · ")));
      bubble.append(meta);
      if (r.verdict === "REVISE" && (g.reasons || []).length) {
        bubble.append(el("div", "meta why", `Not verified: ${g.reasons.join("; ")}.`));
      }
      if (r.file_error) {
        bubble.append(el("div", "meta", `The open file wasn't used: ${r.file_error}`));
      }

      if (calls.length || r.record) {
        const details = el("details");
        details.append(el("summary", "", "How this was answered"));
        const list = el("ol");
        for (const c of calls) {
          const item = el("li");
          item.append(el("code", "", `${c.tool}(${describeArgs(c.args)})`));
          if (c.ok === false) item.append(" — failed");
          list.append(item);
        }
        details.append(list);
        const record = recordName(r.record);
        if (record) {
          const line = el("div", "", "PROV record: ");
          line.append(el("code", "", record));
          details.append(line);
        }
        bubble.append(details);
      }
      return box;
    }
  }

  if (!customElements.get("yprov-ask")) customElements.define("yprov-ask", YprovAsk);
})();
