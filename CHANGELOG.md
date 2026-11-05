# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Added

- MCP server exposing deterministic query tools over an SQLite store of PROV-JSON documents,
  with stdio and streamable HTTP transports.
- Readers for yProv4ML runs (with metric CSVs), yProv4Agents decision records, and yProv4WfS,
  openEO, Ophidia and analytics4yprov workflow traces.
- LangGraph agent with a code-side grounding check (`qa.grounding/v3`): figures must come from
  tool results and be used in the role they were returned for; one revision on failure.
- A PROV record for every answer, with rule inputs so each verdict can be replayed.
- Single-file mode: uploads (`POST /files`) and URLs (`open_file`) pin the agent to one file.
- Conversation memory across turns, keyed by a server-issued conversation id.
- `<yprov-ask>` web component for the yProv Explorer, with a standalone demo page.
- All four MCP tool hints (`readOnlyHint`, `destructiveHint`, `idempotentHint`,
  `openWorldHint`) declared explicitly on every tool.
- MIT license.
- Web console and terminal chat as MCP clients of the server.
- Evaluation harness: computed gold answers, frozen test splits, scorers and audit sheets.
- Docker image with a health check, CORS configuration and a non-root user.
