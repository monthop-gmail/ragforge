# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-03-08

### Added
- Initial release
- 3 RAG templates: No Framework, LangChain, LlamaIndex
- `ragforge.py` CLI for project scaffolding
- FastAPI server with REST endpoints (upload, ingest-url, query, list, delete)
- MCP server (SSE transport) for Claude Code integration
- Docker Compose support with health checks
- CORS middleware with configurable origins
- URL validation (SSRF protection)
- File size limits for uploads
- Non-root Docker user
- CLAUDE.md for each template
- GitHub CI/CD workflow
- Issue templates and PR template
- CONTRIBUTING.md and SECURITY.md
