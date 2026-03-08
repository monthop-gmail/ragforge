# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in RagForge, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email: **monthop@gmail.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will respond within 48 hours and work with you to resolve the issue.

## Security Best Practices

When using RagForge:

1. **Never commit `.env` files** — They contain API keys
2. **Use Docker** — Templates run as non-root user in containers
3. **Set CORS origins** — Restrict `CORS_ORIGINS` in production
4. **Limit file uploads** — Configure `MAX_UPLOAD_SIZE_MB` appropriately
5. **Use HTTPS** — Always use HTTPS in production with a reverse proxy
6. **Rotate API keys** — Regularly rotate your OpenAI API keys

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |
