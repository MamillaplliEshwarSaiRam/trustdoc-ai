# Security Policy

TrustDoc AI is a portfolio/demo RAG application and should not be used as-is for highly sensitive production documents.

## API Keys

- Never commit `.env` or `.streamlit/secrets.toml`.
- Use restricted API keys where possible.
- Rotate a key immediately if it is accidentally committed or shared.

## Uploaded Documents

Uploaded files are processed in the local Streamlit session. If you deploy this app publicly, add authentication, file-size limits, storage controls, and clear data retention rules before accepting private documents.

## Reporting Issues

If you find a security issue, please open a private report if the repository supports GitHub private vulnerability reporting. Otherwise, avoid posting secrets or exploit details publicly; open a minimal issue asking for a secure contact path.
