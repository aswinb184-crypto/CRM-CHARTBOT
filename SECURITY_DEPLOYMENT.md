# Local security and deployment notes

This application has no login or user roles. It is configured for a single trusted
local instance and binds Flask and Streamlit to loopback. Do not expose either
service to a LAN, the public internet, a reverse proxy, or a shared computer. Anyone
able to use the local interface can chat with the application, read its documents,
upload files, and delete individual or all documents.

Removing the former login also removes per-user and per-role authorization. This
configuration is not a substitute for authentication, hospital-level authorization
between users, or a security review. Do not use it with real patient data unless an
independent security and privacy review has approved the deployment.

## Starting locally

Run Flask and Streamlit in separate PowerShell terminals from the project directory:

```powershell
.\.venv\Scripts\python.exe app.py
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

Install the Python dependencies into the project environment with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The current local `.venv` was created with Python 3.14.2; use the same `.venv`
for both processes. Scanned-PDF
and image OCR also needs the separate Tesseract OCR executable installed and
available on `PATH`. `pypdfium2` renders scanned PDF pages for OCR.

The Streamlit server configuration in `.streamlit/config.toml` also pins its bind
address to `127.0.0.1` and keeps XSRF protection enabled. Flask binds to
`127.0.0.1:5001`; it does not honor an environment override for its bind address.
Only configure `BACKEND_URL` to a local HTTP address if Flask is listening on a
different local port. The Streamlit application rejects non-loopback backend URLs.

## Server configuration

- `GROQ_API_KEY` is read by the Flask/RAG server process only. Inject it through a
  trusted local secret manager or process environment; never put it in Streamlit
  widgets, browser code, HTML, source control, or API responses.
- When Flask is started from the project directory, it loads the project `.env`
  without overriding values already set in the process environment. The `.env`
  file is ignored by Git. `GROQ_MODEL` optionally selects the Groq chat model; if
  unset, the app uses its existing `openai/gpt-oss-120b` default. Changing the
  configured key or model requires restarting Flask.
- `APP_HOSPITAL_ID` selects this instance's isolated document and index directory.
  It defaults to `local_hospital` and may contain only letters, digits, `_`, and
  `-`. It is server configuration, not a request parameter. Run separate, secured
  instances with separate data roots if separate datasets are needed.
- `HOSPITAL_DATA_DIR` optionally sets the local data root. Keep it private and
  restrict filesystem permissions to the application account.
- `APP_MAX_REQUEST_MB` (default `25`, range `1`–`100`) sets the request-size cap.
  `APP_MAX_UPLOAD_MB` (default `20`) caps an individual file and must not exceed
  the request limit.
- `APP_RATE_LIMIT_REQUESTS` (default `120`) and
  `APP_RATE_LIMIT_WINDOW_SECONDS` (default `60`) set a per-process, per-IP limit.
  On a single local machine this is not a substitute for user-level throttling.
- `BACKEND_URL` defaults to `http://127.0.0.1:5001` and must remain a loopback HTTP
  address with an explicit port.

The former login variables `APP_USERS_JSON`, `APP_SECRET_KEY`, and
`APP_TOKEN_LIFETIME_SECONDS` are no longer used. The old `/auth/login` and
`/auth/me` routes have been removed; the login configuration error and login 503
should therefore no longer occur.

## Limitations before deployment

- Local-only access is the only access boundary. Loopback binding is not user
  authentication and does not prevent another local process or local user from
  calling the Flask endpoints.
- One process uses one server-selected hospital ID. There is no user-selectable
  hospital boundary, account lifecycle, MFA, or role separation.
- The in-memory rate limiter is process-local and can be bypassed by restarting
  the server. It is not a distributed abuse-control system.
- Documents and indexes are stored locally without application-provided encryption
  at rest, retention management, managed backups, or disaster recovery.
- This application has no login or user-level authorization. All local users of
  this single instance share the same configured hospital documents and can read,
  upload, or delete them. Do not use the current shared-local architecture for
  real patient information without independent security and privacy approval.
- Prompt-injection and secret-request detection are defense-in-depth checks, not a
  guarantee that an LLM can never produce an unsafe answer. Uploaded documents
  remain untrusted input.
- Review Groq's data-processing terms and your deployment configuration before
  sending any sensitive content to the provider. Keep operational logs protected;
  local rotating logs now retain exception tracebacks for diagnosis and are not
  returned to browser clients.
- Do not publish the project directory through a generic static-file server. Exclude
  local source backups such as `app_backup.py` and `streamlit_app_backup.py` from
  deployment artifacts.

No independent security assessment, clinical validation, or privacy-regulation
compliance review has been performed.
