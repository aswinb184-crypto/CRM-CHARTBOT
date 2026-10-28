import hashlib
import ipaddress
import json
import logging
import os
import re
import threading
import time
import uuid
import zipfile
from io import BytesIO
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.exceptions import HTTPException


_project_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_project_dir, ".env"), override=False)


from security import (
    HOSPITAL_ID_PATTERN,
    hospital_documents_dir,
    hospital_index_dir,
    is_sensitive_output,
    is_sensitive_request,
)


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".webp"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_DOCUMENTS_PER_UPLOAD = 10
DOCUMENT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def _environment_limit(name, default, minimum, maximum):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError) as error:
        raise RuntimeError("Invalid server limit configuration.") from error
    if not minimum <= value <= maximum:
        raise RuntimeError("Invalid server limit configuration.")
    return value


MAX_REQUEST_MB = _environment_limit("APP_MAX_REQUEST_MB", 25, 1, 100)
MAX_UPLOAD_MB = _environment_limit("APP_MAX_UPLOAD_MB", 20, 1, MAX_REQUEST_MB)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

app = Flask(__name__)
app.config["DEBUG"] = False
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_MB * 1024 * 1024
app.config["RATE_LIMIT_REQUESTS"] = _environment_limit(
    "APP_RATE_LIMIT_REQUESTS", 120, 1, 10_000
)
app.config["RATE_LIMIT_WINDOW_SECONDS"] = _environment_limit(
    "APP_RATE_LIMIT_WINDOW_SECONDS", 60, 1, 86_400
)

_log_path = os.environ.get(
    "APP_LOG_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".local_logs", "app.log"),
)
os.makedirs(os.path.dirname(os.path.abspath(_log_path)), exist_ok=True)
if not any(
    isinstance(handler, RotatingFileHandler)
    and os.path.abspath(handler.baseFilename) == os.path.abspath(_log_path)
    for handler in app.logger.handlers
):
    _file_handler = RotatingFileHandler(
        _log_path,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    _file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
    )
    app.logger.addHandler(_file_handler)
app.logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

_rate_lock = threading.Lock()
_request_times = {}


def _hospital_id():
    value = os.environ.get("APP_HOSPITAL_ID", "local_hospital")
    if not HOSPITAL_ID_PATTERN.fullmatch(value):
        raise ValueError("APP_HOSPITAL_ID must contain only letters, digits, '_' or '-'.")
    return value


def _documents_dir(hospital_id):
    return hospital_documents_dir(hospital_id)


def _store_dir(hospital_id):
    return hospital_index_dir(hospital_id)


def _manifest_path(hospital_id):
    return os.path.join(_documents_dir(hospital_id), "manifest.json")


def _load_manifest(hospital_id):
    path = _manifest_path(hospital_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError("Invalid document manifest.")
    for record in records:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("id"), str)
            or not DOCUMENT_ID_PATTERN.fullmatch(record["id"])
            or record.get("extension") not in ALLOWED_EXTENSIONS
            or record.get("filename")
            != record["id"] + record["extension"]
            or not isinstance(record.get("name"), str)
            or len(record["name"]) > 180
        ):
            raise ValueError("Invalid document manifest.")
    return records


def _write_manifest(hospital_id, records):
    documents_dir = _documents_dir(hospital_id)
    os.makedirs(documents_dir, exist_ok=True)
    target_path = _manifest_path(hospital_id)
    temporary_path = target_path + "." + uuid.uuid4().hex + ".tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=True)
        os.replace(temporary_path, target_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _clear_index(hospital_id):
    store_dir = _store_dir(hospital_id)
    transaction_id = uuid.uuid4().hex
    moved = []
    try:
        for filename in ("patient_faiss.index", "patient_chunks.pkl"):
            path = os.path.join(store_dir, filename)
            if os.path.exists(path):
                backup_path = path + "." + transaction_id + ".bak"
                os.replace(path, backup_path)
                moved.append((path, backup_path))
    except Exception:
        for path, backup_path in reversed(moved):
            if os.path.exists(backup_path):
                os.replace(backup_path, path)
        raise

    for _path, backup_path in moved:
        os.remove(backup_path)


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(64 * 1024), b""):
            digest.update(block)
    return digest.digest()


def _index_hospital_documents(hospital_id, records):
    indexable_paths = []
    source_names = {}
    seen_file_hashes = set()

    for record in records:
        filename = record.get("filename")
        if not isinstance(filename, str) or os.path.basename(filename) != filename:
            raise ValueError("Invalid document record.")
        path = os.path.join(_documents_dir(hospital_id), filename)
        if record.get("extension") in IMAGE_EXTENSIONS:
            ocr_path = os.path.join(
                _documents_dir(hospital_id),
                record["id"] + "__ocr.txt",
            )
            if os.path.isfile(ocr_path):
                if not os.path.isfile(path):
                    raise ValueError("Uploaded image is missing.")
                file_hash = _file_hash(path)
                if file_hash in seen_file_hashes:
                    continue
                seen_file_hashes.add(file_hash)
                indexable_paths.append(ocr_path)
                source_names[os.path.abspath(ocr_path)] = (
                    record["name"] + " (OCR)"
                )
            else:
                if not os.path.isfile(path):
                    raise ValueError("Uploaded image is missing.")
                raise ValueError("Uploaded image OCR text is missing.")
        else:
            if not os.path.isfile(path):
                raise ValueError("Uploaded document is missing.")
            file_hash = _file_hash(path)
            if file_hash in seen_file_hashes:
                continue
            seen_file_hashes.add(file_hash)
            indexable_paths.append(path)
            source_names[os.path.abspath(path)] = record["name"]

    if not indexable_paths:
        _clear_index(hospital_id)
        return {"documents": 0, "chunks": 0}

    from patient_rag.document_processor import build_patient_index

    result = build_patient_index(
        indexable_paths,
        _store_dir(hospital_id),
        source_names=source_names,
    )
    app.logger.info(
        "Hospital index rebuilt; documents=%d unique_documents=%d chunks=%d",
        len(records),
        result["documents"],
        result["chunks"],
    )
    return result


def _display_filename(filename):
    name = (filename or "").replace("\\", "/").split("/")[-1]
    name = "".join(character for character in name if character.isprintable())
    name = name.lstrip(".")
    return name[:180] or "uploaded_file"


def _validate_upload(extension, content):
    if not content:
        raise ValueError("The uploaded file is empty.")
    if extension == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise ValueError("Invalid file content.")
    elif extension == ".docx":
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                names = set(archive.namelist())
                if (
                    len(names) > 1000
                    or "[Content_Types].xml" not in names
                    or "word/document.xml" not in names
                    or sum(item.file_size for item in archive.infolist())
                    > 50 * 1024 * 1024
                    or any(name.endswith("vbaProject.bin") for name in names)
                ):
                    raise ValueError("Invalid file content.")
                if any(item.flag_bits & 0x1 for item in archive.infolist()):
                    raise ValueError("Invalid file content.")
        except (zipfile.BadZipFile, OSError) as error:
            raise ValueError("Invalid file content.") from error
    elif extension == ".txt":
        if b"\x00" in content:
            raise ValueError("Invalid file content.")
        try:
            content.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                content.decode("cp1252")
            except UnicodeDecodeError as error:
                raise ValueError("Invalid file content.") from error
    elif extension in IMAGE_EXTENSIONS:
        try:
            with Image.open(BytesIO(content)) as image:
                if image.width * image.height > 40_000_000:
                    raise ValueError("Invalid image dimensions.")
                expected_format = {
                    ".png": "PNG",
                    ".jpg": "JPEG",
                    ".jpeg": "JPEG",
                    ".webp": "WEBP",
                }[extension]
                if image.format != expected_format:
                    raise ValueError("Image format does not match its extension.")
                image.verify()
        except (Image.DecompressionBombError, OSError) as error:
            raise ValueError("Invalid image content.") from error
    else:
        raise ValueError("Unsupported file type.")


@app.before_request
def enforce_local_only_and_rate_limit():
    try:
        if not ipaddress.ip_address(request.remote_addr or "").is_loopback:
            return jsonify({
                "status": "error",
                "message": "This service is available only on this computer.",
            }), 403
    except ValueError:
        return jsonify({
            "status": "error",
            "message": "This service is available only on this computer.",
        }), 403

    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        if request.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
            return jsonify({"status": "error", "message": "Request origin not allowed."}), 403
        origin = request.headers.get("Origin")
        if origin:
            allowed_origins = {
                "http://127.0.0.1:5001",
                "http://localhost:5001",
                "http://127.0.0.1:8501",
                "http://localhost:8501",
            }
            if origin.rstrip("/") not in allowed_origins:
                return jsonify({"status": "error", "message": "Request origin not allowed."}), 403

    if request.endpoint == "static":
        return None
    now = time.monotonic()
    address = request.remote_addr or "unknown"
    window = app.config["RATE_LIMIT_WINDOW_SECONDS"]
    cutoff = now - window
    with _rate_lock:
        for key, timestamps in list(_request_times.items()):
            while timestamps and timestamps[0] <= cutoff:
                timestamps.pop(0)
            if not timestamps:
                del _request_times[key]
        timestamps = _request_times.setdefault(address, [])
        if len(timestamps) >= app.config["RATE_LIMIT_REQUESTS"]:
            return jsonify({"status": "error", "message": "Too many requests."}), 429
        timestamps.append(now)
    return None


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"status": "error", "message": "Request is too large."}), 413


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if isinstance(error, HTTPException):
        return jsonify({
            "status": "error",
            "message": "The request could not be completed.",
        }), error.code
    app.logger.exception("Request failed; exception_type=%s", type(error).__name__)
    return jsonify({"status": "error", "message": "The request could not be completed."}), 500


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "chat_configured": bool(os.environ.get("GROQ_API_KEY", "").strip()),
        "model_configured": bool(
            os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        ),
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("question"), str):
        return jsonify({"status": "error", "message": "Invalid request."}), 400

    question = data["question"].strip()
    if not question or len(question) > 4_000:
        return jsonify({"status": "error", "message": "Invalid request."}), 400
    if is_sensitive_request(question):
        return jsonify({
            "status": "success",
            "answer": "I can’t provide secrets, private system information, or unauthorized data.",
        })

    if not os.environ.get("GROQ_API_KEY", "").strip():
        return jsonify({
            "status": "error",
            "message": (
                "The chat service is not configured. Set GROQ_API_KEY in the "
                "project .env file or the Flask PowerShell environment, then "
                "restart Flask."
            ),
        }), 503

    try:
        answer = answer_question(question, _hospital_id())
        if not isinstance(answer, str) or is_sensitive_output(answer):
            answer = "I can’t provide secrets, private system information, or unauthorized data."
    except Exception as error:
        app.logger.exception(
            "Chat processing failed; exception_type=%s",
            type(error).__name__,
        )
        return jsonify({
            "status": "error",
            "message": getattr(
                error,
                "user_message",
                "The assistant could not complete the request. Check the local "
                "Flask log for details.",
            ),
        }), getattr(error, "http_status", 500)

    return jsonify({"status": "success", "answer": answer})


def answer_question(question, hospital_id):
    from document.rag import answer_question as rag_answer_question

    return rag_answer_question(question, hospital_id=hospital_id)


@app.route("/documents", methods=["GET"])
def list_documents():
    records = _load_manifest(_hospital_id())
    return jsonify({
        "status": "success",
        "documents": [
            {"id": record["id"], "name": record["name"]}
            for record in records
        ],
    })


@app.route("/documents/upload", methods=["POST"])
def upload_documents():
    files = request.files.getlist("files")
    if not files or len(files) > MAX_DOCUMENTS_PER_UPLOAD:
        return jsonify({"status": "error", "message": "Invalid upload."}), 400

    prepared = []
    for uploaded_file in files:
        original_name = _display_filename(uploaded_file.filename)
        extension = os.path.splitext(original_name)[1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            return jsonify({"status": "error", "message": "Unsupported file type."}), 415
        content = uploaded_file.stream.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            return jsonify({"status": "error", "message": "File is too large."}), 413
        try:
            _validate_upload(extension, content)
        except ValueError as error:
            message = (
                "The file is empty."
                if str(error) == "The uploaded file is empty."
                else "The file is invalid or corrupted."
            )
            return jsonify({"status": "error", "message": message}), 400
        prepared.append({
            "id": uuid.uuid4().hex,
            "name": original_name,
            "extension": extension,
            "content": content,
        })

    hospital_id = _hospital_id()
    documents_dir = _documents_dir(hospital_id)
    os.makedirs(documents_dir, exist_ok=True)
    existing_records = _load_manifest(hospital_id)
    known_hashes = {}
    for record in existing_records:
        existing_path = os.path.join(documents_dir, record["filename"])
        if os.path.isfile(existing_path):
            known_hashes.setdefault(
                _file_hash(existing_path),
                {"id": record["id"], "name": record["name"]},
            )
    unique_prepared = []
    duplicates = []
    for item in prepared:
        content_hash = hashlib.sha256(item["content"]).digest()
        existing_match = known_hashes.get(content_hash)
        if existing_match is not None:
            duplicates.append(existing_match)
            continue
        known_hashes[content_hash] = {
            "id": item["id"],
            "name": item["name"],
        }
        unique_prepared.append(item)

    if not unique_prepared:
        return jsonify({
            "status": "success",
            "documents": [],
            "duplicates": duplicates,
        }), 200

    stored_records = []
    try:
        for item in unique_prepared:
            filename = item["id"] + item["extension"]
            path = os.path.join(documents_dir, filename)
            record = {
                "id": item["id"],
                "name": item["name"],
                "extension": item["extension"],
                "filename": filename,
            }
            stored_records.append(record)
            with open(path, "xb") as file:
                file.write(item["content"])
            if item["extension"] in IMAGE_EXTENSIONS:
                from patient_rag.document_processor import extract_text_from_image

                extracted_text = extract_text_from_image(item["content"])
                if not extracted_text:
                    raise ValueError("No readable image text.")
                ocr_path = os.path.join(
                    documents_dir,
                    item["id"] + "__ocr.txt",
                )
                with open(ocr_path, "x", encoding="utf-8") as file:
                    file.write("Photo file: {}\n\n{}".format(item["name"], extracted_text))

        records = existing_records + stored_records
        _write_manifest(hospital_id, records)
        _index_hospital_documents(hospital_id, records)
    except Exception as error:
        app.logger.exception(
            "Document upload failed; exception_type=%s",
            type(error).__name__,
        )
        for item in stored_records:
            for filename in (item["filename"], item["id"] + "__ocr.txt"):
                path = os.path.join(documents_dir, filename)
                if os.path.exists(path):
                    os.remove(path)
        if stored_records:
            _write_manifest(hospital_id, existing_records)
        if isinstance(error, (ModuleNotFoundError, ImportError)):
            message = (
                "Document processing is unavailable because a required Python "
                "package is missing. Install the project requirements and retry."
            )
            status_code = 503
        elif isinstance(error, RuntimeError) and any(
            marker in str(error).lower()
            for marker in ("ocr", "tesseract")
        ):
            message = (
                "OCR is unavailable. Install the Tesseract OCR engine and retry "
                "this image or scanned PDF."
            )
            status_code = 422
        elif isinstance(error, ValueError):
            message = (
                "No readable text could be extracted. Verify the file or configure "
                "OCR for scanned documents."
            )
            status_code = 422
        elif isinstance(error, PermissionError):
            message = "The server could not save the file. Check document storage permissions."
            status_code = 500
        else:
            message = (
                "The uploaded documents could not be processed. Review the local "
                "server log and try again."
            )
            status_code = 500
        return jsonify({
            "status": "error",
            "message": message,
        }), status_code

    return jsonify({
        "status": "success",
        "documents": [
            {"id": record["id"], "name": record["name"]}
            for record in stored_records
        ],
        "duplicates": duplicates,
    }), 201


@app.route("/documents/reindex", methods=["POST"])
def reindex_documents():
    hospital_id = _hospital_id()
    try:
        result = _index_hospital_documents(hospital_id, _load_manifest(hospital_id))
    except Exception as error:
        app.logger.exception(
            "Document indexing failed; exception_type=%s",
            type(error).__name__,
        )
        return jsonify({
            "status": "error",
            "message": "The document index could not be rebuilt.",
        }), 500
    return jsonify({"status": "success", "documents": result["documents"]})


def _delete_hospital_documents(hospital_id):
    records = _load_manifest(hospital_id)
    _write_manifest(hospital_id, [])
    try:
        _clear_index(hospital_id)
    except Exception:
        _write_manifest(hospital_id, records)
        raise

    documents_dir = _documents_dir(hospital_id)
    if os.path.isdir(documents_dir):
        for filename in os.listdir(documents_dir):
            path = os.path.join(documents_dir, filename)
            if os.path.isfile(path):
                os.remove(path)
    _write_manifest(hospital_id, [])


@app.route("/documents/<document_id>", methods=["DELETE"])
def delete_document(document_id):
    if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
        return jsonify({"status": "error", "message": "Document not found."}), 404

    hospital_id = _hospital_id()
    try:
        records = _load_manifest(hospital_id)
        target = next((item for item in records if item["id"] == document_id), None)
        if target is None:
            return jsonify({"status": "error", "message": "Document not found."}), 404
        remaining = [item for item in records if item["id"] != document_id]
        _write_manifest(hospital_id, remaining)
        index_rebuilt = False
        _index_hospital_documents(hospital_id, remaining)
        index_rebuilt = True
        documents_dir = _documents_dir(hospital_id)
        for filename in (target["filename"], document_id + "__ocr.txt"):
            path = os.path.join(documents_dir, filename)
            if os.path.exists(path):
                os.remove(path)
    except Exception as error:
        app.logger.exception(
            "Document deletion failed; exception_type=%s",
            type(error).__name__,
        )
        if "records" in locals() and not locals().get("index_rebuilt", False):
            _write_manifest(hospital_id, records)
        return jsonify({
            "status": "error",
            "message": "The document could not be removed.",
        }), 500
    return jsonify({"status": "success"})


@app.route("/documents", methods=["DELETE"])
def remove_all_documents():
    hospital_id = _hospital_id()
    try:
        _delete_hospital_documents(hospital_id)
    except Exception as error:
        app.logger.exception(
            "Document removal failed; exception_type=%s",
            type(error).__name__,
        )
        return jsonify({
            "status": "error",
            "message": "The documents could not be removed.",
        }), 500
    return jsonify({"status": "success"})


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
    )
