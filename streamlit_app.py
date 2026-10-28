
import ipaddress
import os
from urllib.parse import urlsplit

import requests
import streamlit as st

from voice_input import consume_voice_question


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CRM Document Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# BACKEND CONFIGURATION
# ============================================================

BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "http://127.0.0.1:5001",
).rstrip("/")

MAX_UPLOAD_SIZE_MB = 20

UPLOAD_FILE_TYPES = ["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"]


def upload_chat_files(uploaded_files):
    files = [
        (
            "files",
            (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "application/octet-stream",
            ),
        )
        for uploaded_file in uploaded_files
    ]
    return requests.post(
        f"{BACKEND_URL}/documents/upload",
        files=files,
        timeout=180,
    )


SPEECH_COMPONENT_HTML = """
<div class="speech-root">
  <div class="voice-controls" hidden>
    <button class="start-voice" type="button">Start Voice</button>
    <button class="stop-voice" type="button" disabled>Stop Voice</button>
    <button class="stop-speech" type="button">Stop Speech</button>
    <span class="voice-status" role="status" aria-live="polite">Ready</span>
    <span class="speech-status" role="status" aria-live="polite"></span>
  </div>
  <div class="read-control" hidden>
    <button class="read-aloud" type="button">🔊 Read aloud</button>
    <span class="read-status" role="status" aria-live="polite"></span>
  </div>
</div>
"""

SPEECH_COMPONENT_CSS = """
.speech-root {
  color: var(--st-text-color, #1f2937);
  font: inherit;
}
.voice-controls,
.read-control {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}
.voice-controls[hidden],
.read-control[hidden] {
  display: none !important;
}
.speech-root button {
  background: var(--st-secondary-background-color, #f8fafc);
  border: 1px solid var(--st-primary-color, #2563eb);
  border-radius: 0.5rem;
  color: var(--st-text-color, #1f2937);
  cursor: pointer;
  font: inherit;
  line-height: 1.25;
  padding: 0.45rem 0.65rem;
}
.speech-root button:hover:not(:disabled) {
  background: var(--st-primary-color, #2563eb);
  color: white;
}
.speech-root button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
.voice-status,
.speech-status,
.read-status {
  font-size: 0.8rem;
}
"""

SPEECH_COMPONENT_JS = """
const instances = new WeakMap();
const commandEvent = "crm-chat-speech-command";
const statusEvent = "crm-chat-speech-status";

function dispatchStatus(controlId, message) {
  window.dispatchEvent(new CustomEvent(statusEvent, {
    detail: { controlId, message }
  }));
}

function eventId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function permissionMessage(error) {
  if (error === "not-allowed" || error === "service-not-allowed") {
    return "Error: Microphone permission was denied. Allow microphone access and try again.";
  }
  if (error === "audio-capture") {
    return "Error: No microphone was detected.";
  }
  if (error === "network") {
    return "Error: Speech recognition could not connect. Try again.";
  }
  return "Error: Speech recognition failed. Try again.";
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector(".speech-root");
  if (!root) return;

  const voiceControls = root.querySelector(".voice-controls");
  const readControl = root.querySelector(".read-control");
  const startButton = root.querySelector(".start-voice");
  const stopButton = root.querySelector(".stop-voice");
  const stopSpeechButton = root.querySelector(".stop-speech");
  const voiceStatus = root.querySelector(".voice-status");
  const speechStatus = root.querySelector(".speech-status");
  const readButton = root.querySelector(".read-aloud");
  const readStatus = root.querySelector(".read-status");
  const isVoiceControl = data.mode === "voice";
  voiceControls.hidden = !isVoiceControl;
  readControl.hidden = isVoiceControl;

  let state = instances.get(parentElement);
  if (!state) {
    state = {
      recognition: null,
      activeUtterance: null,
      activeControlId: null,
      voiceSubmitted: false,
      manualStop: false,
      voiceMessage: "Ready",
      statusResetToken: data.status_reset_token,
      commandHandler: null,
      statusHandler: null
    };
    instances.set(parentElement, state);
  }

  const recognitionConstructor =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognitionSupported = typeof recognitionConstructor === "function";
  const speechSupported =
    "speechSynthesis" in window &&
    typeof window.SpeechSynthesisUtterance === "function";
  const controlId = data.control_id || "";

  if (state.statusResetToken !== data.status_reset_token) {
    state.statusResetToken = data.status_reset_token;
    if (state.voiceSubmitted) {
      state.voiceSubmitted = false;
      state.manualStop = false;
      state.voiceMessage = "Ready";
    }
  }

  if (isVoiceControl) {
    if (!recognitionSupported) {
      state.voiceMessage =
        "Unsupported: Voice input is not available in this browser.";
    }
    voiceStatus.textContent = state.voiceMessage;
    startButton.disabled =
      !recognitionSupported || Boolean(state.recognition) || state.voiceSubmitted;
    stopButton.disabled = !state.recognition;
    stopSpeechButton.disabled = !speechSupported;
    speechStatus.textContent = speechSupported
      ? "Speech ready."
      : "Text-to-speech is not supported in this browser.";

    if (!state.commandHandler) {
      state.commandHandler = (event) => {
        const detail = event.detail || {};
        if (detail.action === "stop") {
          if (speechSupported) {
            const previousControlId = state.activeControlId;
            state.activeUtterance = null;
            state.activeControlId = null;
            window.speechSynthesis.cancel();
            if (previousControlId) {
              dispatchStatus(previousControlId, "Speech stopped.");
            }
            speechStatus.textContent = "Speech stopped.";
          }
          return;
        }

        if (detail.action !== "speak") return;
        const text = typeof detail.text === "string" ? detail.text.trim() : "";
        if (!text) {
          dispatchStatus(detail.controlId, "There is no response to read.");
          return;
        }
        if (!speechSupported) {
          dispatchStatus(
            detail.controlId,
            "Text-to-speech is not supported in this browser."
          );
          return;
        }

        if (state.activeUtterance) {
          const previousControlId = state.activeControlId;
          state.activeUtterance = null;
          state.activeControlId = null;
          window.speechSynthesis.cancel();
          if (previousControlId) {
            dispatchStatus(previousControlId, "Speech stopped.");
          }
        }

        const utterance = new window.SpeechSynthesisUtterance(text);
        const voices = window.speechSynthesis.getVoices();
        const localVoices = voices.filter((item) => item.localService === true);
        if (!localVoices.length) {
          speechStatus.textContent = "No local speech voice is available.";
          dispatchStatus(
            detail.controlId,
            "No local speech voice is available; the response was not sent to an external service."
          );
          return;
        }
        const preferredLanguage = (navigator.language || "en-US").toLowerCase();
        const preferredLanguagePrefix = preferredLanguage.split("-")[0];
        const voice =
          localVoices.find(
            (item) => (item.lang || "").toLowerCase() === preferredLanguage
          ) ||
          localVoices.find((item) =>
            (item.lang || "").toLowerCase().startsWith(
              preferredLanguagePrefix + "-"
            )
          ) ||
          localVoices.find((item) => /^en([_-]|$)/i.test(item.lang || "")) ||
          localVoices.find((item) => item.default) ||
          localVoices[0];
        if (voice) utterance.voice = voice;

        state.activeUtterance = utterance;
        state.activeControlId = detail.controlId || "";
        speechStatus.textContent = "Speaking assistant response...";
        dispatchStatus(state.activeControlId, "Speaking...");
        utterance.onend = () => {
          if (state.activeUtterance !== utterance) return;
          const finishedControlId = state.activeControlId;
          state.activeUtterance = null;
          state.activeControlId = null;
          speechStatus.textContent = "Speech finished.";
          dispatchStatus(finishedControlId, "Finished.");
        };
        utterance.onerror = () => {
          if (state.activeUtterance !== utterance) return;
          const failedControlId = state.activeControlId;
          state.activeUtterance = null;
          state.activeControlId = null;
          speechStatus.textContent = "Speech could not be played.";
          dispatchStatus(failedControlId, "Speech could not be played.");
        };
        try {
          window.speechSynthesis.speak(utterance);
        } catch (_error) {
          state.activeUtterance = null;
          state.activeControlId = null;
          speechStatus.textContent = "Speech could not be played.";
          dispatchStatus(detail.controlId, "Speech could not be played.");
        }
      };
      window.addEventListener(commandEvent, state.commandHandler);
    }

    if (!state.statusHandler) {
      state.statusHandler = (event) => {
        if (event.detail && event.detail.controlId === controlId) {
          readStatus.textContent = event.detail.message;
        }
      };
      window.addEventListener(statusEvent, state.statusHandler);
    }

    startButton.onclick = () => {
      if (!recognitionSupported || state.recognition || state.voiceSubmitted) {
        return;
      }

      let recognition;
      try {
        recognition = new recognitionConstructor();
      } catch (_error) {
        state.voiceMessage =
          "Error: Could not start speech recognition. Check microphone access and try again.";
        voiceStatus.textContent = state.voiceMessage;
        return;
      }
      state.recognition = recognition;
      state.voiceSubmitted = false;
      state.manualStop = false;
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;
      recognition.lang = navigator.language || "en-US";
      state.voiceMessage = "Listening";
      voiceStatus.textContent = state.voiceMessage;
      startButton.disabled = true;
      stopButton.disabled = false;

      recognition.onresult = (event) => {
        if (state.manualStop || state.voiceSubmitted) return;
        const transcript = Array.from(event.results)
          .filter((result) => result.isFinal)
          .map((result) => result[0].transcript)
          .join(" ")
          .trim();
        if (!transcript) return;

        state.voiceSubmitted = true;
        state.voiceMessage = "Processing recognized question...";
        voiceStatus.textContent = state.voiceMessage;
        startButton.disabled = true;
        stopButton.disabled = true;
        setTriggerValue("voice_question", {
          id: eventId(),
          text: transcript
        });
        recognition.stop();
      };

      recognition.onerror = (event) => {
        if (state.manualStop || state.recognition !== recognition) return;
        state.recognition = null;
        state.voiceMessage =
          event.error === "no-speech"
            ? "Ready — no speech detected. Start again when ready."
            : permissionMessage(event.error);
        voiceStatus.textContent = state.voiceMessage;
        startButton.disabled = !recognitionSupported;
        stopButton.disabled = true;
      };

      recognition.onend = () => {
        if (state.recognition === recognition) state.recognition = null;
        startButton.disabled =
          !recognitionSupported || state.voiceSubmitted;
        stopButton.disabled = true;
        if (state.manualStop) {
          state.manualStop = false;
          state.voiceMessage = "Ready";
        } else if (state.voiceSubmitted) {
          state.voiceMessage = "Processing recognized question...";
        } else if (state.voiceMessage === "Listening") {
          state.voiceMessage = "Ready — no speech detected. Start again when ready.";
        }
        voiceStatus.textContent = state.voiceMessage;
      };

      try {
        recognition.start();
      } catch (error) {
        state.recognition = null;
        state.voiceMessage =
          error && error.name === "NotAllowedError"
            ? permissionMessage("not-allowed")
            : "Error: Could not start speech recognition. Check microphone access and try again.";
        voiceStatus.textContent = state.voiceMessage;
        startButton.disabled = !recognitionSupported;
        stopButton.disabled = true;
      }
    };

    stopButton.onclick = () => {
      const recognition = state.recognition;
      if (!recognition) return;
      state.recognition = null;
      state.manualStop = true;
      state.voiceMessage = "Ready";
      voiceStatus.textContent = state.voiceMessage;
      startButton.disabled = !recognitionSupported;
      stopButton.disabled = true;
      try {
        recognition.abort();
      } catch (_error) {
        state.manualStop = false;
        state.voiceMessage = "Error: Could not stop microphone access.";
        voiceStatus.textContent = state.voiceMessage;
      }
    };

    stopSpeechButton.onclick = () => {
      window.dispatchEvent(new CustomEvent(commandEvent, {
        detail: { action: "stop" }
      }));
    };
  } else {
    const text = typeof data.text === "string" ? data.text.trim() : "";
    readButton.disabled = !speechSupported || !text;
    if (!speechSupported) {
      readStatus.textContent =
        "Text-to-speech is not supported in this browser.";
    } else if (!text) {
      readStatus.textContent = "There is no response to read.";
    }
    readButton.onclick = () => {
      window.dispatchEvent(new CustomEvent(commandEvent, {
        detail: {
          action: "speak",
          controlId,
          text
        }
      }));
    };
  }

  return () => {
    if (state.recognition) {
      state.manualStop = true;
      state.recognition.abort();
      state.recognition = null;
    }
    if (state.commandHandler) {
      window.removeEventListener(commandEvent, state.commandHandler);
      state.commandHandler = null;
    }
    if (state.statusHandler) {
      window.removeEventListener(statusEvent, state.statusHandler);
      state.statusHandler = null;
    }
    if (state.activeUtterance) {
      state.activeUtterance = null;
      state.activeControlId = null;
      window.speechSynthesis.cancel();
    }
    instances.delete(parentElement);
  };
}
"""

_speech_component = st.components.v2.component(
    "crm_speech_controls",
    html=SPEECH_COMPONENT_HTML,
    css=SPEECH_COMPONENT_CSS,
    js=SPEECH_COMPONENT_JS,
)


def render_speech_control(
    mode,
    *,
    key,
    text="",
    status_reset_token=0,
    width="stretch",
):
    return _speech_component(
        key=key,
        data={
            "mode": mode,
            "control_id": key,
            "text": text,
            "status_reset_token": status_reset_token,
        },
        width=width,
        height="content",
        on_voice_question_change=lambda: None,
    )


# Allow only local HTTP backend addresses.
_backend_url = urlsplit(BACKEND_URL)

try:
    _backend_host_is_local = (
        _backend_url.hostname == "localhost"
        or ipaddress.ip_address(
            _backend_url.hostname or ""
        ).is_loopback
    )
except ValueError:
    _backend_host_is_local = False

if (
    _backend_url.scheme != "http"
    or not _backend_host_is_local
    or _backend_url.port is None
    or _backend_url.path not in ("", "/")
    or _backend_url.username
    or _backend_url.password
):
    raise ValueError(
        "BACKEND_URL must be a local HTTP address "
        "with an explicit port, such as "
        "http://127.0.0.1:5001"
    )


# ============================================================
# SERVICES
# ============================================================

SERVICES = {
    "All Services": {
        "icon": "🏥",
        "description": (
            "Ask questions across the available "
            "CRM knowledge base."
        ),
    },
    "Occupational Therapy": {
        "icon": "🧩",
        "description": (
            "Information about occupational therapy "
            "and everyday functional skills."
        ),
    },
    "Physiotherapy": {
        "icon": "🏃",
        "description": (
            "Information about movement, mobility, "
            "and physiotherapy services."
        ),
    },
    "Speech-Language Pathology": {
        "icon": "🗣️",
        "description": (
            "Information about speech, language, "
            "communication, and AAC."
        ),
    },
    "Clinical Psychology": {
        "icon": "🧠",
        "description": (
            "Information about psychological support "
            "and clinical psychology."
        ),
    },
    "Special Education": {
        "icon": "📚",
        "description": (
            "Information about learning support, "
            "IEPs, and special education."
        ),
    },
    "Nutrition": {
        "icon": "🥗",
        "description": (
            "Information about nutrition and "
            "medical nutrition therapy."
        ),
    },
}


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

SUGGESTIONS = {
    "All Services": [
        "What is Occupational Therapy?",
        "What is Speech-Language Pathology?",
        "What does physiotherapy do?",
        "How can speech therapy help children?",
    ],
    "Occupational Therapy": [
        "What is Occupational Therapy?",
        "What does an occupational therapist do?",
        "How can occupational therapy help children?",
        "What skills are supported by occupational therapy?",
    ],
    "Physiotherapy": [
        "What is physiotherapy?",
        "What does a physiotherapist do?",
        "How can physiotherapy help children?",
        "What areas does physiotherapy support?",
    ],
    "Speech-Language Pathology": [
        "What is Speech-Language Pathology?",
        "How can speech therapy help children?",
        "What communication difficulties can speech therapy support?",
        "What is AAC?",
    ],
    "Clinical Psychology": [
        "What is Clinical Psychology?",
        "What does a clinical psychologist do?",
        "What is neuropsychology?",
        "How can psychological support help children?",
    ],
    "Special Education": [
        "What is Special Education?",
        "What is an IEP?",
        "How can special education support children?",
        "What learning difficulties are addressed?",
    ],
    "Nutrition": [
        "What is Medical Nutrition Therapy?",
        "What does a nutritionist do?",
        "How does nutrition support children?",
        "What is nutritional support?",
    ],
}


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

for obsolete_auth_key in (
    "access_token",
    "identity",
    "login_error",
    "login_password",
    "login_username",
):
    st.session_state.pop(obsolete_auth_key, None)

DEFAULTS = {
    "dark_mode": False,
    "selected_service": "All Services",
    "messages": [],
    "backend_status": None,
    "pending_question": None,
    "last_voice_event_id": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# THEME COLORS
# ============================================================

if st.session_state.dark_mode:
    BG = "#0E1525"
    SIDEBAR = "#111827"
    CARD = "#182235"
    TEXT = "#F8FAFC"
    MUTED = "#CBD5E1"
    BORDER = "#334155"
else:
    BG = "#F5F7FB"
    SIDEBAR = "#FFFFFF"
    CARD = "#FFFFFF"
    TEXT = "#111827"
    MUTED = "#475569"
    BORDER = "#E2E8F0"


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BG};
        color: {TEXT};
    }}

    [data-testid="stSidebar"] {{
        background-color: {SIDEBAR};
    }}

    [data-testid="stChatMessage"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }}

    .main-title {{
        font-size: 2rem;
        font-weight: 750;
        margin-bottom: 4px;
    }}

    .subtitle {{
        color: {MUTED};
        margin-bottom: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🤖 CRM Assistant")
    st.caption("Document Intelligence System")

    st.divider()

    st.subheader("📚 Department / Service")

    service_names = list(SERVICES.keys())

    current_service = st.session_state.selected_service
    if current_service not in service_names:
        current_service = "All Services"

    selected_service = st.selectbox(
        "Choose a department or service",
        service_names,
        index=service_names.index(current_service),
    )

    st.session_state.selected_service = selected_service

    service_info = SERVICES[selected_service]

    st.info(
        f"{service_info['icon']} **{selected_service}**\n\n"
        f"{service_info['description']}"
    )

    st.divider()

    st.subheader("🎨 Appearance")

    if st.session_state.dark_mode:
        st.write("🌙 Dark Mode")

        if st.button(
            "☀️ Switch to Light Mode",
            use_container_width=True,
        ):
            st.session_state.dark_mode = False
            st.rerun()
    else:
        st.write("☀️ Light Mode")

        if st.button(
            "🌙 Switch to Dark Mode",
            use_container_width=True,
        ):
            st.session_state.dark_mode = True
            st.rerun()

    st.divider()

    if st.button(
        "➕ New Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.subheader("💡 Suggested Questions")

    for suggestion in SUGGESTIONS[selected_service]:
        if st.button(
            suggestion,
            key=f"suggestion_{selected_service}_{suggestion}",
            use_container_width=True,
        ):
            st.session_state.pending_question = suggestion
            st.rerun()

    st.divider()

    st.caption("Backend address")
    st.code(BACKEND_URL, language=None)

    if st.button(
        "🔌 Check Backend",
        use_container_width=True,
    ):
        try:
            health_response = requests.get(
                f"{BACKEND_URL}/",
                timeout=5,
            )

            if health_response.ok:
                st.session_state.backend_status = True
                st.success("Backend is reachable.")
            else:
                st.session_state.backend_status = False
                st.warning(
                    f"Backend returned HTTP "
                    f"{health_response.status_code}."
                )
        except requests.RequestException:
            st.session_state.backend_status = False
            st.error(
                "Cannot connect to the backend. "
                "Check that Flask is running."
            )


# ============================================================
# MAIN PAGE
# ============================================================

st.markdown(
    '<div class="main-title">CRM Document Intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your organisation’s knowledge base.'
    '</div>',
    unsafe_allow_html=True,
)

st.subheader("💬 Chat with your CRM assistant")

if st.session_state.backend_status is True:
    st.success("Backend connected")
elif st.session_state.backend_status is False:
    st.error("Backend unavailable")
else:
    st.caption("Backend status not checked")

if not st.session_state.messages:
    with st.container(border=True):
        st.markdown("#### Welcome to your CRM assistant")
        st.write(
            "Ask a question about your CRM knowledge base, or attach a "
            "document in the chat box to add it to local search."
        )

        suggestion_columns = st.columns(2)
        for index, suggestion in enumerate(SUGGESTIONS[selected_service][:4]):
            with suggestion_columns[index % 2]:
                if st.button(
                    suggestion,
                    key=f"welcome_suggestion_{selected_service}_{index}",
                    width="stretch",
                ):
                    st.session_state.pending_question = suggestion
                    st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message_index, message in enumerate(st.session_state.messages):
    role = message.get("role", "assistant")
    content = message.get("content", "")
    attachments = message.get("attachments", [])

    avatar = "🧑" if role == "user" else "🤖"

    with st.chat_message(role, avatar=avatar):
        for attachment in attachments:
            st.caption(f"Attached: {attachment}")
        if role == "assistant":
            answer_column, speech_column = st.columns(
                [8, 2],
                gap="small",
                vertical_alignment="top",
            )
            with answer_column:
                st.markdown(content)
            with speech_column:
                render_speech_control(
                    "read",
                    key=f"read_aloud_{message_index}",
                    text=str(content),
                    width="stretch",
                )
        else:
            st.markdown(content)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

voice_component = render_speech_control(
    "voice",
    key="crm_voice_controls",
    status_reset_token=len(st.session_state.messages),
)
voice_event_id, recognized_question = consume_voice_question(
    voice_component.voice_question,
    st.session_state.last_voice_event_id,
)
if voice_event_id != st.session_state.last_voice_event_id:
    st.session_state.last_voice_event_id = voice_event_id
    if recognized_question:
        st.session_state.pending_question = recognized_question

chat_submission = st.chat_input(
    "Ask a question about your CRM documents...",
    accept_file="multiple",
    file_type=UPLOAD_FILE_TYPES,
    max_upload_size=MAX_UPLOAD_SIZE_MB,
)

# Support suggested questions as well as typed questions.
pending_question = st.session_state.pending_question
st.session_state.pending_question = None

if isinstance(chat_submission, str):
    question = chat_submission.strip()
    uploaded_chat_files = []
elif chat_submission is not None:
    question = (chat_submission.text or "").strip()
    uploaded_chat_files = list(chat_submission.files or [])
else:
    question = pending_question or ""
    uploaded_chat_files = []

attachment_names = [
    os.path.basename(uploaded_file.name.replace("\\", "/"))
    for uploaded_file in uploaded_chat_files
]

if uploaded_chat_files:
    try:
        upload_response = upload_chat_files(uploaded_chat_files)
    except requests.RequestException:
        st.session_state.backend_status = False
        st.error(
            "The upload service is unavailable. Check that the Flask backend "
            "is running, then try again."
        )
        question = ""
    else:
        try:
            upload_data = upload_response.json()
        except ValueError:
            upload_data = None

        if upload_response.status_code not in (200, 201):
            error_message = (
                upload_data.get("message")
                if isinstance(upload_data, dict)
                else None
            )
            st.error(
                error_message
                or f"The upload failed (HTTP {upload_response.status_code})."
            )
            question = ""
        elif not isinstance(upload_data, dict) or not isinstance(
            upload_data.get("documents"), list
        ):
            st.error("The backend returned an unexpected upload response.")
            question = ""
        else:
            st.session_state.backend_status = True
            uploaded_documents = upload_data["documents"]
            if uploaded_documents:
                st.success(
                    f"{len(uploaded_documents)} file(s) uploaded and indexed."
                )
            else:
                st.info("The selected file(s) are already uploaded and indexed.")

            if not question:
                question = "Tell me about the file(s) I uploaded."


if question:
    # Display and save the user message.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "attachments": attachment_names,
        }
    )

    with st.chat_message("user", avatar="🧑"):
        for attachment in attachment_names:
            st.caption(f"Attached: {attachment}")
        st.markdown(question)

    # Add the selected service as context.
    question_to_send = question

    if selected_service != "All Services":
        question_to_send = (
            f"Service context: {selected_service}\n\n"
            f"Question: {question}"
        )

    # Ask the Flask backend.
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner(
            "Reading your information and preparing an answer..."
        ):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": question_to_send},
                    timeout=120,
                )

                if response.status_code != 200:
                    try:
                        error_data = response.json()
                    except ValueError:
                        error_data = {}

                    if isinstance(error_data, dict):
                        answer = error_data.get(
                            "message",
                            "The request could not be completed.",
                        )
                    else:
                        answer = (
                            "The request could not be completed."
                        )

                else:
                    data = response.json()

                    if isinstance(data, dict):
                        answer = data.get(
                            "answer",
                            "The backend did not return an answer.",
                        )
                    else:
                        answer = (
                            "The backend returned an unexpected response."
                        )

                st.session_state.backend_status = True

            except requests.exceptions.ConnectionError:
                answer = (
                    "I cannot connect to the Flask backend. "
                    "Please start the backend and try again."
                )
                st.session_state.backend_status = False

            except requests.exceptions.Timeout:
                answer = (
                    "⏳ The request timed out. The backend or AI "
                    "service may be taking too long. Please try again."
                )

            except (requests.RequestException, ValueError):
                answer = (
                    "The response could not be read. "
                    "Please check the backend logs and try again."
                )

            except Exception:
                answer = (
                    "An unexpected error occurred. "
                    "Please check the backend logs."
                )

        st.markdown(str(answer))
        st.caption("🤖 AI response")

    # Save the answer in the conversation.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": str(answer),
        }
    )

    # Refresh to show the complete conversation.
    st.rerun()
