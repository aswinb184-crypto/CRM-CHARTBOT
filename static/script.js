const API_URL = "http://127.0.0.1:5001/chat";

const chatMessages = document.getElementById("chatMessages");
const questionInput = document.getElementById("questionInput");
const sendButton = document.getElementById("sendButton");
const typing = document.getElementById("typing");


// ======================================================
// STOP CURRENT SPEECH
// ======================================================

function stopSpeech() {

    if ("speechSynthesis" in window) {

        window.speechSynthesis.cancel();

        console.log("Previous speech stopped.");

    }
}


// ======================================================
// ADD USER MESSAGE
// ======================================================

function addUserMessage(question) {

    const message = document.createElement("div");

    message.className = "message user-message";

    message.innerHTML = `
        <div class="avatar user-avatar">
            YOU
        </div>

        <div class="message-content">

            <div class="message-name">
                You
            </div>

            <div class="message-bubble">
                ${escapeHTML(question)}
            </div>

        </div>
    `;

    chatMessages.appendChild(message);

    scrollToBottom();
}


// ======================================================
// ADD BOT MESSAGE
// ======================================================

function addBotMessage(answer) {

    const message = document.createElement("div");

    message.className = "message bot-message";

    message.innerHTML = `
        <div class="avatar bot-avatar">
            AI
        </div>

        <div class="message-content">

            <div class="message-name">
                CRM Assistant
            </div>

            <div class="message-bubble">
                ${formatAnswer(answer)}
            </div>

        </div>
    `;

    chatMessages.appendChild(message);

    scrollToBottom();
}


// ======================================================
// FORMAT ANSWER
// ======================================================

function formatAnswer(text) {

    return escapeHTML(text)
        .replace(/\n\n/g, "<br><br>")
        .replace(/\n/g, "<br>");
}


// ======================================================
// ESCAPE HTML
// ======================================================

function escapeHTML(text) {

    const div = document.createElement("div");

    div.textContent = text;

    return div.innerHTML;
}


// ======================================================
// SCROLL CHAT TO BOTTOM
// ======================================================

function scrollToBottom() {

    chatMessages.scrollTop = chatMessages.scrollHeight;
}


// ======================================================
// TYPING INDICATOR
// ======================================================

function showTyping() {

    typing.classList.remove("hidden");

    scrollToBottom();
}


function hideTyping() {

    typing.classList.add("hidden");
}


// ======================================================
// SEND MESSAGE
// ======================================================

async function sendMessage() {

    const question = questionInput.value.trim();

    if (!question) {
        return;
    }


    // IMPORTANT:
    // Stop any previous speech immediately
    // before starting a new question.

    stopSpeech();


    // Add user's message

    addUserMessage(question);


    // Clear input

    questionInput.value = "";


    // Disable send button

    sendButton.disabled = true;


    // Show typing

    showTyping();


    try {

        const response = await fetch(API_URL, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question
            })

        });


        const data = await response.json();


        hideTyping();


        if (data.status === "success") {

            addBotMessage(data.answer);

        } else {

            addBotMessage(
                "Sorry, I couldn't process your question. " +
                (data.message || "")
            );

        }

    } catch (error) {

        hideTyping();

        addBotMessage(
            "Unable to connect to the CRM chatbot server. " +
            "Please make sure the Flask backend is running."
        );

        console.error(error);

    } finally {

        sendButton.disabled = false;

        questionInput.focus();
    }
}


// ======================================================
// SUGGESTION BUTTON
// ======================================================

function askSuggestion(question) {

    // Stop previous speech when suggestion is selected

    stopSpeech();


    questionInput.value = question;

    sendMessage();
}


// ======================================================
// ENTER KEY
// ======================================================

questionInput.addEventListener("keydown", function(event) {

    if (event.key === "Enter" && !event.shiftKey) {

        event.preventDefault();

        sendMessage();

    }

});


// ======================================================
// AUTO RESIZE TEXTAREA
// ======================================================

questionInput.addEventListener("input", function() {

    this.style.height = "auto";

    this.style.height =
        Math.min(this.scrollHeight, 100) + "px";

});


// ======================================================
// STOP SPEECH WHEN PAGE IS CLOSED / REFRESHED
// ======================================================

window.addEventListener("beforeunload", function() {

    stopSpeech();

});