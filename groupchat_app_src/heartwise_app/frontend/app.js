
const API = "/api";
const $ = (id) => document.getElementById(id);

const url = new URL(window.location.href);

if (url.searchParams.has("utm_source")) {
    url.searchParams.delete("utm_source");
    window.history.replaceState({}, "", url.pathname + url.search + url.hash);
}

let token = localStorage.getItem("heartwise_token");
let currentUsername = localStorage.getItem("heartwise_username");
let signupMode = false;
let busy = false;

function showScreen(authenticated) {
  $("auth-screen").hidden = authenticated;
  $("chat-screen").hidden = !authenticated;
  $("account").hidden = !authenticated;
  $("username").textContent = authenticated ? currentUsername : "";
}

function logout() {
  token = null;
  currentUsername = null;
  localStorage.removeItem("heartwise_token");
  localStorage.removeItem("heartwise_username");
  showScreen(false);
}

async function apiRequest(path, options = {}) {
  const response = await fetch(API + path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers
    }
  });

  const data = await response.json();

  if (!response.ok) {
    if (response.status === 401 && path === "/ask") logout();
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : `Request failed (${response.status})`
    );
  }

  return data;
}

$("auth-toggle").addEventListener("click", () => {
  signupMode = !signupMode;
  $("auth-submit").textContent = signupMode ? "Create account" : "Log in";
  $("auth-toggle").textContent = signupMode
    ? "Already have an account? Log in"
    : "New here? Create an account";
  $("auth-error").textContent = "";
});

$("auth-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("auth-error").textContent = "";
  $("auth-submit").disabled = true;

  try {
    const username = $("auth-username").value.trim();
    const password = $("auth-password").value;

    const data = await apiRequest(signupMode ? "/signup" : "/login", {
      method: "POST",
      body: JSON.stringify({ username, password })
    });

    token = data.token;
    currentUsername = data.username;
    localStorage.setItem("heartwise_token", token);
    localStorage.setItem("heartwise_username", currentUsername);
    $("auth-password").value = "";
    showScreen(true);
  } catch (error) {
    $("auth-error").textContent = error.message;
  } finally {
    $("auth-submit").disabled = false;
  }
});

$("logout-btn").addEventListener("click", logout);

function addMessage(sender, content, sources = [], emergency = false) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${sender === "user" ? "user-message" : "assistant-message"}`;

  if (emergency) wrapper.classList.add("emergency-message");

  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = sender === "user" ? "You" : "HeartWise";

  const body = document.createElement("div");
  body.className = "message-content";
  body.textContent = content;

  wrapper.append(label, body);

  if (sources.length) {
    const sourcesBox = document.createElement("div");
    sourcesBox.className = "sources";

    const heading = document.createElement("strong");
    heading.textContent = "References";
    sourcesBox.appendChild(heading);

    for (const source of sources) {
      const link = document.createElement("a");
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = `${source.publisher}: ${source.title}`;
      sourcesBox.appendChild(link);
    }

    wrapper.appendChild(sourcesBox);
  }

  $("messages").appendChild(wrapper);
  $("messages").scrollTop = $("messages").scrollHeight;

  return wrapper;
}

async function sendQuestion(question) {
  question = question.trim();
  if (!question || busy || !token) return;

  busy = true;
  $("send-btn").disabled = true;
  $("question").value = "";
  $("suggestions").hidden = true;

  addMessage("user", question);
  const loadingMessage = addMessage("assistant", "Thinking...");

  try {
    const result = await apiRequest("/ask", {
      method: "POST",
      body: JSON.stringify({ question })
    });

    loadingMessage.remove();
    addMessage(
      "assistant",
      result.answer,
      result.sources || [],
      result.emergency
    );
  } catch (error) {
    loadingMessage.remove();
    addMessage("assistant", `Sorry, I couldn't answer that. ${error.message}`);
  } finally {
    busy = false;
    $("send-btn").disabled = false;
    $("question").focus();
  }
}

$("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion($("question").value);
});

$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("chat-form").requestSubmit();
  }
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    sendQuestion(button.dataset.question);
  });
});

showScreen(Boolean(token));

