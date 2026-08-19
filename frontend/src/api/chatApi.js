import axios from "axios";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  timeout: 30000,
});

export async function sendMessage(sessionId, message) {
  const response = await client.post("/chat", { session_id: sessionId, message });
  return response.data;
}

export async function checkHealth() {
  const response = await client.get("/health");
  return response.data;
}

export async function getEvaluationReport() {
  const response = await client.get("/evaluate/report");
  return response.data;
}

export async function runAcceptanceTests() {
  // Runs 5 real questions through the live chatbot — slower than a typical
  // request, so this gets its own timeout rather than the 30s default.
  const response = await client.post("/evaluate/acceptance", null, { timeout: 60000 });
  return response.data;
}
