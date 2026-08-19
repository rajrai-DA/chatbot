import { useEffect, useRef, useState } from "react";
import { sendMessage } from "../api/chatApi";
import ChatInput from "./ChatInput";
import MessageBubble from "./MessageBubble";

const SESSION_KEY = "wf_chat_session_id";

function getOrCreateSessionId() {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

export default function ChatWindow() {
  const [sessionId] = useState(getOrCreateSessionId);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hi, I'm the Wells Fargo document assistant. Ask me about your deposit account terms, fees, or credit card agreement.",
      citations: [],
    },
  ]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend(text) {
    setMessages((prev) => [...prev, { role: "user", content: text, citations: [] }]);
    setLoading(true);
    try {
      const data = await sendMessage(sessionId, text);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, citations: data.citations || [] },
      ]);
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || "Something went wrong.";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${detail}`, citations: [], isError: true },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="chat-window">
      <div className="chat-history">
        {messages.map((m, i) => (
          <MessageBubble key={i} role={m.role} content={m.content} citations={m.citations} isError={m.isError} />
        ))}
        {loading && (
          <div className="message-bubble assistant loading">
            <div className="message-content">Thinking...</div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <ChatInput onSend={handleSend} disabled={loading} />
    </div>
  );
}
