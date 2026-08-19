import CitationList from "./CitationList";

export default function MessageBubble({ role, content, citations, isError }) {
  const className = `message-bubble ${role}${isError ? " error" : ""}`;
  return (
    <div className={className}>
      <div className="message-content">{content}</div>
      {role === "assistant" && !isError && <CitationList citations={citations} />}
    </div>
  );
}
