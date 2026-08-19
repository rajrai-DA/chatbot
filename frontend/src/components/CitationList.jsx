export default function CitationList({ citations }) {
  if (!citations || citations.length === 0) return null;

  return (
    <div className="citation-list">
      <div className="citation-list-label">Sources</div>
      <ul>
        {citations.map((c, i) => (
          <li key={i}>
            [{i + 1}] {c.document}
            {c.page !== null && c.page !== undefined && c.page !== "" ? ` — p. ${c.page}` : ""}
          </li>
        ))}
      </ul>
    </div>
  );
}
