import { useEffect, useState } from "react";
import { getEvaluationReport, runAcceptanceTests } from "../api/chatApi";

const TABS = ["Pipeline", "Acceptance Tests", "Scores"];

function useEvaluationReport() {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    getEvaluationReport()
      .then((data) => {
        if (!cancelled) setReport(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.response?.data?.detail || err.message || "Failed to load report.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { report, error, loading };
}

function LoadState({ loading, error, loadingLabel }) {
  if (loading) return <p className="eval-summary-line">{loadingLabel}</p>;
  if (error) return <p className="eval-error">{error} — is the backend running on :8000?</p>;
  return null;
}

function PipelineTab() {
  const { report, error, loading } = useEvaluationReport();

  return (
    <div>
      <p className="eval-source-note">
        Live from <code>EVALUATION_REPORT.md</code> — the file every ablation script writes into.
      </p>
      <LoadState loading={loading} error={error} loadingLabel="Loading pipeline report..." />
      {report && (
        <ul className="eval-acceptance-list">
          {report.stages.map((s) => (
            <li key={s.stage} className="eval-acceptance-item pass">
              <div className="eval-acceptance-head">
                <span className="eval-acceptance-q">{s.stage}</span>
              </div>
              <p className="eval-acceptance-note">{s.winner}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function AcceptanceTab() {
  const [run, setRun] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleRun() {
    setLoading(true);
    setError(null);
    try {
      const data = await runAcceptanceTests();
      setRun(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Failed to run acceptance tests.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <p className="eval-source-note">
        Runs the 5 official questions against the live chatbot right now — a "pass" here means it
        refused exactly when it should have (or answered when it should have), not that every
        number in the answer was verified.
      </p>
      <button type="button" className="eval-run-btn" onClick={handleRun} disabled={loading}>
        {loading ? "Running (~20-30s)..." : run ? "Run again" : "Run acceptance tests"}
      </button>
      {error && <p className="eval-error">{error}</p>}
      {run && (
        <>
          <p className="eval-summary-line">
            {run.pass_count}/{run.total} acceptance questions pass
          </p>
          <ul className="eval-acceptance-list">
            {run.results.map((t) => (
              <li key={t.question} className={`eval-acceptance-item ${t.passed ? "pass" : "fail"}`}>
                <div className="eval-acceptance-head">
                  <span className={`eval-badge ${t.passed ? "pass" : "fail"}`}>
                    {t.passed ? "Pass" : "Fail"}
                  </span>
                  <span className="eval-acceptance-q">{t.question}</span>
                </div>
                <p className="eval-acceptance-note">{t.answer}</p>
                {t.citations.length > 0 && (
                  <p className="eval-acceptance-cites">
                    Sources: {t.citations.map((c) => `${c.document} p.${c.page}`).join("; ")}
                  </p>
                )}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function ScoreGroup({ title, items }) {
  if (!items || Object.keys(items).length === 0) return null;
  return (
    <div className="eval-score-group">
      <h3>{title}</h3>
      <div className="eval-score-grid">
        {Object.entries(items).map(([label, value]) => (
          <div key={label} className="eval-score-cell">
            <span className="eval-score-value">{value}</span>
            <span className="eval-score-label">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ScoresTab() {
  const { report, error, loading } = useEvaluationReport();

  return (
    <div>
      <p className="eval-source-note">
        Live from <code>EVALUATION_REPORT.md</code>'s end-to-end RAGAS + DeepEval run.
      </p>
      <LoadState loading={loading} error={error} loadingLabel="Loading scores..." />
      {report && (
        <>
          <ScoreGroup title="Retrieval" items={report.retrieval} />
          <ScoreGroup title="Generation quality" items={report.generation} />
          {report.negative_controls && (
            <p className="eval-summary-line">Negative controls: {report.negative_controls}</p>
          )}
        </>
      )}
    </div>
  );
}

export default function EvaluationPanel() {
  const [tab, setTab] = useState(TABS[0]);

  return (
    <section className="eval-panel">
      <div className="eval-panel-header">
        <h2>Evaluation Dashboard</h2>
        <p>Ablation results, acceptance tests, and end-to-end scores for this pipeline.</p>
      </div>
      <div className="eval-tabs">
        {TABS.map((t) => (
          <button
            key={t}
            type="button"
            className={`eval-tab ${tab === t ? "active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </div>
      <div className="eval-tab-content">
        {tab === "Pipeline" && <PipelineTab />}
        {tab === "Acceptance Tests" && <AcceptanceTab />}
        {tab === "Scores" && <ScoresTab />}
      </div>
    </section>
  );
}
