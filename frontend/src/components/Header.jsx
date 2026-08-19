import wfLogo from "../assets/wf-logo-white.svg";

function NavLink({ children }) {
  return (
    <a href="#" className="wf-navlink" onClick={(e) => e.preventDefault()}>
      {children}
    </a>
  );
}

export default function Header() {
  return (
    <header className="wf-header">
      <div className="wf-header-bar">
        <img src={wfLogo} alt="Wells Fargo" className="wf-logo" />
        <nav className="wf-header-nav">
          <NavLink>ATMs/Locations</NavLink>
          <NavLink>Help</NavLink>
          <NavLink>Español</NavLink>
          <button type="button" className="wf-icon-btn" aria-label="Search">
            <svg viewBox="0 0 20 20" width="18" height="18" fill="none" aria-hidden="true">
              <circle cx="8.5" cy="8.5" r="6.5" stroke="currentColor" strokeWidth="2" />
              <line x1="13.4" y1="13.4" x2="18" y2="18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>
          <button type="button" className="wf-signon-btn">
            Sign On
          </button>
        </nav>
      </div>
      <div className="wf-header-stripe" />
      <div className="wf-header-title">
        <h1>Document Assistant</h1>
        <p>Answers are grounded in your deposit account, fees, and credit card agreement documents.</p>
      </div>
    </header>
  );
}
