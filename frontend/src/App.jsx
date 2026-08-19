import Header from "./components/Header";
import ChatWindow from "./components/ChatWindow";
import EvaluationPanel from "./components/EvaluationPanel";
import "./App.css";

function App() {
  return (
    <div className="app">
      <Header />
      <div className="app-body">
        <div className="app-col app-col-chat">
          <ChatWindow />
        </div>
        <div className="app-col app-col-eval">
          <EvaluationPanel />
        </div>
      </div>
    </div>
  );
}

export default App;
