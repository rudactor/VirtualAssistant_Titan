import "./styles/App.css";
import StartScreen from "./components/StartScreen";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Chat from "./components/Chat";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<StartScreen />} />
        <Route path="/chat" element={<Chat />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
