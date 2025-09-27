import "./styles/main.css";
import StartScreen from "./components/StartScreen";
import { useState } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Chat from "./components/Chat";
import NotFound from "./components/NotFound";

function App() {
  const [auth, setAuth] = useState(true);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<StartScreen />} />
        {auth ? <Route path="/chat" element={<Chat />} /> : "Not found"}

        <Route path="*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
