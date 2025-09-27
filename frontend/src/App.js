import "./styles/main.css";
import StartScreen from "./components/StartScreen";
import { useState, useEffect } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Chat from "./components/Chat";
import NotFound from "./components/NotFound";

function App() {
  const [auth, setAuth] = useState(false);

  useEffect(() => {
    if (localStorage.getItem("token")) {
      setAuth(true);
    }
  }, []);

  return (
    <BrowserRouter>
      <Routes>
        {!auth ? (
          <Route path="/" element={<StartScreen setAuth={setAuth} />} />
        ) : (
          ""
        )}
        {auth ? (
          <Route path="/" element={<Chat />} />
        ) : (
          <Route path="/" element={<StartScreen setAuth={setAuth} />} />
        )}

        <Route path="*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
