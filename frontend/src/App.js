import "./styles/main.css";
import StartScreen from "./components/StartScreen";
import { useState, useEffect } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Chat from "./components/Chat";
import NotFound from "./components/NotFound";

function App() {
  const [auth, setAuth] = useState(false);

  const checkToken = async () => {
    const response = await fetch("http://127.0.0.1:8000/check", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${localStorage.getItem("token")}`,
      },
    });
    const json = await response.json();
    const result = await Promise.resolve(json);
    return result.message;
  };

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
          <Route path="/" element={<Chat checkToken={checkToken} />} />
        ) : (
          <Route path="/" element={<StartScreen setAuth={setAuth} />} />
        )}

        <Route path="*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
