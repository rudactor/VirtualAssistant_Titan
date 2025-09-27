import { useState } from "react";

export default function Authorization({ setPopup }) {
  const [login, setLogin] = useState("");
  const [password, setPassword] = useState("");

  const registerFetch = () => {
    fetch("http://127.0.0.1:8000/auth", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ login, password }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.message === "you are authorized") {
          localStorage.setItem("token", data.token);
          window.location.href = "/";
        }
      })
      .catch((error) => console.error("Ошибка:", error));
  };

  return (
    <div
      className="popup"
      onClick={() => {
        setPopup(false);
      }}
    >
      <div className="popup-container" onClick={(e) => e.stopPropagation()}>
        <h2>Авторизация</h2>
        <input
          placeholder="Логин"
          value={login}
          onChange={(e) => setLogin(e.target.value)}
        />
        <input
          type="password"
          placeholder="Пароль"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button onClick={registerFetch}>Отправить</button>
      </div>
    </div>
  );
}
