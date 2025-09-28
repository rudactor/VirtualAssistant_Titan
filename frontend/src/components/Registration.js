import { useState } from "react";

export default function Registration({ setPopup, setAuth }) {
  const [login, setLogin] = useState("");
  const [password, setPassword] = useState("");

  const [flagCheckReg, setFlagCheckReg] = useState(true);

  const checkReg = () => {
    return login.length > 4 && password.length > 8;
  };

  const registerFetch = () => {
    if (!checkReg()) {
      setFlagCheckReg(false);
      return;
    }

    fetch("http://127.0.0.1:8000/reg", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ login, password }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.message === "successfully") {
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
        <h2>Регистрация</h2>
        <input
          placeholder="Логин"
          value={login}
          onChange={(e) => {
            setLogin(e.target.value);
          }}
        />
        <input
          type="password"
          placeholder="Пароль"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {!flagCheckReg && (
          <p style={{ color: "red" }}>Введите корректные данные</p>
        )}
        <button
          onClick={registerFetch}
          disabled={!checkReg()}
          style={{
            opacity: checkReg() ? 1 : 0.5,
            cursor: checkReg() ? "pointer" : "not-allowed",
          }}
        >
          Отправить
        </button>
      </div>
    </div>
  );
}
