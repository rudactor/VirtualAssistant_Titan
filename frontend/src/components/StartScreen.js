import "../styles/main.css";
import { useState } from "react";
import Registration from "./Registration";
import Authorization from "./Authorization";

export default function StartScreen({ setAuth }) {
  const [popupReg, setPopupReg] = useState(false);
  const [popupAuth, setPopupAuth] = useState(false);

  return (
    <div>
      {popupReg ? (
        <Registration setPopup={setPopupReg} setAuth={setAuth} />
      ) : (
        ""
      )}
      {popupAuth ? <Authorization setPopup={setPopupAuth} /> : ""}
      <div className="container">
        <div className="block-info">
          <h1 className="title">Виртуальный ассистент</h1>
          <p className="small-info">Помощник в любом вопросе.</p>
          <div className="buttons">
            <button
              className="button black-button"
              onClick={() => {
                setPopupAuth(true);
              }}
            >
              Авторизация
            </button>
            <button
              className="button white-button"
              onClick={() => {
                setPopupReg(true);
              }}
            >
              Регистрация
            </button>
          </div>
        </div>
        <img
          src={require("../static/ChatGPT Image 27 сент. 2025 г., 16_35_33 1.png")}
          alt="mainImage"
        />
      </div>
    </div>
  );
}
