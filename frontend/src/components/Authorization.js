export default function Authorization({ setPopup }) {
  return (
    <div
      className="popup"
      onClick={() => {
        setPopup(false);
      }}
    >
      <div className="popup-container" onClick={(e) => e.stopPropagation()}>
        <h2>Авторизация</h2>
        <input placeholder="Логин" />
        <input type="password" placeholder="Пароль" />
        <button>Отправить</button>
      </div>
    </div>
  );
}
