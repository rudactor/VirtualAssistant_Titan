export default function Registration({ setPopup }) {
  return (
    <div
      className="popup"
      onClick={() => {
        setPopup(false);
      }}
    >
      <div className="popup-container" onClick={(e) => e.stopPropagation()}>
        <h2>Регистрация</h2>
        <input placeholder="Логин" />
        <input type="password" placeholder="Пароль" />
        <input placeholder="Ваша роль" />
        <button>Отправить</button>
      </div>
    </div>
  );
}
