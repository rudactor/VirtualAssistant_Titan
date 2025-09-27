import plus from '../static/plus-circle.svg'

export default function Chat() {
    return (
    <div className='container-chat'>
        <div className="leftBar">
            <div className='leftBar-top'>
                <h2>Названия чатов</h2>
                <img src={plus} alt='add-chat' className="plus"/>
            </div>
        </div>
        <div className='rightBar'>
            <h2>Имя Чата</h2>
            <div className='messages'>

            </div>
            <div class="input-wrapper">
                <input type="text" placeholder="Введите текст" />
                <span class="arrow">➔</span>
            </div>
        </div>
    </div>
    )
}