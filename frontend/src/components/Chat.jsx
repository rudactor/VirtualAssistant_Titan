import {useEffect, useState} from 'react'
import plus from '../static/plus-circle.svg'

export default function Chat({checkToken}) {

    const [load, setLoad] = useState(true)
    const [chats, setChats] = useState([])
    const [messages, setMessages] = useState([])

    const [newChatTitle, setNewChatTitle] = useState('');

    const [currentChat, setCurrentChat] = useState('')

    const getChats = async () => { 
            const userId = await checkToken()
            setLoad(true)
            
            const response = await fetch(`http://127.0.0.1:8000/get_chats`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ user_id: userId })
        })
            const json = await response.json()
            const result = await Promise.resolve(json)
            const chats = result.data.flat().map(item => ({
                id: item[0],
                title: item[1]
            }));
            setChats(chats)
            setLoad(false)
        }

    const fetchAddChat = async (title) => {
        const userId = await checkToken()

        const response = await fetch(`http://127.0.0.1:8000/add_chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ title, user_id: userId })
        })
            const json = await response.json()
            const result = await Promise.resolve(json)
    }

    const getMessages = async (chatId, title) => {
        setCurrentChat(title)
        const response = await fetch(`http://127.0.0.1:8000/messages`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ chat_id: chatId })
        }
        )
        const json = await response.json()
        const result = await Promise.resolve(json)
        const messages = result.data.map(msg => ({
            role: msg[0],
            text: msg[1],
            timestamp: msg[2]
        }));

        setMessages(messages)
    }

    useEffect(() => {
        getChats()
    }, [])

    const handleAddChat = () => {
    setChats([{ id: null, title: '', isEditing: true }, ...chats]);
    setNewChatTitle('');
  }

  const handleSaveChat = (index) => {
    if (!newChatTitle.trim()) {
      setChats(chats.filter((_, i) => i !== index));
      return;
    }
    const updatedChats = [...chats];
    updatedChats[index] = { id: Date.now(), title: newChatTitle };
    fetchAddChat(newChatTitle)
    setChats(updatedChats);
  }

    return (
    <div className='container-chat'>
        <div className="leftBar">
            <div className='leftBar-top'>
                <h2>Названия чатов</h2>
                <img src={plus} alt='add-chat' className="plus" onClick={handleAddChat}/>
            </div>
            <div className='chats'>
                {load ? "" : chats.map((chat, index) => (
            chat.isEditing ? (
              <input
                key={index}
                autoFocus
                className="chat-input"
                placeholder="Название чата…"
                value={newChatTitle}
                onChange={e => setNewChatTitle(e.target.value)}
                onBlur={() => handleSaveChat(index)}
                onKeyDown={e => e.key === 'Enter' && handleSaveChat(index)}
              />
            ) : (
              <button
                key={chat.id}
                onClick={() => getMessages(chat.id, chat.title)}
              >
                {chat.title}
              </button>
            )
          ))}
            </div>
        </div>
        <div className='rightBar'>
            <h2>{currentChat}</h2>
            <div className='messages'>
                {messages.map(message => {
                    return (<div className={message.role === 'assistant' ? 'message assistent' : 'message user'}>{message.text}</div>)
                })}
            </div>
            <div className="input-wrapper">
                <input type="text" placeholder="Введите текст" />
                <span className="arrow">➔</span>
            </div>
        </div>
    </div>
    )
}