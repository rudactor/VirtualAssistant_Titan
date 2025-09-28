import {useEffect, useState} from 'react'
import plus from '../static/plus-circle.svg'

export default function Chat({checkToken}) {
  const [load, setLoad] = useState(true)
  const [chats, setChats] = useState([])
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [currentChatId, setCurrentChatId] = useState(0)
  const [newChatTitle, setNewChatTitle] = useState('')
  const [currentChat, setCurrentChat] = useState('')
  const [loadingAssistant, setLoadingAssistant] = useState(false)

  function formatTimestamp(ts) {
    if (!isNaN(ts)) return new Date(ts * 1000).toLocaleString()
    return new Date(ts).toLocaleString()
  }

  const getChats = async () => { 
    const userId = await checkToken()
    setLoad(true)
    const response = await fetch(`http://127.0.0.1:8000/get_chats`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId })
    })
    const json = await response.json()
    const result = await Promise.resolve(json)
    const chats = result.data.flat().map(item => ({ id: item[0], title: item[1] }))
    setChats(chats)
    setLoad(false)
  }

  const fetchAddChat = async (title) => {
    const userId = await checkToken()
    await fetch(`http://127.0.0.1:8000/add_chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, user_id: userId })
    })
  }

  const getMessages = async (chatId, title) => {
    setCurrentChat(title)
    setCurrentChatId(chatId)
    const response = await fetch(`http://127.0.0.1:8000/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chat_id: chatId })
    })
    const json = await response.json()
    const result = await Promise.resolve(json)
    const messages = result.data.map(msg => ({ role: msg[0], text: msg[1], timestamp: msg[2] }))
    setMessages(messages)
  }

  function formatText(text) {
    const parts = text.split(/(\*\*.*?\*\*)/g)
    return parts.map((part, i) => {
      if ((part.startsWith("**") && part.endsWith("**")) || (part.startsWith("*") && part.endsWith("*"))) {
        if (part.startsWith("**") && part.endsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>
        return <strong key={i}>{part.slice(1, -1)}</strong>
      }
      return <span key={i}>{part}</span>
    })
  }

  useEffect(() => { getChats() }, [])

  const handleAddChat = () => {
    setChats([{ id: null, title: '', isEditing: true }, ...chats])
    setNewChatTitle('')
  }

  const handleSaveChat = (index) => {
    if (!newChatTitle.trim()) {
      setChats(chats.filter((_, i) => i !== index))
      return
    }
    const updatedChats = [...chats]
    updatedChats[index] = { id: Date.now(), title: newChatTitle }
    fetchAddChat(newChatTitle)
    setChats(updatedChats)
  }

  const askQuestion = async () => {
    const q = question.trim()
    if (!q || loadingAssistant || !currentChatId) return
    setQuestion('')
    const newMessage = { role: 'user', text: q, timestamp: new Date().toISOString() }
    setMessages(prev => [...prev, newMessage])
    setLoadingAssistant(true)
    try {
      await fetch(`http://127.0.0.1:8000/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chat_id: currentChatId, question: q })
      })
      await getMessages(currentChatId, currentChat)
    } finally {
      setLoadingAssistant(false)
    }
  }

  const handleInputKeyDown = (e) => {
    if (e.nativeEvent.isComposing) return
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      askQuestion()
    }
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
              <button key={chat.id} onClick={() => getMessages(chat.id, chat.title)}>
                {chat.title}
              </button>
            )
          ))}
        </div>
      </div>

      <div className='rightBar'>
        <h2>{currentChat}</h2>
        <div className='messages'>
          {messages.map((message, idx) => (
            <div key={idx} className={message.role === 'assistant' ? 'message assistent' : 'message user'}>
              <p className='content'>{formatText(message.text)}</p>
              <p className='timestamp'>{formatTimestamp(message.timestamp)}</p>
            </div>
          ))}

          {loadingAssistant && (
            <div className='message assistent'>
              <p className='content'>Ассистент печатает...</p>
            </div>
          )}
        </div>

        <div className="input-wrapper">
          <input
            type="text"
            placeholder="Введите текст"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleInputKeyDown}
          />
          <button className="arrow" onClick={askQuestion} disabled={loadingAssistant || !question.trim()}>
            ➔
          </button>
        </div>
      </div>
    </div>
  )
}
