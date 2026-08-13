import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, X, Send, Sparkles, Shield, Zap, RefreshCw, User, ChevronRight, Maximize2, Minimize2 } from 'lucide-react';
import { sendChatMessage, getChatSuggestions } from '../services/api';
import { useAuth } from '../context/AuthContext';
import './ChatbotWidget.css';

const ChatbotWidget = () => {
    const { user } = useAuth();
    const [isOpen, setIsOpen] = useState(false);
    const [isMaximized, setIsMaximized] = useState(false);
    const [messages, setMessages] = useState([
        {
            sender: 'bot',
            text: '👋 Hello! I am your **FedEx DCA AI Assistant**. Ask me anything about high-priority cases, FDCPA compliance checks, agency workload, or auto-assigning invoices!',
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [suggestions, setSuggestions] = useState([]);
    const messagesEndRef = useRef(null);

    useEffect(() => {
        if (isOpen && suggestions.length === 0) {
            fetchSuggestions();
        }
    }, [isOpen]);

    useEffect(() => {
        scrollToBottom();
    }, [messages, loading]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const fetchSuggestions = async () => {
        try {
            const res = await getChatSuggestions();
            setSuggestions(res.data.suggestions || []);
        } catch (err) {
            console.error('Failed to load chat suggestions:', err);
        }
    };

    const handleSend = async (textToSend) => {
        const query = textToSend || input;
        if (!query.trim() || loading) return;

        const userMsg = {
            sender: 'user',
            text: query,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };

        setMessages(prev => [...prev, userMsg]);
        if (!textToSend) setInput('');
        setLoading(true);

        try {
            const response = await sendChatMessage(query, { role: user?.role });
            const botData = response.data;

            const botMsg = {
                sender: 'bot',
                text: botData.reply || "I've processed your request.",
                actionCard: botData.action_card,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            };

            setMessages(prev => [...prev, botMsg]);
        } catch (err) {
            console.error('Chat API Error:', err);
            const errorReply = err.response?.data?.detail 
                ? `⚠️ ${err.response.data.detail}`
                : '⚠️ I encountered a temporary connection issue reaching the operations backend. Please try your request again in a moment.';
            setMessages(prev => [
                ...prev,
                {
                    sender: 'bot',
                    text: errorReply,
                    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                }
            ]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const renderActionCard = (card) => {
        if (!card) return null;

        if (card.type === 'cases_list' && card.cases) {
            return (
                <div className="action-card cases-card">
                    <div className="card-header">
                        <h4>{card.title}</h4>
                        <span className="card-badge">{card.badge}</span>
                    </div>
                    <div className="card-list">
                        {card.cases.map(c => (
                            <div key={c.id} className="case-item">
                                <div className="case-info">
                                    <span className="inv-id">{c.invoice_id}</span>
                                    <span className="cust-name">{c.customer_name}</span>
                                </div>
                                <div className="case-metrics">
                                    <span className="amount">${c.debt_amount?.toLocaleString()}</span>
                                    <span className="priority-badge">P: {c.priority_score}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            );
        }

        if (card.type === 'compliance_result') {
            return (
                <div className="action-card compliance-card">
                    <div className="card-header">
                        <h4>{card.title}</h4>
                        <span className={`card-badge ${card.badge.includes('COMPLIANT') ? 'success' : 'danger'}`}>
                            {card.badge}
                        </span>
                    </div>
                    {card.violations && card.violations.length > 0 && (
                        <div className="violations-list">
                            <h5>Detected Violation Patterns:</h5>
                            <ul>
                                {card.violations.map((v, i) => (
                                    <li key={i}>
                                        <strong>{v.type}</strong> ({v.severity} severity): <em>{v.excerpt || v.disclosure}</em>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            );
        }

        if (card.type === 'assignment_summary') {
            return (
                <div className="action-card assignment-card">
                    <div className="card-header">
                        <h4>{card.title}</h4>
                        <span className="card-badge success">{card.badge}</span>
                    </div>
                    <p className="card-subtext">Optimized based on DCA capabilities & category match.</p>
                </div>
            );
        }

        if (card.type === 'p2p_summary') {
            return (
                <div className="action-card p2p-card">
                    <div className="card-header">
                        <h4>{card.title}</h4>
                        <span className="card-badge success">{card.badge}</span>
                    </div>
                    <div className="card-list">
                        {card.items?.map((item, i) => (
                            <div key={i} className="case-item">
                                <span className="cust-name">{item.label}</span>
                                <span className="priority-badge">{item.value}</span>
                            </div>
                        ))}
                    </div>
                </div>
            );
        }

        return null;
    };

    return (
        <div className="chatbot-widget-container">
            {!isOpen && (
                <button className="chatbot-toggle-btn" onClick={() => setIsOpen(true)}>
                    <Bot className="toggle-icon" size={24} />
                    <span className="toggle-pulse"></span>
                    <span className="toggle-label">FedEx Copilot</span>
                </button>
            )}

            {isOpen && (
                <>
                    {isMaximized && (
                        <div className="chatbot-backdrop" onClick={() => setIsMaximized(false)} />
                    )}
                    <div className={`chatbot-drawer ${isMaximized ? 'maximized' : ''}`}>
                        <div className="chatbot-header">
                            <div className="header-left">
                                <div className="bot-avatar">
                                    <Bot size={20} />
                                </div>
                                <div>
                                    <h3 className="header-title">FedEx DCA AI Assistant</h3>
                                    <div className="header-subtitle">
                                        <span className="online-dot"></span>
                                        <span>DCA Intelligence Operations</span>
                                    </div>
                                </div>
                            </div>
                            <div className="header-actions">
                                <button 
                                    className="header-icon-btn"
                                    onClick={() => setIsMaximized(!isMaximized)}
                                    title={isMaximized ? "Restore size" : "Maximize window"}
                                >
                                    {isMaximized ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
                                </button>
                                <button 
                                    className="header-icon-btn close-btn" 
                                    onClick={() => setIsOpen(false)}
                                    title="Close chat"
                                >
                                    <X size={18} />
                                </button>
                            </div>
                        </div>

                        <div className="chatbot-messages">
                            {messages.map((msg, index) => (
                                <div key={index} className={`message-row ${msg.sender}`}>
                                    {msg.sender === 'bot' && (
                                        <div className="msg-avatar">
                                            <Sparkles size={14} />
                                        </div>
                                    )}
                                    <div className="message-content">
                                        <div className="message-text">
                                            {msg.sender === 'bot' ? (
                                                <ReactMarkdown 
                                                    remarkPlugins={[remarkGfm]}
                                                    components={{
                                                        table: ({node, ...props}) => (
                                                            <div className="markdown-table-wrapper">
                                                                <table {...props} />
                                                            </div>
                                                        ),
                                                        a: ({node, ...props}) => (
                                                            <a target="_blank" rel="noopener noreferrer" {...props} />
                                                        )
                                                    }}
                                                >
                                                    {msg.text}
                                                </ReactMarkdown>
                                            ) : (
                                                <p>{msg.text}</p>
                                            )}
                                        </div>
                                        {renderActionCard(msg.actionCard)}
                                        <span className="message-time">{msg.timestamp}</span>
                                    </div>
                                </div>
                            ))}
                            {loading && (
                                <div className="message-row bot loading">
                                    <div className="msg-avatar">
                                        <Sparkles size={14} />
                                    </div>
                                    <div className="message-content">
                                        <div className="typing-indicator">
                                            <span></span>
                                            <span></span>
                                            <span></span>
                                        </div>
                                    </div>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>

                        {suggestions.length > 0 && (
                            <div className="chatbot-suggestions">
                                {suggestions.map((s, idx) => (
                                    <button key={idx} className="suggestion-pill" onClick={() => handleSend(s)}>
                                        {s}
                                    </button>
                                ))}
                            </div>
                        )}

                        <div className="chatbot-input-area">
                            <textarea
                                className="chat-textarea"
                                placeholder="Ask AI Assistant or paste call transcript..."
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={handleKeyPress}
                                rows={1}
                            />
                            <button className="send-btn" onClick={() => handleSend()} disabled={!input.trim() || loading}>
                                <Send size={16} />
                            </button>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};

export default ChatbotWidget;

