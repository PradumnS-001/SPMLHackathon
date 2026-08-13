import { useState, useRef, useEffect } from 'react';
import { Bell, Search, User, LogOut, ChevronDown, Check, CheckCircle2, AlertTriangle, Info, X, ShieldAlert } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import './Header.css';

export default function Header() {
    const { user, logout } = useAuth();
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const [isNotifOpen, setIsNotifOpen] = useState(false);
    const [notifications, setNotifications] = useState([
        {
            id: 1,
            type: 'danger',
            title: 'FDCPA Violation Flagged',
            message: 'High-severity aggressive statement detected in Call Transcript #TR-8841.',
            time: '5 mins ago',
            read: false
        },
        {
            id: 2,
            type: 'success',
            title: 'Auto-Assignment Complete',
            message: '14 high-priority overdue cases auto-assigned to top DCA agencies.',
            time: '20 mins ago',
            read: false
        },
        {
            id: 3,
            type: 'warning',
            title: 'Agency Capacity Warning',
            message: 'Apex Collections has reached 92% of maximum workload capacity.',
            time: '1 hour ago',
            read: false
        },
        {
            id: 4,
            type: 'info',
            title: 'SLA Audit Passed',
            message: 'Monthly DCA compliance audit score reached 98.4%.',
            time: '3 hours ago',
            read: true
        }
    ]);

    const menuRef = useRef(null);
    const notifRef = useRef(null);

    // Close dropdowns when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                setIsMenuOpen(false);
            }
            if (notifRef.current && !notifRef.current.contains(event.target)) {
                setIsNotifOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const unreadCount = notifications.filter(n => !n.read).length;

    const markAllAsRead = () => {
        setNotifications(prev => prev.map(n => ({ ...n, read: true })));
    };

    const toggleRead = (id) => {
        setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: !n.read } : n));
    };

    const dismissNotification = (id) => {
        setNotifications(prev => prev.filter(n => n.id !== id));
    };

    return (
        <header className="header">
            <div className="search-bar">
                <Search size={18} />
                <input
                    type="text"
                    placeholder="Search cases, agencies, invoices..."
                    className="search-input"
                />
            </div>

            <div className="header-actions">
                <div className="notification-container" ref={notifRef}>
                    <button 
                        className={`header-btn notification-btn ${isNotifOpen ? 'active' : ''}`}
                        onClick={() => {
                            setIsNotifOpen(!isNotifOpen);
                            setIsMenuOpen(false);
                        }}
                        title="Notifications"
                    >
                        <Bell size={20} />
                        {unreadCount > 0 && (
                            <span className="notification-badge">{unreadCount}</span>
                        )}
                    </button>

                    {isNotifOpen && (
                        <div className="notification-dropdown">
                            <div className="notification-header">
                                <div className="notif-header-title">
                                    <h4>Notifications</h4>
                                    {unreadCount > 0 && <span className="unread-pill">{unreadCount} new</span>}
                                </div>
                                {unreadCount > 0 && (
                                    <button className="mark-read-btn" onClick={markAllAsRead}>
                                        <Check size={14} /> Mark all read
                                    </button>
                                )}
                            </div>

                            <div className="notification-list">
                                {notifications.length === 0 ? (
                                    <div className="notification-empty">
                                        <CheckCircle2 size={32} className="empty-icon" style={{ opacity: 0.4 }} />
                                        <p>No active notifications</p>
                                    </div>
                                ) : (
                                    notifications.map(n => (
                                        <div 
                                            key={n.id} 
                                            className={`notification-item ${n.read ? 'read' : 'unread'} type-${n.type}`}
                                            onClick={() => toggleRead(n.id)}
                                        >
                                            <div className={`notif-type-icon icon-${n.type}`}>
                                                {n.type === 'danger' && <ShieldAlert size={16} />}
                                                {n.type === 'warning' && <AlertTriangle size={16} />}
                                                {n.type === 'success' && <CheckCircle2 size={16} />}
                                                {n.type === 'info' && <Info size={16} />}
                                            </div>
                                            <div className="notif-body">
                                                <div className="notif-title-row">
                                                    <span className="notif-title">{n.title}</span>
                                                    <span className="notif-time">{n.time}</span>
                                                </div>
                                                <p className="notif-desc">{n.message}</p>
                                            </div>
                                            <button 
                                                className="notif-dismiss-btn" 
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    dismissNotification(n.id);
                                                }}
                                                title="Dismiss notification"
                                            >
                                                <X size={14} />
                                            </button>
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    )}
                </div>

                <div className="user-menu-container" ref={menuRef}>
                    <button 
                        className="user-menu" 
                        onClick={() => {
                            setIsMenuOpen(!isMenuOpen);
                            setIsNotifOpen(false);
                        }}
                    >
                        <div className="user-avatar">
                            {user?.full_name ? user.full_name.charAt(0).toUpperCase() : <User size={20} />}
                        </div>
                        <div className="user-info">
                            <span className="user-name">{user?.full_name || 'Loading...'}</span>
                            <span className="user-role">{user?.role === 'admin' ? 'System Admin' : 'Agency User'}</span>
                        </div>
                        <ChevronDown size={16} className={`dropdown-icon ${isMenuOpen ? 'open' : ''}`} />
                    </button>

                    {isMenuOpen && (
                        <div className="dropdown-menu">
                            <button className="dropdown-item text-danger" onClick={logout}>
                                <LogOut size={16} />
                                <span>Log Out</span>
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </header>
    );
}

