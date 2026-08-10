import { useState, useRef, useEffect } from 'react';
import { Bell, Search, User, LogOut, ChevronDown } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import './Header.css';

export default function Header() {
    const { user, logout } = useAuth();
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const menuRef = useRef(null);

    // Close menu when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                setIsMenuOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

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
                <button className="header-btn notification-btn">
                    <Bell size={20} />
                    <span className="notification-badge">3</span>
                </button>

                <div className="user-menu-container" ref={menuRef}>
                    <button 
                        className="user-menu" 
                        onClick={() => setIsMenuOpen(!isMenuOpen)}
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
