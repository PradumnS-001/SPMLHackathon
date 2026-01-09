/**
 * Header Component
 */
import { Bell, Search, User } from 'lucide-react';
import './Header.css';

export default function Header() {
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

                <div className="user-menu">
                    <div className="user-avatar">
                        <User size={20} />
                    </div>
                    <div className="user-info">
                        <span className="user-name">Admin User</span>
                        <span className="user-role">System Admin</span>
                    </div>
                </div>
            </div>
        </header>
    );
}
