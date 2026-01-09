/**
 * Sidebar Navigation Component
 */
import { NavLink } from 'react-router-dom';
import {
    LayoutDashboard,
    Briefcase,
    Building2,
    Shield,
    UserCog,
    TrendingUp
} from 'lucide-react';
import './Sidebar.css';

const navItems = [
    { path: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/cases', icon: Briefcase, label: 'Cases' },
    { path: '/agencies', icon: Building2, label: 'Agencies' },
    { path: '/compliance', icon: Shield, label: 'Compliance' },
    { path: '/portal', icon: UserCog, label: 'Agency Portal' },
];

export default function Sidebar() {
    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div className="logo">
                    <TrendingUp size={28} />
                    <div className="logo-text">
                        <span className="logo-title">DCA Manager</span>
                        <span className="logo-subtitle">FedEx</span>
                    </div>
                </div>
            </div>

            <nav className="sidebar-nav">
                {navItems.map(({ path, icon: Icon, label }) => (
                    <NavLink
                        key={path}
                        to={path}
                        className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
                    >
                        <Icon size={20} />
                        <span>{label}</span>
                    </NavLink>
                ))}
            </nav>

            <div className="sidebar-footer">
                <div className="system-status">
                    <div className="status-indicator online"></div>
                    <span>System Online</span>
                </div>
                <div className="version">v1.0.0 MVP</div>
            </div>
        </aside>
    );
}
