import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Briefcase, Building2, Shield } from 'lucide-react';
import './MobileTabBar.css';

const items = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/cases', icon: Briefcase, label: 'Cases' },
  { to: '/agencies', icon: Building2, label: 'Agencies' },
  { to: '/compliance', icon: Shield, label: 'Compliance' },
];

export default function MobileTabBar() {
  return (
    <nav className="mobile-tabbar" role="navigation" aria-label="Main">
      {items.map(item => {
        const Icon = item.icon;
        return (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => `tab-item ${isActive ? 'active' : ''}`}
          >
            <Icon size={20} />
            <span className="tab-label">{item.label}</span>
          </NavLink>
        );
      })}
    </nav>
  );
}
