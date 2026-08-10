import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { TrendingUp, Mail, Lock, ArrowRight } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import './Login.css';

export default function Login() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        try {
            await login(email, password);
            navigate('/dashboard');
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to login. Please check your credentials.');
        } finally {
            setIsLoading(false);
        }
    };

    const loadDemoCredentials = (role) => {
        if (role === 'admin') {
            setEmail('admin@fedex.com');
            setPassword('admin123');
        } else {
            setEmail('agent1@recovermaxsolutions.com');
            setPassword('agent123');
        }
    };

    return (
        <div className="login-container">
            <div className="login-card">
                <div className="login-header">
                    <div className="login-logo">
                        <TrendingUp size={32} />
                    </div>
                    <div>
                        <h1 className="login-title">DCA Manager</h1>
                        <p className="login-subtitle">FedEx Debt Collection Agency System</p>
                    </div>
                </div>

                {error && <div className="error-message">{error}</div>}

                <form className="login-form" onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label>Email Address</label>
                        <div className="input-wrapper">
                            <Mail size={18} className="input-icon" />
                            <input
                                type="email"
                                className="login-input"
                                placeholder="name@company.com"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                            />
                        </div>
                    </div>

                    <div className="form-group">
                        <label>Password</label>
                        <div className="input-wrapper">
                            <Lock size={18} className="input-icon" />
                            <input
                                type="password"
                                className="login-input"
                                placeholder="••••••••"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />
                        </div>
                    </div>

                    <button 
                        type="submit" 
                        className="login-btn"
                        disabled={isLoading || !email || !password}
                    >
                        {isLoading ? (
                            <div className="loader"></div>
                        ) : (
                            <>
                                Sign In
                                <ArrowRight size={18} />
                            </>
                        )}
                    </button>
                </form>

                <div className="demo-credentials">
                    <p>Demo Accounts</p>
                    <div className="demo-buttons">
                        <button 
                            type="button" 
                            className="demo-btn"
                            onClick={() => loadDemoCredentials('admin')}
                        >
                            Admin
                        </button>
                        <button 
                            type="button" 
                            className="demo-btn"
                            onClick={() => loadDemoCredentials('agent')}
                        >
                            Agency User
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
