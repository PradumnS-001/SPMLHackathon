import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { TrendingUp, Mail, Lock, User, Building2, ArrowRight, ArrowLeft } from 'lucide-react';
import { register as registerApi, getAgencies } from '../services/api';
import './Login.css';

export default function Register() {
    const [fullName, setFullName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [role, setRole] = useState('agency');
    const [agencyId, setAgencyId] = useState('');
    const [agencies, setAgencies] = useState([]);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const navigate = useNavigate();

    useEffect(() => {
        // Fetch agencies for the dropdown (public endpoint not needed - we'll handle gracefully)
        const fetchAgencies = async () => {
            try {
                const res = await getAgencies();
                setAgencies(res.data || []);
            } catch {
                // If agencies endpoint requires auth, we'll just let them type an ID
                setAgencies([]);
            }
        };
        fetchAgencies();
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSuccess('');

        if (password !== confirmPassword) {
            setError('Passwords do not match.');
            return;
        }

        if (password.length < 6) {
            setError('Password must be at least 6 characters.');
            return;
        }

        setIsLoading(true);

        try {
            await registerApi({
                email,
                password,
                full_name: fullName,
                role,
                agency_id: role === 'agency' && agencyId ? parseInt(agencyId) : null
            });
            setSuccess('Account created successfully! Redirecting to login...');
            setTimeout(() => navigate('/login'), 2000);
        } catch (err) {
            setError(err.response?.data?.detail || 'Registration failed. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="login-container">
            <div className="login-card register-card">
                <div className="login-header">
                    <div className="login-logo">
                        <TrendingUp size={32} />
                    </div>
                    <div>
                        <h1 className="login-title">Create Account</h1>
                        <p className="login-subtitle">Join the DCA Management System</p>
                    </div>
                </div>

                {error && <div className="error-message">{error}</div>}
                {success && <div className="success-message">{success}</div>}

                <form className="login-form" onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label>Full Name</label>
                        <div className="input-wrapper">
                            <User size={18} className="input-icon" />
                            <input
                                type="text"
                                className="login-input"
                                placeholder="John Doe"
                                value={fullName}
                                onChange={(e) => setFullName(e.target.value)}
                                required
                            />
                        </div>
                    </div>

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

                    <div className="form-row">
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

                        <div className="form-group">
                            <label>Confirm Password</label>
                            <div className="input-wrapper">
                                <Lock size={18} className="input-icon" />
                                <input
                                    type="password"
                                    className="login-input"
                                    placeholder="••••••••"
                                    value={confirmPassword}
                                    onChange={(e) => setConfirmPassword(e.target.value)}
                                    required
                                />
                            </div>
                        </div>
                    </div>

                    <div className="form-group">
                        <label>Role</label>
                        <div className="role-selector">
                            <button
                                type="button"
                                className={`role-btn ${role === 'agency' ? 'active' : ''}`}
                                onClick={() => setRole('agency')}
                            >
                                <Building2 size={16} />
                                Agency User
                            </button>
                            <button
                                type="button"
                                className={`role-btn ${role === 'admin' ? 'active' : ''}`}
                                onClick={() => setRole('admin')}
                            >
                                <User size={16} />
                                Admin
                            </button>
                        </div>
                    </div>

                    {role === 'agency' && (
                        <div className="form-group">
                            <label>Agency</label>
                            <div className="input-wrapper">
                                <Building2 size={18} className="input-icon" />
                                {agencies.length > 0 ? (
                                    <select
                                        className="login-input login-select"
                                        value={agencyId}
                                        onChange={(e) => setAgencyId(e.target.value)}
                                    >
                                        <option value="">Select your agency</option>
                                        {agencies.map(a => (
                                            <option key={a.id} value={a.id}>{a.name}</option>
                                        ))}
                                    </select>
                                ) : (
                                    <input
                                        type="number"
                                        className="login-input"
                                        placeholder="Agency ID"
                                        value={agencyId}
                                        onChange={(e) => setAgencyId(e.target.value)}
                                    />
                                )}
                            </div>
                        </div>
                    )}

                    <button 
                        type="submit" 
                        className="login-btn"
                        disabled={isLoading || !email || !password || !fullName}
                    >
                        {isLoading ? (
                            <div className="loader"></div>
                        ) : (
                            <>
                                Create Account
                                <ArrowRight size={18} />
                            </>
                        )}
                    </button>
                </form>

                <div className="login-footer">
                    <p>Already have an account? <Link to="/login" className="register-link"><ArrowLeft size={14} /> Back to Login</Link></p>
                </div>
            </div>
        </div>
    );
}
