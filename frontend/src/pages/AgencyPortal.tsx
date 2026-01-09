/**
 * Agency Portal Page
 * DCA agent working interface
 */
import { useState, useEffect } from 'react';
import {
    Phone,
    Mail,
    CheckCircle,
    Clock,
    DollarSign,
    AlertCircle,
    User
} from 'lucide-react';
import { getAgencyCases, resolveCase, Case } from '../services/api';
import './AgencyPortal.css';

export default function AgencyPortal() {
    const [cases, setCases] = useState<Case[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedCase, setSelectedCase] = useState<Case | null>(null);
    const [resolveAmount, setResolveAmount] = useState('');

    // Demo: Using agency ID 1
    const agencyId = 1;

    useEffect(() => {
        loadCases();
    }, []);

    const loadCases = async () => {
        try {
            const response = await getAgencyCases(agencyId);
            setCases(response.data);
        } catch (error) {
            console.error('Failed to load cases:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleResolve = async (caseItem: Case) => {
        const amount = parseFloat(resolveAmount);
        if (isNaN(amount) || amount <= 0) return;

        try {
            await resolveCase(caseItem.id, amount, 'Resolved via portal');
            setSelectedCase(null);
            setResolveAmount('');
            loadCases();
        } catch (error) {
            console.error('Failed to resolve case:', error);
        }
    };

    const formatCurrency = (value: number) =>
        new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);

    const getP2PClass = (score: number) => {
        if (score >= 0.7) return 'high';
        if (score >= 0.4) return 'medium';
        return 'low';
    };

    const activeCases = cases.filter(c => c.status !== 'resolved');
    const resolvedCases = cases.filter(c => c.status === 'resolved');

    if (loading) {
        return (
            <div className="loading-container">
                <div className="spinner"></div>
                <span>Loading your cases...</span>
            </div>
        );
    }

    return (
        <div className="portal-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Agency Portal</h1>
                    <p className="page-subtitle">Your assigned cases and tasks</p>
                </div>
            </div>

            {/* Quick Stats */}
            <div className="portal-stats">
                <div className="portal-stat">
                    <Clock size={20} />
                    <span className="stat-value">{activeCases.length}</span>
                    <span className="stat-label">Active Cases</span>
                </div>
                <div className="portal-stat success">
                    <CheckCircle size={20} />
                    <span className="stat-value">{resolvedCases.length}</span>
                    <span className="stat-label">Resolved</span>
                </div>
                <div className="portal-stat">
                    <DollarSign size={20} />
                    <span className="stat-value">
                        {formatCurrency(activeCases.reduce((sum, c) => sum + c.debt_amount, 0))}
                    </span>
                    <span className="stat-label">Total Outstanding</span>
                </div>
            </div>

            {/* Cases List */}
            <div className="portal-content">
                <div className="cases-list">
                    <h3 className="section-title">Active Cases ({activeCases.length})</h3>

                    {activeCases.length === 0 ? (
                        <div className="empty-state">
                            <CheckCircle size={48} />
                            <p>No active cases assigned</p>
                        </div>
                    ) : (
                        activeCases.map((caseItem) => (
                            <div
                                key={caseItem.id}
                                className={`case-card ${selectedCase?.id === caseItem.id ? 'selected' : ''}`}
                                onClick={() => setSelectedCase(caseItem)}
                            >
                                <div className="case-header">
                                    <span className="case-invoice">{caseItem.invoice_id}</span>
                                    <span className={`p2p-badge ${getP2PClass(caseItem.p2p_score)}`}>
                                        P2P: {(caseItem.p2p_score * 100).toFixed(0)}%
                                    </span>
                                </div>

                                <div className="case-customer">
                                    <User size={16} />
                                    {caseItem.customer_name}
                                </div>

                                <div className="case-details">
                                    <span className="case-amount">{formatCurrency(caseItem.debt_amount)}</span>
                                    <span className="case-overdue">{caseItem.days_overdue} days overdue</span>
                                </div>

                                <div className="case-actions">
                                    <button className="action-btn" title="Call">
                                        <Phone size={16} />
                                    </button>
                                    <button className="action-btn" title="Email">
                                        <Mail size={16} />
                                    </button>
                                    <button
                                        className="action-btn resolve"
                                        title="Mark Resolved"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setSelectedCase(caseItem);
                                        }}
                                    >
                                        <CheckCircle size={16} />
                                    </button>
                                </div>
                            </div>
                        ))
                    )}
                </div>

                {/* Case Detail Panel */}
                <div className="case-detail">
                    {selectedCase ? (
                        <>
                            <h3 className="section-title">Case Details</h3>

                            <div className="detail-card">
                                <div className="detail-header">
                                    <span className="detail-invoice">{selectedCase.invoice_id}</span>
                                    <span className={`badge badge-${selectedCase.status}`}>
                                        {selectedCase.status.replace('_', ' ')}
                                    </span>
                                </div>

                                <div className="detail-section">
                                    <h4>Customer Information</h4>
                                    <div className="detail-row">
                                        <span>Name</span>
                                        <span>{selectedCase.customer_name}</span>
                                    </div>
                                    <div className="detail-row">
                                        <span>Email</span>
                                        <span>{selectedCase.customer_email}</span>
                                    </div>
                                    <div className="detail-row">
                                        <span>Phone</span>
                                        <span>{selectedCase.customer_phone}</span>
                                    </div>
                                </div>

                                <div className="detail-section">
                                    <h4>Debt Information</h4>
                                    <div className="detail-row">
                                        <span>Amount</span>
                                        <span className="amount">{formatCurrency(selectedCase.debt_amount)}</span>
                                    </div>
                                    <div className="detail-row">
                                        <span>Days Overdue</span>
                                        <span>{selectedCase.days_overdue}</span>
                                    </div>
                                    <div className="detail-row">
                                        <span>Segment</span>
                                        <span className="capitalize">{selectedCase.segment}</span>
                                    </div>
                                    <div className="detail-row">
                                        <span>P2P Score</span>
                                        <span className={`p2p-text ${getP2PClass(selectedCase.p2p_score)}`}>
                                            {(selectedCase.p2p_score * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                </div>

                                {selectedCase.status !== 'resolved' && (
                                    <div className="resolve-section">
                                        <h4>Resolve Case</h4>
                                        <div className="resolve-form">
                                            <input
                                                type="number"
                                                placeholder="Amount Recovered"
                                                value={resolveAmount}
                                                onChange={(e) => setResolveAmount(e.target.value)}
                                                className="form-input"
                                            />
                                            <button
                                                className="btn btn-success"
                                                onClick={() => handleResolve(selectedCase)}
                                                disabled={!resolveAmount}
                                            >
                                                Mark Resolved
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </>
                    ) : (
                        <div className="empty-detail">
                            <AlertCircle size={48} />
                            <p>Select a case to view details</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
