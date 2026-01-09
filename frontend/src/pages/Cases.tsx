/**
 * Cases Page
 * Case management with filtering and assignment
 */
import { useState, useEffect } from 'react';
import {
    Filter,
    PlayCircle,
    CheckCircle,
    AlertCircle,
    Search,
    ChevronDown
} from 'lucide-react';
import { getCases, assignCases, Case, AssignmentResult } from '../services/api';
import './Cases.css';

export default function Cases() {
    const [cases, setCases] = useState<Case[]>([]);
    const [loading, setLoading] = useState(true);
    const [assigning, setAssigning] = useState(false);
    const [filter, setFilter] = useState({ status: '', segment: '' });
    const [assignmentResults, setAssignmentResults] = useState<AssignmentResult[] | null>(null);

    useEffect(() => {
        loadCases();
    }, [filter]);

    const loadCases = async () => {
        setLoading(true);
        try {
            const params: Record<string, any> = {};
            if (filter.status) params.status = filter.status;
            if (filter.segment) params.segment = filter.segment;

            const response = await getCases(params);
            setCases(response.data);
        } catch (error) {
            console.error('Failed to load cases:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleAssign = async () => {
        setAssigning(true);
        try {
            const response = await assignCases();
            setAssignmentResults(response.data.assignments);
            loadCases(); // Refresh the list
        } catch (error) {
            console.error('Assignment failed:', error);
        } finally {
            setAssigning(false);
        }
    };

    const getP2PClass = (score: number) => {
        if (score >= 0.7) return 'high';
        if (score >= 0.4) return 'medium';
        return 'low';
    };

    const formatCurrency = (value: number) =>
        new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);

    return (
        <div className="cases-page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Case Management</h1>
                    <p className="page-subtitle">View and manage debt collection cases</p>
                </div>
                <button
                    className="btn btn-primary"
                    onClick={handleAssign}
                    disabled={assigning}
                >
                    <PlayCircle size={16} />
                    {assigning ? 'Assigning...' : 'Auto-Assign Cases'}
                </button>
            </div>

            {/* Assignment Results Banner */}
            {assignmentResults && assignmentResults.length > 0 && (
                <div className="assignment-banner">
                    <CheckCircle size={20} />
                    <span>
                        Successfully assigned {assignmentResults.length} cases.
                        Methods used: {assignmentResults.filter(r => r.method === 'ai').length} AI,
                        {assignmentResults.filter(r => r.method === 'fallback').length} Fallback
                    </span>
                    <button onClick={() => setAssignmentResults(null)}>×</button>
                </div>
            )}

            {/* Filters */}
            <div className="filters-bar">
                <div className="filter-group">
                    <Filter size={16} />
                    <select
                        value={filter.status}
                        onChange={(e) => setFilter(f => ({ ...f, status: e.target.value }))}
                        className="form-select"
                    >
                        <option value="">All Statuses</option>
                        <option value="unassigned">Unassigned</option>
                        <option value="assigned">Assigned</option>
                        <option value="in_progress">In Progress</option>
                        <option value="resolved">Resolved</option>
                    </select>
                </div>

                <div className="filter-group">
                    <select
                        value={filter.segment}
                        onChange={(e) => setFilter(f => ({ ...f, segment: e.target.value }))}
                        className="form-select"
                    >
                        <option value="">All Segments</option>
                        <option value="retail">Retail</option>
                        <option value="commercial">Commercial</option>
                        <option value="international">International</option>
                    </select>
                </div>

                <span className="case-count">
                    {cases.length} cases found
                </span>
            </div>

            {/* Cases Table */}
            {loading ? (
                <div className="loading-container">
                    <div className="spinner"></div>
                    <span>Loading cases...</span>
                </div>
            ) : (
                <div className="card">
                    <div className="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Invoice</th>
                                    <th>Customer</th>
                                    <th>Amount</th>
                                    <th>Days Overdue</th>
                                    <th>Segment</th>
                                    <th>P2P Score</th>
                                    <th>Status</th>
                                    <th>Agency</th>
                                </tr>
                            </thead>
                            <tbody>
                                {cases.map((caseItem) => (
                                    <tr key={caseItem.id}>
                                        <td className="invoice-id">{caseItem.invoice_id}</td>
                                        <td>
                                            <div className="customer-info">
                                                <span className="customer-name">{caseItem.customer_name}</span>
                                                <span className="customer-id">{caseItem.customer_id}</span>
                                            </div>
                                        </td>
                                        <td className="amount">{formatCurrency(caseItem.debt_amount)}</td>
                                        <td>
                                            <span className={`days-badge ${caseItem.days_overdue > 60 ? 'danger' : caseItem.days_overdue > 30 ? 'warning' : 'normal'}`}>
                                                {caseItem.days_overdue} days
                                            </span>
                                        </td>
                                        <td>
                                            <span className={`segment-badge ${caseItem.segment}`}>
                                                {caseItem.segment}
                                            </span>
                                        </td>
                                        <td>
                                            <div className="p2p-score">
                                                <div className="p2p-bar">
                                                    <div
                                                        className={`p2p-fill ${getP2PClass(caseItem.p2p_score)}`}
                                                        style={{ width: `${caseItem.p2p_score * 100}%` }}
                                                    ></div>
                                                </div>
                                                <span>{(caseItem.p2p_score * 100).toFixed(0)}%</span>
                                            </div>
                                        </td>
                                        <td>
                                            <span className={`badge badge-${caseItem.status}`}>
                                                {caseItem.status.replace('_', ' ')}
                                            </span>
                                        </td>
                                        <td>
                                            {caseItem.agency_id ? (
                                                <span className="agency-assigned">Agency #{caseItem.agency_id}</span>
                                            ) : (
                                                <span className="text-muted">—</span>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}
