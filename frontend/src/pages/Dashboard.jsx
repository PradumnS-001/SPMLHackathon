/**
 * Dashboard Page
 * Executive overview with KPIs and analytics
 */
import { useState, useEffect } from 'react';
import {
    TrendingUp,
    TrendingDown,
    DollarSign,
    Users,
    Clock,
    CheckCircle,
    AlertTriangle,
    Zap
} from 'lucide-react';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell
} from 'recharts';
import { getDashboardStats, getCasesByStatus, getAgencyPerformance } from '../services/api';
import './Dashboard.css';

const COLORS = ['#3B82F6', '#F59E0B', '#8B5CF6', '#10B981', '#EF4444'];

export default function Dashboard() {
    const [stats, setStats] = useState(null);
    const [casesByStatus, setCasesByStatus] = useState([]);
    const [agencyPerf, setAgencyPerf] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            const [statsRes, statusRes, perfRes] = await Promise.all([
                getDashboardStats(),
                getCasesByStatus(),
                getAgencyPerformance(5)
            ]);
            setStats(statsRes.data);
            setCasesByStatus(statusRes.data);
            setAgencyPerf(perfRes.data);
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="loading-container">
                <div className="spinner"></div>
                <span>Loading dashboard...</span>
            </div>
        );
    }

    const formatCurrency = (value) =>
        new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value);

    return (
        <div className="dashboard">
            <div className="page-header">
                <div>
                    <h1 className="page-title">Dashboard</h1>
                    <p className="page-subtitle">Real-time overview of debt collection operations</p>
                </div>
                <button className="btn btn-primary" onClick={loadData}>
                    <Zap size={16} />
                    Refresh Data
                </button>
            </div>

            {/* KPI Grid */}
            <div className="kpi-grid">
                <div className="kpi-card">
                    <div className="kpi-icon">
                        <DollarSign size={24} />
                    </div>
                    <div className="kpi-content">
                        <span className="kpi-label">Total Debt</span>
                        <span className="kpi-value">{formatCurrency(stats?.total_debt || 0)}</span>
                        <div className="kpi-change positive">
                            <TrendingUp size={14} />
                            <span>{formatCurrency(stats?.total_recovered || 0)} recovered</span>
                        </div>
                    </div>
                </div>

                <div className="kpi-card success">
                    <div className="kpi-icon success">
                        <CheckCircle size={24} />
                    </div>
                    <div className="kpi-content">
                        <span className="kpi-label">Recovery Rate</span>
                        <span className="kpi-value">{stats?.recovery_rate?.toFixed(1) || 0}%</span>
                        <div className="kpi-change positive">
                            <TrendingUp size={14} />
                            <span>{stats?.resolved_cases || 0} cases resolved</span>
                        </div>
                    </div>
                </div>

                <div className="kpi-card warning">
                    <div className="kpi-icon warning">
                        <Clock size={24} />
                    </div>
                    <div className="kpi-content">
                        <span className="kpi-label">Avg Days Overdue</span>
                        <span className="kpi-value">{stats?.avg_days_overdue?.toFixed(0) || 0}</span>
                        <div className="kpi-change negative">
                            <TrendingDown size={14} />
                            <span>{stats?.unassigned_cases || 0} unassigned</span>
                        </div>
                    </div>
                </div>

                <div className="kpi-card info">
                    <div className="kpi-icon info">
                        <Users size={24} />
                    </div>
                    <div className="kpi-content">
                        <span className="kpi-label">SLA Compliance</span>
                        <span className="kpi-value">{stats?.sla_compliance?.toFixed(1) || 0}%</span>
                        <div className="kpi-change positive">
                            <TrendingUp size={14} />
                            <span>{stats?.total_cases || 0} total cases</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Charts Row */}
            <div className="charts-row">
                <div className="card chart-card">
                    <div className="card-header">
                        <h3 className="card-title">Cases by Status</h3>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={casesByStatus}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={100}
                                    paddingAngle={2}
                                    dataKey="count"
                                    nameKey="status"
                                    label={({ status, percentage }) => `${status}: ${percentage}%`}
                                >
                                    {casesByStatus.map((_, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="card chart-card">
                    <div className="card-header">
                        <h3 className="card-title">Agency Performance</h3>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={agencyPerf} layout="vertical">
                                <XAxis type="number" domain={[0, 100]} />
                                <YAxis type="category" dataKey="agency_name" width={120} />
                                <Tooltip />
                                <Bar dataKey="recovery_rate" fill="#4D148C" radius={[0, 4, 4, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Agency Leaderboard */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Agency Leaderboard</h3>
                    <span className="text-muted">Top performing agencies</span>
                </div>
                <div className="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Agency</th>
                                <th>Cases</th>
                                <th>Resolved</th>
                                <th>Recovery Rate</th>
                                <th>Avg Resolution</th>
                                <th>Compliance</th>
                            </tr>
                        </thead>
                        <tbody>
                            {agencyPerf.map((agency, index) => (
                                <tr key={agency.agency_id}>
                                    <td>
                                        <span className={`rank rank-${index + 1}`}>#{index + 1}</span>
                                    </td>
                                    <td className="agency-name">{agency.agency_name}</td>
                                    <td>{agency.total_cases}</td>
                                    <td>{agency.resolved_cases}</td>
                                    <td>
                                        <span className={`recovery-rate ${agency.recovery_rate >= 70 ? 'high' : agency.recovery_rate >= 50 ? 'medium' : 'low'}`}>
                                            {agency.recovery_rate.toFixed(1)}%
                                        </span>
                                    </td>
                                    <td>{agency.avg_resolution_days.toFixed(0)} days</td>
                                    <td>
                                        <div className="compliance-bar">
                                            <div
                                                className="compliance-fill"
                                                style={{ width: `${agency.compliance_score * 100}%` }}
                                            ></div>
                                            <span>{(agency.compliance_score * 100).toFixed(0)}%</span>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
