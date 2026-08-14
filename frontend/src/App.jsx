import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ChatbotWidget from './components/ChatbotWidget';
import MobileTabBar from './components/MobileTabBar';
import Dashboard from './pages/Dashboard';
import Cases from './pages/Cases';
import Agencies from './pages/Agencies';
import Compliance from './pages/Compliance';
import AgencyPortal from './pages/AgencyPortal';
import Login from './pages/Login';
import Register from './pages/Register';

// Protected Route Wrapper
const ProtectedRoute = ({ children }) => {
    const { user, loading } = useAuth();
    const location = useLocation();

    if (loading) {
        return <div style={{ display: 'flex', height: '100vh', alignItems: 'center', justifyContent: 'center' }}>Loading...</div>;
    }

    if (!user) {
        return <Navigate to="/login" state={{ from: location }} replace />;
    }

    return children;
};

// Admin Route Wrapper
const AdminRoute = ({ children }) => {
    const { user } = useAuth();
    
    // Only allow admins, redirect others to their portal
    if (user?.role !== 'admin') {
        return <Navigate to="/portal" replace />;
    }

    return children;
};

// Smart redirect for the root path
const RoleBasedRedirect = () => {
    const { user } = useAuth();
    return user?.role === 'admin' ? (
        <Navigate to="/dashboard" replace />
    ) : (
        <Navigate to="/portal" replace />
    );
};

// Layout for authenticated pages
const AppLayout = ({ children }) => (
    <div className="app-layout">
        <Sidebar />
        <div className="main-content">
            <Header />
            {children}
            <ChatbotWidget />
            <MobileTabBar />
        </div>
    </div>
);

function App() {
    return (
        <AuthProvider>
            <BrowserRouter>
                <Routes>
                    {/* Public Route */}
                    <Route path="/login" element={<Login />} />
                    <Route path="/register" element={<Register />} />

                    {/* Protected Routes */}
                    <Route
                        path="/*"
                        element={
                            <ProtectedRoute>
                                <AppLayout>
                                    <Routes>
                                        <Route path="/" element={<RoleBasedRedirect />} />
                                        
                                        {/* Admin Only Routes */}
                                        <Route path="/dashboard" element={<AdminRoute><Dashboard /></AdminRoute>} />
                                        <Route path="/cases" element={<AdminRoute><Cases /></AdminRoute>} />
                                        <Route path="/agencies" element={<AdminRoute><Agencies /></AdminRoute>} />
                                        <Route path="/compliance" element={<AdminRoute><Compliance /></AdminRoute>} />
                                        
                                        {/* Available to both, but primarily for Agencies */}
                                        <Route path="/portal" element={<AgencyPortal />} />
                                    </Routes>
                                </AppLayout>
                            </ProtectedRoute>
                        }
                    />
                </Routes>
            </BrowserRouter>
        </AuthProvider>
    );
}

export default App;
