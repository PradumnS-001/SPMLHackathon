import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import Cases from './pages/Cases';
import Agencies from './pages/Agencies';
import Compliance from './pages/Compliance';
import AgencyPortal from './pages/AgencyPortal';

function App() {
    return (
        <BrowserRouter>
            <div className="app-layout">
                <Sidebar />
                <div className="main-content">
                    <Header />
                    <Routes>
                        <Route path="/" element={<Navigate to="/dashboard" replace />} />
                        <Route path="/dashboard" element={<Dashboard />} />
                        <Route path="/cases" element={<Cases />} />
                        <Route path="/agencies" element={<Agencies />} />
                        <Route path="/compliance" element={<Compliance />} />
                        <Route path="/portal" element={<AgencyPortal />} />
                    </Routes>
                </div>
            </div>
        </BrowserRouter>
    );
}

export default App;
