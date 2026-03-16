import './App.css';
import { Routes, Route } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import Home from './pages/Home'; // Changed from AnswerSheet to Home

function App() {
    return (
        <Routes>
            <Route path="/" element={<LoginPage />} />
            <Route path="/home" element={<Home />} /> {/* Changed to Home */}
        </Routes>
    );
}

export default App;
