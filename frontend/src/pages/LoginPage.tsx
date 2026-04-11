import { useState } from 'react'
import { useNavigate } from 'react-router-dom';

export default function LoginPage(){
    const navigate = useNavigate();

    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const [error, setError] = useState('')

    const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if(username === 'admin@gmail.com' && password === 'admin'){
            localStorage.setItem('username', username);
            localStorage.setItem('password', password);
            navigate('/home');
        }
        else{
            setError('Invalid credentials');
        }
    };

    return (
        <>
            <div className="w-screen h-screen flex items-center justify-center" style={{backgroundColor: '#BDE8F5'}}>
                <div className="flex w-[800px] h-[500px] bg-white shadow-xl">
                    <div className="w-1/2 flex flex-col justify-center px-12" style={{backgroundColor: '#0F2854'}}>
                        <h1 className="text-4xl font-light mb-8 text-white" style={{color: '#BDE8F5'}}>welcome back</h1>
                        <form onSubmit={handleLogin} className="flex flex-col gap-4">
                            <input 
                                type="email" 
                                placeholder="email" 
                                value={username} 
                                onChange={(e) => setUsername(e.target.value)} 
                                className="bg-transparent border-b border-white/30 py-2 text-white placeholder:text-white/50 outline-none focus:border-white transition-colors"
                            />
                            <input 
                                type="password" 
                                placeholder="password" 
                                value={password} 
                                onChange={(e) => setPassword(e.target.value)} 
                                className="bg-transparent border-b border-white/30 py-2 text-white placeholder:text-white/50 outline-none focus:border-white transition-colors"
                            />
                            {error && <p className="text-sm text-white/80">invalid credentials</p>}
                            <button 
                                type="submit" 
                                className="mt-4 py-2 px-6 w-fit text-sm uppercase tracking-wider transition-colors duration-300"
                                style={{backgroundColor: '#BDE8F5', color: '#0F2854'}}
                            >
                                sign in
                            </button>
                        </form>
                    </div>
                    <div className="w-1/2 flex items-center justify-center bg-white">
                        <div className="text-center">
                            <h1 className="text-5xl font-light tracking-tight" style={{color: '#0F2854'}}>Auto<span className="font-medium">Corrector</span></h1>
                            <div className="w-12 h-0.5 mx-auto mt-4" style={{backgroundColor: '#BDE8F5'}}></div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    )
}
