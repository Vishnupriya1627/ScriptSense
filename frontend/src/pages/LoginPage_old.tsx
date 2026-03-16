import { useState } from 'react'
import { useNavigate } from 'react-router-dom';

export default function LoginPage(){
    const navigate = useNavigate();

    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const [error, setError] = useState('')

    const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if(username === 'admin@rait.ac.in' && password === 'admin'){
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
            <div className="w-screen h-screen flex items-center justify-center bg-white">
                <div className="flex bg-white h-3/4 w-1/2 rounded-2xl">
                    <div className="h-full w-1/2 bg-dy-red rounded-l-2xl">
                        <form onSubmit={handleLogin} className="flex flex-col justify-center items-center h-full text-black">
                            <h1 className="text-4xl font-bold mb-6 text-white">Login</h1>
                            <input type="email" placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} className="w-3/4 h-10 mb-3 border border-gray-400 rounded-md px-2"/>
                            <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} className="w-3/4 h-10 mb-3 border border-gray-400 rounded-md px-2"/>
                            <button type="submit" className="w-3/4 h-10 bg-dy-peach font-semibold text-xl text-dy-red rounded-md hover:text-white transition-colors ease-in-out duration-300">Login</button>
                            {error?<p className='pt-2 text-white'>Invalid credentials</p>:null}
                        </form>
                        
                    </div>
                    <div className="h-full w-1/2 rounded-r-2xl bg-login flex items-center justify-center">
                        <h1 className='text-dy-red font-bold text-3xl'> AutoCorrector </h1>
                    </div>

                </div>
            </div>
        </>
    )
}