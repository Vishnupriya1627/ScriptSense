import { useState } from 'react';

export default function Panel() {
    const [fileName, setFileName] = useState('');

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            const file = files[0];
            setFileName(file.name); // Update state with the selected file's name
        }
    };

    return (
        <div className="w-1/4 h-full bg-dy-red shadow-inner flex flex-col items-center py-5">
            <h1 className="text-white text-4xl font-bold">AutoChecker</h1>
            <form className="flex-col mt-10 bg-white rounded-2xl w-3/4 h-1/3 items-center py-5 px-5">
                <input 
                    type="file"
                    accept=".txt, .doc, .docx, .pdf"
                    className="hidden"
                    id="file"
                    onChange={handleFileChange} // Handle file input change event
                />
                <div className="flex items-center">
                    <label className="font-bold text-xl">Answer Key: </label>
                    <label 
                        htmlFor="file"
                        className="bg-dy-red px-5 py-3 rounded-full ml-5 text-white font-semibold hover:bg-dy-peach hover:text-dy-red transition-colors duration-300 ease-in">Upload</label>
                
                </div>
                {fileName && (
                    <p className="mt-4 text-gray-700 font-semibold text-sm">Selected File: {fileName}</p>
                )}
                {fileName ? <button className="mt-12 ml-16 text-white font-semibold bg-dy-red px-5 py-3 rounded-3xl hover:bg-dy-peach hover:text-dy-red transition-colors duration-300 ease-in">AutoCorrect</button> : null }
            </form>

            
        </div>
    );
}
