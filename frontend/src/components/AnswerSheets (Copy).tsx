import { useRef, useState } from 'react';

export default function AnswerSheet() {

    const [fileNames, setFileNames] = useState<string[]>([]);

    const fileInputRef = useRef<HTMLInputElement>(null);

    // Function to trigger the file input when the button is clicked
    const handleButtonClick = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click();
        }
    };
    // Function to handle file selection
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files ? Array.from(e.target.files) : [];  // Convert FileList to Array if not null
        const names = files.map(file => file.name); // Extract file names
        setFileNames(names);  // Store file names in state
    };

    return (
        <div className="w-3/4 h-screen flex items-center justify-center bg-white">
            <div className="w-5/6 h-5/6 rounded-3xl bg-dy-peach bg-opacity-50 border-dashed border-dy-red border-2 flex flex-col items-center justify-center">

                {/* Hidden file input */}
                <input 
                    type="file" 
                    multiple 
                    accept=".txt" 
                    ref={fileInputRef} 
                    onChange={handleFileChange}
                    className="hidden"
                />

                {/* Button disappears once files are uploaded */}
                {fileNames.length === 0 && (
                    <button 
                        className="bg-dy-red text-white font-bold text-2xl opacity-100 px-5 py-2 rounded-full"
                        onClick={handleButtonClick}
                    >
                        Add Answer Sheets+
                    </button>
                )}

                {/* Display boxes with file names */}
                {fileNames.length > 0 && (
                    <div className="grid grid-cols-4 grid-row-5 gap-6 w-full px-10 mt-5">
                        {fileNames.map((name, index) => (
                            <div 
                                key={index} 
                                className="bg-white text-dy-red border border-dy-red rounded-xl p-4 flex items-center justify-center shadow-lg"
                            >
                                {name}
                            </div>
                        ))}
                    </div>
                )}

            </div>
        </div>
    );
}