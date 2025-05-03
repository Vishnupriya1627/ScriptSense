import React, { useState, useRef } from "react";
import axios from "axios"; // Ensure axios is imported

export default function AnswerSheet() {
    const [answerKey, setAnswerKey] = useState<string>("");
    const [fileNames, setFileNames] = useState<string[]>([]);
    const [files, setFiles] = useState<File[]>([]);
    const [similarityData, setSimilarityData] = useState<Record<number, [number, number]>>({});
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [marks, setMarks] = useState<number>(0);
    const [corrected, setCorrected] = useState(false)
    const [image, setImage] = useState<File | null>(null);
    const [imageName, setImageName] = useState<string | null>("Add Image");
    const [openedSheet, setOpenedSheet] = useState<number|null>(null)

    const handleOpenSheet = (index: number) => {
        setOpenedSheet(index);
    };
    
    const handleCloseSheet = () => {
        setOpenedSheet(null);
    };

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            const file = files[0];
            setImage(file); // Store the actual File object
            setImageName(file.name); // Store the file name
        }
    };

    const handleMarksChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setMarks(Number(e.target.value));
    };

    const handleAnswerTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setAnswerKey(e.target.value);
    }


    const handleButtonClick = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click();
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files ? Array.from(e.target.files) : [];
        const names = files.map(file => file.name); // Get the file names
        const selectedFiles = e.target.files ? Array.from(e.target.files) : [];

        setFileNames(names); // Store the file names in state
        setFiles(selectedFiles); // Store the actual file objects
    };

    const handleCorrecting = () =>{
        setCorrected(true);
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        

        if (!answerKey || files.length === 0) {
            alert("Please upload both an answer key and answer sheets");
            return;
        }

        const formData = new FormData();
        formData.append("answer_key_text", answerKey); // Append the actual answer key file
        formData.append("answer_key_diagram", image as Blob); // Append the actual image file
        files.forEach((file) => {
            formData.append("answer_sheets", file); // Append each answer sheet file
        });

        try {
            const response = await axios.post('http://localhost:8000/similarity', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            console.log("Response:", response.data);
            setSimilarityData(response.data);
            setCorrected(false);
             // Set similarity scores to the state
        } catch (error) {
            console.error("Error uploading files:", error);
        }
    };

    return (
        <div className="w-screen h-screen flex">
            <div className="w-1/4 h-full bg-dy-red shadow-inner flex flex-col items-center py-5">
                <h1 className="text-white text-4xl font-bold">AutoChecker</h1>
                <form className="flex-col mt-10 bg-white rounded-2xl w-5/6 h-3/4 items-center py-5 px-5" onSubmit={handleSubmit}>


{/* Answer Text Input */}
<div className="mt-4">
    <label className="block font-semibold text-lg">Textual Key:</label>
    <textarea
        className="w-full p-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-dy-red"
        placeholder="Enter answer text here..."
        rows={4}
        onChange={handleAnswerTextChange}
    ></textarea>
</div>

{/* Answer Diagram Input */}
<div className="mt-4 flex">
    <label className="block font-semibold text-lg">Diagramatic Key:</label>
    <input
        type="file"
        id="image"
        accept="image/*"
        className="hidden"
        onChange={handleImageChange}  // Define this function to handle image uploads
    />
    <label
        htmlFor="image"
        className="bg-dy-red px-8 py-3 rounded-full ml-5 text-white font-semibold hover:bg-dy-peach hover:text-dy-red transition-colors duration-300 ease-in">
            {imageName}</label>
</div>

                    <div className="flex w-full mt-3">
                        <label className="font-bold text-xl">Total Marks:</label>
                        <input onChange={handleMarksChange} className="bg-dy-peach rounded-3xl w-1/4 ml-6 py-1 text-center font-bold text-dy-red" type="number"/>
                    </div>
                    {answerKey && image ? (
                        <button className="mt-5 ml-16 text-white font-semibold bg-dy-red px-5 py-3 rounded-3xl hover:bg-dy-peach hover:text-dy-red transition-colors duration-300 ease-in" onClick={handleCorrecting}>
                            {corrected ? "AutoCorrecting...": "AutoCorrect"}
                        </button>
                    ) : null}
                </form>
            </div>
            <div className="w-3/4 h-screen flex items-center justify-center bg-white">
                <div className="w-5/6 h-5/6 rounded-3xl bg-dy-peach bg-opacity-50 border-dashed border-dy-red border-2 flex flex-col items-center justify-center">
                    <input
                        type="file"
                        multiple
                        accept=".txt, .pdf"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        className="hidden"
                    />
                    {fileNames.length === 0 && (
                        <button
                            className="bg-dy-red text-white font-bold text-2xl opacity-100 px-5 py-2 rounded-full"
                            onClick={handleButtonClick}
                        >
                            Add Answer Sheets+
                        </button>
                    )}
                    {fileNames.length > 0 && openedSheet == null && (
    <div className="grid grid-cols-4 grid-row-5 gap-6 w-full px-10 mt-5">
        {fileNames.map((name, index) => {
            const textSimilarity = similarityData[index]?.[0] as number;
            const diagramSimilarity = similarityData[index]?.[1] as number;
            const averageSimilarity = (textSimilarity + diagramSimilarity) / 2;
            const totalScore = (averageSimilarity * (marks as number)).toFixed(1);

            return (
                <button
                    key={index}
                    className="bg-white text-dy-red border border-dy-red rounded-xl p-4 flex flex-col items-center justify-center shadow-lg transition-transform duration-300 ease-in-out"
                    onClick={() => handleOpenSheet(index)}
                >
                    <p className="font-bold">{name}</p>

                    {similarityData[index] && (
                        <div className="mt-2 flex flex-col items-center">
                            <p className="text-sm text-gray-600">Text Similarity: {textSimilarity.toFixed(2)}</p>
                            <p className="text-sm text-gray-600">Diagram Similarity: {diagramSimilarity.toFixed(2)}</p>
                            <p className="mt-2 font-semibold text-dy-red">Total Score: {totalScore}/{marks.toString()}</p>
                        </div>
                    )}
                </button>
            );
        })}
    </div>
)}

{/* Expanded View */}
{openedSheet !== null && (
    <div className="w-full h-full flex flex-col items-center justify-center bg-white rounded-3xl p-6 shadow-lg">
        <h2 className="text-2xl font-bold text-dy-red">Answer Sheet Details</h2>
        <p className="text-lg mt-2 font-semibold">{fileNames[openedSheet]}</p>

        {similarityData[openedSheet] && (
            <div className="mt-4 text-center">
                <p className="text-gray-700">Text Similarity: {similarityData[openedSheet][0].toFixed(2)}</p>
                <p className="text-gray-700">Diagram Similarity: {similarityData[openedSheet][1].toFixed(2)}</p>
                <p className="text-xl font-bold text-dy-red mt-3">
                    Total Score: {((similarityData[openedSheet][0] + similarityData[openedSheet][1]) / 2 * marks).toFixed(1)}/{marks}
                </p>
            </div>
        )}

        <button
            onClick={handleCloseSheet}
            className="mt-6 px-6 py-2 bg-dy-red text-white rounded-xl hover:bg-dy-peach hover:text-dy-red transition-colors duration-300 ease-in-out"
        >
            Back
        </button>
    </div>
)}


                </div>
            </div>
        </div>
    );
}
