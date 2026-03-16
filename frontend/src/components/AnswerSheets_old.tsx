import { useRef, useState } from 'react';
import axios from 'axios';

interface Question {
    id: number;
    studentAnswer: string;
    keyAnswer: string;
    keyDiagram: File | null;
    marks: number;
    diagramName: string;
}

interface Result {
    totalObtained: number;
    totalPossible: number;
    percentage: number;
    breakdown: {
        questionId: number;
        obtained: number;
        possible: number;
        textSimilarity: number;
        diagramSimilarity: number;
    }[];
}

export default function AnswerSheet() {
    // Step management
    const [step, setStep] = useState<1 | 2 | 3>(1);
    const [numQuestions, setNumQuestions] = useState<number>(0);
    
    // Questions state
    const [questions, setQuestions] = useState<Question[]>([]);
    
    // Answer sheets
    const [answerSheetFiles, setAnswerSheetFiles] = useState<File[]>([]);
    const [answerSheetNames, setAnswerSheetNames] = useState<string[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);
    
    // Results
    const [results, setResults] = useState<Result | null>(null);
    const [loading, setLoading] = useState(false);
    
    // Pagination for questions
    const [currentPage, setCurrentPage] = useState(0);
    const questionsPerPage = 3;

    // ============= STEP 1: Set Number of Questions =============
    const handleNumQuestionsSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (numQuestions > 0) {
            // Initialize empty questions array
            const initialQuestions = Array.from({ length: numQuestions }, (_, i) => ({
                id: i + 1,
                studentAnswer: '',
                keyAnswer: '',
                keyDiagram: null,
                marks: 0,
                diagramName: 'Add diagram'
            }));
            setQuestions(initialQuestions);
            setStep(2);
            setCurrentPage(0);
        }
    };

    // ============= STEP 2: Question Input Forms =============
    const handleQuestionChange = (index: number, field: keyof Question, value: any) => {
        const updated = [...questions];
        updated[index] = { ...updated[index], [field]: value };
        setQuestions(updated);
    };

    const handleDiagramUpload = (index: number, file: File | null) => {
        const updated = [...questions];
        updated[index] = {
            ...updated[index],
            keyDiagram: file,
            diagramName: file ? file.name : 'Add diagram'
        };
        setQuestions(updated);
    };

    const handleNextPage = () => {
        if ((currentPage + 1) * questionsPerPage < questions.length) {
            setCurrentPage(currentPage + 1);
        }
    };

    const handlePrevPage = () => {
        if (currentPage > 0) {
            setCurrentPage(currentPage - 1);
        }
    };

    const validateQuestions = () => {
        return questions.every(q => 
            q.keyAnswer.trim() !== '' && 
            q.marks > 0
        );
    };

    const proceedToUpload = () => {
        if (validateQuestions()) {
            setStep(3);
        } else {
            alert('Please fill all answer keys and marks for each question');
        }
    };

    // ============= STEP 3: Upload Answer Sheets & Evaluate =============
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files ? Array.from(e.target.files) : [];
        const names = files.map(file => file.name);
        setAnswerSheetFiles(files);
        setAnswerSheetNames(names);
    };

    const handleButtonClick = () => {
        if (fileInputRef.current) {
            fileInputRef.current.click();
        }
    };

    const handleEvaluate = async () => {
        if (answerSheetFiles.length === 0) {
            alert('Please upload answer sheets');
            return;
        }

        setLoading(true);
        const formData = new FormData();

        // Add questions as JSON
        formData.append('questions', JSON.stringify(questions.map(q => ({
            keyAnswer: q.keyAnswer,
            marks: q.marks
        }))));

        // Add diagram files with indexing
        questions.forEach((q, idx) => {
            if (q.keyDiagram) {
                formData.append(`diagram_${idx}`, q.keyDiagram);
            }
        });

        // Add answer sheets
        answerSheetFiles.forEach(file => {
            formData.append('answer_sheets', file);
        });

        try {
            const response = await axios.post('http://localhost:8080/similarity', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setResults(response.data);
        } catch (error) {
            console.error('Evaluation failed:', error);
            alert('Evaluation failed. Check console for details.');
        } finally {
            setLoading(false);
        }
    };

    // ============= RENDER =============
    return (
        <div className="w-screen h-screen flex" style={{backgroundColor: '#BDE8F5'}}>
            {/* Left Panel - Dynamic based on step */}
            <div className="w-1/3 h-full flex flex-col items-center py-8 px-6" style={{backgroundColor: '#0F2854'}}>
                <h1 className="text-3xl font-light tracking-wide mb-8" style={{color: '#BDE8F5'}}>
                    AutoChecker
                </h1>

                {/* STEP 1: Number of Questions */}
                {step === 1 && (
                    <form onSubmit={handleNumQuestionsSubmit} className="w-full mt-20">
                        <label className="text-sm uppercase tracking-wider" style={{color: '#BDE8F5'}}>
                            Number of Questions
                        </label>
                        <input
                            type="number"
                            min="1"
                            value={numQuestions || ''}
                            onChange={(e) => setNumQuestions(parseInt(e.target.value) || 0)}
                            className="w-full mt-2 p-3 bg-white/10 border border-white/20 rounded-md text-white placeholder:text-white/50 outline-none focus:border-white"
                            placeholder="e.g., 10"
                        />
                        <button
                            type="submit"
                            className="w-full mt-6 py-3 text-sm uppercase tracking-wider transition-colors"
                            style={{backgroundColor: '#BDE8F5', color: '#0F2854'}}
                        >
                            Continue
                        </button>
                    </form>
                )}

                {/* STEP 2: Question Input Forms with Pagination */}
                {step === 2 && questions.length > 0 && (
                    <div className="w-full h-full flex flex-col">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="text-lg" style={{color: '#BDE8F5'}}>
                                Questions {currentPage * questionsPerPage + 1} - {Math.min((currentPage + 1) * questionsPerPage, questions.length)} of {questions.length}
                            </h2>
                            <div className="flex gap-2">
                                <button
                                    onClick={handlePrevPage}
                                    disabled={currentPage === 0}
                                    className="px-3 py-1 text-sm disabled:opacity-30"
                                    style={{color: '#BDE8F5'}}
                                >
                                    ← Prev
                                </button>
                                <button
                                    onClick={handleNextPage}
                                    disabled={(currentPage + 1) * questionsPerPage >= questions.length}
                                    className="px-3 py-1 text-sm disabled:opacity-30"
                                    style={{color: '#BDE8F5'}}
                                >
                                    Next →
                                </button>
                            </div>
                        </div>

                        <div className="flex-1 overflow-y-auto space-y-6 pr-2">
                            {questions
                                .slice(currentPage * questionsPerPage, (currentPage + 1) * questionsPerPage)
                                .map((question, idx) => {
                                    const actualIndex = currentPage * questionsPerPage + idx;
                                    return (
                                        <div key={question.id} className="space-y-3 p-4 bg-white/5 rounded-lg">
                                            <h3 className="font-medium" style={{color: '#BDE8F5'}}>
                                                Question {question.id}
                                            </h3>
                                            
                                            {/* Key Answer Text */}
                                            <div className="space-y-1">
                                                <label className="text-xs" style={{color: '#BDE8F5'}}>Answer Key</label>
                                                <textarea
                                                    value={question.keyAnswer}
                                                    onChange={(e) => handleQuestionChange(actualIndex, 'keyAnswer', e.target.value)}
                                                    className="w-full p-2 bg-white/10 border border-white/20 rounded-md text-white placeholder:text-white/50 text-sm outline-none focus:border-white"
                                                    rows={2}
                                                    placeholder="Enter model answer..."
                                                />
                                            </div>
                                            
                                            {/* Key Diagram */}
                                            <div className="space-y-1">
                                                <label className="text-xs" style={{color: '#BDE8F5'}}>Diagram (Optional)</label>
                                                <div className="flex items-center">
                                                    <input
                                                        type="file"
                                                        id={`diagram-${question.id}`}
                                                        accept="image/*"
                                                        className="hidden"
                                                        onChange={(e) => handleDiagramUpload(actualIndex, e.target.files?.[0] || null)}
                                                    />
                                                    <label
                                                        htmlFor={`diagram-${question.id}`}
                                                        className="px-3 py-1 text-xs cursor-pointer transition-colors"
                                                        style={{backgroundColor: '#BDE8F5', color: '#0F2854'}}
                                                    >
                                                        {question.diagramName}
                                                    </label>
                                                </div>
                                            </div>
                                            
                                            {/* Marks */}
                                            <div className="space-y-1">
                                                <label className="text-xs" style={{color: '#BDE8F5'}}>Marks</label>
                                                <input
                                                    type="number"
                                                    min="0"
                                                    value={question.marks || ''}
                                                    onChange={(e) => handleQuestionChange(actualIndex, 'marks', parseInt(e.target.value) || 0)}
                                                    className="w-20 p-1 bg-white/10 border border-white/20 rounded-md text-white text-center outline-none focus:border-white"
                                                />
                                            </div>
                                        </div>
                                    );
                                })}
                        </div>

                        {/* Submit Button */}
                        <button
                            onClick={proceedToUpload}
                            className="w-full mt-6 py-3 text-sm uppercase tracking-wider transition-colors"
                            style={{backgroundColor: '#BDE8F5', color: '#0F2854'}}
                        >
                            Continue to Upload Sheets
                        </button>
                    </div>
                )}

                {/* STEP 3: Upload Answer Sheets */}
                {step === 3 && (
                    <div className="w-full flex flex-col items-center">
                        <input
                            type="file"
                            multiple
                            accept=".txt,.pdf,.doc,.docx"
                            ref={fileInputRef}
                            onChange={handleFileChange}
                            className="hidden"
                        />
                        
                        {answerSheetNames.length === 0 ? (
                            <>
                                <p className="mb-4 text-sm" style={{color: '#BDE8F5'}}>
                                    Upload student answer sheets
                                </p>
                                <button
                                    onClick={handleButtonClick}
                                    className="px-6 py-3 text-sm uppercase tracking-wider transition-colors"
                                    style={{backgroundColor: '#BDE8F5', color: '#0F2854'}}
                                >
                                    Select Files
                                </button>
                            </>
                        ) : (
                            <div className="w-full space-y-4">
                                <div className="flex justify-between items-center">
                                    <p style={{color: '#BDE8F5'}}>
                                        {answerSheetNames.length} sheet(s) selected
                                    </p>
                                    <button
                                        onClick={handleButtonClick}
                                        className="text-xs underline"
                                        style={{color: '#BDE8F5'}}
                                    >
                                        Change
                                    </button>
                                </div>
                                
                                <button
                                    onClick={handleEvaluate}
                                    disabled={loading}
                                    className="w-full py-3 text-sm uppercase tracking-wider transition-colors disabled:opacity-50"
                                    style={{backgroundColor: '#BDE8F5', color: '#0F2854'}}
                                >
                                    {loading ? 'Evaluating...' : 'Evaluate All'}
                                </button>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Right Panel - Results Display */}
            <div className="w-2/3 h-full p-8 overflow-y-auto">
                <div className="w-full h-full bg-white rounded-2xl p-8 shadow-lg">
                    {!results ? (
                        <div className="flex flex-col items-center justify-center h-full text-gray-400">
                            {step === 3 && answerSheetNames.length > 0 ? (
                                <p>Ready to evaluate. Click "Evaluate All"</p>
                            ) : (
                                <p>Configure questions and upload answer sheets to begin</p>
                            )}
                        </div>
                    ) : (
                        <div className="space-y-8">
                            <h2 className="text-2xl font-light" style={{color: '#0F2854'}}>Evaluation Results</h2>
                            
                            {/* Summary Card */}
                            <div className="p-6 rounded-lg" style={{backgroundColor: '#BDE8F5'}}>
                                <div className="flex justify-between items-center">
                                    <div>
                                        <p className="text-sm uppercase tracking-wider" style={{color: '#0F2854'}}>Total Score</p>
                                        <p className="text-4xl font-light mt-2" style={{color: '#0F2854'}}>
                                            {results.totalObtained.toFixed(1)}/{results.totalPossible}
                                        </p>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-sm uppercase tracking-wider" style={{color: '#0F2854'}}>Percentage</p>
                                        <p className="text-4xl font-light mt-2" style={{color: '#0F2854'}}>
                                            {results.percentage.toFixed(1)}%
                                        </p>
                                    </div>
                                </div>
                            </div>

                            {/* Per-Question Breakdown */}
                            <div className="space-y-4">
                                <h3 className="font-medium" style={{color: '#0F2854'}}>Question Breakdown</h3>
                                {results.breakdown.map((b, i) => (
                                    <div key={i} className="flex items-center justify-between p-4 border border-gray-100 rounded-lg">
                                        <div>
                                            <p className="font-medium" style={{color: '#0F2854'}}>Q{b.questionId}</p>
                                            <div className="flex gap-4 mt-1 text-sm text-gray-500">
                                                <span>Text: {(b.textSimilarity * 100).toFixed(0)}%</span>
                                                <span>Diagram: {(b.diagramSimilarity * 100).toFixed(0)}%</span>
                                            </div>
                                        </div>
                                        <p className="text-xl font-light" style={{color: '#0F2854'}}>
                                            {b.obtained.toFixed(1)}/{b.possible}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
