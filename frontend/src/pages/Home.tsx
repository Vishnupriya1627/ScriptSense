import React, { useState } from "react";
import axios from "axios";

interface Question {
  id: number;
  keyAnswer: string;
  keyDiagram: File | null;
  keyDiagramName: string;
  studentSheet: File | null;
  studentSheetName: string;
  marks: number;
  textWeight: number;
  diagramWeight: number;
  result?: {
    obtained: number;
    textSimilarity: number;
    diagramSimilarity: number;
    keyPoints?: string[];
    remarks?: string;
  } | null;
  analysis?: {
    strengths: string[];
    improvements: string[];
    suggestions: string[];
    blooms: { level: string; required: number; demonstrated: number }[];
  } | null;
  analysisOpen?: boolean;
}

interface EvaluationResult {
  totalObtained: number;
  totalPossible: number;
  percentage: number;
  breakdown: {
    questionId: number;
    obtained: number;
    possible: number;
    textSimilarity: number;
    diagramSimilarity: number;
    textWeight: number;
    diagramWeight: number;
  }[];
}

const BACKEND_URL = "https://mackenzie-unfilterable-kirby.ngrok-free.dev";

const BLOOM_COLORS: Record<string, { bg: string; border: string }> = {
  Remember:   { bg: "#DBEAFE", border: "#3B82F6" },
  Understand: { bg: "#D1FAE5", border: "#10B981" },
  Apply:      { bg: "#FEF9C3", border: "#EAB308" },
  Analyse:    { bg: "#FFE4E6", border: "#F43F5E" },
  Evaluate:   { bg: "#EDE9FE", border: "#8B5CF6" },
  Create:     { bg: "#FFEDD5", border: "#F97316" },
};

export default function Home() {
  const [numQuestions, setNumQuestions] = useState<number>(0);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentPage, setCurrentPage] = useState(0);
  const [isSetupComplete, setIsSetupComplete] = useState(false);
  const [evaluationResult, setEvaluationResult] = useState<EvaluationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [analysing, setAnalysing] = useState(false);

  // ============= SETUP =============
  const handleNumQuestionsSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (numQuestions > 0) {
      const initialQuestions = Array.from({ length: numQuestions }, (_, i) => ({
        id: i + 1,
        keyAnswer: "",
        keyDiagram: null,
        keyDiagramName: "Add diagram",
        studentSheet: null,
        studentSheetName: "Upload answer",
        marks: 0,
        textWeight: 1.0,
        diagramWeight: 0.0,
        result: null,
        analysis: null,
        analysisOpen: false,
      }));
      setQuestions(initialQuestions);
      setIsSetupComplete(true);
      setCurrentPage(0);
    }
  };

  // ============= QUESTION HANDLERS =============
  const handleQuestionChange = (index: number, field: keyof Question, value: any) => {
    setQuestions((prev) => {
      const updated = [...prev];
      updated[index] = { ...updated[index], [field]: value };
      if (field === "keyDiagram") {
        updated[index].textWeight = value ? 0.5 : 1.0;
        updated[index].diagramWeight = value ? 0.5 : 0.0;
      }
      if (field === "textWeight" || field === "diagramWeight") {
        const total = updated[index].textWeight + updated[index].diagramWeight;
        if (total !== 1.0 && total > 0) {
          if (field === "textWeight") {
            updated[index].diagramWeight = Number((1.0 - updated[index].textWeight).toFixed(1));
          } else {
            updated[index].textWeight = Number((1.0 - updated[index].diagramWeight).toFixed(1));
          }
        }
      }
      if (updated[index].result) {
        updated[index].result = null;
        updated[index].analysis = null;
        setEvaluationResult(null);
      }
      return updated;
    });
  };

  const handleKeyDiagramUpload = (index: number, file: File | null) => {
    handleQuestionChange(index, "keyDiagram", file);
    handleQuestionChange(index, "keyDiagramName", file ? file.name : "Add diagram");
  };

  const handleStudentSheetUpload = (index: number, file: File | null) => {
    setQuestions((prev) => {
      const updated = [...prev];
      updated[index] = {
        ...updated[index],
        studentSheet: file,
        studentSheetName: file ? file.name : "Upload answer",
        result: null,
        analysis: null,
      };
      return updated;
    });
    setEvaluationResult(null);
  };

  // ============= NAVIGATION =============
  const handleNext = () => { if (currentPage + 1 < questions.length) setCurrentPage(currentPage + 1); };
  const handlePrev = () => { if (currentPage > 0) setCurrentPage(currentPage - 1); };

  // ============= EVALUATION =============
  const calculateOverallResults = (updatedQuestions: Question[]) => {
    let totalObtained = 0;
    let totalPossible = 0;
    const breakdown = [];
    for (const q of updatedQuestions) {
      if (q.result) {
        totalObtained += q.result.obtained;
        totalPossible += q.marks;
        breakdown.push({
          questionId: q.id,
          obtained: q.result.obtained,
          possible: q.marks,
          textSimilarity: q.result.textSimilarity,
          diagramSimilarity: q.result.diagramSimilarity,
          textWeight: q.textWeight,
          diagramWeight: q.diagramWeight,
        });
      }
    }
    setEvaluationResult({
      totalObtained: Number(totalObtained.toFixed(2)),
      totalPossible,
      percentage: totalPossible > 0 ? Number(((totalObtained / totalPossible) * 100).toFixed(1)) : 0,
      breakdown,
    });
  };

  const performEvaluation = async (question: Question, currentIndex: number) => {
    setLoading(true);
    const formData = new FormData();
    formData.append("questions", JSON.stringify([{
      keyAnswer: question.keyAnswer,
      marks: question.marks,
      textWeight: question.textWeight,
      diagramWeight: question.diagramWeight,
    }]));
    if (question.keyDiagram) formData.append("diagram_0", question.keyDiagram);
    if (question.studentSheet) formData.append("answer_sheets", question.studentSheet);

    try {
      console.log("🚀 Sending request to backend...");
      const response = await axios.post(`${BACKEND_URL}/similarity`, formData, {
        headers: { "Content-Type": "multipart/form-data", "ngrok-skip-browser-warning": "true" },
        timeout: 600000,
        onUploadProgress: (e) => {
          if (e.total) console.log(`📤 Upload: ${Math.round((e.loaded * 100) / e.total)}%`);
        },
      });

      const result = response.data;
      console.log("✅ Response:", result);

      setQuestions((prev) => {
        const updated = [...prev];
        updated[currentIndex] = {
          ...updated[currentIndex],
          result: {
            obtained: result.score || 0,
            textSimilarity: 0,
            diagramSimilarity: 0,
            keyPoints: result.key_points || [],
            remarks: result.per_question?.[0]?.remarks || "",
          },
          analysis: null,
          analysisOpen: false,
        };
        calculateOverallResults(updated);
        return updated;
      });
    } catch (error: any) {
      console.error("❌ Evaluation failed:", error);
      alert(`Evaluation failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const evaluateCurrentQuestion = async () => {
    const question = questions[currentPage];
    if (!question.keyAnswer.trim()) { alert("Please enter an answer key"); return; }
    if (!question.studentSheet) { alert("Please upload student's answer sheet"); return; }
    if (question.marks <= 0) { alert("Please enter marks for this question"); return; }
    await performEvaluation(question, currentPage);
  };

  const evaluateAllQuestions = async () => {
    const currentQuestions = [...questions];
    for (let i = 0; i < currentQuestions.length; i++) {
      const q = currentQuestions[i];
      if (!q.keyAnswer.trim()) { alert(`Q${i+1} missing answer key`); setCurrentPage(i); return; }
      if (!q.studentSheet) { alert(`Q${i+1} missing student sheet`); setCurrentPage(i); return; }
      if (q.marks <= 0) { alert(`Q${i+1} has invalid marks`); setCurrentPage(i); return; }
    }
    for (let i = 0; i < currentQuestions.length; i++) {
      await performEvaluation(currentQuestions[i], i);
    }
  };

  // ============= ANALYSIS =============
  const performAnalysis = async (index: number) => {
    const q = questions[index];
    if (!q.result) return;

    // toggle close if already open and loaded
    if (q.analysisOpen && q.analysis) {
      setQuestions((prev) => {
        const updated = [...prev];
        updated[index] = { ...updated[index], analysisOpen: false };
        return updated;
      });
      return;
    }

    // open panel
    setQuestions((prev) => {
      const updated = [...prev];
      updated[index] = { ...updated[index], analysisOpen: true };
      return updated;
    });

    // if already loaded, just show it
    if (q.analysis) return;

    setAnalysing(true);
    try {
      const response = await axios.post(
        `${BACKEND_URL}/analyse`,
        {
          keyAnswer: q.keyAnswer,
          keyPoints: q.result.keyPoints || [],
          score: q.result.obtained,
          total: q.marks,
          remarks: q.result.remarks || "",
        },
        {
          headers: { "Content-Type": "application/json", "ngrok-skip-browser-warning": "true" },
          timeout: 300000,
        }
      );

      const data = response.data;
      setQuestions((prev) => {
        const updated = [...prev];
        updated[index] = { ...updated[index], analysis: data, analysisOpen: true };
        return updated;
      });
    } catch (error: any) {
      console.error("❌ Analysis failed:", error);
      alert(`Analysis failed: ${error.message}`);
      setQuestions((prev) => {
        const updated = [...prev];
        updated[index] = { ...updated[index], analysisOpen: false };
        return updated;
      });
    } finally {
      setAnalysing(false);
    }
  };

  // ============= BLOOM'S CHART =============
  const BloomsChart = ({ blooms }: { blooms: { level: string; required: number; demonstrated: number }[] }) => {
    return (
      <div className="mt-4">
        <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: "#0F2854" }}>
          Bloom's Taxonomy Coverage
        </p>
        <div className="flex items-center gap-4 mb-2 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-sm inline-block" style={{ backgroundColor: "#0F2854" }}></span>
            Required by answer key
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-sm inline-block" style={{ backgroundColor: "#BDE8F5" }}></span>
            Demonstrated by student
          </span>
        </div>
        <div className="space-y-2">
          {blooms.map((b) => {
            const color = BLOOM_COLORS[b.level] || { bg: "#F3F4F6", border: "#6B7280" };
            return (
              <div key={b.level}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-medium" style={{ color: "#0F2854" }}>{b.level}</span>
                  <span className="text-gray-500">{b.demonstrated}% / {b.required}%</span>
                </div>
                {/* Required bar */}
                <div className="w-full bg-gray-100 rounded-full h-2 mb-0.5">
                  <div
                    className="h-2 rounded-full transition-all duration-500"
                    style={{ width: `${b.required}%`, backgroundColor: "#0F2854" }}
                  />
                </div>
                {/* Demonstrated bar */}
                <div className="w-full bg-gray-100 rounded-full h-2">
                  <div
                    className="h-2 rounded-full transition-all duration-500"
                    style={{ width: `${b.demonstrated}%`, backgroundColor: color.border }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // ============= RENDER =============
  if (!isSetupComplete) {
    return (
      <div className="w-screen h-screen flex items-center justify-center" style={{ backgroundColor: "#BDE8F5" }}>
        <div className="bg-white p-12 rounded-2xl shadow-lg w-96">
          <h1 className="text-3xl font-light mb-8" style={{ color: "#0F2854" }}>AutoChecker</h1>
          <form onSubmit={handleNumQuestionsSubmit} className="space-y-6">
            <div className="space-y-2">
              <label className="text-sm uppercase tracking-wider" style={{ color: "#0F2854" }}>
                Number of Questions
              </label>
              <input
                type="number" min="1" value={numQuestions || ""}
                onChange={(e) => setNumQuestions(parseInt(e.target.value) || 0)}
                className="w-full p-3 border border-gray-200 rounded-md outline-none focus:border-2 transition-colors"
                placeholder="e.g., 10" autoFocus
              />
            </div>
            <button type="submit" className="w-full py-3 text-sm uppercase tracking-wider transition-colors rounded-md"
              style={{ backgroundColor: "#0F2854", color: "#BDE8F5" }}>
              Start
            </button>
          </form>
        </div>
      </div>
    );
  }

  const currentQuestion = questions[currentPage];

  return (
    <div className="w-screen h-screen flex flex-col" style={{ backgroundColor: "#BDE8F5" }}>
      {/* HEADER */}
      <div className="h-16 bg-white shadow-sm flex items-center justify-between px-8">
        <h1 className="text-xl font-light" style={{ color: "#0F2854" }}>AutoChecker</h1>
        <span style={{ color: "#0F2854" }}>Question {currentPage + 1} of {questions.length}</span>
      </div>

      {/* MAIN */}
      <div className="flex-1 flex p-6 gap-6 overflow-hidden">

        {/* LEFT PANEL */}
        <div className="w-1/4 bg-white rounded-2xl shadow-sm p-4 flex flex-col">
          <h2 className="text-sm uppercase tracking-wider mb-4" style={{ color: "#0F2854" }}>Questions</h2>
          <div className="flex-1 overflow-y-auto space-y-2">
            {questions.map((q, idx) => {
              const isActive = idx === currentPage;
              const isCompleted = q.result != null;
              const isIncomplete = !q.keyAnswer.trim() || !q.studentSheet || q.marks <= 0;
              return (
                <button key={q.id} onClick={() => setCurrentPage(idx)}
                  className={`w-full p-3 rounded-lg text-left transition-all flex items-center justify-between ${isActive ? "ring-2" : "hover:bg-gray-50"}`}
                  style={{ backgroundColor: isActive ? "#BDE8F5" : "white" }}>
                  <span style={{ color: "#0F2854" }}>Q{q.id}</span>
                  <div className="flex gap-2">
                    {isCompleted && <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded-full">✓</span>}
                    {isIncomplete && !isCompleted && <span className="text-xs px-2 py-1 bg-yellow-100 text-yellow-700 rounded-full">!</span>}
                  </div>
                </button>
              );
            })}
          </div>
          {evaluationResult && (
            <div className="mt-4 p-4 rounded-lg" style={{ backgroundColor: "#0F2854" }}>
              <p className="text-sm text-white/80">Total Score</p>
              <p className="text-2xl font-light text-white">{evaluationResult.totalObtained}/{evaluationResult.totalPossible}</p>
              <p className="text-sm text-white/80 mt-1">{evaluationResult.percentage}%</p>
            </div>
          )}
        </div>

        {/* RIGHT PANEL */}
        <div className="flex-1 bg-white rounded-2xl shadow-sm p-6 flex flex-col overflow-y-auto">
          {/* Header */}
          <div className="flex flex-col gap-4 mb-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-light" style={{ color: "#0F2854" }}>Question {currentQuestion.id}</h2>
              <div className="flex items-center gap-3">
                <label className="text-sm" style={{ color: "#0F2854" }}>Marks:</label>
                <input type="number" min="0" value={currentQuestion.marks || ""}
                  onChange={(e) => handleQuestionChange(currentPage, "marks", parseInt(e.target.value) || 0)}
                  className="w-20 p-2 border border-gray-200 rounded-md text-center outline-none focus:border-2" />
              </div>
            </div>

            {/* Weight Controls */}
            <div className="flex items-center gap-6 p-3 bg-gray-50 rounded-lg">
              <span className="text-sm font-medium" style={{ color: "#0F2854" }}>Weights:</span>
              <div className="flex items-center gap-2 flex-1">
                <span className="text-xs text-gray-600 min-w-[60px]">Text:</span>
                <input type="range" min="0" max="1" step="0.1" value={currentQuestion.textWeight}
                  onChange={(e) => handleQuestionChange(currentPage, "textWeight", parseFloat(e.target.value))}
                  disabled={!currentQuestion.keyAnswer.trim()} className="flex-1 h-2 rounded-lg appearance-none cursor-pointer"
                  style={{ background: `linear-gradient(to right, #0F2854 0%, #0F2854 ${currentQuestion.textWeight * 100}%, #E5E7EB ${currentQuestion.textWeight * 100}%, #E5E7EB 100%)` }} />
                <span className="text-sm font-medium min-w-[50px]" style={{ color: "#0F2854" }}>
                  {(currentQuestion.textWeight * 100).toFixed(0)}%
                </span>
              </div>
              <div className="flex items-center gap-2 flex-1">
                <span className="text-xs text-gray-600 min-w-[60px]">Diagram:</span>
                <input type="range" min="0" max="1" step="0.1" value={currentQuestion.diagramWeight}
                  onChange={(e) => handleQuestionChange(currentPage, "diagramWeight", parseFloat(e.target.value))}
                  disabled={!currentQuestion.keyDiagram} className="flex-1 h-2 rounded-lg appearance-none cursor-pointer"
                  style={{ background: `linear-gradient(to right, #0F2854 0%, #0F2854 ${currentQuestion.diagramWeight * 100}%, #E5E7EB ${currentQuestion.diagramWeight * 100}%, #E5E7EB 100%)` }} />
                <span className="text-sm font-medium min-w-[50px]" style={{ color: "#0F2854" }}>
                  {(currentQuestion.diagramWeight * 100).toFixed(0)}%
                </span>
              </div>
              {!currentQuestion.keyDiagram && <span className="text-xs text-gray-400 italic">Add diagram to enable weight adjustment</span>}
            </div>
          </div>

          {/* Side-by-side columns */}
          <div className="flex gap-6 min-h-[320px]">
            {/* ANSWER KEY */}
            <div className="flex-1 flex flex-col bg-gray-50 rounded-xl p-4">
              <h3 className="text-sm font-medium mb-3" style={{ color: "#0F2854" }}>ANSWER KEY</h3>
              <div className="flex-1 flex flex-col gap-4">
                <div className="flex-1">
                  <label className="text-xs text-gray-500">Text Answer</label>
                  <textarea value={currentQuestion.keyAnswer}
                    onChange={(e) => handleQuestionChange(currentPage, "keyAnswer", e.target.value)}
                    className="w-full h-40 p-3 bg-white border border-gray-200 rounded-lg text-sm outline-none focus:border-2 resize-none"
                    placeholder="Enter model answer..." />
                </div>
                <div>
                  <label className="text-xs text-gray-500">Diagram (Optional)</label>
                  <div className="mt-1">
                    <input type="file" id={`key-diagram-${currentQuestion.id}`} accept="image/*" className="hidden"
                      onChange={(e) => handleKeyDiagramUpload(currentPage, e.target.files?.[0] || null)} />
                    <label htmlFor={`key-diagram-${currentQuestion.id}`}
                      className="inline-block px-4 py-2 text-sm bg-white border border-gray-200 rounded-lg cursor-pointer hover:border-gray-300 transition-colors">
                      {currentQuestion.keyDiagramName}
                    </label>
                  </div>
                </div>
              </div>
            </div>

            {/* STUDENT ANSWER */}
            <div className="flex-1 flex flex-col bg-gray-50 rounded-xl p-4">
              <h3 className="text-sm font-medium mb-3" style={{ color: "#0F2854" }}>STUDENT ANSWER</h3>
              <div className="flex-1 flex flex-col gap-4">
                <div className="flex-1 flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-6">
                  {!currentQuestion.studentSheet ? (
                    <>
                      <input type="file" id={`student-sheet-${currentQuestion.id}`} accept=".pdf,.txt,.jpg,.png"
                        className="hidden" onChange={(e) => handleStudentSheetUpload(currentPage, e.target.files?.[0] || null)} />
                      <label htmlFor={`student-sheet-${currentQuestion.id}`} className="cursor-pointer text-center">
                        <div className="w-16 h-16 mx-auto mb-3 rounded-full flex items-center justify-center" style={{ backgroundColor: "#BDE8F5" }}>
                          <span className="text-2xl" style={{ color: "#0F2854" }}>+</span>
                        </div>
                        <p className="text-sm font-medium" style={{ color: "#0F2854" }}>Upload Answer Sheet</p>
                        <p className="text-xs text-gray-400 mt-1">PDF, Image, or TXT</p>
                      </label>
                    </>
                  ) : (
                    <div className="w-full text-center">
                      <div className="w-16 h-16 mx-auto mb-3 rounded-full flex items-center justify-center bg-green-100">
                        <span className="text-2xl text-green-600">✓</span>
                      </div>
                      <p className="text-sm font-medium truncate max-w-full" style={{ color: "#0F2854" }}>
                        {currentQuestion.studentSheetName}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {(currentQuestion.studentSheet.size / 1024).toFixed(1)} KB
                      </p>
                      <button onClick={() => handleStudentSheetUpload(currentPage, null)}
                        className="text-xs text-red-500 mt-2 hover:underline">Remove</button>
                    </div>
                  )}
                </div>

                {/* Score card */}
                {currentQuestion.result && (
                  <div className="p-4 rounded-lg" style={{ backgroundColor: "#BDE8F5" }}>
                    <div className="flex justify-between items-center">
                      <span className="text-sm" style={{ color: "#0F2854" }}>Score:</span>
                      <span className="text-xl font-light" style={{ color: "#0F2854" }}>
                        {currentQuestion.result.obtained.toFixed(1)}/{currentQuestion.marks}
                      </span>
                    </div>
                    {currentQuestion.result.remarks && (
                      <p className="text-xs text-gray-600 mt-2 italic">{currentQuestion.result.remarks}</p>
                    )}
                    {/* Analyse button */}
                    <button
                      onClick={() => performAnalysis(currentPage)}
                      disabled={analysing}
                      className="mt-3 w-full py-2 text-xs font-medium rounded-lg transition-colors disabled:opacity-50"
                      style={{ backgroundColor: "#0F2854", color: "#BDE8F5" }}
                    >
                      {analysing && currentQuestion.analysisOpen
                        ? "⏳ Analysing..."
                        : currentQuestion.analysisOpen && currentQuestion.analysis
                        ? "▲ Hide Analysis"
                        : "🔍 Analyse Answer"}
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ===== ANALYSIS PANEL ===== */}
          {currentQuestion.analysisOpen && (
            <div className="mt-6 border border-gray-200 rounded-2xl overflow-hidden">
              <div className="p-4" style={{ backgroundColor: "#0F2854" }}>
                <h3 className="text-sm font-semibold text-white uppercase tracking-wider">
                  📊 Answer Analysis — Question {currentQuestion.id}
                </h3>
              </div>

              {!currentQuestion.analysis ? (
                <div className="p-8 text-center text-gray-400 text-sm">
                  ⏳ Generating analysis... this may take 1-2 minutes
                </div>
              ) : (
                <div className="p-5 grid grid-cols-2 gap-6">
                  {/* LEFT: Feedback cards */}
                  <div className="space-y-4">
                    {/* Strengths */}
                    <div className="bg-green-50 border border-green-200 rounded-xl p-4">
                      <p className="text-xs font-semibold text-green-700 uppercase tracking-wider mb-2">✅ Strengths</p>
                      <ul className="space-y-1">
                        {currentQuestion.analysis.strengths.map((s, i) => (
                          <li key={i} className="text-sm text-green-800 flex gap-2">
                            <span>•</span><span>{s}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Areas to improve */}
                    <div className="bg-red-50 border border-red-200 rounded-xl p-4">
                      <p className="text-xs font-semibold text-red-700 uppercase tracking-wider mb-2">⚠️ Areas to Improve</p>
                      <ul className="space-y-1">
                        {currentQuestion.analysis.improvements.map((s, i) => (
                          <li key={i} className="text-sm text-red-800 flex gap-2">
                            <span>•</span><span>{s}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Suggestions */}
                    <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
                      <p className="text-xs font-semibold text-blue-700 uppercase tracking-wider mb-2">💡 Suggestions</p>
                      <ul className="space-y-1">
                        {currentQuestion.analysis.suggestions.map((s, i) => (
                          <li key={i} className="text-sm text-blue-800 flex gap-2">
                            <span>•</span><span>{s}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  {/* RIGHT: Bloom's chart */}
                  <div className="bg-gray-50 rounded-xl p-4">
                    <BloomsChart blooms={currentQuestion.analysis.blooms} />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ACTION BUTTONS */}
          <div className="mt-6 flex items-center justify-between">
            <div className="flex gap-3">
              <button onClick={handlePrev} disabled={currentPage === 0}
                className="px-4 py-2 text-sm border border-gray-200 rounded-lg disabled:opacity-30 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
                style={{ color: "#0F2854" }}>← Previous</button>
              <button onClick={handleNext} disabled={currentPage === questions.length - 1}
                className="px-4 py-2 text-sm border border-gray-200 rounded-lg disabled:opacity-30 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
                style={{ color: "#0F2854" }}>Next →</button>
            </div>
            <div className="flex gap-3">
              <button onClick={evaluateCurrentQuestion} disabled={loading}
                className="px-6 py-2 text-sm rounded-lg transition-colors disabled:opacity-50"
                style={{ backgroundColor: "#BDE8F5", color: "#0F2854" }}>
                {loading ? "Evaluating..." : "Evaluate Current"}
              </button>
              <button onClick={evaluateAllQuestions} disabled={loading}
                className="px-6 py-2 text-sm rounded-lg transition-colors disabled:opacity-50"
                style={{ backgroundColor: "#0F2854", color: "#BDE8F5" }}>
                Evaluate All
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}