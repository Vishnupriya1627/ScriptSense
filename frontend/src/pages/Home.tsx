import React, { useState, useRef } from "react";
import axios from "axios";

interface Question {
  id: number;
  keyAnswer: string;
  keyDiagram: File | null;
  keyDiagramName: string;
  studentSheet: File | null;
  studentSheetName: string;
  marks: number;
  textWeight: number; // Added text weight
  diagramWeight: number; // Added diagram weight
  result?: {
    obtained: number;
    textSimilarity: number;
    diagramSimilarity: number;
  } | null;
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
    textWeight: number; // Added to breakdown
    diagramWeight: number; // Added to breakdown
  }[];
}

export default function Home() {
  // ============= SETUP =============
  const [numQuestions, setNumQuestions] = useState<number>(0);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentPage, setCurrentPage] = useState(0);
  const [isSetupComplete, setIsSetupComplete] = useState(false);
  const [evaluationResult, setEvaluationResult] =
    useState<EvaluationResult | null>(null);
  const [loading, setLoading] = useState(false);

  const questionsPerPage = 1;

  // ============= STEP 1: INITIAL SETUP =============
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
        textWeight: 1.0, // Default: 100% text
        diagramWeight: 0.0, // Default: 0% diagram
        result: null,
      }));
      setQuestions(initialQuestions);
      setIsSetupComplete(true);
      setCurrentPage(0);
    }
  };

  // ============= QUESTION HANDLERS =============
  const handleQuestionChange = (
    index: number,
    field: keyof Question,
    value: any,
  ) => {
    setQuestions((prevQuestions) => {
      const updated = [...prevQuestions];
      updated[index] = { ...updated[index], [field]: value };

      // Auto-adjust weights when diagram is added/removed
      if (field === "keyDiagram") {
        if (value) {
          // Diagram added - set balanced weights
          updated[index].textWeight = 0.5;
          updated[index].diagramWeight = 0.5;
        } else {
          // Diagram removed - set text weight to 100%
          updated[index].textWeight = 1.0;
          updated[index].diagramWeight = 0.0;
        }
      }

      // Ensure weights sum to 1.0
      if (field === "textWeight" || field === "diagramWeight") {
        const total = updated[index].textWeight + updated[index].diagramWeight;
        if (total !== 1.0 && total > 0) {
          if (field === "textWeight") {
            updated[index].diagramWeight = Number(
              (1.0 - updated[index].textWeight).toFixed(1),
            );
          } else {
            updated[index].textWeight = Number(
              (1.0 - updated[index].diagramWeight).toFixed(1),
            );
          }
        }
      }

      // Clear result when changes are made
      if (updated[index].result) {
        updated[index].result = null;
        setEvaluationResult(null);
      }
      return updated;
    });
  };

  const handleKeyDiagramUpload = (index: number, file: File | null) => {
    console.log("🖼️ Key diagram upload:", file?.name, file?.type, file?.size);
    handleQuestionChange(index, "keyDiagram", file);
    handleQuestionChange(
      index,
      "keyDiagramName",
      file ? file.name : "Add diagram",
    );
  };

  const handleStudentSheetUpload = (index: number, file: File | null) => {
    console.log("📎 Student sheet upload attempt:", {
      fileName: file?.name,
      fileType: file?.type,
      fileSize: file?.size,
      questionIndex: index,
    });

    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        console.log("✅ File read successfully, size:", e.total);
      };
      reader.onerror = (e) => {
        console.error("❌ File read failed:", e);
      };
      reader.readAsArrayBuffer(file.slice(0, 100));
    }

    setQuestions((prevQuestions) => {
      const updated = [...prevQuestions];
      updated[index] = {
        ...updated[index],
        studentSheet: file,
        studentSheetName: file ? file.name : "Upload answer",
        result: null,
      };
      return updated;
    });

    setEvaluationResult(null);
  };

  // ============= NAVIGATION =============
  const handleNext = () => {
    if ((currentPage + 1) * questionsPerPage < questions.length) {
      setCurrentPage(currentPage + 1);
    }
  };

  const handlePrev = () => {
    if (currentPage > 0) {
      setCurrentPage(currentPage - 1);
    }
  };

  // ============= EVALUATION =============
  const performEvaluation = async (
    question: Question,
    currentIndex: number,
  ) => {
    setLoading(true);

    const formData = new FormData();

    const questionsJson = JSON.stringify([
      {
        keyAnswer: question.keyAnswer,
        marks: question.marks,
        textWeight: question.textWeight,
        diagramWeight: question.diagramWeight,
      },
    ]);
    formData.append("questions", questionsJson);
    console.log("📦 FormData - questions:", questionsJson);

    if (question.keyDiagram) {
      formData.append("diagram_0", question.keyDiagram);
      console.log(
        "📦 FormData - diagram_0:",
        question.keyDiagram.name,
        question.keyDiagram.size,
      );
    }

    if (question.studentSheet) {
      formData.append("answer_sheets", question.studentSheet);
      console.log(
        "📦 FormData - answer_sheets:",
        question.studentSheet.name,
        question.studentSheet.size,
      );
    }

    try {
      console.log("🚀 Sending request to backend...");
      const response = await axios.post(
        " https://mackenzie-unfilterable-kirby.ngrok-free.dev/similarity",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
            // "cf-access-bypass": "true"
            "ngrok-skip-browser-warning": "true",
          },
          onUploadProgress: (progressEvent) => {
            if (progressEvent.total) {
              console.log(
                `📤 Upload progress: ${Math.round((progressEvent.loaded * 100) / progressEvent.total)}%`,
              );
            }
          },
        },
      );

      console.log("✅ Response received:", response.data);

      const result = response.data;

      setQuestions((prevQuestions) => {
        const updated = [...prevQuestions];

        updated[currentIndex].result = {
          obtained: result.score || 0, // ✅ backend gives score
          textSimilarity: 0, // ❌ not available anymore
          diagramSimilarity: 0, // ❌ not available anymore
        };

        return updated;
      });

      // Update overall results
      setTimeout(() => {
        setQuestions((prevQuestions) => {
          calculateOverallResults(prevQuestions);
          return prevQuestions;
        });
      }, 0);
    } catch (error: any) {
      console.error("❌ Evaluation failed:", error);
      if (error.response) {
        console.error("Response data:", error.response.data);
        console.error("Response status:", error.response.status);
      }
      alert(`Evaluation failed: ${error.message}. Check console for details.`);
    } finally {
      setLoading(false);
    }
  };

  const evaluateCurrentQuestion = async () => {
    const currentIndex = currentPage;

    setQuestions((prevQuestions) => {
      const question = prevQuestions[currentIndex];

      console.log("🔍 Evaluating current question:", {
        questionId: question.id,
        hasKeyAnswer: !!question.keyAnswer,
        keyAnswerLength: question.keyAnswer.length,
        hasKeyDiagram: !!question.keyDiagram,
        hasStudentSheet: !!question.studentSheet,
        studentSheetName: question.studentSheet?.name,
        studentSheetSize: question.studentSheet?.size,
        marks: question.marks,
        textWeight: question.textWeight,
        diagramWeight: question.diagramWeight,
      });

      if (!question.keyAnswer.trim()) {
        alert("Please enter an answer key");
        return prevQuestions;
      }
      if (!question.studentSheet) {
        console.error("❌ No student sheet file object found!");
        alert("Please upload student's answer sheet");
        return prevQuestions;
      }
      if (question.marks <= 0) {
        alert("Please enter marks for this question");
        return prevQuestions;
      }

      performEvaluation(question, currentIndex);

      return prevQuestions;
    });
  };

  const evaluateAllQuestions = async () => {
    setLoading(true);

    // Get current questions state
    const currentQuestions = [...questions];

    // Validate all questions
    for (let i = 0; i < currentQuestions.length; i++) {
      const q = currentQuestions[i];
      if (!q.keyAnswer.trim()) {
        alert(`Question ${i + 1} is missing answer key`);
        setCurrentPage(i);
        setLoading(false);
        return;
      }
      if (!q.studentSheet) {
        alert(`Question ${i + 1} is missing student answer sheet`);
        setCurrentPage(i);
        setLoading(false);
        return;
      }
      if (q.marks <= 0) {
        alert(`Question ${i + 1} has invalid marks`);
        setCurrentPage(i);
        setLoading(false);
        return;
      }
    }

    // Evaluate one by one
    for (let i = 0; i < currentQuestions.length; i++) {
      const q = currentQuestions[i];
      console.log(
        `📝 Evaluating question ${i + 1}/${currentQuestions.length}...`,
      );
      await performEvaluation(q, i);
    }

    setLoading(false);
  };

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
      percentage:
        totalPossible > 0
          ? Number(((totalObtained / totalPossible) * 100).toFixed(1))
          : 0,
      breakdown,
    });
  };

  // ============= RENDER =============
  if (!isSetupComplete) {
    return (
      <div
        className="w-screen h-screen flex items-center justify-center"
        style={{ backgroundColor: "#BDE8F5" }}
      >
        <div className="bg-white p-12 rounded-2xl shadow-lg w-96">
          <h1 className="text-3xl font-light mb-8" style={{ color: "#0F2854" }}>
            AutoChecker
          </h1>
          <form onSubmit={handleNumQuestionsSubmit} className="space-y-6">
            <div className="space-y-2">
              <label
                className="text-sm uppercase tracking-wider"
                style={{ color: "#0F2854" }}
              >
                Number of Questions
              </label>
              <input
                type="number"
                min="1"
                value={numQuestions || ""}
                onChange={(e) => setNumQuestions(parseInt(e.target.value) || 0)}
                className="w-full p-3 border border-gray-200 rounded-md outline-none focus:border-2 transition-colors"
                style={{ focusBorderColor: "#0F2854" }}
                placeholder="e.g., 10"
                autoFocus
              />
            </div>
            <button
              type="submit"
              className="w-full py-3 text-sm uppercase tracking-wider transition-colors rounded-md"
              style={{ backgroundColor: "#0F2854", color: "#BDE8F5" }}
            >
              Start
            </button>
          </form>
        </div>
      </div>
    );
  }

  const currentQuestion = questions[currentPage];

  return (
    <div
      className="w-screen h-screen flex flex-col"
      style={{ backgroundColor: "#BDE8F5" }}
    >
      {/* ========== HEADER ========== */}
      <div className="h-16 bg-white shadow-sm flex items-center justify-between px-8">
        <h1 className="text-xl font-light" style={{ color: "#0F2854" }}>
          AutoChecker
        </h1>
        <div className="flex items-center gap-4">
          <span style={{ color: "#0F2854" }}>
            Question {currentPage + 1} of {questions.length}
          </span>
        </div>
      </div>

      {/* ========== MAIN CONTENT ========== */}
      <div className="flex-1 flex p-6 gap-6 overflow-hidden">
        {/* ===== LEFT PANEL - Question Navigation ===== */}
        <div className="w-1/4 bg-white rounded-2xl shadow-sm p-4 flex flex-col">
          <h2
            className="text-sm uppercase tracking-wider mb-4"
            style={{ color: "#0F2854" }}
          >
            Questions
          </h2>
          <div className="flex-1 overflow-y-auto space-y-2">
            {questions.map((q, idx) => {
              const isActive = idx === currentPage;
              const isCompleted = q.result !== null;
              const isIncomplete =
                !q.keyAnswer.trim() || !q.studentSheet || q.marks <= 0;

              return (
                <button
                  key={q.id}
                  onClick={() => setCurrentPage(idx)}
                  className={`w-full p-3 rounded-lg text-left transition-all flex items-center justify-between
                                        ${isActive ? "ring-2" : "hover:bg-gray-50"}`}
                  style={{
                    backgroundColor: isActive ? "#BDE8F5" : "white",
                    ringColor: "#0F2854",
                  }}
                >
                  <span style={{ color: "#0F2854" }}>Q{q.id}</span>
                  <div className="flex gap-2">
                    {isCompleted && (
                      <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded-full">
                        ✓
                      </span>
                    )}
                    {isIncomplete && !isCompleted && (
                      <span className="text-xs px-2 py-1 bg-yellow-100 text-yellow-700 rounded-full">
                        !
                      </span>
                    )}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Marks Summary */}
          {evaluationResult && (
            <div
              className="mt-4 p-4 rounded-lg"
              style={{ backgroundColor: "#0F2854" }}
            >
              <p className="text-sm text-white/80">Total Score</p>
              <p className="text-2xl font-light text-white">
                {evaluationResult.totalObtained}/
                {evaluationResult.totalPossible}
              </p>
              <p className="text-sm text-white/80 mt-1">
                {evaluationResult.percentage}%
              </p>
            </div>
          )}
        </div>

        {/* ===== RIGHT PANEL - Current Question ===== */}
        <div className="flex-1 bg-white rounded-2xl shadow-sm p-6 flex flex-col overflow-hidden">
          {/* Question Header with Weight Sliders */}
          <div className="flex flex-col gap-4 mb-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-light" style={{ color: "#0F2854" }}>
                Question {currentQuestion.id}
              </h2>
              <div className="flex items-center gap-3">
                <label className="text-sm" style={{ color: "#0F2854" }}>
                  Marks:
                </label>
                <input
                  type="number"
                  min="0"
                  value={currentQuestion.marks || ""}
                  onChange={(e) =>
                    handleQuestionChange(
                      currentPage,
                      "marks",
                      parseInt(e.target.value) || 0,
                    )
                  }
                  className="w-20 p-2 border border-gray-200 rounded-md text-center outline-none focus:border-2"
                  style={{ focusBorderColor: "#0F2854" }}
                />
              </div>
            </div>

            {/* Weight Controls */}
            <div className="flex items-center gap-6 p-3 bg-gray-50 rounded-lg">
              <span
                className="text-sm font-medium"
                style={{ color: "#0F2854" }}
              >
                Weights:
              </span>

              {/* Text Weight Slider */}
              <div className="flex items-center gap-2 flex-1">
                <span className="text-xs text-gray-600 min-w-[60px]">
                  Text:
                </span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={currentQuestion.textWeight}
                  onChange={(e) =>
                    handleQuestionChange(
                      currentPage,
                      "textWeight",
                      parseFloat(e.target.value),
                    )
                  }
                  disabled={!currentQuestion.keyAnswer.trim()}
                  className="flex-1 h-2 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, #0F2854 0%, #0F2854 ${currentQuestion.textWeight * 100}%, #E5E7EB ${currentQuestion.textWeight * 100}%, #E5E7EB 100%)`,
                  }}
                />
                <span
                  className="text-sm font-medium min-w-[50px]"
                  style={{ color: "#0F2854" }}
                >
                  {(currentQuestion.textWeight * 100).toFixed(0)}%
                </span>
              </div>

              {/* Diagram Weight Slider */}
              <div className="flex items-center gap-2 flex-1">
                <span className="text-xs text-gray-600 min-w-[60px]">
                  Diagram:
                </span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={currentQuestion.diagramWeight}
                  onChange={(e) =>
                    handleQuestionChange(
                      currentPage,
                      "diagramWeight",
                      parseFloat(e.target.value),
                    )
                  }
                  disabled={!currentQuestion.keyDiagram}
                  className="flex-1 h-2 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, #0F2854 0%, #0F2854 ${currentQuestion.diagramWeight * 100}%, #E5E7EB ${currentQuestion.diagramWeight * 100}%, #E5E7EB 100%)`,
                  }}
                />
                <span
                  className="text-sm font-medium min-w-[50px]"
                  style={{ color: "#0F2854" }}
                >
                  {(currentQuestion.diagramWeight * 100).toFixed(0)}%
                </span>
              </div>

              {!currentQuestion.keyDiagram && (
                <span className="text-xs text-gray-400 italic">
                  Add diagram to enable weight adjustment
                </span>
              )}
            </div>
          </div>

          {/* Side-by-side columns */}
          <div className="flex-1 flex gap-6 min-h-0">
            {/* ANSWER KEY COLUMN */}
            <div className="flex-1 flex flex-col bg-gray-50 rounded-xl p-4">
              <h3
                className="text-sm font-medium mb-3"
                style={{ color: "#0F2854" }}
              >
                ANSWER KEY
              </h3>

              <div className="flex-1 flex flex-col gap-4">
                <div className="flex-1">
                  <label className="text-xs text-gray-500">Text Answer</label>
                  <textarea
                    value={currentQuestion.keyAnswer}
                    onChange={(e) =>
                      handleQuestionChange(
                        currentPage,
                        "keyAnswer",
                        e.target.value,
                      )
                    }
                    className="w-full h-40 p-3 bg-white border border-gray-200 rounded-lg text-sm outline-none focus:border-2 resize-none"
                    style={{ focusBorderColor: "#0F2854" }}
                    placeholder="Enter model answer..."
                  />
                </div>

                <div>
                  <label className="text-xs text-gray-500">
                    Diagram (Optional)
                  </label>
                  <div className="mt-1">
                    <input
                      type="file"
                      id={`key-diagram-${currentQuestion.id}`}
                      accept="image/*"
                      className="hidden"
                      onChange={(e) =>
                        handleKeyDiagramUpload(
                          currentPage,
                          e.target.files?.[0] || null,
                        )
                      }
                    />
                    <label
                      htmlFor={`key-diagram-${currentQuestion.id}`}
                      className="inline-block px-4 py-2 text-sm bg-white border border-gray-200 rounded-lg cursor-pointer hover:border-gray-300 transition-colors"
                    >
                      {currentQuestion.keyDiagramName}
                    </label>
                  </div>
                </div>
              </div>
            </div>

            {/* STUDENT ANSWER COLUMN */}
            <div className="flex-1 flex flex-col bg-gray-50 rounded-xl p-4">
              <h3
                className="text-sm font-medium mb-3"
                style={{ color: "#0F2854" }}
              >
                STUDENT ANSWER
              </h3>

              <div className="flex-1 flex flex-col gap-4">
                <div className="flex-1 flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-6">
                  {!currentQuestion.studentSheet ? (
                    <>
                      <input
                        type="file"
                        id={`student-sheet-${currentQuestion.id}`}
                        accept=".pdf,.txt,.jpg,.png"
                        className="hidden"
                        onChange={(e) =>
                          handleStudentSheetUpload(
                            currentPage,
                            e.target.files?.[0] || null,
                          )
                        }
                      />
                      <label
                        htmlFor={`student-sheet-${currentQuestion.id}`}
                        className="cursor-pointer text-center"
                      >
                        <div
                          className="w-16 h-16 mx-auto mb-3 rounded-full flex items-center justify-center"
                          style={{ backgroundColor: "#BDE8F5" }}
                        >
                          <span
                            className="text-2xl"
                            style={{ color: "#0F2854" }}
                          >
                            +
                          </span>
                        </div>
                        <p
                          className="text-sm font-medium"
                          style={{ color: "#0F2854" }}
                        >
                          Upload Answer Sheet
                        </p>
                        <p className="text-xs text-gray-400 mt-1">
                          PDF, Image, or TXT
                        </p>
                      </label>
                    </>
                  ) : (
                    <div className="w-full text-center">
                      <div className="w-16 h-16 mx-auto mb-3 rounded-full flex items-center justify-center bg-green-100">
                        <span className="text-2xl text-green-600">✓</span>
                      </div>
                      <p
                        className="text-sm font-medium truncate max-w-full"
                        style={{ color: "#0F2854" }}
                      >
                        {currentQuestion.studentSheetName}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {(currentQuestion.studentSheet.size / 1024).toFixed(1)}{" "}
                        KB
                      </p>
                      <button
                        onClick={() =>
                          handleStudentSheetUpload(currentPage, null)
                        }
                        className="text-xs text-red-500 mt-2 hover:underline"
                      >
                        Remove
                      </button>
                    </div>
                  )}
                </div>

                {/* Result Display with Weight Info */}
                {currentQuestion.result && (
                  <div
                    className="p-4 rounded-lg"
                    style={{ backgroundColor: "#BDE8F5" }}
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-sm" style={{ color: "#0F2854" }}>
                        Score:
                      </span>
                      <span
                        className="text-xl font-light"
                        style={{ color: "#0F2854" }}
                      >
                        {currentQuestion.result.obtained.toFixed(1)}/
                        {currentQuestion.marks}
                      </span>
                    </div>
                    <div className="flex justify-between text-xs mt-2 text-gray-600">
                      <span>
                        Text:{" "}
                        {(currentQuestion.result.textSimilarity * 100).toFixed(
                          0,
                        )}
                        %
                        <span className="ml-1 text-gray-400">
                          (weight:{" "}
                          {(currentQuestion.textWeight * 100).toFixed(0)}%)
                        </span>
                      </span>
                      <span>
                        Diagram:{" "}
                        {(
                          currentQuestion.result.diagramSimilarity * 100
                        ).toFixed(0)}
                        %
                        <span className="ml-1 text-gray-400">
                          (weight:{" "}
                          {(currentQuestion.diagramWeight * 100).toFixed(0)}%)
                        </span>
                      </span>
                    </div>
                    <div className="text-xs text-right mt-1 text-gray-500">
                      Weighted avg:{" "}
                      {(
                        currentQuestion.result.textSimilarity *
                          currentQuestion.textWeight +
                        currentQuestion.result.diagramSimilarity *
                          currentQuestion.diagramWeight
                      ).toFixed(2)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ===== ACTION BUTTONS ===== */}
          <div className="mt-6 flex items-center justify-between">
            <div className="flex gap-3">
              <button
                onClick={handlePrev}
                disabled={currentPage === 0}
                className="px-4 py-2 text-sm border border-gray-200 rounded-lg disabled:opacity-30 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
                style={{ color: "#0F2854" }}
              >
                ← Previous
              </button>
              <button
                onClick={handleNext}
                disabled={currentPage === questions.length - 1}
                className="px-4 py-2 text-sm border border-gray-200 rounded-lg disabled:opacity-30 disabled:cursor-not-allowed hover:bg-gray-50 transition-colors"
                style={{ color: "#0F2854" }}
              >
                Next →
              </button>
            </div>

            <div className="flex gap-3">
              <button
                onClick={evaluateCurrentQuestion}
                disabled={loading}
                className="px-6 py-2 text-sm rounded-lg transition-colors disabled:opacity-50"
                style={{ backgroundColor: "#BDE8F5", color: "#0F2854" }}
              >
                {loading ? "Evaluating..." : "Evaluate Current"}
              </button>
              <button
                onClick={evaluateAllQuestions}
                disabled={loading}
                className="px-6 py-2 text-sm rounded-lg transition-colors disabled:opacity-50"
                style={{ backgroundColor: "#0F2854", color: "#BDE8F5" }}
              >
                Evaluate All
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
