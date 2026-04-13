import React, { useState } from "react";
import axios from "axios";

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

interface Question {
  id: number;
  keyAnswer: string;
  keyDiagram: File | null;
  keyDiagramName: string;
  marks: number;
  textWeight: number;
  diagramWeight: number;
}

interface PerQuestion {
  score: number;
  marks: number;
  remarks: string;
  emb_score?: number;
  llm_score?: number;
  band?: string;
  band_score?: number;
}

interface StudentResult {
  student_name: string;
  student_index: number;
  score: number;
  total: number;
  key_points: string[];
  student_answer: string[];
  per_question: PerQuestion[];
  analysis?: Analysis | null;
  analysisOpen?: boolean;
  cluster_id?: number;
  band?: string;
  band_score?: number;
}

interface Analysis {
  strengths: string[];
  improvements: string[];
  suggestions: string[];
  blooms: { level: string; required: number; demonstrated: number }[];
  band?: string;
  band_score?: number;
}

interface BatchResult {
  batch_size: number;
  average_score: number;
  highest_score: number;
  lowest_score: number;
  total_marks: number;
  students: StudentResult[];
  // flat keys for single-student backwards compat
  score?: number;
  total?: number;
  key_points?: string[];
  per_question?: PerQuestion[];
}

// ─────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────

const BACKEND_URL = "https://mackenzie-unfilterable-kirby.ngrok-free.dev";

const BLOOM_COLORS: Record<string, { bg: string; border: string }> = {
  Remember:   { bg: "#DBEAFE", border: "#3B82F6" },
  Understand: { bg: "#D1FAE5", border: "#10B981" },
  Apply:      { bg: "#FEF9C3", border: "#EAB308" },
  Analyse:    { bg: "#FFE4E6", border: "#F43F5E" },
  Evaluate:   { bg: "#EDE9FE", border: "#8B5CF6" },
  Create:     { bg: "#FFEDD5", border: "#F97316" },
};

const BAND_COLORS: Record<string, { bg: string; text: string }> = {
  "Excellent":      { bg: "#D1FAE5", text: "#065F46" },
  "Good":           { bg: "#DBEAFE", text: "#1E40AF" },
  "Average":        { bg: "#FEF9C3", text: "#92400E" },
  "Below Average":  { bg: "#FFE4E6", text: "#9F1239" },
  "Poor":           { bg: "#F3F4F6", text: "#374151" },
  "N/A":            { bg: "#F3F4F6", text: "#374151" },
};

// ─────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────

const BloomsChart = ({ blooms }: { blooms: { level: string; required: number; demonstrated: number }[] }) => (
  <div className="mt-2">
    <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: "#0F2854" }}>
      Bloom's Taxonomy Coverage
    </p>
    <div className="flex items-center gap-4 mb-2 text-xs text-gray-500">
      <span className="flex items-center gap-1">
        <span className="w-3 h-3 rounded-sm inline-block" style={{ backgroundColor: "#0F2854" }} />
        Required
      </span>
      <span className="flex items-center gap-1">
        <span className="w-3 h-3 rounded-sm inline-block" style={{ backgroundColor: "#BDE8F5" }} />
        Demonstrated
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
            <div className="w-full bg-gray-100 rounded-full h-2 mb-0.5">
              <div className="h-2 rounded-full transition-all duration-500"
                style={{ width: `${b.required}%`, backgroundColor: "#0F2854" }} />
            </div>
            <div className="w-full bg-gray-100 rounded-full h-2">
              <div className="h-2 rounded-full transition-all duration-500"
                style={{ width: `${b.demonstrated}%`, backgroundColor: color.border }} />
            </div>
          </div>
        );
      })}
    </div>
  </div>
);

const BandBadge = ({ band }: { band: string }) => {
  const colors = BAND_COLORS[band] || BAND_COLORS["N/A"];
  return (
    <span className="text-xs font-semibold px-2 py-1 rounded-full"
      style={{ backgroundColor: colors.bg, color: colors.text }}>
      {band}
    </span>
  );
};

// ─────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────

export default function Home() {
  // ── Setup state ──────────────────────────────
  const [numQuestions, setNumQuestions] = useState(0);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [isSetupComplete, setIsSetupComplete] = useState(false);
  const [currentPage, setCurrentPage] = useState(0);

  // ── Student sheets state ──────────────────────
  const [studentSheets, setStudentSheets] = useState<
    { file: File | null; name: string }[]
  >([{ file: null, name: "" }]);

  // ── Results state ─────────────────────────────
  const [batchResult, setBatchResult] = useState<BatchResult | null>(null);
  const [selectedStudentIdx, setSelectedStudentIdx] = useState(0);
  const [analysingIndex, setAnalysingIndex] = useState<number | null>(null);

  // ── Loading ───────────────────────────────────
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState("");

  // ═════════════════════════════════════════════
  // SETUP
  // ═════════════════════════════════════════════

  const handleNumQuestionsSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (numQuestions < 1) return;
    setQuestions(
      Array.from({ length: numQuestions }, (_, i) => ({
        id: i + 1,
        keyAnswer: "",
        keyDiagram: null,
        keyDiagramName: "Add diagram",
        marks: 0,
        textWeight: 1.0,
        diagramWeight: 0.0,
      }))
    );
    setIsSetupComplete(true);
    setCurrentPage(0);
  };

  // ═════════════════════════════════════════════
  // QUESTION HANDLERS
  // ═════════════════════════════════════════════

  const handleQuestionChange = (index: number, field: keyof Question, value: any) => {
    setQuestions((prev) => {
      const updated = [...prev];
      updated[index] = { ...updated[index], [field]: value };
      return updated;
    });
  };

  // ═════════════════════════════════════════════
  // STUDENT SHEET HANDLERS
  // ═════════════════════════════════════════════

  const addStudentSlot = () =>
    setStudentSheets((prev) => [...prev, { file: null, name: "" }]);

  const removeStudentSlot = (idx: number) =>
    setStudentSheets((prev) => prev.filter((_, i) => i !== idx));

  const updateStudentSheet = (idx: number, file: File | null) =>
    setStudentSheets((prev) => {
      const updated = [...prev];
      updated[idx] = { ...updated[idx], file };
      return updated;
    });

  const updateStudentName = (idx: number, name: string) =>
    setStudentSheets((prev) => {
      const updated = [...prev];
      updated[idx] = { ...updated[idx], name };
      return updated;
    });

  // ═════════════════════════════════════════════
  // EVALUATION
  // ═════════════════════════════════════════════

  const evaluateAll = async () => {
    // Validate questions
    for (let i = 0; i < questions.length; i++) {
      if (!questions[i].keyAnswer.trim()) { alert(`Q${i + 1}: Please enter an answer key`); setCurrentPage(i); return; }
      if (questions[i].marks <= 0)         { alert(`Q${i + 1}: Please enter valid marks`);  setCurrentPage(i); return; }
    }
    // Validate sheets
    const validSheets = studentSheets.filter((s) => s.file !== null);
    if (validSheets.length === 0) { alert("Please upload at least one student answer sheet"); return; }

    setLoading(true);
    setBatchResult(null);

    try {
      const formData = new FormData();
      formData.append(
        "questions",
        JSON.stringify(
          questions.map((q) => ({
            keyAnswer: q.keyAnswer,
            marks: q.marks,
            textWeight: q.textWeight,
            diagramWeight: q.diagramWeight,
          }))
        )
      );

      const names: string[] = [];
      validSheets.forEach((s, i) => {
        formData.append("answer_sheets", s.file as File);
        names.push(s.name.trim() || `Student_${i + 1}`);
      });
      formData.append("student_names", JSON.stringify(names));

      // Stage labels for UX feedback
      setLoadingStage("📄 Stage 1/3 — Uni-MuMER reading handwriting...");
      console.log("🚀 Sending batch request...");

      const response = await axios.post(`${BACKEND_URL}/similarity`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          "ngrok-skip-browser-warning": "true",
        },
        timeout: 1200000, // 20 min for large batches
        onUploadProgress: (e) => {
          if (e.total)
            setLoadingStage(
              `📤 Uploading... ${Math.round((e.loaded * 100) / e.total)}%`
            );
        },
      });

      setLoadingStage("✅ Processing complete!");
      const data: BatchResult = response.data;
      console.log("✅ Batch response:", data);

      // Normalise: if backend returned single-student flat shape, wrap it
      if (!data.students && (data as any).score !== undefined) {
        const flat = data as any;
        const wrapped: BatchResult = {
          batch_size: 1,
          average_score: flat.score,
          highest_score: flat.score,
          lowest_score:  flat.score,
          total_marks:   flat.total,
          students: [{
            student_name:   names[0] || "Student 1",
            student_index:  0,
            score:          flat.score,
            total:          flat.total,
            key_points:     flat.key_points || [],
            student_answer: flat.student_answer || [],
            per_question:   flat.per_question || [],
            analysis: null,
            analysisOpen: false,
          }],
        };
        setBatchResult(wrapped);
      } else {
        // Ensure analysis/analysisOpen defaults
        data.students = data.students.map((s) => ({
          ...s,
          analysis: null,
          analysisOpen: false,
        }));
        setBatchResult(data);
      }

      setSelectedStudentIdx(0);
    } catch (error: any) {
      console.error("❌ Evaluation failed:", error);
      alert(`Evaluation failed: ${error.message}`);
    } finally {
      setLoading(false);
      setLoadingStage("");
    }
  };

  // ═════════════════════════════════════════════
  // ANALYSIS (per student)
  // ═════════════════════════════════════════════

  const performAnalysis = async (studentIdx: number) => {
    if (!batchResult) return;
    const student = batchResult.students[studentIdx];
    if (!student) return;

    // Toggle close
    if (student.analysisOpen && student.analysis) {
      setBatchResult((prev) => {
        if (!prev) return prev;
        const updated = [...prev.students];
        updated[studentIdx] = { ...updated[studentIdx], analysisOpen: false };
        return { ...prev, students: updated };
      });
      return;
    }

    // Open panel
    setBatchResult((prev) => {
      if (!prev) return prev;
      const updated = [...prev.students];
      updated[studentIdx] = { ...updated[studentIdx], analysisOpen: true };
      return { ...prev, students: updated };
    });

    if (student.analysis) return; // already loaded

    setAnalysingIndex(studentIdx);
    try {
      const primaryQ = questions[0];
      const response = await axios.post(
        `${BACKEND_URL}/analyse`,
        {
          keyAnswer:  primaryQ?.keyAnswer || "",
          keyPoints:  student.key_points || [],
          score:      student.score,
          total:      student.total,
          remarks:    student.per_question?.[0]?.remarks || "",
        },
        {
          headers: {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
          },
          timeout: 300000,
        }
      );

      const data: Analysis = response.data;
      console.log("✅ Analysis response:", data);

      setBatchResult((prev) => {
        if (!prev) return prev;
        const updated = [...prev.students];
        updated[studentIdx] = {
          ...updated[studentIdx],
          analysis: data,
          analysisOpen: true,
          band: data.band || updated[studentIdx].band,
        };
        return { ...prev, students: updated };
      });
    } catch (error: any) {
      console.error("❌ Analysis failed:", error);
      alert(`Analysis failed: ${error.message}`);
      setBatchResult((prev) => {
        if (!prev) return prev;
        const updated = [...prev.students];
        updated[studentIdx] = { ...updated[studentIdx], analysisOpen: false };
        return { ...prev, students: updated };
      });
    } finally {
      setAnalysingIndex(null);
    }
  };

  // ═════════════════════════════════════════════
  // RENDER — SETUP SCREEN
  // ═════════════════════════════════════════════

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
                placeholder="e.g., 5" autoFocus
              />
            </div>
            <button type="submit"
              className="w-full py-3 text-sm uppercase tracking-wider transition-colors rounded-md"
              style={{ backgroundColor: "#0F2854", color: "#BDE8F5" }}>
              Start
            </button>
          </form>
        </div>
      </div>
    );
  }

  const currentQuestion = questions[currentPage];
  const selectedStudent = batchResult?.students[selectedStudentIdx];

  // ═════════════════════════════════════════════
  // RENDER — MAIN
  // ═════════════════════════════════════════════

  return (
    <div className="w-screen min-h-screen flex flex-col" style={{ backgroundColor: "#BDE8F5" }}>

      {/* HEADER */}
      <div className="h-16 bg-white shadow-sm flex items-center justify-between px-8">
        <h1 className="text-xl font-light" style={{ color: "#0F2854" }}>AutoChecker</h1>
        <span className="text-sm" style={{ color: "#0F2854" }}>
          {questions.length} question{questions.length > 1 ? "s" : ""} · {studentSheets.filter(s => s.file).length} sheet{studentSheets.filter(s => s.file).length !== 1 ? "s" : ""} uploaded
        </span>
      </div>

      <div className="flex-1 flex p-6 gap-6 overflow-hidden">

        {/* ── LEFT SIDEBAR ───────────────────────────────────── */}
        <div className="w-64 flex flex-col gap-4">

          {/* Question list */}
          <div className="bg-white rounded-2xl shadow-sm p-4 flex flex-col">
            <h2 className="text-xs uppercase tracking-wider mb-3" style={{ color: "#0F2854" }}>Questions</h2>
            <div className="space-y-1">
              {questions.map((q, idx) => {
                const isActive     = idx === currentPage;
                const isIncomplete = !q.keyAnswer.trim() || q.marks <= 0;
                return (
                  <button key={q.id} onClick={() => setCurrentPage(idx)}
                    className={`w-full p-2 rounded-lg text-left text-sm flex items-center justify-between transition-all ${isActive ? "ring-2" : "hover:bg-gray-50"}`}
                    style={{ backgroundColor: isActive ? "#BDE8F5" : "white" }}>
                    <span style={{ color: "#0F2854" }}>Q{q.id} — {q.marks || "?"} marks</span>
                    {isIncomplete && (
                      <span className="text-xs px-1.5 py-0.5 bg-yellow-100 text-yellow-700 rounded-full">!</span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Student sheets */}
          <div className="bg-white rounded-2xl shadow-sm p-4 flex flex-col flex-1 overflow-hidden">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs uppercase tracking-wider" style={{ color: "#0F2854" }}>Student Sheets</h2>
              <button onClick={addStudentSlot}
                className="text-xs px-2 py-1 rounded-lg transition-colors"
                style={{ backgroundColor: "#BDE8F5", color: "#0F2854" }}>
                + Add
              </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-2">
              {studentSheets.map((s, idx) => (
                <div key={idx} className="border border-gray-100 rounded-xl p-3 space-y-2">

                  {/* Name input */}
                  <input
                    type="text"
                    value={s.name}
                    onChange={(e) => updateStudentName(idx, e.target.value)}
                    placeholder={`Student ${idx + 1} name`}
                    className="w-full text-xs p-1.5 border border-gray-200 rounded-md outline-none focus:border-blue-300"
                  />

                  {/* File upload */}
                  {!s.file ? (
                    <>
                      <input type="file" id={`sheet-${idx}`} accept=".pdf,.jpg,.png"
                        className="hidden"
                        onChange={(e) => updateStudentSheet(idx, e.target.files?.[0] || null)} />
                      <label htmlFor={`sheet-${idx}`}
                        className="flex items-center justify-center gap-1 w-full py-2 border-2 border-dashed border-gray-200 rounded-lg cursor-pointer hover:border-gray-300 transition-colors text-xs text-gray-400">
                        <span>+</span> Upload sheet
                      </label>
                    </>
                  ) : (
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="text-green-500 text-sm">✓</span>
                        <span className="text-xs truncate text-gray-600">{s.file.name}</span>
                      </div>
                      <button onClick={() => updateStudentSheet(idx, null)}
                        className="text-xs text-red-400 hover:text-red-600 ml-1 shrink-0">✕</button>
                    </div>
                  )}

                  {/* Remove slot */}
                  {studentSheets.length > 1 && (
                    <button onClick={() => removeStudentSlot(idx)}
                      className="text-xs text-gray-400 hover:text-red-500 w-full text-center">
                      Remove slot
                    </button>
                  )}
                </div>
              ))}
            </div>

            {/* Evaluate button */}
            <button
              onClick={evaluateAll}
              disabled={loading}
              className="mt-3 w-full py-2.5 text-sm font-medium rounded-xl transition-colors disabled:opacity-50"
              style={{ backgroundColor: "#0F2854", color: "#BDE8F5" }}>
              {loading ? loadingStage || "Processing..." : `Evaluate ${studentSheets.filter(s => s.file).length} Sheet${studentSheets.filter(s => s.file).length !== 1 ? "s" : ""}`}
            </button>
          </div>

        </div>

        {/* ── RIGHT PANEL ────────────────────────────────────── */}
        <div className="flex-1 flex flex-col gap-4 overflow-hidden">

          {/* ── Question editor ─────────────────────────────── */}
          <div className="bg-white rounded-2xl shadow-sm p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-light" style={{ color: "#0F2854" }}>
                Question {currentQuestion.id}
              </h2>
              <div className="flex items-center gap-3">
                <label className="text-sm" style={{ color: "#0F2854" }}>Marks:</label>
                <input type="number" min="0" value={currentQuestion.marks || ""}
                  onChange={(e) => handleQuestionChange(currentPage, "marks", parseInt(e.target.value) || 0)}
                  className="w-20 p-2 border border-gray-200 rounded-md text-center outline-none focus:border-2" />
                <div className="flex gap-2">
                  <button onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
                    disabled={currentPage === 0}
                    className="px-3 py-1 text-sm border border-gray-200 rounded-lg disabled:opacity-30 hover:bg-gray-50"
                    style={{ color: "#0F2854" }}>←</button>
                  <button onClick={() => setCurrentPage(p => Math.min(questions.length - 1, p + 1))}
                    disabled={currentPage === questions.length - 1}
                    className="px-3 py-1 text-sm border border-gray-200 rounded-lg disabled:opacity-30 hover:bg-gray-50"
                    style={{ color: "#0F2854" }}>→</button>
                </div>
              </div>
            </div>
            <textarea
              value={currentQuestion.keyAnswer}
              onChange={(e) => handleQuestionChange(currentPage, "keyAnswer", e.target.value)}
              className="w-full h-32 p-3 bg-gray-50 border border-gray-200 rounded-lg text-sm outline-none focus:border-2 resize-none"
              placeholder="Enter model answer for this question..."
            />
          </div>

          {/* ── Results panel ───────────────────────────────── */}
          {batchResult && (
            <div className="flex-1 flex flex-col gap-4 overflow-hidden">

              {/* Batch summary bar */}
              <div className="bg-white rounded-2xl shadow-sm p-4 flex items-center gap-6">
                <div>
                  <p className="text-xs text-gray-500">Students</p>
                  <p className="text-2xl font-light" style={{ color: "#0F2854" }}>{batchResult.batch_size}</p>
                </div>
                <div className="h-10 w-px bg-gray-200" />
                <div>
                  <p className="text-xs text-gray-500">Average</p>
                  <p className="text-2xl font-light" style={{ color: "#0F2854" }}>
                    {batchResult.average_score}/{batchResult.total_marks}
                  </p>
                </div>
                <div className="h-10 w-px bg-gray-200" />
                <div>
                  <p className="text-xs text-gray-500">Highest</p>
                  <p className="text-xl font-light text-green-600">{batchResult.highest_score}</p>
                </div>
                <div className="h-10 w-px bg-gray-200" />
                <div>
                  <p className="text-xs text-gray-500">Lowest</p>
                  <p className="text-xl font-light text-red-500">{batchResult.lowest_score}</p>
                </div>
                <div className="h-10 w-px bg-gray-200" />
                {/* Mini score bar chart */}
                <div className="flex-1 flex items-end gap-1 h-10">
                  {batchResult.students.map((s, i) => {
                    const pct = batchResult.total_marks > 0 ? (s.score / batchResult.total_marks) * 100 : 0;
                    return (
                      <button key={i} onClick={() => setSelectedStudentIdx(i)}
                        title={`${s.student_name}: ${s.score}/${s.total}`}
                        className={`flex-1 rounded-t transition-all ${i === selectedStudentIdx ? "opacity-100" : "opacity-60 hover:opacity-80"}`}
                        style={{ height: `${Math.max(10, pct)}%`, backgroundColor: "#0F2854" }} />
                    );
                  })}
                </div>
              </div>

              {/* Student tabs + detail */}
              <div className="flex-1 flex gap-4 overflow-hidden">

                {/* Student list */}
                <div className="w-52 bg-white rounded-2xl shadow-sm p-3 overflow-y-auto">
                  <p className="text-xs uppercase tracking-wider mb-2 px-1" style={{ color: "#0F2854" }}>Results</p>
                  {batchResult.students.map((s, i) => {
                    const isActive = i === selectedStudentIdx;
                    const pct      = batchResult.total_marks > 0
                      ? Math.round((s.score / batchResult.total_marks) * 100)
                      : 0;
                    return (
                      <button key={i} onClick={() => setSelectedStudentIdx(i)}
                        className={`w-full p-2.5 rounded-xl text-left mb-1 transition-all ${isActive ? "ring-2" : "hover:bg-gray-50"}`}
                        style={{ backgroundColor: isActive ? "#BDE8F5" : "white" }}>
                        <p className="text-sm font-medium truncate" style={{ color: "#0F2854" }}>{s.student_name}</p>
                        <div className="flex items-center justify-between mt-1">
                          <p className="text-xs text-gray-500">{s.score}/{s.total}</p>
                          {s.band && <BandBadge band={s.band} />}
                        </div>
                        {/* Mini progress bar */}
                        <div className="w-full bg-gray-100 rounded-full h-1 mt-1.5">
                          <div className="h-1 rounded-full" style={{ width: `${pct}%`, backgroundColor: "#0F2854" }} />
                        </div>
                      </button>
                    );
                  })}
                </div>

                {/* Student detail */}
                {selectedStudent && (
                  <div className="flex-1 bg-white rounded-2xl shadow-sm p-5 overflow-y-auto">

                    {/* Student header */}
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-light" style={{ color: "#0F2854" }}>
                          {selectedStudent.student_name}
                        </h3>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-2xl font-light" style={{ color: "#0F2854" }}>
                            {selectedStudent.score}/{selectedStudent.total}
                          </span>
                          {selectedStudent.band && <BandBadge band={selectedStudent.band} />}
                          {selectedStudent.cluster_id !== undefined && (
                            <span className="text-xs px-2 py-0.5 bg-gray-100 text-gray-500 rounded-full">
                              Cluster {selectedStudent.cluster_id}
                            </span>
                          )}
                        </div>
                      </div>
                      <button
                        onClick={() => performAnalysis(selectedStudentIdx)}
                        disabled={analysingIndex === selectedStudentIdx}
                        className="px-4 py-2 text-xs font-medium rounded-lg transition-colors disabled:opacity-50"
                        style={{ backgroundColor: "#0F2854", color: "#BDE8F5" }}>
                        {analysingIndex === selectedStudentIdx
                          ? "⏳ Analysing..."
                          : selectedStudent.analysisOpen && selectedStudent.analysis
                          ? "▲ Hide Analysis"
                          : "🔍 Analyse Answer"}
                      </button>
                    </div>

                    {/* Per-question breakdown */}
                    {selectedStudent.per_question?.length > 0 && (
                      <div className="mb-4">
                        <p className="text-xs uppercase tracking-wider text-gray-500 mb-2">Score Breakdown</p>
                        <div className="space-y-2">
                          {selectedStudent.per_question.map((pq, qi) => (
                            <div key={qi} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                              <span className="text-sm font-medium w-8" style={{ color: "#0F2854" }}>Q{qi + 1}</span>
                              <div className="flex-1">
                                <div className="flex justify-between text-xs mb-1">
                                  <span className="text-gray-600">{pq.remarks || "—"}</span>
                                  <span className="font-medium" style={{ color: "#0F2854" }}>{pq.score}/{pq.marks}</span>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-1.5">
                                  <div className="h-1.5 rounded-full" style={{
                                    width: `${pq.marks > 0 ? (pq.score / pq.marks) * 100 : 0}%`,
                                    backgroundColor: "#0F2854",
                                  }} />
                                </div>
                                {(pq.emb_score !== undefined || pq.llm_score !== undefined) && (
                                  <div className="flex gap-3 mt-1 text-xs text-gray-400">
                                    {pq.emb_score !== undefined && <span>Emb: {pq.emb_score}</span>}
                                    {pq.llm_score !== undefined && <span>LLM: {pq.llm_score}</span>}
                                  </div>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Key points */}
                    {selectedStudent.key_points?.length > 0 && (
                      <div className="mb-4">
                        <p className="text-xs uppercase tracking-wider text-gray-500 mb-2">Extracted Key Points</p>
                        <div className="flex flex-wrap gap-2">
                          {selectedStudent.key_points.map((kp, i) => (
                            <span key={i} className="text-xs px-2 py-1 bg-blue-50 text-blue-700 rounded-full">{kp}</span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Analysis panel */}
                    {selectedStudent.analysisOpen && (
                      <div className="border border-gray-200 rounded-2xl overflow-hidden mt-2">
                        <div className="p-3" style={{ backgroundColor: "#0F2854" }}>
                          <h4 className="text-sm font-semibold text-white uppercase tracking-wider">
                            📊 Answer Analysis
                          </h4>
                        </div>

                        {!selectedStudent.analysis ? (
                          <div className="p-8 text-center text-gray-400 text-sm">
                            ⏳ Generating analysis... this may take 1–2 minutes
                          </div>
                        ) : (
                          <div className="p-5 grid grid-cols-2 gap-6">
                            {/* Feedback cards */}
                            <div className="space-y-3">
                              <div className="bg-green-50 border border-green-200 rounded-xl p-3">
                                <p className="text-xs font-semibold text-green-700 uppercase tracking-wider mb-2">✅ Strengths</p>
                                <ul className="space-y-1">
                                  {selectedStudent.analysis.strengths.map((s, i) => (
                                    <li key={i} className="text-sm text-green-800 flex gap-2"><span>•</span><span>{s}</span></li>
                                  ))}
                                </ul>
                              </div>
                              <div className="bg-red-50 border border-red-200 rounded-xl p-3">
                                <p className="text-xs font-semibold text-red-700 uppercase tracking-wider mb-2">⚠️ Improvements</p>
                                <ul className="space-y-1">
                                  {selectedStudent.analysis.improvements.map((s, i) => (
                                    <li key={i} className="text-sm text-red-800 flex gap-2"><span>•</span><span>{s}</span></li>
                                  ))}
                                </ul>
                              </div>
                              <div className="bg-blue-50 border border-blue-200 rounded-xl p-3">
                                <p className="text-xs font-semibold text-blue-700 uppercase tracking-wider mb-2">💡 Suggestions</p>
                                <ul className="space-y-1">
                                  {selectedStudent.analysis.suggestions.map((s, i) => (
                                    <li key={i} className="text-sm text-blue-800 flex gap-2"><span>•</span><span>{s}</span></li>
                                  ))}
                                </ul>
                              </div>
                            </div>

                            {/* Bloom's chart */}
                            <div className="bg-gray-50 rounded-xl p-4">
                              <BloomsChart blooms={selectedStudent.analysis.blooms} />
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Empty state */}
          {!batchResult && !loading && (
            <div className="flex-1 bg-white rounded-2xl shadow-sm flex items-center justify-center">
              <div className="text-center text-gray-400">
                <p className="text-4xl mb-3">📝</p>
                <p className="text-sm">Fill in the answer keys, upload student sheets,</p>
                <p className="text-sm">then click <strong>Evaluate</strong> to begin.</p>
              </div>
            </div>
          )}

          {/* Loading state */}
          {loading && (
            <div className="flex-1 bg-white rounded-2xl shadow-sm flex items-center justify-center">
              <div className="text-center">
                <div className="w-12 h-12 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"
                  style={{ borderTopColor: "#0F2854" }} />
                <p className="text-sm font-medium" style={{ color: "#0F2854" }}>{loadingStage || "Processing..."}</p>
                <p className="text-xs text-gray-400 mt-2">
                  Uni-MuMER → Qwen → Clustering. This takes a few minutes.
                </p>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}