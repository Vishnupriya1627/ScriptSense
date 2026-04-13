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
}

// ─────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────

const BACKEND_URL = "https://mackenzie-unfilterable-kirby.ngrok-free.dev";

const BLOOM_COLORS: Record<string, { bg: string; bar: string }> = {
  Remember:   { bg: "#EFF6FF", bar: "#3B82F6" },
  Understand: { bg: "#F0FDF4", bar: "#22C55E" },
  Apply:      { bg: "#FEFCE8", bar: "#EAB308" },
  Analyse:    { bg: "#FFF1F2", bar: "#F43F5E" },
  Evaluate:   { bg: "#F5F3FF", bar: "#8B5CF6" },
  Create:     { bg: "#FFF7ED", bar: "#F97316" },
};

const BAND_COLORS: Record<string, { bg: string; text: string }> = {
  "Excellent":     { bg: "#D1FAE5", text: "#065F46" },
  "Good":          { bg: "#DBEAFE", text: "#1E40AF" },
  "Average":       { bg: "#FEF9C3", text: "#92400E" },
  "Below Average": { bg: "#FFE4E6", text: "#9F1239" },
  "Poor":          { bg: "#F3F4F6", text: "#374151" },
  "N/A":           { bg: "#F3F4F6", text: "#374151" },
};

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/** Convert backend flat result → StudentResult */
const toStudent = (flat: any, idx: number, name: string): StudentResult => ({
  student_name:   name || flat.student_name || `Student ${idx + 1}`,
  student_index:  idx,
  score:          flat.score          ?? 0,
  total:          flat.total          ?? 0,
  key_points:     flat.key_points     ?? [],
  student_answer: flat.student_answer ?? [],
  per_question:   flat.per_question   ?? [],
  analysis:       null,
  analysisOpen:   false,
  cluster_id:     flat.cluster_id,
  band:           flat.band,
  band_score:     flat.band_score,
});

/** Wrap any backend shape into BatchResult */
const normaliseBatchResult = (data: any, names: string[]): BatchResult => {
  // ── Case 1: backend returned an array (multiple sheets) ──────────────
  if (Array.isArray(data)) {
    const students = data.map((flat, i) =>
      toStudent(flat, i, names[i] ?? flat.student_name ?? `Student ${i + 1}`)
    );
    const scores = students.map((s) => s.score);
    return {
      batch_size:    students.length,
      average_score: parseFloat(
        (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2)
      ),
      highest_score: Math.max(...scores),
      lowest_score:  Math.min(...scores),
      total_marks:   students[0]?.total ?? 0,
      students,
    };
  }

  // ── Case 2: backend already returned batch shape ──────────────────────
  if (data.students) {
    const students = (data.students as any[]).map((s, i) =>
      toStudent(s, i, names[i] ?? s.student_name ?? `Student ${i + 1}`)
    );
    return {
      batch_size:    data.batch_size    ?? students.length,
      average_score: data.average_score ?? 0,
      highest_score: data.highest_score ?? 0,
      lowest_score:  data.lowest_score  ?? 0,
      total_marks:   data.total_marks   ?? students[0]?.total ?? 0,
      students,
    };
  }

  // ── Case 3: single flat result ────────────────────────────────────────
  const student = toStudent(data, 0, names[0] ?? "Student 1");
  return {
    batch_size:    1,
    average_score: student.score,
    highest_score: student.score,
    lowest_score:  student.score,
    total_marks:   student.total,
    students:      [student],
  };
};

// ─────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────

const BandBadge = ({ band }: { band: string }) => {
  const c = BAND_COLORS[band] ?? BAND_COLORS["N/A"];
  return (
    <span
      className="text-xs font-semibold px-2 py-1 rounded-full"
      style={{ backgroundColor: c.bg, color: c.text }}
    >
      {band}
    </span>
  );
};

const ScoreRing = ({ score, total }: { score: number; total: number }) => {
  const pct   = total > 0 ? score / total : 0;
  const r     = 28;
  const circ  = 2 * Math.PI * r;
  const dash  = circ * pct;
  const color = pct >= 0.75 ? "#22C55E" : pct >= 0.5 ? "#F59E0B" : "#F43F5E";
  return (
    <svg width="72" height="72" viewBox="0 0 72 72">
      <circle cx="36" cy="36" r={r} fill="none" stroke="#E5E7EB" strokeWidth="6" />
      <circle
        cx="36" cy="36" r={r}
        fill="none"
        stroke={color}
        strokeWidth="6"
        strokeDasharray={`${dash} ${circ - dash}`}
        strokeLinecap="round"
        transform="rotate(-90 36 36)"
        style={{ transition: "stroke-dasharray 0.6s ease" }}
      />
      <text x="36" y="40" textAnchor="middle" fontSize="13" fontWeight="600" fill="#0F2854">
        {score}/{total}
      </text>
    </svg>
  );
};

const BloomsChart = ({
  blooms,
}: {
  blooms: { level: string; required: number; demonstrated: number }[];
}) => (
  <div>
    <p className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: "#0F2854" }}>
      Bloom's Taxonomy
    </p>
    <div className="flex items-center gap-4 mb-3 text-xs text-gray-400">
      <span className="flex items-center gap-1">
        <span className="w-3 h-1.5 rounded-sm inline-block bg-gray-300" /> Required
      </span>
      <span className="flex items-center gap-1">
        <span
          className="w-3 h-1.5 rounded-sm inline-block"
          style={{ backgroundColor: "#0F2854" }}
        />{" "}
        Demonstrated
      </span>
    </div>
    <div className="space-y-3">
      {blooms.map((b) => {
        const c = BLOOM_COLORS[b.level] ?? { bg: "#F9FAFB", bar: "#6B7280" };
        return (
          <div key={b.level} className="rounded-lg p-2" style={{ backgroundColor: c.bg }}>
            <div className="flex justify-between text-xs mb-1.5">
              <span className="font-medium" style={{ color: "#0F2854" }}>
                {b.level}
              </span>
              <span className="text-gray-500">
                {b.demonstrated}% / {b.required}%
              </span>
            </div>
            {/* Required track */}
            <div className="w-full bg-gray-200 rounded-full h-1.5 mb-1">
              <div
                className="h-1.5 rounded-full bg-gray-400 transition-all duration-500"
                style={{ width: `${b.required}%` }}
              />
            </div>
            {/* Demonstrated track */}
            <div className="w-full bg-gray-200 rounded-full h-1.5">
              <div
                className="h-1.5 rounded-full transition-all duration-500"
                style={{ width: `${b.demonstrated}%`, backgroundColor: c.bar }}
              />
            </div>
          </div>
        );
      })}
    </div>
  </div>
);

// ─────────────────────────────────────────────
// Student Answer display
// ─────────────────────────────────────────────

const StudentAnswerPanel = ({ lines }: { lines: string[] }) => {
  const [expanded, setExpanded] = useState(false);
  const fullText = lines.join(" ");
  const preview  = fullText.slice(0, 320);
  const isTrunc  = fullText.length > 320;

  return (
    <div className="mb-5">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-semibold uppercase tracking-wider text-gray-400">
          📄 Extracted &amp; Cleaned Answer
        </p>
        {isTrunc && (
          <button
            onClick={() => setExpanded((v) => !v)}
            className="text-xs px-2 py-0.5 rounded-full border transition-colors"
            style={{ borderColor: "#BDE8F5", color: "#0F2854" }}
          >
            {expanded ? "Show less" : "Show full"}
          </button>
        )}
      </div>
      <div
        className="bg-gray-50 border border-gray-100 rounded-xl p-4 text-sm leading-relaxed text-gray-700 whitespace-pre-wrap"
        style={{ fontFamily: "'Georgia', serif" }}
      >
        {expanded || !isTrunc ? fullText : preview + (isTrunc ? "…" : "")}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────

export default function Home() {
  // ── Setup state ──────────────────────────────
  const [numQuestions, setNumQuestions]       = useState(0);
  const [questions, setQuestions]             = useState<Question[]>([]);
  const [isSetupComplete, setIsSetupComplete] = useState(false);
  const [currentPage, setCurrentPage]         = useState(0);

  // ── Student sheets ────────────────────────────
  const [studentSheets, setStudentSheets] = useState<{ file: File | null; name: string }[]>([
    { file: null, name: "" },
  ]);

  // ── Results ───────────────────────────────────
  const [batchResult, setBatchResult]               = useState<BatchResult | null>(null);
  const [selectedStudentIdx, setSelectedStudentIdx] = useState(0);
  const [analysingIndex, setAnalysingIndex]         = useState<number | null>(null);

  // ── Loading ───────────────────────────────────
  const [loading, setLoading]           = useState(false);
  const [loadingStage, setLoadingStage] = useState("");

  // ═════════════════════════════════════════════
  // SETUP
  // ═════════════════════════════════════════════

  const handleNumQuestionsSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const n = Math.floor(numQuestions);
    if (!n || n < 1) return;
    setQuestions(
      Array.from({ length: n }, (_, i) => ({
        id:            i + 1,
        keyAnswer:     "",
        keyDiagram:    null,
        keyDiagramName:"Add diagram",
        marks:         0,
        textWeight:    1.0,
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
    setStudentSheets((p) => [...p, { file: null, name: "" }]);

  const removeStudentSlot = (idx: number) =>
    setStudentSheets((p) => p.filter((_, i) => i !== idx));

  const updateStudentSheet = (idx: number, file: File | null) =>
    setStudentSheets((p) => {
      const u = [...p];
      u[idx]  = { ...u[idx], file };
      return u;
    });

  const updateStudentName = (idx: number, name: string) =>
    setStudentSheets((p) => {
      const u = [...p];
      u[idx]  = { ...u[idx], name };
      return u;
    });

  // ═════════════════════════════════════════════
  // EVALUATION
  // ═════════════════════════════════════════════

  const evaluateAll = async () => {
    for (let i = 0; i < questions.length; i++) {
      if (!questions[i].keyAnswer.trim()) {
        alert(`Q${i + 1}: Please enter an answer key`);
        setCurrentPage(i);
        return;
      }
      if (questions[i].marks <= 0) {
        alert(`Q${i + 1}: Please enter valid marks`);
        setCurrentPage(i);
        return;
      }
    }

    const validSheets = studentSheets.filter((s) => s.file !== null);
    if (validSheets.length === 0) {
      alert("Please upload at least one student answer sheet");
      return;
    }

    setLoading(true);
    setBatchResult(null);

    try {
      const formData = new FormData();

      formData.append(
        "questions",
        JSON.stringify(
          questions.map((q) => ({
            keyAnswer:     q.keyAnswer,
            marks:         q.marks,
            textWeight:    q.textWeight,
            diagramWeight: q.diagramWeight,
          }))
        )
      );

      // Append optional key diagrams
      questions.forEach((q, i) => {
        if (q.keyDiagram) {
          formData.append(`key_diagram_${i}`, q.keyDiagram);
        }
      });

      const names: string[] = [];
      validSheets.forEach((s, i) => {
        formData.append("answer_sheets", s.file as File);
        names.push(s.name.trim() || `Student_${i + 1}`);
      });
      formData.append("student_names", JSON.stringify(names));

      setLoadingStage("📄 Stage 1/3 — Uni-MuMER reading handwriting...");

      const response = await axios.post(`${BACKEND_URL}/similarity`, formData, {
        headers: {
          "Content-Type":              "multipart/form-data",
          "ngrok-skip-browser-warning":"true",
        },
        timeout: 1_200_000,
        onUploadProgress: (e) => {
          if (e.total) {
            setLoadingStage(
              `📤 Uploading... ${Math.round((e.loaded * 100) / e.total)}%`
            );
          }
        },
      });

      setLoadingStage("✅ Processing complete!");
      const raw = response.data;
      console.log("✅ Raw backend response:", raw);

      const normalised = normaliseBatchResult(raw, names);
      console.log("✅ Normalised batch result:", normalised);

      setBatchResult(normalised);
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
  // ANALYSIS
  // ═════════════════════════════════════════════

  const performAnalysis = async (studentIdx: number) => {
    if (!batchResult) return;
    const student = batchResult.students[studentIdx];
    if (!student) return;

    // Toggle close
    if (student.analysisOpen && student.analysis) {
      setBatchResult((prev) => {
        if (!prev) return prev;
        const u       = [...prev.students];
        u[studentIdx] = { ...u[studentIdx], analysisOpen: false };
        return { ...prev, students: u };
      });
      return;
    }

    // Open panel
    setBatchResult((prev) => {
      if (!prev) return prev;
      const u       = [...prev.students];
      u[studentIdx] = { ...u[studentIdx], analysisOpen: true };
      return { ...prev, students: u };
    });

    if (student.analysis) return;

    setAnalysingIndex(studentIdx);
    try {
      const primaryQ = questions[0];
      const response = await axios.post(
        `${BACKEND_URL}/analyse`,
        {
          keyAnswer: primaryQ?.keyAnswer || "",
          keyPoints: student.key_points  || [],
          score:     student.score,
          total:     student.total,
          remarks:   student.per_question?.[0]?.remarks || "",
        },
        {
          headers: {
            "Content-Type":              "application/json",
            "ngrok-skip-browser-warning":"true",
          },
          timeout: 300_000,
        }
      );

      const data: Analysis = response.data;
      setBatchResult((prev) => {
        if (!prev) return prev;
        const u       = [...prev.students];
        u[studentIdx] = {
          ...u[studentIdx],
          analysis:     data,
          analysisOpen: true,
          band:         data.band       ?? u[studentIdx].band,
          band_score:   data.band_score ?? u[studentIdx].band_score,
        };
        return { ...prev, students: u };
      });
    } catch (error: any) {
      console.error("❌ Analysis failed:", error);
      alert(`Analysis failed: ${error.message}`);
      setBatchResult((prev) => {
        if (!prev) return prev;
        const u       = [...prev.students];
        u[studentIdx] = { ...u[studentIdx], analysisOpen: false };
        return { ...prev, students: u };
      });
    } finally {
      setAnalysingIndex(null);
    }
  };

  // ═════════════════════════════════════════════
  // RENDER — SETUP
  // ═════════════════════════════════════════════

  if (!isSetupComplete) {
    return (
      <div
        className="w-screen h-screen flex items-center justify-center"
        style={{ backgroundColor: "#BDE8F5" }}
      >
        <div className="bg-white p-12 rounded-2xl shadow-lg w-96">
          <h1 className="text-3xl font-light mb-2" style={{ color: "#0F2854" }}>
            AutoChecker
          </h1>
          <p className="text-sm text-gray-400 mb-8">AI-powered answer sheet evaluator</p>
          <form onSubmit={handleNumQuestionsSubmit} className="space-y-6">
            <div className="space-y-2">
              <label className="text-sm uppercase tracking-wider" style={{ color: "#0F2854" }}>
                Number of Questions
              </label>
              <input
                type="number"
                min="1"
                value={numQuestions || ""}
                onChange={(e) => setNumQuestions(parseInt(e.target.value) || 0)}
                className="w-full p-3 border border-gray-200 rounded-md outline-none focus:border-blue-300 transition-colors"
                placeholder="e.g., 5"
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

  const currentQuestion  = questions[currentPage];
  const selectedStudent  = batchResult?.students[selectedStudentIdx];
  const uploadedCount    = studentSheets.filter((s) => s.file).length;

  // ═════════════════════════════════════════════
  // RENDER — MAIN
  // ═════════════════════════════════════════════

  return (
    <div className="w-screen min-h-screen flex flex-col" style={{ backgroundColor: "#BDE8F5" }}>

      {/* HEADER */}
      <div className="h-16 bg-white shadow-sm flex items-center justify-between px-8 shrink-0">
        <h1 className="text-xl font-light" style={{ color: "#0F2854" }}>AutoChecker</h1>
        <span className="text-sm text-gray-400">
          {questions.length} question{questions.length !== 1 ? "s" : ""}
          &nbsp;·&nbsp;
          {uploadedCount} sheet{uploadedCount !== 1 ? "s" : ""} uploaded
        </span>
      </div>

      <div className="flex-1 flex p-6 gap-6 overflow-hidden">

        {/* ── LEFT SIDEBAR ─────────────────────────────── */}
        <div className="w-64 flex flex-col gap-4 shrink-0">

          {/* Question list */}
          <div className="bg-white rounded-2xl shadow-sm p-4">
            <h2 className="text-xs uppercase tracking-wider mb-3 text-gray-400">Questions</h2>
            <div className="space-y-1">
              {questions.map((q, idx) => {
                const isActive     = idx === currentPage;
                const isIncomplete = !q.keyAnswer.trim() || q.marks <= 0;
                return (
                  <button
                    key={q.id}
                    onClick={() => setCurrentPage(idx)}
                    className={`w-full p-2 rounded-lg text-left text-sm flex items-center justify-between transition-all ${
                      isActive ? "ring-2" : "hover:bg-gray-50"
                    }`}
                    style={{ backgroundColor: isActive ? "#BDE8F5" : "white" }}
                  >
                    <span style={{ color: "#0F2854" }}>
                      Q{q.id} — {q.marks || "?"} marks
                    </span>
                    {isIncomplete && (
                      <span className="text-xs px-1.5 py-0.5 bg-yellow-100 text-yellow-700 rounded-full">
                        !
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Student sheets */}
          <div className="bg-white rounded-2xl shadow-sm p-4 flex flex-col flex-1 overflow-hidden">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs uppercase tracking-wider text-gray-400">Student Sheets</h2>
              <button
                onClick={addStudentSlot}
                className="text-xs px-2 py-1 rounded-lg"
                style={{ backgroundColor: "#BDE8F5", color: "#0F2854" }}
              >
                + Add
              </button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-2">
              {studentSheets.map((s, idx) => (
                <div key={idx} className="border border-gray-100 rounded-xl p-3 space-y-2">
                  <input
                    type="text"
                    value={s.name}
                    onChange={(e) => updateStudentName(idx, e.target.value)}
                    placeholder={`Student ${idx + 1} name`}
                    className="w-full text-xs p-1.5 border border-gray-200 rounded-md outline-none focus:border-blue-300"
                  />

                  {!s.file ? (
                    <>
                      <input
                        type="file"
                        id={`sheet-${idx}`}
                        accept=".pdf,.jpg,.png"
                        className="hidden"
                        onChange={(e) =>
                          updateStudentSheet(idx, e.target.files?.[0] || null)
                        }
                      />
                      <label
                        htmlFor={`sheet-${idx}`}
                        className="flex items-center justify-center gap-1 w-full py-2 border-2 border-dashed border-gray-200 rounded-lg cursor-pointer hover:border-gray-300 text-xs text-gray-400"
                      >
                        + Upload sheet
                      </label>
                    </>
                  ) : (
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="text-green-500 text-sm">✓</span>
                        <span className="text-xs truncate text-gray-600">{s.file.name}</span>
                      </div>
                      <button
                        onClick={() => updateStudentSheet(idx, null)}
                        className="text-xs text-red-400 hover:text-red-600 ml-1 shrink-0"
                      >
                        ✕
                      </button>
                    </div>
                  )}

                  {studentSheets.length > 1 && (
                    <button
                      onClick={() => removeStudentSlot(idx)}
                      className="text-xs text-gray-400 hover:text-red-500 w-full text-center"
                    >
                      Remove slot
                    </button>
                  )}
                </div>
              ))}
            </div>

            <button
              onClick={evaluateAll}
              disabled={loading}
              className="mt-3 w-full py-2.5 text-sm font-medium rounded-xl transition-colors disabled:opacity-50"
              style={{ backgroundColor: "#0F2854", color: "#BDE8F5" }}
            >
              {loading
                ? loadingStage || "Processing..."
                : `Evaluate ${uploadedCount} Sheet${uploadedCount !== 1 ? "s" : ""}`}
            </button>
          </div>

        </div>

        {/* ── RIGHT PANEL ──────────────────────────────── */}
        <div className="flex-1 flex flex-col gap-4 overflow-hidden min-w-0">

          {/* Question editor */}
          <div className="bg-white rounded-2xl shadow-sm p-6 shrink-0">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-light" style={{ color: "#0F2854" }}>
                Question {currentQuestion.id}
              </h2>
              <div className="flex items-center gap-3">
                <label className="text-sm" style={{ color: "#0F2854" }}>Marks:</label>
                <input
                  type="number"
                  min="0"
                  value={currentQuestion.marks || ""}
                  onChange={(e) =>
                    handleQuestionChange(currentPage, "marks", parseInt(e.target.value) || 0)
                  }
                  className="w-20 p-2 border border-gray-200 rounded-md text-center outline-none focus:border-blue-300"
                />
                <div className="flex gap-2">
                  <button
                    onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
                    disabled={currentPage === 0}
                    className="px-3 py-1 text-sm border border-gray-200 rounded-lg disabled:opacity-30 hover:bg-gray-50"
                    style={{ color: "#0F2854" }}
                  >
                    ←
                  </button>
                  <button
                    onClick={() =>
                      setCurrentPage((p) => Math.min(questions.length - 1, p + 1))
                    }
                    disabled={currentPage === questions.length - 1}
                    className="px-3 py-1 text-sm border border-gray-200 rounded-lg disabled:opacity-30 hover:bg-gray-50"
                    style={{ color: "#0F2854" }}
                  >
                    →
                  </button>
                </div>
              </div>
            </div>
            <textarea
              value={currentQuestion.keyAnswer}
              onChange={(e) => handleQuestionChange(currentPage, "keyAnswer", e.target.value)}
              className="w-full h-28 p-3 bg-gray-50 border border-gray-200 rounded-lg text-sm outline-none focus:border-blue-300 resize-none"
              placeholder="Enter model answer for this question..."
            />
          </div>

          {/* ── RESULTS ──────────────────────────────────── */}
          {batchResult && (
            <div className="flex-1 flex flex-col gap-4 overflow-hidden">

              {/* Batch summary bar */}
              <div className="bg-white rounded-2xl shadow-sm p-4 flex items-center gap-6 shrink-0">
                <div>
                  <p className="text-xs text-gray-400">Students</p>
                  <p className="text-2xl font-light" style={{ color: "#0F2854" }}>
                    {batchResult.batch_size}
                  </p>
                </div>
                <div className="h-10 w-px bg-gray-100" />
                <div>
                  <p className="text-xs text-gray-400">Average</p>
                  <p className="text-2xl font-light" style={{ color: "#0F2854" }}>
                    {batchResult.average_score}
                    <span className="text-sm text-gray-400">/{batchResult.total_marks}</span>
                  </p>
                </div>
                <div className="h-10 w-px bg-gray-100" />
                <div>
                  <p className="text-xs text-gray-400">Highest</p>
                  <p className="text-xl font-light text-green-500">{batchResult.highest_score}</p>
                </div>
                <div className="h-10 w-px bg-gray-100" />
                <div>
                  <p className="text-xs text-gray-400">Lowest</p>
                  <p className="text-xl font-light text-red-400">{batchResult.lowest_score}</p>
                </div>
                <div className="h-10 w-px bg-gray-100" />
                {/* Mini bar chart */}
                <div className="flex-1 flex items-end gap-1 h-10">
                  {batchResult.students.map((s, i) => {
                    const pct    = batchResult.total_marks > 0
                      ? (s.score / batchResult.total_marks) * 100
                      : 0;
                    const active = i === selectedStudentIdx;
                    return (
                      <button
                        key={i}
                        onClick={() => setSelectedStudentIdx(i)}
                        title={`${s.student_name}: ${s.score}/${s.total}`}
                        className={`flex-1 rounded-t transition-all ${
                          active ? "opacity-100" : "opacity-40 hover:opacity-70"
                        }`}
                        style={{
                          height:          `${Math.max(8, pct)}%`,
                          backgroundColor: "#0F2854",
                        }}
                      />
                    );
                  })}
                </div>
              </div>

              {/* Student list + detail */}
              <div className="flex-1 flex gap-4 overflow-hidden">

                {/* Student list */}
                <div className="w-52 bg-white rounded-2xl shadow-sm p-3 overflow-y-auto shrink-0">
                  <p className="text-xs uppercase tracking-wider mb-2 px-1 text-gray-400">
                    Results
                  </p>
                  {batchResult.students.map((s, i) => {
                    const isActive   = i === selectedStudentIdx;
                    const pct        = batchResult.total_marks > 0
                      ? Math.round((s.score / batchResult.total_marks) * 100)
                      : 0;
                    const scoreColor = pct >= 75 ? "#22C55E" : pct >= 50 ? "#F59E0B" : "#F43F5E";
                    return (
                      <button
                        key={i}
                        onClick={() => setSelectedStudentIdx(i)}
                        className={`w-full p-2.5 rounded-xl text-left mb-1 transition-all ${
                          isActive ? "ring-2" : "hover:bg-gray-50"
                        }`}
                        style={{ backgroundColor: isActive ? "#EFF6FF" : "white" }}
                      >
                        <p className="text-sm font-medium truncate" style={{ color: "#0F2854" }}>
                          {s.student_name}
                        </p>
                        <div className="flex items-center justify-between mt-1">
                          <p className="text-xs font-semibold" style={{ color: scoreColor }}>
                            {s.score}/{s.total}
                            <span className="text-gray-400 font-normal ml-1">({pct}%)</span>
                          </p>
                          {s.band && <BandBadge band={s.band} />}
                        </div>
                        <div className="w-full bg-gray-100 rounded-full h-1 mt-1.5">
                          <div
                            className="h-1 rounded-full transition-all duration-500"
                            style={{ width: `${pct}%`, backgroundColor: scoreColor }}
                          />
                        </div>
                      </button>
                    );
                  })}
                </div>

                {/* ── STUDENT DETAIL ── */}
                {selectedStudent && (
                  <div className="flex-1 bg-white rounded-2xl shadow-sm p-5 overflow-y-auto min-w-0">

                    {/* Header row */}
                    <div className="flex items-start justify-between mb-5">
                      <div className="flex items-center gap-4">
                        <ScoreRing score={selectedStudent.score} total={selectedStudent.total} />
                        <div>
                          <h3 className="text-xl font-light" style={{ color: "#0F2854" }}>
                            {selectedStudent.student_name}
                          </h3>
                          <div className="flex items-center gap-2 mt-1">
                            {(() => {
                              const pct   = selectedStudent.total > 0
                                ? (selectedStudent.score / selectedStudent.total) * 100
                                : 0;
                              const label = pct >= 75
                                ? "Excellent"
                                : pct >= 60
                                ? "Good"
                                : pct >= 40
                                ? "Average"
                                : "Below Average";
                              return <BandBadge band={selectedStudent.band ?? label} />;
                            })()}
                            {selectedStudent.cluster_id !== undefined && (
                              <span className="text-xs px-2 py-0.5 bg-gray-100 text-gray-400 rounded-full">
                                Cluster {selectedStudent.cluster_id}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={() => performAnalysis(selectedStudentIdx)}
                        disabled={analysingIndex === selectedStudentIdx}
                        className="px-4 py-2 text-xs font-medium rounded-lg transition-colors disabled:opacity-50"
                        style={{ backgroundColor: "#0F2854", color: "#BDE8F5" }}
                      >
                        {analysingIndex === selectedStudentIdx
                          ? "⏳ Analysing..."
                          : selectedStudent.analysisOpen && selectedStudent.analysis
                          ? "▲ Hide Analysis"
                          : "🔍 Deep Analysis"}
                      </button>
                    </div>

                    {/* ── Score Breakdown ── */}
                    {selectedStudent.per_question?.length > 0 && (
                      <div className="mb-5">
                        <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">
                          Score Breakdown
                        </p>
                        <div className="space-y-2">
                          {selectedStudent.per_question.map((pq, qi) => {
                            const pct = pq.marks > 0 ? (pq.score / pq.marks) * 100 : 0;
                            const col = pct >= 75 ? "#22C55E" : pct >= 50 ? "#F59E0B" : "#F43F5E";
                            return (
                              <div key={qi} className="rounded-xl border border-gray-100 p-3">
                                <div className="flex items-center justify-between mb-1">
                                  <span className="text-sm font-semibold" style={{ color: "#0F2854" }}>
                                    Q{qi + 1}
                                  </span>
                                  <div className="flex items-center gap-3">
                                    {pq.emb_score !== undefined && (
                                      <span className="text-xs text-gray-400">
                                        Emb{" "}
                                        <span className="font-medium text-gray-600">
                                          {pq.emb_score}
                                        </span>
                                      </span>
                                    )}
                                    {pq.llm_score !== undefined && (
                                      <span className="text-xs text-gray-400">
                                        LLM{" "}
                                        <span className="font-medium text-gray-600">
                                          {pq.llm_score}
                                        </span>
                                      </span>
                                    )}
                                    <span className="text-sm font-bold" style={{ color: col }}>
                                      {pq.score}/{pq.marks}
                                    </span>
                                  </div>
                                </div>
                                <div className="w-full bg-gray-100 rounded-full h-2 mb-2">
                                  <div
                                    className="h-2 rounded-full transition-all duration-500"
                                    style={{ width: `${pct}%`, backgroundColor: col }}
                                  />
                                </div>
                                {pq.remarks && (
                                  <p className="text-xs text-gray-500 leading-relaxed">
                                    {pq.remarks}
                                  </p>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* ── Extracted & Cleaned Answer ── */}
                    {selectedStudent.student_answer?.length > 0 && (
                      <StudentAnswerPanel lines={selectedStudent.student_answer} />
                    )}

                    {/* ── Key Points ── */}
                    {selectedStudent.key_points?.length > 0 && (
                      <div className="mb-5">
                        <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-2">
                          🔑 Key Points Identified
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {selectedStudent.key_points.map((kp, i) => (
                            <span
                              key={i}
                              className="text-xs px-3 py-1 rounded-full border"
                              style={{
                                backgroundColor: "#EFF6FF",
                                borderColor:     "#BFDBFE",
                                color:           "#1D4ED8",
                              }}
                            >
                              {kp}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* ── Analysis Panel ── */}
                    {selectedStudent.analysisOpen && (
                      <div className="border border-gray-100 rounded-2xl overflow-hidden mt-2">
                        <div className="px-5 py-3" style={{ backgroundColor: "#0F2854" }}>
                          <h4 className="text-sm font-semibold text-white uppercase tracking-wider">
                            📊 Deep Answer Analysis
                          </h4>
                        </div>

                        {!selectedStudent.analysis ? (
                          <div className="p-8 text-center text-gray-400 text-sm">
                            <div
                              className="w-8 h-8 border-4 border-gray-200 border-t-blue-500 rounded-full animate-spin mx-auto mb-3"
                              style={{ borderTopColor: "#0F2854" }}
                            />
                            Generating analysis… this may take 1–2 minutes
                          </div>
                        ) : (
                          <div className="p-5 grid grid-cols-2 gap-5">
                            {/* Feedback */}
                            <div className="space-y-3">
                              <div className="bg-green-50 border border-green-100 rounded-xl p-3">
                                <p className="text-xs font-semibold text-green-700 uppercase tracking-wider mb-2">
                                  ✅ Strengths
                                </p>
                                <ul className="space-y-1.5">
                                  {selectedStudent.analysis.strengths.map((s, i) => (
                                    <li key={i} className="text-sm text-green-800 flex gap-2">
                                      <span className="shrink-0">•</span>
                                      <span>{s}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                              <div className="bg-red-50 border border-red-100 rounded-xl p-3">
                                <p className="text-xs font-semibold text-red-700 uppercase tracking-wider mb-2">
                                  ⚠️ Needs Improvement
                                </p>
                                <ul className="space-y-1.5">
                                  {selectedStudent.analysis.improvements.map((s, i) => (
                                    <li key={i} className="text-sm text-red-800 flex gap-2">
                                      <span className="shrink-0">•</span>
                                      <span>{s}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                              <div className="bg-blue-50 border border-blue-100 rounded-xl p-3">
                                <p className="text-xs font-semibold text-blue-700 uppercase tracking-wider mb-2">
                                  💡 Suggestions
                                </p>
                                <ul className="space-y-1.5">
                                  {selectedStudent.analysis.suggestions.map((s, i) => (
                                    <li key={i} className="text-sm text-blue-800 flex gap-2">
                                      <span className="shrink-0">•</span>
                                      <span>{s}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </div>

                            {/* Bloom's */}
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
                <p className="text-sm mt-1">
                  then click <strong className="text-gray-600">Evaluate</strong> to begin.
                </p>
              </div>
            </div>
          )}

          {/* Loading state */}
          {loading && (
            <div className="flex-1 bg-white rounded-2xl shadow-sm flex items-center justify-center">
              <div className="text-center">
                <div
                  className="w-12 h-12 border-4 border-gray-100 rounded-full animate-spin mx-auto mb-4"
                  style={{ borderTopColor: "#0F2854" }}
                />
                <p className="text-sm font-medium" style={{ color: "#0F2854" }}>
                  {loadingStage || "Processing..."}
                </p>
                <p className="text-xs text-gray-400 mt-2">
                  Uni-MuMER → Qwen → Evaluation · This takes a few minutes
                </p>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}