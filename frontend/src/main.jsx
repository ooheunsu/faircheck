import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const EXAMPLE_QUESTIONS = [
  "커피 가맹점주인데 계약서상 보장된 영업지역 100m 안에 본사가 직영점을 열겠다고 합니다.",
  "원사업자가 목적물을 받은 지 60일이 지났는데도 하도급대금을 지급하지 않고 있습니다.",
  "가맹본부가 계약서에 없는 식자재를 반드시 본사 지정업체에서만 사라고 합니다.",
];

const LOADING_STEPS = [
  "질문 쟁점을 분류하고 있어요",
  "유사 공정위 의결서를 검색하고 있어요",
  "의결서에 나온 관련 법령을 확인하고 있어요",
  "질문과 직접 관련된 법령을 다시 찾고 있어요",
  "근거를 합쳐 진단 답변을 작성하고 있어요",
];

function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  const resultRef = useRef(null);

  async function handleSubmit(event) {
    event.preventDefault();
    const cleaned = question.trim();
    if (!cleaned) {
      setError("진단할 상황을 먼저 입력해주세요.");
      return;
    }

    setError("");
    setResult(null);
    setIsLoading(true);
    setActiveStep(0);

    const stepTimer = window.setInterval(() => {
      setActiveStep((step) => Math.min(step + 1, LOADING_STEPS.length - 1));
    }, 9000);

    try {
      const response = await fetch(`${API_BASE_URL}/api/risk-analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: cleaned,
          provider: "gemini",
          query_analysis: "auto",
          include_prompt: false,
        }),
      });

      const body = await response.json();
      if (!response.ok) {
        throw new Error(formatApiError(body));
      }

      setResult(body);
      window.setTimeout(() => {
        resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } catch (requestError) {
      setError(requestError.message || "분석 중 오류가 발생했습니다.");
    } finally {
      window.clearInterval(stepTimer);
      setIsLoading(false);
      setActiveStep(-1);
    }
  }

  return (
    <main className="app-shell">
      <section className="query-panel">
        <div className="brand-lockup">
          <p className="eyebrow">공정거래 위반 가능성 사전진단</p>
          <h1>FairCheck</h1>
          <p className="lead">
            거래 상황을 입력하면 유사 공정위 의결서와 관련 법령을 근거로
            <br />
            위반 가능성을 사전 점검합니다.
          </p>
        </div>

        <form className="query-form" onSubmit={handleSubmit}>
          <label htmlFor="question">상황 입력</label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="예: 커피 가맹점주인데 계약서상 보장된 영업지역 100m 안에 본사가 직영점을 열겠다고 합니다."
            rows={5}
          />
          <div className="form-footer">
            <p>내 입장, 상대방, 문제 행위, 계약 내용, 불이익, 기간·금액·거리 등을 함께 적어주세요.</p>
            <button type="submit" disabled={isLoading || !question.trim()}>
              {isLoading ? "진단 중" : "진단하기"}
            </button>
          </div>
        </form>

        <div className="example-row" aria-label="질문 예시">
          {EXAMPLE_QUESTIONS.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => setQuestion(example)}
              disabled={isLoading}
            >
              {example}
            </button>
          ))}
        </div>
      </section>

      {error && <div className="error-banner">{error}</div>}
      {isLoading && <LoadingTrace activeStep={activeStep} />}

      <section ref={resultRef} className="result-region">
        {result && <RiskResult result={result} />}
      </section>
    </main>
  );
}

function LoadingTrace({ activeStep }) {
  return (
    <section className="trace-panel" aria-live="polite">
      <div className="trace-header">
        <span className="spinner" />
        <div>
          <h2>분석 중입니다</h2>
          <p>요청이 처리되는 동안 FairCheck의 분석 순서를 안내합니다.</p>
        </div>
      </div>
      <ol>
        {LOADING_STEPS.map((step, index) => (
          <li
            key={step}
            className={index <= activeStep ? "active" : ""}
          >
            <span>{index + 1}</span>
            {step}
          </li>
        ))}
      </ol>
    </section>
  );
}

function RiskResult({ result }) {
  const riskLevel = extractRiskLevel(result.answer);
  const [activeStatute, setActiveStatute] = useState(null);
  const additionalInfoItems = buildAdditionalInfoItems(result);

  return (
    <article className="result-layout">
      <header className="result-header">
        <div>
          <h2>진단 결과</h2>
          {result.analysis?.interpreted_issue && (
            <p className="result-topic">{result.analysis.interpreted_issue}</p>
          )}
        </div>
      </header>

      {result.search_query && (
        <p className="search-query">검색 질의: {result.search_query}</p>
      )}

      <div className="result-grid">
        <section className="answer-card">
          <AnswerText
            text={result.answer || ""}
            result={result}
            onOpenStatute={setActiveStatute}
            riskLevel={riskLevel}
          />
        </section>

        <aside className="evidence-stack">
          <EvidenceSection
            title="핵심 의결서"
            emptyText="답변에서 직접 인용된 의결서가 없습니다."
            items={result.used_decision_references}
            type="decision"
          />
          <EvidenceSection
            title="핵심 법령"
            emptyText="답변에서 직접 인용된 법령이 없습니다."
            items={result.used_statute_references}
            type="statute"
            onOpenStatute={setActiveStatute}
          />
        </aside>
      </div>

      {additionalInfoItems.length > 0 && (
        <section className="plain-panel additional-info-card">
          <h3>➕ 질의에 추가하면 좋은 정보</h3>
          <ul className="analysis-info-list">
            {additionalInfoItems.map((fact) => (
              <li key={fact}>{fact}</li>
            ))}
          </ul>
          {result.analysis?.suggested_question && (
            <p className="suggested-question">{result.analysis.suggested_question}</p>
          )}
        </section>
      )}

      <p className="result-disclaimer">
        ⚠️ 이 서비스는 공정위 의결서와 보유 법령 DB를 바탕으로 한 사전 점검 도구이며,
        최종 법률 판단이나 법률 자문이 아닙니다. ⚠️
      </p>

      <AdditionalEvidence result={result} onOpenStatute={setActiveStatute} />
      {activeStatute && (
        <StatuteModal statute={activeStatute} onClose={() => setActiveStatute(null)} />
      )}
    </article>
  );
}

function AnswerText({ text, result, onOpenStatute, riskLevel }) {
  const parsed = useMemo(() => parseAnalysisAnswer(text), [text]);
  return (
    <AnalysisCards
      parsed={parsed}
      result={result}
      onOpenStatute={onOpenStatute}
      riskLevel={riskLevel}
    />
  );
}

function AnalysisCards({ parsed, result, onOpenStatute, riskLevel }) {
  const decisionReferences = collectReferences(result, "decision");
  const statuteReferences = collectReferences(result, "statute");
  const decisionItems = parsed.coreItems.filter((item) => item.kind === "decision");
  const statuteItems = parsed.coreItems.filter((item) => item.kind === "statute");
  const hasCoreItems = decisionItems.length > 0 || parsed.coreNotes.length > 0;

  return (
    <section className="analysis-card analysis-answer-card">
      {riskLevel && (
        <div className="analysis-risk-row">
          <span className={`risk-badge ${riskLevelClass(riskLevel)}`}>
            {riskBadgeLabel(riskLevel)}
          </span>
        </div>
      )}

      {parsed.summary.length > 0 && (
        <AnalysisSection>
          <p className="analysis-section-title">진단 요약</p>
          {parsed.summary.map((paragraph) => (
            <p key={paragraph}>{paragraph}</p>
          ))}
        </AnalysisSection>
      )}

      {hasCoreItems && (
        <AnalysisSection>
          <p className="analysis-section-title">유사한 의결서</p>

          {decisionItems.map((item, index) => (
            <DecisionAnalysisBlock
              key={`decision-${item.referenceId || item.title || index}`}
              item={item}
              reference={findDecisionReference(item, decisionReferences)}
            />
          ))}

          {parsed.coreNotes.map((note) => (
            <p key={note} className="analysis-note">{note}</p>
          ))}
        </AnalysisSection>
      )}

      {statuteItems.length > 0 && (
        <AnalysisSection>
          <p className="analysis-section-title">관련 법령</p>
          {statuteItems.map((item, index) => (
            <StatuteAnalysisBlock
              key={`statute-${item.referenceId || item.title || index}`}
              item={item}
              reference={findStatuteReference(item, statuteReferences)}
              onOpenStatute={onOpenStatute}
            />
          ))}
        </AnalysisSection>
      )}
    </section>
  );
}

function AnalysisSection({ children }) {
  return (
    <section className="analysis-section">
      {children}
    </section>
  );
}

function DecisionAnalysisBlock({ item, reference }) {
  const title = item.title || reference?.title || "의결서명 확인 필요";
  const href = reference?.pdf_source ? decisionPdfUrl(reference.pdf_source) : "";
  const decisionMeta = [item.violationType || reference?.violation_type, reference?.industry]
    .filter(Boolean)
    .join(" · ");

  return (
    <article className="analysis-evidence-block">
      <div className="analysis-field">
        <p className="analysis-label">의결서명</p>
        {href ? (
          <a
            className="analysis-title-link"
            href={href}
            target="_blank"
            rel="noreferrer"
            title="의결서 PDF 원문 열기"
          >
            {title}
          </a>
        ) : (
          <p className="analysis-title-text">{title}</p>
        )}
        {decisionMeta && (
          <p className="analysis-decision-meta">{decisionMeta}</p>
        )}
      </div>

      {item.reason && (
        <div className="analysis-field analysis-reason-field">
          <p>{item.reason}</p>
        </div>
      )}
    </article>
  );
}

function StatuteAnalysisBlock({ item, reference, onOpenStatute }) {
  const title = item.title || statuteReferenceTitle(reference) || "관련 법령 확인 필요";
  const content = item.content || cleanStatuteContent(reference?.content || "");
  const connection = item.connection;

  return (
    <article className="analysis-evidence-block">
      <div className="analysis-field">
        <p className="analysis-label">법령명</p>
        {reference ? (
          <button
            type="button"
            className="analysis-title-link analysis-title-button"
            onClick={() => onOpenStatute?.(reference)}
            title="법령 원문 보기"
          >
            {title}
          </button>
        ) : (
          <p className="analysis-title-text statute-title-text">{title}</p>
        )}
      </div>

      {content && (
        <div className="statute-emphasis-box">
          {content}
        </div>
      )}

      {connection && (
        <div className="analysis-field analysis-connection-field">
          <p className="analysis-section-title">사용자 상황과의 연결</p>
          <p>{connection}</p>
        </div>
      )}

    </article>
  );
}

function parseAnalysisAnswer(answer = "") {
  const parsed = {
    summary: [],
    coreItems: [],
    coreNotes: [],
  };
  let section = "";
  let currentItem = null;
  let currentField = "";

  for (const rawLine of answer.split("\n")) {
    const line = rawLine.trim();
    if (!line || shouldHideAnswerLine(line)) continue;

    if (/^1\.\s*진단\s*요약/.test(line)) {
      section = "summary";
      currentItem = null;
      currentField = "";
      continue;
    }

    if (/^2\.\s*핵심\s*근거/.test(line)) {
      section = "core";
      currentItem = null;
      currentField = "";
      continue;
    }

    if (/^3\.\s*확인\s*필요\s*사항/.test(line)) {
      section = "confirmation";
      currentItem = null;
      currentField = "";
      continue;
    }

    if (/^\d+\.\s/.test(line)) {
      section = "other";
      currentItem = null;
      currentField = "";
      continue;
    }

    if (section === "summary") {
      const summaryText = summaryLineText(line);
      if (summaryText) parsed.summary.push(summaryText);
      continue;
    }

    if (section !== "core") continue;
    if (isReferenceOnlyLine(line)) continue;

    const labeled = parseLabeledLine(line);
    if (labeled) {
      const label = normalizeLabel(labeled.label);
      const value = cleanAnswerText(labeled.value);
      const referenceId = extractPrimaryReferenceId(labeled.value);

      if (label === "의결서명") {
        currentItem = {
          kind: "decision",
          title: value,
          referenceId,
          violationType: "",
          reason: "",
        };
        parsed.coreItems.push(currentItem);
        currentField = "title";
        continue;
      }

      if (label === "관련법령") {
        currentItem = {
          kind: "statute",
          title: value,
          referenceId,
          content: "",
          connection: "",
          extraCheck: "",
        };
        parsed.coreItems.push(currentItem);
        currentField = "title";
        continue;
      }

      if (currentItem?.kind === "decision" && label === "위반유형") {
        currentItem.violationType = value;
        currentField = "violationType";
        continue;
      }

      if (currentItem?.kind === "decision" && label === "핵심근거") {
        currentItem.reason = value;
        currentField = "reason";
        continue;
      }

      if (currentItem?.kind === "statute" && label === "법내용") {
        currentItem.content = value;
        currentField = "content";
        continue;
      }

      if (currentItem?.kind === "statute" && label === "사용자상황과의연결") {
        currentItem.connection = value;
        currentField = "connection";
        continue;
      }

      if (currentItem?.kind === "statute" && label === "추가확인필요") {
        currentItem.extraCheck = value;
        currentField = "extraCheck";
        continue;
      }
    }

    const continuation = cleanAnswerText(line);
    if (!continuation) continue;

    if (currentItem && currentField && currentField !== "title") {
      currentItem[currentField] = appendSentence(currentItem[currentField], continuation);
    } else {
      parsed.coreNotes.push(continuation);
    }
  }

  parsed.summary = uniqueTexts(parsed.summary);
  parsed.coreNotes = uniqueTexts(parsed.coreNotes);
  parsed.coreItems = parsed.coreItems.filter((item) => {
    if (item.kind === "decision") return item.title || item.violationType || item.reason;
    return item.title || item.content || item.connection;
  });

  return parsed;
}

function summaryLineText(line) {
  const labeled = parseLabeledLine(line);
  if (labeled && normalizeLabel(labeled.label) === "위반가능성") return "";
  return cleanAnswerText(line);
}

function parseLabeledLine(line = "") {
  const withoutBullet = line.replace(/^\s*[-*]\s*/, "").trim();
  const match = withoutBullet.match(/^([^:：]+?)\s*[:：]\s*(.+)$/);
  if (!match) return null;
  return {
    label: match[1],
    value: match[2],
  };
}

function normalizeLabel(label = "") {
  const compact = label.replace(/\s+/g, "");
  if (compact.includes("의결서명")) return "의결서명";
  if (compact.includes("관련법령")) return "관련법령";
  if (compact.includes("위반유형")) return "위반유형";
  if (compact.includes("핵심근거") || compact.includes("관련근거")) return "핵심근거";
  if (compact.includes("법내용") || compact.includes("원문")) return "법내용";
  if (
    compact.includes("사용자상황") ||
    compact.includes("상황과의연결") ||
    compact.includes("유사점")
  ) {
    return "사용자상황과의연결";
  }
  if (compact.includes("추가확인") || compact.includes("확인필요")) return "추가확인필요";
  if (compact.includes("위반가능성")) return "위반가능성";
  return compact;
}

function cleanAnswerText(value = "") {
  return value
    .replace(/\*\*/g, "")
    .replace(/^#{1,6}\s*/, "")
    .replace(/^\s*[-*]\s*/, "")
    .replace(/^\d+\.\s*/, "")
    .replace(/\[(?:문서|법령)[^\]]+\]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function appendSentence(current = "", next = "") {
  if (!current) return next;
  if (!next) return current;
  return `${current} ${next}`;
}

function uniqueTexts(items = []) {
  return [...new Set(items.map((item) => cleanAnswerText(item)).filter(Boolean))];
}

function collectReferences(result, type) {
  const keys = type === "decision"
    ? ["used_decision_references", "additional_decision_references", "decision_references"]
    : ["used_statute_references", "additional_statute_references", "statute_references"];
  const references = [];
  const seen = new Set();

  for (const key of keys) {
    for (const reference of result?.[key] || []) {
      const id = reference.reference_id || JSON.stringify(reference);
      if (seen.has(id)) continue;
      seen.add(id);
      references.push(reference);
    }
  }

  return references;
}

function findDecisionReference(item, references = []) {
  if (item.referenceId) {
    const byId = references.find((reference) => reference.reference_id === item.referenceId);
    if (byId) return byId;
  }

  const title = normalizeSearchText(item.title);
  if (!title) return null;

  return references.find((reference) => {
    const referenceTitle = normalizeSearchText(reference.title);
    return referenceTitle && (title.includes(referenceTitle) || referenceTitle.includes(title));
  }) || null;
}

function findStatuteReference(item, references = []) {
  if (item.referenceId) {
    const byId = references.find((reference) => reference.reference_id === item.referenceId);
    if (byId) return byId;
  }

  const title = normalizeSearchText(item.title);
  if (!title) return null;

  return references.find((reference) => {
    const candidates = statuteTitleCandidates(
      reference.law_title || "",
      reference.jo_number || "",
      reference.jo_title || ""
    ).map(normalizeSearchText);

    return candidates.some((candidate) => (
      candidate && (title.includes(candidate) || candidate.includes(title))
    ));
  }) || null;
}

function statuteReferenceTitle(reference) {
  if (!reference) return "";
  const lawTitle = reference.law_title || "";
  const joNumber = reference.jo_number || "";
  const joTitle = reference.jo_title || "";
  const joPart = joNumber && joTitle ? `${joNumber}(${joTitle})` : [joNumber, joTitle].filter(Boolean).join(" ");
  return [lawTitle, joPart].filter(Boolean).join(" ");
}

function normalizeSearchText(value = "") {
  return cleanAnswerText(value).replace(/\s+/g, "");
}

function extractPrimaryReferenceId(value = "") {
  const match = value.match(/\[((?:문서|법령)\s*\d+)/);
  return match ? match[1].replace(/\s+/g, " ") : "";
}

function linkAnswerReferences(text, result, onOpenStatute) {
  if (!text) return null;

  const decisionPdfById = new Map(
    (result?.decision_references || [])
      .filter((reference) => reference.pdf_source)
      .map((reference) => [reference.reference_id, reference.pdf_source])
  );
  const answerLinks = (result?.decision_references || [])
    .filter((reference) => reference.title && reference.pdf_source)
    .map((reference) => ({
      text: reference.title,
      href: decisionPdfUrl(reference.pdf_source),
      external: true,
      title: "의결서 PDF 원문 열기",
    }))
    .concat(statuteAnswerLinks(result?.statute_references || [], onOpenStatute))
    .sort((left, right) => right.text.length - left.text.length);

  return answerDisplayLines(text).map((line, lineIndex) => {
    const displayLine = decorateRiskLevelText(line);
    if (isReferenceOnlyLine(displayLine)) return null;

    const pieces = renderAnswerLine(displayLine, lineIndex, decisionPdfById, answerLinks);
    const lineClass = answerLineClass(displayLine);

    if (lineClass === "answer-section-title") {
      return (
        <h4 key={lineIndex} className={lineClass}>
          {pieces.length ? pieces : "\u00A0"}
        </h4>
      );
    }

    return (
      <p key={lineIndex} className={lineClass}>
        {pieces.length ? pieces : "\u00A0"}
      </p>
    );
  });
}

function statuteAnswerLinks(statuteReferences, onOpenStatute) {
  const links = [];

  for (const reference of statuteReferences) {
    const lawTitle = reference.law_title || "";
    const joNumber = reference.jo_number || "";
    const joTitle = reference.jo_title || "";
    const href = `#${referenceDomId(reference.reference_id)}`;
    const candidates = statuteTitleCandidates(lawTitle, joNumber, joTitle);

    for (const text of candidates) {
      links.push({
        text,
        href,
        external: false,
        onClick: () => onOpenStatute?.(reference),
        title: "법령 근거 보기",
      });
    }
  }

  return links;
}

function statuteTitleCandidates(lawTitle, joNumber, joTitle) {
  const joNumbers = joNumberVariants(joNumber);
  const candidates = [
    joTitle || "",
  ];

  for (const number of joNumbers) {
    candidates.push(
      [lawTitle, number && joTitle && `${number}(${joTitle})`].filter(Boolean).join(" "),
      [lawTitle, number && joTitle && `${number} (${joTitle})`].filter(Boolean).join(" "),
      [lawTitle, number].filter(Boolean).join(" "),
      lawTitle && number && joTitle ? `${lawTitle}${number}(${joTitle})` : "",
      lawTitle && number ? `${lawTitle}${number}` : "",
      number && joTitle ? `${number}(${joTitle})` : "",
      number && joTitle ? `${number} (${joTitle})` : ""
    );
  }

  return [...new Set(candidates.filter(Boolean))];
}

function joNumberVariants(joNumber = "") {
  const compact = joNumber.replace(/\s+/g, "");
  const spaced = compact.replace(/조의(\d+)/, "조의 $1");
  return [...new Set([joNumber, compact, spaced].filter(Boolean))];
}

function renderAnswerLine(line, lineIndex, decisionPdfById, answerLinks) {
  const pieces = [];
  const referencePattern = /\[(문서\s*\d+(?:\s*-\s*근거\s*\d+)?|법령\s*\d+)\]/g;
  let cursor = 0;
  let pieceIndex = 0;

  while (cursor < line.length) {
    referencePattern.lastIndex = cursor;
    const referenceMatch = referencePattern.exec(line);
    const titleMatch = findNextAnswerLink(line, cursor, answerLinks);

    const nextReferenceIndex = referenceMatch?.index ?? Number.POSITIVE_INFINITY;
    const nextTitleIndex = titleMatch?.index ?? Number.POSITIVE_INFINITY;
    const nextIndex = Math.min(nextReferenceIndex, nextTitleIndex);

    if (!Number.isFinite(nextIndex)) {
      pieces.push(line.slice(cursor));
      break;
    }

    if (nextIndex > cursor) {
      pieces.push(line.slice(cursor, nextIndex));
    }

    if (nextTitleIndex <= nextReferenceIndex && titleMatch) {
      const isButtonLink = typeof titleMatch.onClick === "function";
      pieces.push(
        isButtonLink ? (
          <button
            key={`${lineIndex}-title-${pieceIndex}`}
            type="button"
            className="inline-ref title-inline-ref inline-ref-button"
            onClick={titleMatch.onClick}
            title={titleMatch.title}
          >
            {titleMatch.text}
          </button>
        ) : (
          <a
            key={`${lineIndex}-title-${pieceIndex}`}
            href={titleMatch.href}
            className="inline-ref title-inline-ref"
            target={titleMatch.external ? "_blank" : undefined}
            rel={titleMatch.external ? "noreferrer" : undefined}
            title={titleMatch.title}
          >
            {titleMatch.text}
          </a>
        )
      );
      cursor = titleMatch.index + titleMatch.text.length;
    } else if (referenceMatch) {
      pieces.push(
        renderReferenceLink(
          referenceMatch[0],
          referenceMatch[1],
          decisionPdfById,
          `${lineIndex}-ref-${pieceIndex}`
        )
      );
      cursor = referencePattern.lastIndex;
    }

    pieceIndex += 1;
  }

  return pieces;
}

function findNextAnswerLink(line, startIndex, answerLinks) {
  let bestMatch = null;

  for (const link of answerLinks) {
    const index = line.indexOf(link.text, startIndex);
    if (index === -1) continue;
    if (!bestMatch || index < bestMatch.index) {
      bestMatch = { ...link, index };
    }
  }

  return bestMatch;
}

function renderReferenceLink(label, referenceText, decisionPdfById, key) {
  const normalizedReference = referenceText.replace(/\s+/g, " ").trim();
  const decisionMatch = normalizedReference.match(/^문서\s*(\d+)/);
  const decisionId = decisionMatch ? `문서 ${decisionMatch[1]}` : "";
  const pdfSource = decisionPdfById.get(decisionId);

  // 사용자 화면에서는 [문서 1], [법령 1] 같은 내부 근거번호를 숨깁니다.
  // API 응답에는 그대로 남아 있어 디버깅과 근거 매칭에 계속 사용할 수 있습니다.
  if (!pdfSource) return null;

  if (pdfSource) {
    return null;
  }

  return null;
}

function answerDisplayLines(text) {
  const lines = text.split("\n");
  const visibleLines = [];
  let skippingConfirmationSection = false;

  for (const line of lines) {
    const trimmed = line.trim();

    if (/^3\.\s*확인\s*필요\s*사항/.test(trimmed)) {
      skippingConfirmationSection = true;
      continue;
    }

    if (/^\d+\.\s/.test(trimmed)) {
      skippingConfirmationSection = false;
    }

    if (skippingConfirmationSection || shouldHideAnswerLine(line)) {
      continue;
    }

    visibleLines.push(line);
  }

  return visibleLines;
}

function isReferenceOnlyLine(line) {
  return /^\s*-\s*근거\s*번호\s*:\s*(\[(?:문서\s*\d+(?:\s*-\s*근거\s*\d+)?|법령\s*\d+)\][,\s]*)+\s*$/.test(
    line
  );
}

function shouldHideAnswerLine(line) {
  const trimmed = line.trim();
  return (
    /^4\.\s*주의/.test(trimmed) ||
    /최종\s*법률\s*판단|법률\s*자문/.test(trimmed)
  );
}

function answerLineClass(line) {
  const trimmed = line.trim();
  if (!trimmed) return "blank-line";
  if (/^\d+\.\s/.test(trimmed)) return "answer-section-title";
  if (/^-\s*(위반 가능성|의결서명|관련 법령|법 내용|사용자 상황|추가 확인|최종 판단|핵심 근거|위반유형)/.test(trimmed)) {
    return "answer-key-line";
  }
  if (/^\s*-\s/.test(trimmed)) return "answer-bullet-line";
  return "";
}

function EvidenceSection({ title, emptyText, items = [], type, onOpenStatute }) {
  return (
    <section className="evidence-panel">
      <h3>{title}</h3>
      {items.length === 0 ? (
        <p className="empty-state">{emptyText}</p>
      ) : (
        <div className="card-list">
          {items.map((item) => (
            <EvidenceCard
              key={`${type}-${item.reference_id}`}
              item={item}
              type={type}
              onOpenStatute={onOpenStatute}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function EvidenceCard({ item, type, onOpenStatute }) {
  const title = type === "decision"
    ? item.title || item.reference_id
    : `${item.law_title || ""} ${item.jo_number || ""}`.trim();
  const subtitle = type === "decision"
    ? [item.violation_type, item.industry].filter(Boolean).join(" · ")
    : item.jo_title;

  return (
    <article className="evidence-card" id={referenceDomId(item.reference_id)}>
      {type === "decision" && item.pdf_source ? (
        <h4>
          <a
            className="title-link"
            href={decisionPdfUrl(item.pdf_source)}
            target="_blank"
            rel="noreferrer"
            title="의결서 PDF 원문 열기"
          >
            {title}
          </a>
        </h4>
      ) : (
        <h4>
          {type === "statute" && item.content ? (
            <button
              type="button"
              className="title-link title-button"
              onClick={() => onOpenStatute?.(item)}
              title="법령 원문 보기"
            >
              {title}
            </button>
          ) : (
            title
          )}
        </h4>
      )}
      {subtitle && <p>{subtitle}</p>}
      {type === "statute" && item.content && (
        <p className="statute-excerpt">원문 일부: {statuteExcerpt(item.content)}</p>
      )}
    </article>
  );
}

function AdditionalEvidence({ result, onOpenStatute }) {
  const additionalCount =
    (result.additional_decision_references?.length || 0) +
    (result.additional_statute_references?.length || 0);

  if (additionalCount === 0) return null;

  return (
    <details className="additional-panel">
      <summary>추가로 검토된 유사 근거 {additionalCount}개</summary>
      <div className="additional-grid">
        <EvidenceSection
          title="의결서"
          emptyText="추가 의결서가 없습니다."
          items={result.additional_decision_references}
          type="decision"
        />
        <EvidenceSection
          title="법령"
          emptyText="추가 법령이 없습니다."
          items={result.additional_statute_references}
          type="statute"
          onOpenStatute={onOpenStatute}
        />
      </div>
    </details>
  );
}

function StatuteModal({ statute, onClose }) {
  const title = [statute.law_title, statute.jo_number].filter(Boolean).join(" ");

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section
        className="statute-modal"
        role="dialog"
        aria-modal="true"
        aria-label="법령 원문"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <p className="eyebrow">{statute.reference_id}</p>
            <h3>{title || "법령 원문"}</h3>
            {statute.jo_title && <p>{statute.jo_title}</p>}
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="닫기">
            닫기
          </button>
        </div>
        <div className="modal-body">
          <pre>{cleanStatuteContent(statute.content || "법령 원문이 없습니다.")}</pre>
        </div>
      </section>
    </div>
  );
}

function referenceDomId(referenceId = "") {
  return `ref-${referenceId.replace(/\s+/g, "-").replace(/[^0-9A-Za-z가-힣-]/g, "")}`;
}

function decisionPdfUrl(pdfSource = "") {
  return `${API_BASE_URL}/api/decision-pdfs/${encodeURIComponent(pdfSource)}`;
}

function cleanStatuteContent(content = "") {
  return content
    .replace(/^법률명:\s*.*$/gm, "")
    .replace(/^조문번호:\s*.*$/gm, "")
    .replace(/^조문제목:\s*.*$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function buildAdditionalInfoItems(result) {
  const items = [
    ...extractConfirmationItems(result.answer || ""),
    ...extractAdditionalCheckItems(result.answer || ""),
    ...(result.analysis?.missing_facts || []),
  ];

  return [...new Set(items.map((item) => cleanAnswerText(item)).filter(Boolean))];
}

function extractConfirmationItems(answer = "") {
  const lines = answer.split("\n");
  const items = [];
  let inSection = false;
  let currentItem = "";

  for (const line of lines) {
    const trimmed = line.trim();
    if (/^3\.\s*확인\s*필요\s*사항/.test(trimmed)) {
      inSection = true;
      continue;
    }

    if (inSection && /^\d+\.\s/.test(trimmed)) {
      break;
    }

    if (!inSection || !trimmed) {
      continue;
    }

    if (/^-\s+/.test(trimmed)) {
      if (currentItem) items.push(currentItem);
      currentItem = cleanAnswerText(trimmed.replace(/^-\s+/, ""));
    } else if (currentItem) {
      currentItem = appendSentence(currentItem, cleanAnswerText(trimmed));
    }
  }

  if (currentItem) items.push(currentItem);
  return items;
}

function extractAdditionalCheckItems(answer = "") {
  const items = [];

  for (const rawLine of answer.split("\n")) {
    const line = rawLine.trim();
    const labeled = parseLabeledLine(line);
    if (!labeled || normalizeLabel(labeled.label) !== "추가확인필요") continue;
    const item = cleanAnswerText(labeled.value);
    if (item) items.push(item);
  }

  return items;
}

function decorateRiskLevelText(line) {
  return line.replace(
    /(위반\s*가능성\s*:\s*)(🔴|🟡|🔵|💬)?\s*(높음|중간|낮음|판단\s*보류|판단보류)/,
    (_, prefix, _existingIcon, level) => `${prefix}${riskLevelLabel(level)}`
  );
}

function statuteExcerpt(content = "") {
  const cleaned = cleanStatuteContent(content).replace(/\s+/g, " ").trim();
  if (cleaned.length <= 120) return cleaned;
  return `${cleaned.slice(0, 120)}...`;
}

function getStatusLabel(resultType) {
  if (resultType === "risk_analysis") return "진단 가능";
  if (resultType === "needs_clarification") return "추가 정보 필요";
  return "서비스 범위 밖";
}

function extractRiskLevel(answer = "") {
  const match = answer.match(/위반\s*가능성\s*:\s*(?:🔴|🟡|🔵|💬)?\s*(높음|중간|낮음|판단\s*보류|판단보류)/);
  return normalizeRiskLevel(match?.[1] || "");
}

function riskLevelClass(riskLevel = "") {
  const normalized = normalizeRiskLevel(riskLevel);
  if (normalized === "높음") return "high";
  if (normalized === "중간") return "medium";
  if (normalized === "낮음") return "low";
  return "hold";
}

function riskLevelLabel(riskLevel = "") {
  const normalized = normalizeRiskLevel(riskLevel);
  if (normalized === "높음") return "🔴 높음";
  if (normalized === "중간") return "🟡 중간";
  if (normalized === "낮음") return "🔵 낮음";
  return "💬 판단보류";
}

function riskBadgeLabel(riskLevel = "") {
  const normalized = normalizeRiskLevel(riskLevel);
  if (normalized === "높음") return "🔴 위반 가능성 높음";
  if (normalized === "중간") return "🟡 위반 가능성 중간";
  if (normalized === "낮음") return "🔵 위반 가능성 낮음";
  return "💬 판단보류";
}

function normalizeRiskLevel(riskLevel = "") {
  const compact = riskLevel.replace(/\s+/g, "");
  if (compact === "높음") return "높음";
  if (compact === "중간") return "중간";
  if (compact === "낮음") return "낮음";
  return "판단보류";
}

function formatApiError(body) {
  if (Array.isArray(body?.detail)) {
    return body.detail.map((item) => item.msg).join("\n");
  }
  if (typeof body?.detail === "string") return body.detail;
  return "요청을 처리하지 못했습니다.";
}

createRoot(document.getElementById("root")).render(<App />);
