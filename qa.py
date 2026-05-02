from langchain_community.vectorstores import FAISS
import re
import json
import os
from difflib import SequenceMatcher
from functools import lru_cache
from embeddings_provider import LocalEmbeddings

try:
    import google.generativeai as genai
except ImportError:
    genai = None


JSON_QA_PATH = os.path.join("data", "qa_data.json")


def resolve_rupp_dataset_path():
    """Prefer the curated 500-sample dataset when available."""
    preferred = os.path.join("data", "rupp_dataset_500.json")
    fallback = os.path.join("data", "rupp_dataset.json")
    if os.path.exists(preferred):
        return preferred
    return fallback


def normalize_text(text):
    """Normalize user and stored text for robust matching."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s\u1780-\u17FF]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


EN_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "how", "in", "is",
    "it", "of", "on", "or", "the", "to", "what", "when", "where", "which", "who",
    "why", "with", "your", "you", "me", "my", "we", "our", "us", "can", "could",
    "would", "should", "tell", "about", "please", "rupp", "much", "many", "does",
}

EN_GREETINGS = {
    "hi", "hello", "hey", "yo", "sup", "morning", "afternoon", "evening",
    "good morning", "good afternoon", "good evening",
}

KM_GREETINGS = {
    "សួស្តី", "ជំរាបសួរ", "ជម្រាបសួរ", "ជំរាបលា",
}

KM_FEE_MARKERS = {"ថ្លៃសិក្សា", "តម្លៃប៉ុន្មាន", "ប៉ុន្មាន", "ថ្លៃ", "សិក្សា", "តម្លៃ"}
EN_FEE_MARKERS = {"fee", "fees", "tuition", "cost", "price"}

INTENT_TERMS = {
    "fee", "fees", "tuition", "cost", "price",
    "major", "majors", "faculty", "department", "program", "course",
    "location", "located", "address",
}

# -----------------------------------------------------------------------
# TERM_ALIASES: expand short/informal user terms into dataset-friendly ones
# -----------------------------------------------------------------------
TERM_ALIASES = {
    # Abbreviations
    "dse": "data science and engineering",
    "cs": "computer science",
    "math": "mathematics",
    "maths": "mathematics",
    "mathematic": "mathematics",
    "ite": "it engineering",
    "network": "telecommunication network",
    "networking": "telecommunication network",
    "uni": "university",
    "loc": "location",
    # Fee synonyms
    "tuition": "fee",
    "cost": "fee",
    "price": "fee",
    # Admission synonyms
    "enroll": "admission",
    "enrol": "admission",
    "enrollment": "admission",
    "admission": "application",
    "apply": "application",
    "register": "application",
    "registration": "application",
    "entran": "entrance",
    # ---- NEW: subject/major name variants ----
    "chemical": "chemistry",
    "chem": "chemistry",
    "bio": "biology",
    "biological": "biology",
    "phys": "physics",
    "physic": "physics",
    "econ": "economics",
    "economic": "economics",
    "socio": "sociology",
    "sociological": "sociology",
    "geo": "geography",
    "geographical": "geography",
    "phil": "philosophy",
    "philosophical": "philosophy",
    "hist": "history",
    "historical": "history",
    "law": "law",
    "legal": "law",
    "mgmt": "management",
    "acct": "accounting",
    "fin": "finance",
    "financial": "finance",
    "it": "information technology",
    "ict": "information technology",
    "eng": "engineering",
    "enviro": "environmental science",
    "environ": "environmental science",
    "environmental": "environmental science",
    "arch": "architecture",
    "civil": "civil engineering",
    "mech": "mechanical engineering",
    "elec": "electrical engineering",
    "agri": "agriculture",
    "agricultural": "agriculture",
    "edu": "education",
    "pedagogy": "education",
    "khmer": "khmer literature",
    "literature": "khmer literature",
    "french": "french language",
    "english": "english language",
    "russian": "russian language",
    "chinese": "chinese language",
    "japanese": "japanese language",
    "korean": "korean language",
    "tourism": "tourism management",
    "hotel": "hotel management",
    "media": "media and communication",
    "journalism": "media and communication",
    "public admin": "public administration",
    "pol sci": "political science",
    "politic": "political science",
    "political": "political science",
    "psychology": "psychology",
    "psych": "psychology",
    "statistics": "statistics",
    "stat": "statistics",
    "actuarial": "actuarial science",
}

# -----------------------------------------------------------------------
# KM_MAJOR_ALIASES: Khmer informal → canonical Khmer major name
# -----------------------------------------------------------------------
KM_MAJOR_ALIASES = {
    # Existing
    "គីមី": "គីមីវិទ្យា",
    "គណិត": "គណិតវិទ្យា",
    "សង្គម": "សង្គមវិទ្យា",
    "ជីវ": "ជីវវិទ្យា",
    "រូប": "រូបវិទ្យា",
    # NEW: Computer Science variants
    "វិទ្យាសាស្ត្រកំព្យូទ័រ": "computer science",   # maps to EN for lookup
    "វិទ្យាសាស្ត្រ​កំព្យូទ័រ": "computer science",
    "សាស្ត្រកំព្យូទ័រ": "computer science",
    "កំព្យូទ័រ": "computer science",
    # IT Engineering
    "វិស្វកម្មព័ត៌មានវិទ្យា": "it engineering",
    "ព័ត៌មានវិទ្យា": "information technology",
    # History
    "ប្រវត្តិ": "ប្រវត្តិវិទ្យា",
    "ប្រវតិ": "ប្រវត្តិវិទ្យា",
    # Geography
    "ភូគោល": "ភូគោលវិទ្យា",
    # Economics
    "សេដ្ឋកិច្ច": "សេដ្ឋកិច្ច",
    # Law
    "ច្បាប់": "ច្បាប់",
    # Environment
    "បរិស្ថាន": "បរិស្ថានវិទ្យា",
    # Math shorthand
    "គណិតសាស្ត្រ": "គណិតវិទ្យា",
}

# Mapping from Khmer aliases that resolve to English dataset keys
KM_TO_EN_MAP = {
    "វិទ្យាសាស្ត្រកំព្យូទ័រ": "computer science",
    "វិទ្យាសាស្ត្រ​កំព្យូទ័រ": "computer science",
    "សាស្ត្រកំព្យូទ័រ": "computer science",
    "កំព្យូទ័រ": "computer science",
    "វិស្វកម្មព័ត៌មានវិទ្យា": "it engineering",
    "ព័ត៌មានវិទ្យា": "information technology",
}


def extract_terms(text):
    """Extract meaningful terms and remove common stopwords."""
    normalized = normalize_text(text)
    for src, dst in TERM_ALIASES.items():
        normalized = re.sub(rf"\b{re.escape(src)}\b", dst, normalized)
    words = re.findall(r"\w+", normalized)
    return {w for w in words if len(w) > 2 and w not in EN_STOPWORDS}


def fuzzy_ratio(a, b):
    """Return fuzzy similarity score in [0,1]."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def extract_relevant_sentences(chunks, question, max_sentences=4):
    """Extract concise, question-aligned sentences from retrieved chunks."""
    q_terms = extract_terms(question)
    if not q_terms:
        q_terms = {w for w in normalize_text(question).split() if len(w) > 2}

    candidates = []
    for chunk in chunks:
        cleaned = clean_text(chunk)
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        for sent in parts:
            sent = sent.strip()
            if len(sent) < 35:
                continue
            sent_norm = normalize_text(sent)
            sent_terms = set(re.findall(r"\w+", sent_norm))
            overlap = len(q_terms & sent_terms)
            if overlap > 0:
                candidates.append((overlap, len(sent), sent))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1]))
    selected = []
    seen = set()
    for _, _, sent in candidates:
        sig = normalize_text(sent)
        if sig in seen:
            continue
        seen.add(sig)
        selected.append(sent)
        if len(selected) >= max_sentences:
            break

    if not selected:
        return None
    return " ".join(selected)


@lru_cache(maxsize=1)
def load_json_qa():
    if not os.path.exists(JSON_QA_PATH):
        return []

    with open(JSON_QA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        items = raw_data.get("qa", [])
    elif isinstance(raw_data, list):
        items = raw_data
    else:
        items = []

    normalized_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        keywords = item.get("keywords", [])
        if not question or not answer:
            continue
        if not isinstance(keywords, list):
            keywords = []
        normalized_items.append(
            {
                "question": question,
                "answer": answer,
                "keywords": [normalize_text(str(k)) for k in keywords],
            }
        )
    return normalized_items


@lru_cache(maxsize=1)
def load_rupp_dataset_qa():
    dataset_path = resolve_rupp_dataset_path()
    if not os.path.exists(dataset_path):
        return []

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    dataset = raw_data.get("dataset", []) if isinstance(raw_data, dict) else []

    normalized_items = []
    for item in dataset:
        if not isinstance(item, dict):
            continue
        question_en = str(item.get("question_en", "")).strip()
        question_kh = str(item.get("question_kh", "")).strip()
        answer_en = str(item.get("answer_en", "")).strip()
        answer_kh = str(item.get("answer_kh", "")).strip()
        category = str(item.get("category", "")).strip()
        if not (question_en or question_kh):
            continue
        if not (answer_en or answer_kh):
            continue
        normalized_items.append(
            {
                "question_en": question_en,
                "question_kh": question_kh,
                "answer_en": answer_en,
                "answer_kh": answer_kh,
                "category": category,
            }
        )
    return normalized_items


def contains_khmer(text):
    return any("\u1780" <= ch <= "\u17FF" for ch in text)


def detect_greeting(question):
    """
    Detect greetings robustly — but never fire if the message also contains
    an intent term (fee, major, location, etc.).
    """
    normalized = normalize_text(question)
    if not normalized:
        return False

    # If the question has any intent term, it is NOT a greeting.
    if any(marker in normalized for marker in EN_FEE_MARKERS):
        return False
    if any(marker in question for marker in KM_FEE_MARKERS):
        return False
    if any(term in normalized for term in INTENT_TERMS):
        return False
    # Check for major/subject words via aliases
    for alias in TERM_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return False

    if contains_khmer(question):
        return any(greet in question for greet in KM_GREETINGS)

    if normalized in EN_GREETINGS:
        return True
    if normalized.startswith(("hi ", "hello ", "hey ")):
        return True
    return len(normalized.split()) <= 3 and any(g in normalized for g in {"hi", "hello", "hey"})


def greeting_response(use_khmer):
    if use_khmer:
        return "RUPP Assistant:\nសួស្តី! ខ្ញុំជាជំនួយការ RUPP។ អ្នកអាចសួរអំពីថ្លៃសិក្សា ជំនាញ ទីតាំង ឬការចូលរៀន។"
    return "RUPP Assistant:\nHello! I am the RUPP Assistant. You can ask about tuition fees, majors, location, or admissions."


@lru_cache(maxsize=1)
def get_major_terms():
    """Extract known major names from fee questions in the dataset."""
    items = load_rupp_dataset_qa()
    en_terms = set()
    kh_terms = set()

    for item in items:
        if item.get("category") != "fees":
            continue

        q_en = normalize_text(item.get("question_en", ""))
        m_en = re.search(r"for\s+(.+?)\s+at\s+rupp", q_en)
        if m_en:
            term = m_en.group(1).strip()
            term = re.sub(r"^the\s+", "", term)
            term = re.sub(r"\s+major$", "", term)
            term = term.strip()
            if term:
                en_terms.add(term)

        q_kh = item.get("question_kh", "")
        m_kh = re.search(r"ជំនាញ\s*(.+?)\s*នៅ", q_kh)
        if m_kh:
            term = m_kh.group(1).strip()
            term = re.sub(r"^ផ្នែក\s*", "", term)
            if term:
                kh_terms.add(term)

    return en_terms, kh_terms


@lru_cache(maxsize=1)
def get_fee_answer_maps():
    """Build fast lookup maps for fee answers by major name."""
    en_map = {}
    kh_map = {}
    for item in load_rupp_dataset_qa():
        if item.get("category") != "fees":
            continue

        q_en = normalize_text(item.get("question_en", ""))
        q_kh = item.get("question_kh", "")

        m_en = re.search(r"for\s+(?:the\s+)?(.+?)\s+(?:major|program)\s+at\s+rupp", q_en)
        if m_en and item.get("answer_en"):
            major_en = re.sub(r"^the\s+", "", m_en.group(1).strip())
            if major_en:
                en_map[major_en] = item.get("answer_en")

        m_kh = re.search(r"ជំនាញ\s*(.+?)\s*នៅ", q_kh)
        if m_kh and item.get("answer_kh"):
            major_kh = re.sub(r"^ផ្នែក\s*", "", m_kh.group(1).strip())
            if major_kh:
                kh_map[major_kh] = item.get("answer_kh")

    return en_map, kh_map


def _match_it_engineering_fee(question, en_map):
    """Prefer the IT Engineering fee for IT/ITE queries before generic aliasing.

    The generic alias expansion can accidentally rewrite `IT Engineering` into a
    broader phrase that matches other engineering majors. This helper keeps the
    explicit IT/ITE path stable.
    """
    q_raw = question.lower()
    q_norm = normalize_text(question)

    if not re.search(r"\b(ite|it engineering)\b", q_raw) and not re.search(
        r"\b(it engineering|ite)\b", q_norm
    ):
        return None

    return en_map.get("it engineering") or en_map.get("information technology")


def _apply_term_aliases(text_norm):
    """Apply all TERM_ALIASES substitutions on a normalized string."""
    for src, dst in TERM_ALIASES.items():
        text_norm = re.sub(rf"\b{re.escape(src)}\b", dst, text_norm)
    return text_norm


def find_fee_answer_by_major(question):
    """
    Directly match fee questions to known majors.
    Handles short queries like 'history fee' or 'chemical fee'.
    """
    use_khmer = contains_khmer(question)
    question_norm = _apply_term_aliases(normalize_text(question))
    en_map, kh_map = get_fee_answer_maps()

    if not use_khmer:
        it_engineering_answer = _match_it_engineering_fee(question, en_map)
        if it_engineering_answer:
            return it_engineering_answer

    if use_khmer:
        # 1) Check KM_TO_EN_MAP first (Khmer phrases that map to English keys)
        for km_phrase, en_key in KM_TO_EN_MAP.items():
            if km_phrase in question:
                if en_key in en_map:
                    return en_map[en_key]

        # 2) Khmer alias → canonical Khmer key
        for alias, canonical in KM_MAJOR_ALIASES.items():
            if alias in question:
                if canonical in kh_map:
                    return kh_map[canonical]
                # Also try the canonical in en_map
                if canonical in en_map:
                    return en_map[canonical]

        # 3) Direct substring match in kh_map
        for major_kh, answer_kh in kh_map.items():
            if major_kh in question:
                return answer_kh

        # 4) Fuzzy match Khmer majors
        best_ratio = 0.0
        best_answer = None
        for major_kh, answer_kh in kh_map.items():
            r = fuzzy_ratio(question, major_kh)
            if r > best_ratio:
                best_ratio = r
                best_answer = answer_kh
        if best_ratio >= 0.6:
            return best_answer

        return None

    # ---- English path ----
    # Substring match after alias expansion
    for major_en, answer_en in en_map.items():
        if major_en in question_norm:
            return answer_en

    # Fuzzy match against en_map keys
    best_ratio = 0.0
    best_answer = None
    for major_en, answer_en in en_map.items():
        r = fuzzy_ratio(question_norm, major_en)
        # Also check if the major name appears as a substring of the question words
        q_words = set(question_norm.split())
        m_words = set(major_en.split())
        word_overlap = len(q_words & m_words) / max(len(m_words), 1)
        combined = max(r, word_overlap)
        if combined > best_ratio:
            best_ratio = combined
            best_answer = answer_en

    if best_ratio >= 0.55:
        return best_answer

    return None


def _has_fee_intent(question, question_norm):
    """Return True if the question is clearly about fees/tuition."""
    if contains_khmer(question):
        return any(marker in question for marker in KM_FEE_MARKERS)
    return any(marker in question_norm for marker in EN_FEE_MARKERS)


def _has_major_in_question(question, question_norm):
    """Return True if a recognizable major name exists in the question."""
    en_terms, kh_terms = get_major_terms()
    use_khmer = contains_khmer(question)

    if use_khmer:
        if any(term in question for term in kh_terms):
            return True
        if any(alias in question for alias in KM_MAJOR_ALIASES):
            return True
        if any(alias in question for alias in KM_TO_EN_MAP):
            return True
        return False

    # English: check aliases + dataset terms
    for alias in TERM_ALIASES:
        expanded = TERM_ALIASES[alias]
        if re.search(rf"\b{re.escape(alias)}\b", question_norm):
            if expanded in " ".join(en_terms) or expanded == "fee":
                return True
    return any(term in question_norm for term in en_terms)


def needs_fee_clarification(question):
    """Return True if question asks fee but does not specify a known major."""
    question_norm = _apply_term_aliases(normalize_text(question))

    if not _has_fee_intent(question, question_norm):
        return False

    # Do not force major clarification for admissions/application fee questions.
    admission_markers_en = {"application", "admission", "entrance", "exam"}
    admission_markers_kh = {"ពាក្យសុំ", "ចូលរៀន", "ប្រឡង"}
    if contains_khmer(question):
        if any(m in question for m in admission_markers_kh):
            return False
    else:
        if any(m in question_norm for m in admission_markers_en):
            return False

    return not _has_major_in_question(question, question_norm)


@lru_cache(maxsize=1)
def build_enrollment_help():
    """Build a concise enrollment guidance response from admissions entries."""
    period = None
    documents = None
    form_location = None

    for item in load_rupp_dataset_qa():
        if item.get("category") != "admissions":
            continue

        q = normalize_text(item.get("question_en", ""))
        ans = item.get("answer_en", "")

        if "application period" in q and not period:
            period = ans
        elif "required" in q and "admission" in q and not documents:
            documents = ans
        elif "application form" in q and not form_location:
            form_location = ans

    steps = []
    if period:
        steps.append(period)
    if documents:
        steps.append(documents)
    if form_location:
        steps.append(form_location)

    if not steps:
        return None

    lines = ["To enroll in RUPP, please follow these steps:"]
    for idx, step in enumerate(steps, start=1):
        lines.append(f"{idx}. {step}")
    return "\n".join(lines)


def find_rupp_dataset_answer(question):
    """Find best answer from RUPP bilingual dataset."""
    question_norm = _apply_term_aliases(normalize_text(question))
    question_words = extract_terms(question)
    use_khmer = contains_khmer(question)

    asks_fee = any(t in question_words for t in {"fee", "fees", "tuition", "cost", "price"})
    asks_major = any(t in question_words for t in {"major", "faculty", "department", "program", "course"})
    asks_admission = any(t in question_words for t in {"application", "admission", "document", "documents", "form"})
    asks_exam = any(t in question_words for t in {"exam", "entrance"})

    items = load_rupp_dataset_qa()
    if not items:
        return None

    # 1) Exact match
    for item in items:
        q_en = normalize_text(item["question_en"]) if item["question_en"] else ""
        q_kh = normalize_text(item["question_kh"]) if item["question_kh"] else ""
        if question_norm and (question_norm == q_en or question_norm == q_kh):
            if use_khmer and item["answer_kh"]:
                return item["answer_kh"]
            return item["answer_en"] or item["answer_kh"]

    # 2) Token overlap + fuzzy scoring
    best_score = 0.0
    second_best_score = 0.0
    best_informative_overlap = 0
    best_item = None

    for item in items:
        combined_question = f"{item['question_en']} {item['question_kh']} {item['category']}"
        item_words = extract_terms(combined_question)
        overlap = question_words & item_words
        informative_overlap = overlap - INTENT_TERMS
        score = float(len(overlap))

        q_en_norm = normalize_text(item["question_en"])
        q_kh_norm = normalize_text(item["question_kh"])
        fuzzy = max(fuzzy_ratio(question_norm, q_en_norm), fuzzy_ratio(question_norm, q_kh_norm))
        score += fuzzy * 4.0

        if item["category"] == "fees" and asks_fee:
            score += 2
        if item["category"] == "majors" and asks_major:
            score += 2
        if item["category"] == "admissions" and asks_admission:
            score += 3
        if asks_exam and ("exam" in item_words or "entrance" in item_words):
            score += 3

        if q_en_norm and q_en_norm in question_norm:
            score += 1.5
        if q_kh_norm and q_kh_norm in question_norm:
            score += 1.5

        if asks_exam and item["category"] in {"fees", "majors"}:
            score -= 2
        if asks_admission and item["category"] in {"fees", "majors"}:
            score -= 1

        if score > best_score:
            second_best_score = best_score
            best_score = score
            best_informative_overlap = len(informative_overlap)
            best_item = item
        elif score > second_best_score:
            second_best_score = score

    if best_item and best_score >= 2.5:
        if best_informative_overlap < 1 and best_score < 4.5:
            return None
        if best_score - second_best_score < 0.7 and best_score < 6.5:
            return None
        if use_khmer and best_item["answer_kh"]:
            return best_item["answer_kh"]
        return best_item["answer_en"] or best_item["answer_kh"]

    return None


def find_json_answer(question):
    """Try to answer from JSON data before using vector search."""
    question_norm = _apply_term_aliases(normalize_text(question))
    question_words = extract_terms(question)

    qa_items = load_json_qa()
    if not qa_items:
        return None

    for item in qa_items:
        if normalize_text(item["question"]) == question_norm:
            return item["answer"]

    best_score = 0
    best_answer = None
    for item in qa_items:
        item_words = extract_terms(item["question"])
        keyword_words = set(item["keywords"])
        overlap_score = len(question_words & item_words)
        keyword_score = len(question_words & keyword_words) * 2
        total_score = overlap_score + keyword_score
        if total_score > best_score:
            best_score = total_score
            best_answer = item["answer"]

    if best_score >= 2:
        return best_answer
    return None


def clean_text(text):
    """Clean and format text for better readability."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:!?()\-\'\"]+', '', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    return text.strip()


def get_question_keywords(question):
    """Extract key terms from question."""
    question_lower = question.lower()
    keywords = []
    if any(word in question_lower for word in ['where', 'location', 'located', 'address', 'place']):
        keywords.extend(['address', 'located', 'street', 'phnom penh', 'cambodia', 'russian', 'boulevard'])
    if any(word in question_lower for word in ['program', 'course', 'study', 'major', 'degree']):
        keywords.extend(['bachelor', 'master', 'program', 'degree', 'faculty', 'department'])
    if 'what' in question_lower and 'rupp' in question_lower:
        keywords.extend(['university', 'royal', 'phnom penh', 'established', 'founded'])
    return keywords


def score_relevance(text, question, keywords):
    """Score how relevant a text chunk is to the question."""
    text_lower = text.lower()
    score = 0
    for keyword in keywords:
        if keyword in text_lower:
            score += 2
    question_words = question.lower().split()
    for word in question_words:
        if len(word) > 3 and word in text_lower:
            score += 1
    return score


def _is_too_short_and_vague(question_norm):
    """
    Return True only if the query is very short AND has no recognizable intent.
    Short fee/major queries like 'history fee' are NOT vague.
    """
    words = question_norm.split()
    if len(words) > 4:
        return False
    # Has a fee marker → not vague
    if any(m in question_norm for m in EN_FEE_MARKERS):
        return False
    # Has an alias that maps to a subject/major → not vague
    for alias in TERM_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", question_norm):
            return False
    # Has an intent term → not vague
    if any(t in question_norm for t in INTENT_TERMS):
        return False
    return len(words) <= 3


def ask_question(question):
    try:
        use_khmer = contains_khmer(question)
        question_norm = _apply_term_aliases(normalize_text(question))

        # 1) Greeting detection (safe: won't fire if fee/major intent present)
        if detect_greeting(question):
            return greeting_response(use_khmer)

        # 2) Enrollment/admission shortcut
        if not use_khmer and any(
            t in question_norm for t in {"enroll", "enrol", "enrollment", "admission", "apply"}
        ):
            enrollment_help = build_enrollment_help()
            if enrollment_help:
                return f"RUPP Assistant:\n{enrollment_help}"

        # 3) Fee clarification only when major is truly missing
        if needs_fee_clarification(question):
            if use_khmer:
                return (
                    "RUPP Assistant:\n"
                    "សូមបញ្ជាក់ឈ្មោះជំនាញដែលអ្នកចង់សួរថ្លៃសិក្សា "
                    "(ឧទាហរណ៍៖ សង្គមវិទ្យា ឬ Computer Science)។"
                )
            return (
                "RUPP Assistant:\n"
                "Please specify the major/program for the fee question "
                "(for example: Sociology or Computer Science)."
            )

        # 4) Direct fee-by-major lookup (handles short queries like "history fee")
        direct_fee_answer = find_fee_answer_by_major(question)
        if direct_fee_answer:
            return f"RUPP Assistant:\n{direct_fee_answer}"

        # 5) Dataset QA scoring
        rupp_answer = find_rupp_dataset_answer(question)
        if rupp_answer:
            return f"RUPP Assistant:\n{rupp_answer}"

        # 6) JSON QA
        json_answer = find_json_answer(question)
        if json_answer:
            return f"RUPP Assistant:\n{json_answer}"

        # 7) Fee intent with no match → helpful error
        asks_fee = _has_fee_intent(question, question_norm)
        if asks_fee:
            if use_khmer:
                return (
                    "RUPP Assistant:\n"
                    "ខ្ញុំមិនទាន់អាចផ្គូផ្គងជំនាញនេះក្នុងទិន្នន័យថ្លៃសិក្សាបានទេ។ "
                    "សូមសរសេរឈ្មោះជំនាញឲ្យបានច្បាស់ (ឧទាហរណ៍៖ គីមីវិទ្យា, សង្គមវិទ្យា)។"
                )
            return (
                "RUPP Assistant:\n"
                "I could not match that major in the tuition dataset. "
                "Please provide the exact program name (for example: Chemistry, Sociology)."
            )

        # 8) Short vague queries (not fee/major related)
        if _is_too_short_and_vague(question_norm):
            if use_khmer:
                return "RUPP Assistant:\nសូមសរសេរសំណួរឲ្យច្បាស់ជាងនេះបន្តិច (ឧទាហរណ៍៖ តើ RUPP គឺជាអ្វី?)."
            return "RUPP Assistant:\nPlease ask a more specific question (for example: What is RUPP?)."

        # 9) Vector / FAISS search
        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        embeddings = LocalEmbeddings(model_name=model_name)
        if not os.path.exists(os.path.join("rupp_index", "index.faiss")):
            if use_khmer:
                return "RUPP Assistant:\n\nខ្ញុំមិនអាចរកឃើញចម្លើយនៅក្នុង JSON dataset ឬ PDF index ទេ។"
            return "RUPP Assistant:\n\nI cannot find this answer in the JSON dataset or PDF index."

        try:
            vectorstore = FAISS.load_local("rupp_index", embeddings)
        except KeyError as e:
            if "__fields_set__" in str(e):
                if use_khmer:
                    return (
                        "RUPP Assistant:\n\n"
                        "FAISS index ចាស់មិនត្រូវគ្នាជាមួយ dependency បច្ចុប្បន្នទេ។ "
                        "សូមដំណើរការ `python rag_system.py` ដើម្បីបង្កើត index ថ្មី។"
                    )
                return (
                    "RUPP Assistant:\n\n"
                    "The local FAISS index is incompatible with current dependencies. "
                    "Please run `python rag_system.py` to rebuild the index."
                )
            raise

        keywords = get_question_keywords(question)
        retrieval_results = vectorstore.similarity_search_with_score(question, k=8)
        if not retrieval_results:
            fallback_docs = vectorstore.similarity_search(question, k=5)
            retrieval_results = [(doc, 0.0) for doc in fallback_docs]

        if not retrieval_results:
            return "I cannot find this information in RUPP documents."

        scored_docs = []
        for rank, (doc, _) in enumerate(retrieval_results):
            text = doc.page_content.strip()
            if len(text) > 30:
                lexical_score = score_relevance(text, question, keywords)
                term_overlap = len(extract_terms(question) & extract_terms(text))
                dense_rank_bonus = max(0, 8 - rank)
                score = (lexical_score * 2) + term_overlap + dense_rank_bonus
                scored_docs.append((score, text))

        scored_docs.sort(reverse=True, key=lambda x: x[0])

        if not scored_docs:
            return "I cannot find this information in RUPP documents."

        if scored_docs[0][0] < 4:
            if use_khmer:
                return "RUPP Assistant:\n\nខ្ញុំមិនមានបរិបទគ្រប់គ្រាន់ក្នុងឯកសារ RUPP សម្រាប់សំណួរនេះទេ។"
            return "RUPP Assistant:\n\nI do not have enough supporting context in RUPP documents to answer this accurately."

        top_chunks = [text for _, text in scored_docs[:4]]
        extracted_answer = extract_relevant_sentences(top_chunks, question)
        cleaned_text = clean_text(extracted_answer or top_chunks[0])

        if len(cleaned_text) > 500:
            sentences = cleaned_text.split('. ')
            result = ""
            for sentence in sentences:
                if len(result) + len(sentence) < 450:
                    result += sentence + ". "
                else:
                    break
            cleaned_text = result.strip()
            if not cleaned_text.endswith('.'):
                cleaned_text += "..."

        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if gemini_api_key and genai is not None:
            try:
                genai.configure(api_key=gemini_api_key)
                gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
                model = genai.GenerativeModel(gemini_model)

                context_text = "\n\n".join(clean_text(chunk) for chunk in top_chunks)
                prompt = (
                    "You are a careful university assistant. "
                    "Answer using only the provided context. "
                    "Do not invent facts, numbers, dates, or requirements. "
                    "If the context is insufficient, reply that you do not know from the provided documents. "
                    "Use the same language as the question and keep the answer concise.\n\n"
                    f"Context:\n{context_text}\n\n"
                    f"Question: {question}"
                )
                generated = model.generate_content(prompt)
                if generated and getattr(generated, "text", None):
                    return f"🎓 RUPP Assistant:\n\n{generated.text.strip()}"
            except Exception:
                pass

        return f"🎓 RUPP Assistant:\n\n{cleaned_text}"

    except Exception as e:
        return f"Error: {str(e)}"