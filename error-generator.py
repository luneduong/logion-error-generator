"""
error_generator.py

Synthetic scribal error generator.

NEW TAXONOMY:
    Morphological                    -- same word stem but wrong inflection (case/tense/person/mood/number)
    Addition / Omission              -- word added or dropped
    Phonological-Orthographic        -- different word that is visually/aurally similar to the correct one
    Lexical / Contextual             -- Lexical: spelling error, letter corruption/ Contextual: completely unrelated word substitution
    Diacritic / Punctuation          -- accent/breathing change, added or removed punctuation
    Word Order / Transposition       -- words swapped position

OLD TAXONOMY:
    add, delete, change, accent, word_order, punctuation
    Old "change" is resolved at runtime into new-taxonomy subcategories 
    derived from the re-annotated dataset.
"""

import json
import math
import random
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ErrorEvent:
    """
    Record of one atomic error injection.

    word_index     -- index in the post-error word list where the change sits
                      (for omission: index where deletion occurred;
                       for addition: index of the inserted word)
    original_form  -- token before the error (none for pure additions)
    corrupted_form -- token after the error (none for pure omissions)
    depends_on     -- index into the parent ErrorResult.events list of the
                      event the error is contingent on; None = independent.
                      Set by compose_errors().
    """
    category: str
    word_index: int
    original_form: Optional[str]
    corrupted_form: Optional[str]
    depends_on: Optional[int] = None


@dataclass
class ErrorResult:
    words: list
    events: list = field(default_factory=list)
    success: bool = True

    def text(self) -> str:
        return " ".join(self.words)


class GenerationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Lexical resources
# ---------------------------------------------------------------------------

# Closed-class insertion pool for Addition / Omission and Lexical function-word swap.
# Disproportionately particles/conjunctions/pronouns.
CLOSED_CLASS_POOL = [
    "δέ", "καί", "γάρ", "οὖν", "μέν", "τε", "ἀλλά", "ὅτι", "τοῦτο",
    "αὐτός", "αὐτοῦ", "αὐτῷ", "αὐτόν", "τις", "τι", "ἄρα", "εἰ", "ὡς",
    "ὅτε", "ἐπεί", "ὥστε", "ἵνα", "ἐάν", "εἴτε", "πρὸς", "περί", "ἐπί",
]

# Preposition swap table: when a preposition is the target, draw the replacement
# from the same functional pool.
PREPOSITION_POOL = [
    "κατά", "παρά", "περί", "πρός", "ἐπί", "διά", "μετά", "ὑπό", "ἀπό",
    "ἐκ", "εἰς", "ἐν", "σύν", "ἀντί", "ἀνά",
]

# Conjunction/particle swap pool
PARTICLE_POOL = [
    "δέ", "γάρ", "οὖν", "μέν", "τε", "ἀλλά", "ὅτι", "ὡς", "ἐπεί",
    "ὥστε", "ἵνα", "καίτοι", "ὅμως", "πλήν", "πλέον",
]

# Greek diacritical combining marks.
GREEK_COMBINING_MARKS = [
    "\u0301",  # acute / oxia
    "\u0300",  # grave / varia
    "\u0342",  # circumflex / perispomeni
    "\u0313",  # smooth breathing / psili
    "\u0314",  # rough breathing / dasia
    "\u0345",  # iota subscript / ypogegrammeni
    "\u0308",  # diaeresis / dialytika
]

# Iotacism confusion group (tbd).
IOTACISM_GROUPS = [
    ["η", "ι", "υ"],       # /i/ merger
    ["ει", "ι", "η"],      # /i/ merger (digraph form)
    ["οι", "υ", "ι"],      # /i/ merger
    ["ω", "ο"],            # /o/ merger
    ["αι", "ε"],           # /e/ merger
]

# Consonant confusion groups (tbd).
CONSONANT_CONFUSIONS = [
    # Single/double consonant confusion
    ("σσ", "σ"),  ("λλ", "λ"),  ("ρρ", "ρ"),  ("ττ", "τ"),
    ("γγ", "γ"),  ("νν", "ν"),  ("ππ", "π"),  ("κκ", "κ"),
    ("φ",  "π"),  ("θ",  "τ"),  ("χ",  "κ"),
    ("ν",  "μ"),
    ("λ",  "ρ"),
    ("ζ",  "σ"),
]

# Attested phonological confusions.
# Used as the primary sampling pool in generate_phonological(); algorithmic
# operations as fallback for novel confusions.
ATTESTED_CONFUSION_PAIRS = [
    ("περιχαίροντες", "περιχαίνοντες"),   
    ("παρέφθειρε",   "παρέσπειρε"),
    ("εὐθηνούμενον", "εὐθυνόμενον"),
    ("βιώσιμα",      "βιωτικά"),
    ("βληθείσης",    "ἀμβλυνθείσης"),
    ("συγκλῶ",       "συγκυκλούμενος"),
    ("πείσμασι",     "πιέσμασι"),
    ("ἀφεθείς",      "ἀφεθῇς"),
    ("μικρᾶς",       "μακρᾶς"),
    ("μεμνῆσθαι",   "μεμυῆσθαι"),
    ("ἀποφήσειας",   "ἀπορήσειας"),
    ("ἐφίμερον",     "ἐφήμερον"),
    ("ὑποτρέχον",    "ὑπερέχον"),
    ("οἶδνα",        "ὕδνα"),
    ("οἶδνον",       "ὕδνον"),
    ("θεωρίζουσαν",  "θεατρίζουσαν"),
    ("προσκαταβαλλόμενος", "προκαταβαλλόμενος"),
    ("προηκούσης",   "προσηκούσης"),
    ("προήκοντα",    "προσήκοντα"),
    ("ἐπομένῳ",      "ἐρομένῳ"),
    ("ἑλκομένων",    "ἠλκωμένων"),
    ("Κωνταντίνου",  "Κωνσταντίνου"),
    ("ἐλπισμένων",   "ἠλπισμένων"),
    ("ἐξεναντίωται", "ἐξηναντίωται"),
    ("κυλινδούμενος","καλινδούμενος"),
    ("γλυκοθυμίας",  "γλυκυθυμίας"),
    ("ὑψηλογῶν",     "ὑψηλολογῶν"),
    ("προεχειρησάμην","προεχειρισάμην"),
    ("ἀρρύονται",    "ἀρύονται"),
    ("ἀνετάτετο",    "ἀνετέτατο"),
    ("ἀνιόντων",     "ἀνιώντων"),
    ("ἀναγνώριζε",   "γνώριζε"),
    ("ἀμβλότητος",   "ἀμβλύτητος"),
    ("γραφήσεται",   "ἐπιγραφήσεται"),
    ("σύντομος",     "σύντονος"),
    ("περιφέρομαι",  "περιρρέομαι"),
]

# Suffix swap table for Morphological function.
# Key = deaccented suffix to detect; value = list of replacement suffixes.
# Covers: nominal case endings, verbal person/number/tense/mood endings (tbd).
SUFFIX_SWAP_TABLE = {
    # Nouns: 1st/2nd declension
    "ης":  ["ην", "ῃ", "αν", "α", "ας"],
    "ην":  ["ης", "ῃ", "αν", "α"],
    "ῃ":   ["ης", "ην", "αν", "ει", "ῃς", "η"],
    "ου":  ["ον", "ῳ", "ους", "ω"],
    "ον":  ["ου", "ῳ", "ους", "α"],
    "ῳ":   ["ου", "ον", "οις"],
    "ους": ["ον", "ου", "οις"],
    "οις": ["ους", "ου", "ον"],
    "ας":  ["α", "ᾳ", "αν", "ης", "ε", "αμεν", "ασα", "αντος"],
    "α":   ["ας", "ᾳ", "αν", "ης", "ε", "αμεν", "ατε"],
    "ᾳ":   ["ας", "α", "αν"],
    "αν":  ["ας", "α", "ᾳ", "αμεν", "ατε"],
    # Nouns: 3rd declension
    "ος":  ["ον", "ῳ", "ους", "ες"],
    "ες":  ["ος", "ων", "ι", "ε", "εν"],
    "ων":  ["ος", "ες", "ι", "α", "ουσα", "ον", "οντος"],
    "ι":   ["ος", "ων", "ες", "α"],
    # Verbs: active indicative present/imperfect
    "ω":   ["εις", "ει", "ομεν", "ετε", "ουσι"],
    "εις": ["ω", "ει", "ομεν"],
    "ει":  ["ω", "εις", "ομεν", "εν"],
    "ομεν":["ω", "ετε", "ουσι"],
    "ετε": ["ω", "ομεν", "ουσι"],
    "ε":   ["ες", "εν", "ον"],  # imperfect 3sg
    "εν":  ["ε", "ες", "ον"],    # imperfect 3sg
    # Verbs: aorist active
    "αμεν":["α", "ατε", "αν"],
    "ατε": ["α", "αμεν", "αν"],
    # Verbs: optative/subjunctive
    "οιτο":["αιτο", "οιτε", "οιεν"],
    "αιτο":["οιτο", "αιεν"],
    "ῃς":  ["εις", "ῃ", "ης"],
    # Verbs: Perfect active
    "κα":  ["κας", "κε", "καμεν"],
    "κας": ["κα", "κε"],
    "κε":  ["κα", "κας", "κεν"],
    # Verbs: Infinitive
    "ειν": ["ει", "ε", "ων"],
    "αι":  ["ειν", "ε", "ας"],
    "ναι": ["ειν", "αι"],
    # Participles
    "ουσα":["ων", "ον", "ουσης"],
    "οντος":["ων", "ουσα", "οντι"],
    "ασα": ["ας", "αντος", "αν"],
}

# Empirical split for old "change" category -> new taxonomy
OLD_CHANGE_SPLIT = {
    "Morphological":                   0.515,
    "Phonological-Orthographic":       0.190,
    "Lexical / Contextual":            0.180,
    "Addition / Omission":             0.103,
    "Diacritic / Punctuation":         0.010,
}

# Old -> new taxonomy 1:1 mappings
OLD_TO_NEW_DIRECT = {
    "add":        "Addition / Omission",
    "delete":     "Addition / Omission",
    "accent":     "Diacritic / Punctuation",
    "word_order": "Word Order / Transposition",
    "punctuation":"Diacritic / Punctuation",
}

# Empirical positional bias per category.
# Beta distribution shape parameters (alpha, beta) fitted to real-data mean positions:
#   Addition/Omission: mean=0.331  (front-loaded)
#   Diacritic:         mean=0.366  (slightly front)
#   Lexical:           mean=0.485  (near-uniform)
#   Morphological:     mean=0.550  (mid-late)
#   Word Order:        mean=0.543  (mid)
#   Phonological:      mean=0.598  (late)
# Parameters chosen so that Beta.mean = alpha/(alpha+beta) matches empirical mean.
POSITION_BIAS_DEFAULTS = {
    "Addition / Omission":        (1.0, 2.0),   # mean ~0.333
    "Diacritic / Punctuation":    (1.2, 2.0),   # mean ~0.375
    "Lexical / Contextual":       (1.2, 1.2),   # mean ~0.500
    "Morphological":              (1.5, 1.2),   # mean ~0.556
    "Word Order / Transposition": (1.5, 1.3),   # mean ~0.536
    "Phonological-Orthographic":  (2.0, 1.3),   # mean ~0.606
}

# Default target weights.
# Pass as category_weights to generate_corpus() for calibrated generation.
NEW_TAXONOMY_WEIGHTS = {
    "Morphological":               0.352,
    "Addition / Omission":         0.286,
    "Phonological-Orthographic":   0.129,
    "Lexical / Contextual":        0.122,
    "Diacritic / Punctuation":     0.066,
    "Word Order / Transposition":  0.028,
}

# Default target weights for old taxonomy.
OLD_TAXONOMY_WEIGHTS = {
    "add":        0.143,
    "delete":     0.143,
    "change":     0.500,
    "accent":     0.066,
    "word_order": 0.028,
    "punctuation":0.120,
}

# Dual-error pairs observed in dataset.
# Used as default dual_error_pairs in generate_corpus().
EMPIRICAL_DUAL_PAIRS = [
    ("Morphological",          "Lexical / Contextual"),
    ("Morphological",          "Diacritic / Punctuation"),
    ("Addition / Omission",    "Morphological"),
    ("Morphological",          "Addition / Omission"),
    ("Diacritic / Punctuation","Morphological"),
]

OLD_DUAL_PAIRS = [
    ("change", "accent"),
    ("delete", "change"),
    ("change", "add"),
    ("accent", "change"),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _deaccent(w: str) -> str:
    """Strip all combining diacritical marks.
    """
    d = unicodedata.normalize("NFD", w)
    return "".join(c for c in d if unicodedata.category(c) != "Mn")


def _strip_punct(w: str) -> str:
    return w.strip(".,·;·!?\"'()[]†#— ")

_GREEK_VOWELS = set("αεηιοωυΑΕΗΙΟΩΥ")
_ACCENT_MARKS    = {"\u0301", "\u0300", "\u0342"}
_BREATHING_MARKS = {"\u0313", "\u0314"}


def _transfer_accent(original: str, corrupted: str) -> str:
    """
     Transfer accent from 'original' onto 'corrupted'.

    Called when category function produces corrupted form that has lost 
    all diacritical marks. Place the accent at the same vowel position relative
    to the end of the word as in the original, clamped to the corrupted word length.

    If corrupted already carries combining mark, return unchanged.
    If neither word has a vowel, return corrupted unchanged.

    (1) find the accented vowel in original and its marks
    (2) place the accent at the same rank in the corrupted word
    """
    corr_nfd = unicodedata.normalize("NFD", corrupted)
    if any(unicodedata.category(c) == "Mn" for c in corr_nfd):
        return corrupted

    orig_nfd = unicodedata.normalize("NFD", original)

    # Build list of (nfd_index, base_char) for non-combining chars in original
    orig_base = [(i, c) for i, c in enumerate(orig_nfd)
                 if unicodedata.category(c) != "Mn"]
    # List of positions in orig_base that are vowels
    orig_vowel_base_indices = [k for k, (_, c) in enumerate(orig_base)
                                if c.lower() in _GREEK_VOWELS]

    if not orig_vowel_base_indices:
        return corrupted

    # Find which vowel carries the accent (search from end)
    marks_to_transfer = []
    vowel_rank_from_end = 0  # 0 = last vowel, 1 = penultimate, etc.
    for rank, k in enumerate(reversed(orig_vowel_base_indices)):
        nfd_idx = orig_base[k][0]
        # Collect combining marks immediately after this base char
        marks = []
        j = nfd_idx + 1
        while j < len(orig_nfd) and unicodedata.category(orig_nfd[j]) == "Mn":
            if orig_nfd[j] in _ACCENT_MARKS or orig_nfd[j] in _BREATHING_MARKS:
                marks.append(orig_nfd[j])
            j += 1
        if marks:
            marks_to_transfer = marks
            vowel_rank_from_end = rank
            break

    if not marks_to_transfer:
        # No accented vowel found — add a plain acute to the last vowel of corrupted
        marks_to_transfer = ["́"]
        vowel_rank_from_end = 0

    corr_base = [(i, c) for i, c in enumerate(corr_nfd)
                 if unicodedata.category(c) != "Mn"]
    corr_vowel_base_indices = [k for k, (_, c) in enumerate(corr_base)
                                if c.lower() in _GREEK_VOWELS]

    if not corr_vowel_base_indices:
        return corrupted

    n_corr_vowels = len(corr_vowel_base_indices)
    # Clamp rank to available vowels
    rank = min(vowel_rank_from_end, n_corr_vowels - 1)
    # k-th vowel from the end = index (n-1-rank) in the vowel list
    target_k = corr_vowel_base_indices[n_corr_vowels - 1 - rank]
    target_nfd_idx = corr_base[target_k][0]

    new_nfd = (corr_nfd[:target_nfd_idx + 1]
               + "".join(marks_to_transfer)
               + corr_nfd[target_nfd_idx + 1:])
    return unicodedata.normalize("NFC", new_nfd)


def _beta_sample(rng: random.Random, alpha: float, beta: float) -> float:
    """Sample from Beta(alpha, beta) using the rng instance."""
    return rng.betavariate(alpha, beta)


def _position_weighted_index(
    words: list,
    rng: random.Random,
    bias: tuple,  # (alpha, beta) for Beta distribution
    exclude: set = None,
) -> Optional[int]:
    """
    Select a word index from `words` with probability proportional to
    a Beta(alpha, beta) distribution over normalised sentence positions.

    The Beta distribution is evaluated at n evenly-spaced positions across
    [0, 1] and used as unnormalised weights for random.choices().

    Args:
        words   -- word list (must be non-empty)
        rng     -- seeded random.Random
        bias    -- (alpha, beta) shape parameters
        exclude -- set of indices to exclude from selection

    Returns:
        Selected index, or None if all positions are excluded.
    """
    if not words:
        return None
    exclude = exclude or set()
    n = len(words)
    alpha, beta = bias

    if n == 1:
        return None if 0 in exclude else 0

    # Compute Beta PDF weight at each word position
    def beta_pdf(x, a, b):
        # Proportional to x^(a-1) * (1-x)^(b-1); avoid 0^0
        if x <= 0.0:
            x = 1e-9
        if x >= 1.0:
            x = 1 - 1e-9
        return (x ** (a - 1)) * ((1 - x) ** (b - 1))

    positions = [i / (n - 1) for i in range(n)]
    weights = [
        beta_pdf(positions[i], alpha, beta) if i not in exclude else 0.0
        for i in range(n)
    ]
    total = sum(weights)
    if total == 0.0:
        return None

    return rng.choices(range(n), weights=weights, k=1)[0]


def load_vocab(path: str) -> list:
    """Load a one-word-per-line Greek vocabulary file."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Category functions — NEW TAXONOMY
# ---------------------------------------------------------------------------

def generate_morphological(
    words: list,
    rng: random.Random,
    position_bias: tuple = None,
    min_stem_len: int = 3,
) -> ErrorResult:
    """
    MORPHOLOGICAL: correct word stem with wrong inflectional ending.

    Detects the word's suffix by finding the longest match in SUFFIX_SWAP_TABLE, 
    then replaces the suffix with a randomly chosen alternative from the table.

    For short function words (≤4 base characters) with no suffix-table match,
    falls back to a whole-word swap within a small paradigm set (articles,
    pronouns).

    Args:
        position_bias -- (alpha, beta) Beta params; defaults to empirical mid-late
        min_stem_len  -- minimum base-letter stem length before suffix starts
    """
    bias = position_bias or POSITION_BIAS_DEFAULTS["Morphological"]
    if not words:
        return ErrorResult(words=words, success=False)

    
    SHORT_PARADIGMS = {
        # articles
        "τον": ["την", "το", "τω", "τοις", "τους", "τα"],
        "την": ["τον", "το", "τω", "τοις", "τας"],
        "το":  ["τον", "την", "τω", "τα"],
        "τω":  ["τον", "την", "το", "τοις"],
        "τοις":["τους", "τω", "τα"],
        "τους":["τοις", "τα", "τον"],
        "τα":  ["τους", "τοις", "το"],
        "του": ["τον", "τω", "τους", "τοις"],
        "της": ["τον", "την", "τω", "τοις"],
        "τας": ["τους", "τα", "την"],
        "των": ["τοις", "τους", "τα"],
        # pronouns
        "μου": ["μοι", "με", "μην"],
        "μοι": ["μου", "με"],
        "με":  ["μοι", "μου"],
        "σοι": ["σου", "σε", "σην"],
        "σου": ["σοι", "σε"],
        "σε":  ["σοι", "σου"],
        "αυτον":["αυτου", "αυτω", "αυτους", "αυτοις"],
        "αυτου":["αυτον", "αυτω", "αυτοις"],
        "αυτω": ["αυτον", "αυτου", "αυτοις"],
        # relative pronouns
        "ον":  ["ου", "ω", "οις", "ους"],
        "ους": ["ον", "ου", "ω", "οις"],
        "ων":  ["ους", "οις", "ον"],
        "οις": ["ους", "ων", "ω", "ον"],
        # particles with morphological variants
        "οσα": ["οσον", "οσοι", "οσης"],
        "οσον":["οσα", "οσοι"],
        "οσοι":["οσα", "οσον"],
    }

    # Words that must not be processed by the suffix table because their
    # final characters happen to match inflectional suffixes despite being
    # uninflecting particles, prepositions, or conjunctions.
    SUFFIX_BLOCKLIST = {
        "κατα", "παρα", "περι", "προς", "επι", "δια", "μετα", "υπο", "απο",
        "εκ", "εις", "εν", "συν", "αντι", "ανα", "αρα", "ουν", "γαρ", "δε",
        "μεν", "τε", "αλλα", "οτι", "ως", "ωστε", "επει", "ινα", "εαν",
        "ωσπερ", "καθα", "αμα", "αμφω", "πλην", "πλεον", "ουτε", "ουδε",
        "μηδε", "ητε", "ητοι", "ιδου", "ιδε", "ναι", "ου", "ουχ", "ουκ",
        "μη", "μητε", "που", "που", "νυν", "τοτε", "οτε", "ενθα", "αυτα",
        "ταδε", "τοιαδε", "τοιαυτα", "τοσαυτα", "παντα", "μονα", "ολα",
    }

    def find_suffix_match(w: str):
        """Return (stem, old_suffix, candidates) or None."""
        base = _deaccent(_strip_punct(w)).lower()
        if base in SUFFIX_BLOCKLIST:
            return None
        for length in range(min(6, len(base) - min_stem_len), 0, -1):
            suf = base[-length:]
            if suf in SUFFIX_SWAP_TABLE:
                stem_len = len(base) - length
                if stem_len < min_stem_len:
                    continue
                stem = w[:len(w) - length]  # preserve original accents on stem
                candidates = [c for c in SUFFIX_SWAP_TABLE[suf] if c != suf]
                if candidates:
                    return stem, suf, candidates
        return None

    def find_paradigm_match(w: str):
        """Return list of alternants for short function words, or None."""
        base = _deaccent(_strip_punct(w)).lower()
        if base in SHORT_PARADIGMS:
            return SHORT_PARADIGMS[base]
        return None

    candidates = []
    for i, w in enumerate(words):
        if find_suffix_match(w) or find_paradigm_match(w):
            candidates.append(i)

    if not candidates:
        return ErrorResult(words=words, success=False)

    candidate_set = set(candidates)
    all_indices = list(range(len(words)))
    bias_weights = []
    alpha, beta_v = bias
    n = len(words)
    def beta_pdf(x, a, b):
        if x <= 0: x = 1e-9
        if x >= 1: x = 1 - 1e-9
        return (x ** (a - 1)) * ((1 - x) ** (b - 1))
    positions = [i / max(n - 1, 1) for i in range(n)]
    for i in range(n):
        bias_weights.append(beta_pdf(positions[i], alpha, beta_v) if i in candidate_set else 0.0)
    total_w = sum(bias_weights)
    if total_w == 0:
        return ErrorResult(words=words, success=False)

    target_idx = rng.choices(all_indices, weights=bias_weights, k=1)[0]
    original_word = words[target_idx]

    match = find_suffix_match(original_word)
    if match:
        stem, old_suf, suf_candidates = match
        new_suffix = rng.choice(suf_candidates)
        corrupted_word = stem + new_suffix
    else:
        alternants = find_paradigm_match(original_word)
        corrupted_word = rng.choice(alternants)

    if corrupted_word == original_word:
        return ErrorResult(words=words, success=False)

    corrupted_word = _transfer_accent(original_word, corrupted_word)
    new_words = words[:target_idx] + [corrupted_word] + words[target_idx + 1:]
    event = ErrorEvent(
        category="Morphological",
        word_index=target_idx,
        original_form=original_word,
        corrupted_form=corrupted_word,
    )
    return ErrorResult(words=new_words, events=[event])


def generate_addition_omission(
    words: list,
    rng: random.Random,
    position_bias: tuple = None,
    omit_ratio: float = 0.70,
    haplography_bonus: float = 2.0,
    closed_class_vocab: list = None,
) -> ErrorResult:
    """
    ADDITION / OMISSION: word dropped or inserted.

    Omission (70% by default):
        Selects a word for deletion, with a bonus weight on words whose first
        3 characters match an adjacent word (haplography/eyeskip trigger).

    Addition (30% by default):
        Inserts a word from the closed-class vocabulary at a position weighted
        toward the front of the sentence.

    Args:
        omit_ratio        -- fraction of calls that should produce omissions
        haplography_bonus -- weight multiplier on words adjacent to a prefix-match
                             neighbour
        closed_class_vocab -- word pool for additions; defaults to CLOSED_CLASS_POOL
    """
    bias = position_bias or POSITION_BIAS_DEFAULTS["Addition / Omission"]
    pool = closed_class_vocab or CLOSED_CLASS_POOL

    if not words:
        return ErrorResult(words=words, success=False)

    is_omission = rng.random() < omit_ratio

    if is_omission:
        if len(words) < 1:
            return ErrorResult(words=words, success=False)

        n = len(words)
        alpha, beta_v = bias
        def beta_pdf(x, a, b):
            if x <= 0: x = 1e-9
            if x >= 1: x = 1 - 1e-9
            return (x ** (a - 1)) * ((1 - x) ** (b - 1))
        positions = [i / max(n - 1, 1) for i in range(n)]
        weights = [beta_pdf(positions[i], alpha, beta_v) for i in range(n)]

        for i in range(n):
            base = _deaccent(words[i])[:3].lower()
            if base:
                if i > 0 and _deaccent(words[i-1])[:3].lower() == base:
                    weights[i] *= haplography_bonus
                if i < n - 1 and _deaccent(words[i+1])[:3].lower() == base:
                    weights[i] *= haplography_bonus

        del_idx = rng.choices(range(n), weights=weights, k=1)[0]
        deleted_word = words[del_idx]
        new_words = words[:del_idx] + words[del_idx + 1:]
        event = ErrorEvent(
            category="Addition / Omission",
            word_index=del_idx,
            original_form=deleted_word,
            corrupted_form=None,
        )
        return ErrorResult(words=new_words, events=[event])

    else:
        insert_idx = _position_weighted_index(
            words + ["__end__"],
            rng, bias
        )
        if insert_idx is None:
            insert_idx = rng.randint(0, len(words))
        insert_idx = min(insert_idx, len(words))
        insert_word = rng.choice(pool)
        new_words = words[:insert_idx] + [insert_word] + words[insert_idx:]
        event = ErrorEvent(
            category="Addition / Omission",
            word_index=insert_idx,
            original_form=None,
            corrupted_form=insert_word,
        )
        return ErrorResult(words=new_words, events=[event])


def generate_phonological(
    words: list,
    rng: random.Random,
    position_bias: tuple = None,
    confusion_weights: tuple = (0.50, 0.30, 0.20),
    attested_pair_weight: float = 0.65,
    max_retries: int = 3,
) -> ErrorResult:
    """
    PHONOLOGICAL-ORTHOGRAPHIC: ddifferent word that is visually/aurally similar to the correct one

      (1) (probability=attested_pair_weight): try to apply a known confusion
        pair from the real dataset by finding a substring in the target word that
        matches one half of an attested pair and replacing it with the other half.
      (2) (fallback): apply one of three algorithmic confusion operations
        weighted by confusion_weights:
          - Iotacism (0.50): substitute within a vowel confusion group (η/ι/υ etc.)
          - Consonant confusion (0.30): gemination, aspiration, place confusion
          - Vowel quantity (0.20): ο/ω or ε/αι substitution

    The result is validated to ensure the change in base letters, retried up to max_retries times.

    Args:
        confusion_weights     -- (iotacism, consonant, vowel_qty) operation weights
        attested_pair_weight  -- probability of attempting an attested-pair match
                                 before falling back to algorithmic operations
        max_retries           -- max attempts before returning success=False
    """
    bias = position_bias or POSITION_BIAS_DEFAULTS["Phonological-Orthographic"]
    pairs = ATTESTED_CONFUSION_PAIRS

    if not words:
        return ErrorResult(words=words, success=False)

    candidate_indices = [
        i for i, w in enumerate(words)
        if len(_deaccent(_strip_punct(w))) >= 2
    ]
    if not candidate_indices:
        return ErrorResult(words=words, success=False)

    n = len(words)
    alpha, beta_v = bias
    def beta_pdf(x, a, b):
        if x <= 0: x = 1e-9
        if x >= 1: x = 1 - 1e-9
        return (x ** (a - 1)) * ((1 - x) ** (b - 1))
    positions = [i / max(n - 1, 1) for i in range(n)]
    cand_set = set(candidate_indices)
    w_list = [beta_pdf(positions[i], alpha, beta_v) if i in cand_set else 0.0 for i in range(n)]
    total_w = sum(w_list)
    if total_w == 0:
        return ErrorResult(words=words, success=False)
    target_idx = rng.choices(range(n), weights=w_list, k=1)[0]
    original_word = words[target_idx]

    def _apply_iotacism(word: str, rng: random.Random) -> Optional[str]:
        """Substitute one vowel/digraph for another within a confusion group."""
        base = _deaccent(word).lower()
        digraph_groups = [g for g in IOTACISM_GROUPS if any(len(v) > 1 for v in g)]
        monograph_groups = [g for g in IOTACISM_GROUPS if all(len(v) == 1 for v in g)]
        for group in digraph_groups + monograph_groups:
            present = [(v, base.find(v)) for v in group if v in base]
            if len(present) >= 1:
                original_vowel, pos = rng.choice(present)
                candidates = [v for v in group if v != original_vowel]
                if not candidates:
                    continue
                replacement = rng.choice(candidates)
                new_base = base[:pos] + replacement + base[pos + len(original_vowel):]
                nfd = unicodedata.normalize("NFD", word)
                nfd_base = _deaccent(word).lower()
                idx = nfd_base.find(original_vowel)
                if idx >= 0:
                    base_pos = 0
                    nfd_pos = 0
                    while nfd_pos < len(nfd) and base_pos < idx:
                        if unicodedata.category(nfd[nfd_pos]) != "Mn":
                            base_pos += 1
                        nfd_pos += 1
                    span_end = nfd_pos
                    while span_end < len(nfd) and (
                        span_end - nfd_pos < len(original_vowel) or
                        unicodedata.category(nfd[span_end]) == "Mn"
                    ):
                        span_end += 1
                    corrupted_nfd = nfd[:nfd_pos] + replacement + nfd[span_end:]
                    return unicodedata.normalize("NFC", corrupted_nfd)
        return None

    def _apply_consonant_confusion(word: str, rng: random.Random) -> Optional[str]:
        """Gemination, aspiration, or place-of-articulation confusion."""
        base = _deaccent(word).lower()
        applicable = [(old, new) for old, new in CONSONANT_CONFUSIONS if old in base]
        if not applicable:
            return None
        old_c, new_c = rng.choice(applicable)
        idx = base.find(old_c)
        corrupted = word[:idx] + new_c + word[idx + len(old_c):]
        return corrupted

    def _apply_vowel_quantity(word: str, rng: random.Random) -> Optional[str]:
        """ο/ω or ε/αι quantity confusion."""
        pairs_vq = [("ω", "ο"), ("ο", "ω"), ("αι", "ε"), ("ε", "αι")]
        base = _deaccent(word).lower()
        applicable = [(o, n) for o, n in pairs_vq if o in base]
        if not applicable:
            return None
        old_v, new_v = rng.choice(applicable)
        idx = base.find(old_v)
        corrupted = word[:idx] + new_v + word[idx + len(old_v):]
        return corrupted

    ops = ["iotacism", "consonant", "vowel_qty"]
    op_weights = list(confusion_weights)

    for attempt in range(max_retries):
        corrupted_word = None

        # Stage 1: try attested pair match (bidirectional).
        if pairs and rng.random() < attested_pair_weight:
            orig_base_word = _deaccent(_strip_punct(original_word)).lower()
            # Build bidirectional candidate list
            bidir_pairs = []
            for o, c in pairs:
                o_base = _deaccent(_strip_punct(o)).lower()
                c_base = _deaccent(_strip_punct(c)).lower()
                if o_base in orig_base_word:
                    bidir_pairs.append((o, c, o_base, c_base))
                elif c_base in orig_base_word:
                    bidir_pairs.append((c, o, c_base, o_base))  # reversed
            if bidir_pairs:
                orig_p, conf_p, orig_b, conf_b = rng.choice(bidir_pairs)
                if orig_p in original_word:
                    corrupted_word = original_word.replace(orig_p, conf_p, 1)
                else:
                    idx = orig_base_word.find(orig_b)
                    if idx >= 0:
                        corrupted_word = original_word[:idx] + conf_p + original_word[idx + len(orig_p):]
                    else:
                        corrupted_word = None

        if corrupted_word is None:
            operation = rng.choices(ops, weights=op_weights, k=1)[0]
            if operation == "iotacism":
                corrupted_word = _apply_iotacism(original_word, rng)
            elif operation == "consonant":
                corrupted_word = _apply_consonant_confusion(original_word, rng)
            else:
                corrupted_word = _apply_vowel_quantity(original_word, rng)

        if corrupted_word is None:
            continue

        # Validate: base letters must have changed
        if _deaccent(corrupted_word).lower() != _deaccent(original_word).lower():
            corrupted_word = _transfer_accent(original_word, corrupted_word)
            new_words = words[:target_idx] + [corrupted_word] + words[target_idx + 1:]
            event = ErrorEvent(
                category="Phonological-Orthographic",
                word_index=target_idx,
                original_form=original_word,
                corrupted_form=corrupted_word,
            )
            return ErrorResult(words=new_words, events=[event])

    return ErrorResult(words=words, success=False)


def generate_lexical(
    words: list,
    rng: random.Random,
    position_bias: tuple = None,
    vocab: list = None,
    subtype_weights: tuple = (0.45, 0.35, 0.20),
) -> ErrorResult:
    """
    LEXICAL / CONTEXTUAL: spelling error, letter corruption, or unrelated word swap.

    Three sub-types, sampled with subtype_weights:

      spelling_corruption (0.45):
          One of: letter drop (delete 1 non-initial consonant),
                  letter duplication (double a consonant),
                  letter transposition (swap two adjacent letters).
          Targets letters in the word body.

      different_word (0.35):
          Replace the target word with a word drawn from the vocab argument.
          If no vocab is provided, falls back to the closed-class pool. 
          The replacement must differ from the original.

      function_word_swap (0.20):
          If the target word is a preposition, replace from PREPOSITION_POOL.
          If it is a particle/conjunction, replace from PARTICLE_POOL.
          Falls back to different_word if the target is neither.

    Args:
        vocab           -- open-class word pool (list or None)
        subtype_weights -- (spelling_corruption, different_word, func_swap) weights
    """
    bias = position_bias or POSITION_BIAS_DEFAULTS["Lexical / Contextual"]
    if not words:
        return ErrorResult(words=words, success=False)

    target_idx = _position_weighted_index(words, rng, bias)
    if target_idx is None:
        return ErrorResult(words=words, success=False)
    original_word = words[target_idx]

    subtypes = ["spelling", "different_word", "func_swap"]
    subtype = rng.choices(subtypes, weights=list(subtype_weights), k=1)[0]

    corrupted_word = None

    if subtype == "spelling":
        base = list(_deaccent(_strip_punct(original_word)).lower())
        if len(base) < 3:
            subtype = "different_word"
        else:
            ops = ["drop", "duplicate", "transpose"]
            op = rng.choice(ops)
            interior = list(range(1, len(base) - 1))
            if not interior:
                subtype = "different_word"
            else:
                if op == "drop":
                    drop_idx = rng.choice(interior)
                    base.pop(drop_idx)
                elif op == "duplicate":
                    dup_idx = rng.choice(interior)
                    base.insert(dup_idx, base[dup_idx])
                elif op == "transpose":
                    if len(interior) >= 2:
                        i = rng.choice(interior[:-1])
                        base[i], base[i + 1] = base[i + 1], base[i]
                    else:
                        subtype = "different_word"
                if subtype == "spelling":
                    candidate = "".join(base)
                    # Reject if corruption removed all vowels (e.g. πρὸς->πρς)
                    if any(c in "αεηιοωυαεηιοωυ" for c in candidate):
                        corrupted_word = candidate

    if subtype == "different_word" or (subtype == "spelling" and corrupted_word is None):
        pool = vocab if vocab else CLOSED_CLASS_POOL
        candidates = [w for w in pool if w != original_word]
        if not candidates:
            return ErrorResult(words=words, success=False)
        corrupted_word = rng.choice(candidates)

    elif subtype == "func_swap":
        base_clean = _deaccent(_strip_punct(original_word)).lower()
        if base_clean in [_deaccent(p).lower() for p in PREPOSITION_POOL]:
            pool = [p for p in PREPOSITION_POOL if _deaccent(p).lower() != base_clean]
        elif base_clean in [_deaccent(p).lower() for p in PARTICLE_POOL]:
            pool = [p for p in PARTICLE_POOL if _deaccent(p).lower() != base_clean]
        else:
            pool = vocab if vocab else CLOSED_CLASS_POOL
            pool = [w for w in pool if w != original_word]
        if not pool:
            return ErrorResult(words=words, success=False)
        corrupted_word = rng.choice(pool)

    if corrupted_word is None or corrupted_word == original_word:
        return ErrorResult(words=words, success=False)

    corrupted_word = _transfer_accent(original_word, corrupted_word)
    new_words = words[:target_idx] + [corrupted_word] + words[target_idx + 1:]
    event = ErrorEvent(
        category="Lexical / Contextual",
        word_index=target_idx,
        original_form=original_word,
        corrupted_form=corrupted_word,
    )
    return ErrorResult(words=new_words, events=[event])


def generate_diacritic(
    words: list,
    rng: random.Random,
    position_bias: tuple = None,
    operation_weights: tuple = (0.50, 0.30, 0.20),
) -> ErrorResult:
    """
    DIACRITIC / PUNCTUATION: accent/breathing mark change, or punctuation change.

    Three sub-operations weighted by operation_weights:
        drop (0.50)      -- remove one combining mark
        swap (0.30)      -- replace one combining mark with a different one
        insert (0.20)    -- add a combining mark to an unaccented word

    Operates at NFD combining-character level (Unicode category Mn),
    matching preprocess-char-data.py's deaccent() convention.
    """
    bias = position_bias or POSITION_BIAS_DEFAULTS["Diacritic / Punctuation"]
    if not words:
        return ErrorResult(words=words, success=False)

    accented = [
        i for i, w in enumerate(words)
        if any(unicodedata.category(c) == "Mn"
               for c in unicodedata.normalize("NFD", w))
    ]
    unaccented = [
        i for i, w in enumerate(words)
        if not any(unicodedata.category(c) == "Mn"
                   for c in unicodedata.normalize("NFD", w))
        and len(_deaccent(w)) >= 2
        and any(c.lower() in _GREEK_VOWELS for c in _deaccent(w))
    ]

    ops = ["drop", "swap", "insert"]
    op_w = list(operation_weights)
    if not accented:
        op_w[0] = 0.0; op_w[1] = 0.0
    if not unaccented:
        op_w[2] = 0.0
    if sum(op_w) == 0:
        return ErrorResult(words=words, success=False)

    operation = rng.choices(ops, weights=op_w, k=1)[0]

    if operation in ("drop", "swap"):
        n = len(words)
        alpha, beta_v = bias
        def beta_pdf(x, a, b):
            if x <= 0: x = 1e-9
            if x >= 1: x = 1 - 1e-9
            return (x ** (a - 1)) * ((1 - x) ** (b - 1))
        pos_list = [i / max(n - 1, 1) for i in range(n)]
        acc_set = set(accented)
        wts = [beta_pdf(pos_list[i], alpha, beta_v) if i in acc_set else 0.0 for i in range(n)]
        if sum(wts) == 0:
            return ErrorResult(words=words, success=False)
        target_idx = rng.choices(range(n), weights=wts, k=1)[0]

        original_word = words[target_idx]
        decomposed = list(unicodedata.normalize("NFD", original_word))
        mark_positions = [i for i, c in enumerate(decomposed) if unicodedata.category(c) == "Mn"]
        pos = rng.choice(mark_positions)

        if operation == "drop":
            del decomposed[pos]
        else:  # swap
            current = decomposed[pos]
            pool = [m for m in GREEK_COMBINING_MARKS if m != current]
            decomposed[pos] = rng.choice(pool)

    else:
        n = len(words)
        alpha, beta_v = bias
        def beta_pdf(x, a, b):
            if x <= 0: x = 1e-9
            if x >= 1: x = 1 - 1e-9
            return (x ** (a - 1)) * ((1 - x) ** (b - 1))
        pos_list = [i / max(n - 1, 1) for i in range(n)]
        una_set = set(unaccented)
        wts = [beta_pdf(pos_list[i], alpha, beta_v) if i in una_set else 0.0 for i in range(n)]
        if sum(wts) == 0:
            return ErrorResult(words=words, success=False)
        target_idx = rng.choices(range(n), weights=wts, k=1)[0]

        original_word = words[target_idx]
        decomposed = list(unicodedata.normalize("NFD", original_word))
        vowel_positions = [
            i for i, c in enumerate(decomposed)
            if unicodedata.category(c) != "Mn" and c.lower() in _GREEK_VOWELS
        ]
        if not vowel_positions:
            return ErrorResult(words=words, success=False)
        mark = rng.choice(GREEK_COMBINING_MARKS)
        decomposed.insert(rng.choice(vowel_positions) + 1, mark)

    corrupted_word = unicodedata.normalize("NFC", "".join(decomposed))
    if corrupted_word == original_word:
        return ErrorResult(words=words, success=False)

    new_words = words[:target_idx] + [corrupted_word] + words[target_idx + 1:]
    event = ErrorEvent(
        category="Diacritic / Punctuation",
        word_index=target_idx,
        original_form=original_word,
        corrupted_form=corrupted_word,
    )
    return ErrorResult(words=new_words, events=[event])


def generate_word_order(
    words: list,
    rng: random.Random,
    position_bias: tuple = None,
    max_distance: int = 1,
) -> ErrorResult:
    """
    WORD ORDER / TRANSPOSITION: swap two adjacent or near-adjacent words.

    Default max_distance=1 (adjacent transposition).
    """
    bias = position_bias or POSITION_BIAS_DEFAULTS["Word Order / Transposition"]
    if len(words) < 2:
        return ErrorResult(words=words, success=False)

    exclude = {len(words) - 1}
    i = _position_weighted_index(words, rng, bias, exclude=exclude)
    if i is None:
        i = rng.randint(0, len(words) - 2)

    max_j = min(len(words) - 1, i + max_distance)
    j = rng.randint(i + 1, max_j)

    new_words = words.copy()
    new_words[i], new_words[j] = new_words[j], new_words[i]

    event = ErrorEvent(
        category="Word Order / Transposition",
        word_index=i,
        original_form=f"{words[i]} … {words[j]}",
        corrupted_form=f"{new_words[i]} … {new_words[j]}",
    )
    return ErrorResult(words=new_words, events=[event])


# ---------------------------------------------------------------------------
# Category registries
# ---------------------------------------------------------------------------

NEW_TAXONOMY_REGISTRY: dict = {
    "Morphological":               generate_morphological,
    "Addition / Omission":         generate_addition_omission,
    "Phonological-Orthographic":   generate_phonological,
    "Lexical / Contextual":        generate_lexical,
    "Diacritic / Punctuation":     generate_diacritic,
    "Word Order / Transposition":  generate_word_order,
}

OLD_TAXONOMY_REGISTRY: dict = {
    "add":        ("Addition / Omission",        {"omit_ratio": 0.0}),
    "delete":     ("Addition / Omission",        {"omit_ratio": 1.0}),
    "accent":     ("Diacritic / Punctuation",    {}),
    "word_order": ("Word Order / Transposition", {}),
    "punctuation":("Diacritic / Punctuation",    {}),
    # "change" resolved dynamically
}


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def _call_new_category(
    category: str,
    words: list,
    rng: random.Random,
    vocab: list = None,
    position_bias: dict = None,
) -> ErrorResult:
    """
    Call the appropriate new-taxonomy function for `category`,
    forwarding only the kwargs each function accepts.
    """
    fn = NEW_TAXONOMY_REGISTRY.get(category)
    if fn is None:
        raise GenerationError(
            f"Unknown new-taxonomy category '{category}'. "
            f"Valid: {list(NEW_TAXONOMY_REGISTRY)}"
        )

    # Per-category position bias override
    pb = None
    if position_bias and category in position_bias:
        pb = position_bias[category]

    if category == "Morphological":
        return fn(words, rng, position_bias=pb)
    elif category == "Addition / Omission":
        return fn(words, rng, position_bias=pb)
    elif category == "Phonological-Orthographic":
        return fn(words, rng, position_bias=pb)
    elif category == "Lexical / Contextual":
        return fn(words, rng, position_bias=pb, vocab=vocab)
    elif category == "Diacritic / Punctuation":
        return fn(words, rng, position_bias=pb)
    elif category == "Word Order / Transposition":
        return fn(words, rng, position_bias=pb)
    else:
        return fn(words, rng)


def _resolve_old_change(rng: random.Random) -> str:
    """
    Resolve old "change" to a new-taxonomy category using the empirical split
    derived from the 287-record re-annotated Psellos dataset (§3 of spec).
    """
    cats = list(OLD_CHANGE_SPLIT.keys())
    wts = list(OLD_CHANGE_SPLIT.values())
    return rng.choices(cats, weights=wts, k=1)[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_single_error(
    text: str,
    category: str,
    rng: random.Random,
    taxonomy: str = "new",
    vocab: list = None,
    position_bias: dict = None,
) -> ErrorResult:
    """
    Apply one category function to a clean sentence.

    Args:
        text          -- clean input sentence
        category      -- category name (new or old taxonomy depending on `taxonomy`)
        rng           -- seeded random.Random
        taxonomy      -- "new" (default) or "old"
        vocab         -- word pool for Lexical/Addition functions
        position_bias -- dict {category_name: (alpha, beta)} to override defaults

    Returns:
        ErrorResult
    """
    words = text.split()

    if taxonomy == "new":
        return _call_new_category(
            category, words, rng,
            vocab=vocab, position_bias=position_bias,
        )

    elif taxonomy == "old":
        if category == "change":
            resolved = _resolve_old_change(rng)
            return _call_new_category(
                resolved, words, rng,
                vocab=vocab, position_bias=position_bias,
            )
        elif category in OLD_TAXONOMY_REGISTRY:
            new_cat, extra_kwargs = OLD_TAXONOMY_REGISTRY[category]
            fn = NEW_TAXONOMY_REGISTRY[new_cat]
            pb = (position_bias or {}).get(new_cat)
            # Merge extra_kwargs (e.g. omit_ratio override for add/delete)
            if new_cat == "Addition / Omission":
                return fn(words, rng, position_bias=pb, **extra_kwargs)
            elif new_cat == "Phonological-Orthographic":
                return fn(words, rng, position_bias=pb, **extra_kwargs)
            elif new_cat == "Lexical / Contextual":
                return fn(words, rng, position_bias=pb, vocab=vocab, **extra_kwargs)
            return fn(words, rng, position_bias=pb, **extra_kwargs)
        else:
            raise GenerationError(
                f"Unknown old-taxonomy category '{category}'. "
                f"Valid: {list(OLD_TAXONOMY_REGISTRY) + ['change']}"
            )
    else:
        raise GenerationError(f"Unknown taxonomy '{taxonomy}'. Use 'new' or 'old'.")


def compose_errors(
    text: str,
    categories: list,
    rng: random.Random,
    taxonomy: str = "new",
    vocab: list = None,
    position_bias: dict = None,
    contingent: bool = True,
) -> ErrorResult:
    """
    Apply two or more category functions in sequence, feeding each output into
    the next. Records dependency relationships in ErrorEvent.depends_on.

    Args:
        categories  -- ordered list of category names to apply
        contingent  -- if True, each event records depends_on = index of prior event. 
                        If False, events are independent (coincidental co-occurrence; depends_on stays None).

    Returns:
        ErrorResult with one event per successfully applied category.
        Stops and returns success=False if any intermediate step fails.
    """
    current_words = text.split()
    all_events = []

    for category in categories:
        step_result = generate_single_error(
            " ".join(current_words),
            category,
            rng,
            taxonomy=taxonomy,
            vocab=vocab,
            position_bias=position_bias,
        )

        if not step_result.success:
            return ErrorResult(words=current_words, events=all_events, success=False)

        for event in step_result.events:
            if contingent and all_events:
                event.depends_on = len(all_events) - 1
            all_events.append(event)

        current_words = step_result.words

    return ErrorResult(words=current_words, events=all_events, success=True)


def generate_corpus(
    clean_sentences: list,
    rng: random.Random,
    category_weights: dict = None,
    taxonomy: str = "new",
    vocab: list = None,
    dual_error_rate: float = 0.101,
    dual_error_pairs: list = None,
    position_bias: dict = None,
    max_retries_per_sentence: int = 5,
) -> list:
    """
    Generate a synthetic (corrupt, correct) corpus from a list of clean sentences.

    Args:
        clean_sentences  -- list of clean Greek sentences
        rng              -- seeded random.Random (seed=42 by convention)
        category_weights -- dict {category_name: weight}. Weights normalised
                            internally; need not sum to 1. Defaults to
                            NEW_TAXONOMY_WEIGHTS (empirical real-data distribution).
        taxonomy         -- "new" or "old" (see generate_single_error)
        vocab            -- word pool for Lexical / Addition functions
        dual_error_rate  -- fraction of examples receiving compose_errors()
                            (default 0.101 = real-world 10.1% dual-error rate)
        dual_error_pairs -- list of (cat_a, cat_b) tuples for dual-error examples.
                            Defaults to EMPIRICAL_DUAL_PAIRS from the dataset.
        position_bias    -- dict {category: (alpha, beta)} to override defaults
        max_retries_per_sentence -- retry budget

    Returns:
        list[dict] with keys:
            id, original, corrupt, events
            events: list of dicts with keys:
                category, word_index, original_form, corrupted_form, depends_on
    """
    weights = category_weights or (NEW_TAXONOMY_WEIGHTS if taxonomy == "new" else OLD_TAXONOMY_WEIGHTS)
    pairs = dual_error_pairs or (EMPIRICAL_DUAL_PAIRS if taxonomy == "new" else OLD_DUAL_PAIRS)

    # Normalise weights
    categories = list(weights.keys())
    raw_weights = list(weights.values())
    total = sum(raw_weights)
    norm_weights = [w / total for w in raw_weights]

    corpus = []
    for i, sentence in enumerate(clean_sentences):
        result = ErrorResult(words=sentence.split(), success=False)

        for _ in range(max_retries_per_sentence):
            is_dual = rng.random() < dual_error_rate

            if is_dual:
                cat_a, cat_b = rng.choice(pairs)
                result = compose_errors(
                    sentence, [cat_a, cat_b], rng,
                    taxonomy=taxonomy, vocab=vocab,
                    position_bias=position_bias,
                )
            else:
                category = rng.choices(categories, weights=norm_weights, k=1)[0]
                result = generate_single_error(
                    sentence, category, rng,
                    taxonomy=taxonomy, vocab=vocab,
                    position_bias=position_bias,
                )

            if result.success and result.text() != sentence:
                break

        if not result.success:
            continue

        corpus.append({
            "id": i,
            "original": sentence,
            "corrupt": result.text(),
            "events": [
                {
                    "category":       e.category,
                    "word_index":     e.word_index,
                    "original_form":  e.original_form,
                    "corrupted_form": e.corrupted_form,
                    "depends_on":     e.depends_on,
                }
                for e in result.events
            ],
        })

    return corpus


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(
        description="Generate a synthetic Greek scribal error corpus."
    )
    parser.add_argument("--input",    required=True,  help="Input .txt file (one sentence per line)")
    parser.add_argument("--output",   required=True,  help="Output .jsonl file path")
    parser.add_argument("--taxonomy", default="new",  choices=["new", "old"], help="Taxonomy to use")
    parser.add_argument("--seed",     default=42,     type=int, help="Random seed")
    parser.add_argument("--dual_rate",default=0.101,  type=float, help="Dual-error rate (default: 0.101)")
    parser.add_argument("--vocab",      default=None,  help="Path to vocabulary .txt file")
    parser.add_argument("--min_length", default=5, type=int,
                        help="Minimum word count per sentence (default: 5). "
                             "Filters out short fragment lines before generation.")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        all_lines = [unicodedata.normalize("NFC", line.strip()) for line in f if line.strip()]

    # Apply minimum word-count filter to remove short fragments that produce
    # low quality training pairs. Default 5 words matches the real-data
    # minimum meaningful sentence length observed in the Psellos corpus.
    sentences = [s for s in all_lines if len(s.split()) >= args.min_length]
    skipped = len(all_lines) - len(sentences)
    if skipped:
        print(f"Filtered {skipped} sentences below --min_length {args.min_length} words.")

    vocab = load_vocab(args.vocab) if args.vocab else None
    rng = random.Random(args.seed)

    corpus = generate_corpus(
        sentences,
        rng=rng,
        taxonomy=args.taxonomy,
        vocab=vocab,
        dual_error_rate=args.dual_rate,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for record in corpus:
            f.write(_json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Generated {len(corpus)} records from {len(sentences)} sentences -> {args.output}")
