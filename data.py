"""Most of the data processing code including several methods for extracting the core business information from 10-K filings, as well as a stratified sampling function to reduce the dataset size while preserving class distribution."""""

import os

import re

import pandas as pd

from bs4 import BeautifulSoup

import spacy
from spacy.matcher import Matcher

# --- CONFIG ---
CSV_PATH = "FundamentalsNANew.csv"
FILINGS_DIR = "10k_filings_new"
CACHE_FILE = "preprocessed_filings.pkl"

# --- LOAD LABELS ---
df = pd.read_csv(CSV_PATH).drop_duplicates(subset=["cik"])
df["cik"] = df["cik"].dropna().astype(int).astype(str)  # ensure no zero padding

def loadOrBuildData():
    # --- LOAD OR BUILD PREPROCESSED DATA ---
    if os.path.exists(CACHE_FILE):
        print(f"⚡ Loading preprocessed data from {CACHE_FILE} ...")
        df_valid = pd.read_pickle(CACHE_FILE)
        print(f"✅ Loaded {len(df_valid)} preprocessed filings.\n")
    else:
        print("Reading and cleaning filings (first time, this may take a while)...")
        valid_rows = []

        for i, row in df.iterrows():
            cik = row["cik"]
            file_path = os.path.join(FILINGS_DIR, f"{cik}_10-K.html")
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        html = f.read()
                    soup = BeautifulSoup(html, "html.parser")
                    text = soup.get_text(separator=" ")
                    text = re.sub(r"\s+", " ", text)

                    row_dict = row.to_dict()
                    row_dict["text"] = text
                    valid_rows.append(row_dict)
                    print(f"Read {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
        df_valid.to_pickle(CACHE_FILE)
        print(f"💾 Saved preprocessed data to {CACHE_FILE}")
        print(f"✅ Loaded {len(df_valid)} filings out of {len(df)} companies.\n")

    return df_valid

def clean_10k_text(text: str) -> str:
    """
    A robust function to clean 10-K filing text by removing the table of 
    contents and other noise while preserving the narrative text that 
    follows section headings.

    Args:
        text (str): The raw text from a 10-K filing as a single string.

    Returns:
        str: The cleaned and processed text, with the core narrative preserved.
    """
    # 1. Start by finding the Table of Contents to define the working area.
    # This helps prevent accidental removal of legitimate content that might appear before it.
    toc_match = re.search(r'(?i)TABLE OF CONTENTS', text)
    if toc_match:
        text = text[toc_match.start():]

    # 2. Correctly remove the Table of Contents.
    # This regex now correctly uses a single '?' for a non-greedy match.
    # It removes content from "TABLE OF CONTENTS" up to the first major section.
    text = re.sub(r'(?is)TABLE OF CONTENTS.*?(?=PART\s+I\b|ITEM\s+1\b)', '', text)

    # 3. As a fallback, ensure the text starts with the first major section.
    # This cleans up any lingering introductory text if the TOC wasn't found or formatted unusually.
    first_item_match = re.search(r'(?i)(PART\s+I\b|ITEM\s+1\b)', text)
    if first_item_match:
        text = text[first_item_match.start():]

    # 4. Remove any remaining HTML/XML tags that might have been missed.
    text = re.sub(r'<[^>]+>', ' ', text)

    # 5. Remove non-alphanumeric characters, but preserve essential punctuation.
    # This is more lenient and keeps sentence structure.
    text = re.sub(r'[^\w\s.,-]', '', text)

    # 6. Normalize all whitespace (spaces, tabs, newlines) to a single space.
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_core_business_info(text: str) -> str:
    """
    Extracts and cleans the 'Item 1. Business' section from a 10-K filing.

    Uses a tempered-greedy regex pattern to capture the content between
    'Item 1.' and 'Item 1A.', preventing catastrophic backtracking even
    on large filings.

    Args:
        text (str): The raw text from a 10-K filing as a single string.

    Returns:
        str: The cleaned 'Item 1. Business' section text, or an empty string
             if the section cannot be reliably identified.
    """
    # 1. Remove HTML/XBRL tags for cleaner pattern matching
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. Define robust regex to capture between "Item 1." and "Item 1A."
    pattern = re.compile(
        r'(?is)'                           # case-insensitive, dot matches newline
        r'ITEM\s*1\.'                      # match "ITEM 1."
        r'(?:(?!ITEM\s*1A\.).)*'           # tempered greedy: anything not followed by "ITEM 1A."
        r'ITEM\s*1A\.',                    # stop right before "ITEM 1A."
    )

    # 3. Search for the match
    match = pattern.search(text)
    if not match:
        return ""

    core_text = match.group(0)

    # 4. Clean the extracted text
    cleaned_text = re.sub(r'[^\w\s.,-]', '', core_text)  # keep essential punctuation
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # normalize whitespace

    return cleaned_text

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Load the spaCy model once to be reused.
nlp = spacy.load("en_core_web_sm")

nlp.max_length = 3000000  # Increase max length if needed

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def extract_core_business_info_nlp_stemming_lemma(text: str) -> str:
    """
    Extracts core business information (Items 1 and 1A) from a 10-K filing using NLP with spaCy.
    Handles various header formats (e.g., "Item 1", "ITEM I", "Item 1A").
    """
    # 1. Light pre-cleaning to handle HTML tags and excessive newlines.
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'(\n\s*)+\n', '\n', text)  # Consolidate multiple newlines

    # Process the text with spaCy, disabling unnecessary components for speed
    doc = nlp(text, disable=["ner", "parser", "tagger"])

    # 2. Set up the spaCy Matcher with flexible patterns for section headers.
    matcher = Matcher(nlp.vocab)

    # # Define patterns for Item 1 and Item 1A, covering Roman and Arabic numerals, and upper/lower case
    # item_1_patterns = [
    #     [{"LOWER": "item"}, {"TEXT": {"IN": ["1", "i", "I"]}, "OP": "?"}, {"TEXT": {"IN": [".", ":"]}, "OP": "?"}, {"LOWER": "business"}],
    # ]
    # item_1a_patterns = [
    #     [{"LOWER": "item"}, {"TEXT": {"IN": ["1", "i", "I"]}, "OP": "?"}, {"LOWER": "a"}, {"TEXT": {"IN": [".", ":"]}, "OP": "?"}, {"LOWER": "risk"}, {"LOWER": "factors"}],
    #     [{"LOWER": "item"}, {"TEXT": {"IN": ["1a", "ia", "IA"]}, "OP": "?"}, {"TEXT": {"IN": [".", ":"]}, "OP": "?"}, {"LOWER": "risk"}, {"LOWER": "factors"}]
    # ]

    # Match exactly "ITEM 1"
    item_1_patterns = [
        [{"LOWER": "item"}, {"TEXT": {"REGEX": "1"}}, {"TEXT": {"IN": [".", ":"]}, "OP": "?"}]
    ]

    # Match exactly "ITEM 1A"
    item_1a_patterns = [
        [{"LOWER": "item"}, {"TEXT": {"REGEX": "1a"}}, {"TEXT": {"IN": [".", ":"]}, "OP": "?"}]
    ]

    # Add the patterns to the matcher
    matcher.add("ITEM_1_BUSINESS", item_1_patterns)
    matcher.add("ITEM_1A_RISK_FACTORS", item_1a_patterns)

    # 3. Find all matches in the document
    matches = matcher(doc)

    # Store matches, ensuring only the first occurrence of each header is kept
    found_headers = {}
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        if label not in found_headers:
            found_headers[label] = doc[start:end].start_char

    if not found_headers:
        return ""  # No headers found

    # Sort headers by their position in the text
    sorted_headers = sorted(found_headers.items(), key=lambda item: item[1])

    # 4. Extract content between the desired headers
    core_content = []
    sections_to_extract = ["ITEM_1_BUSINESS", "ITEM_1A_RISK_FACTORS"]

    # header_map = {label: pos for label, pos in sorted_headers}

    for i, (label, start_pos) in enumerate(sorted_headers):
        if label in sections_to_extract:
            # Determine the end position by finding the start of the next section
            end_pos = len(text)
            if i + 1 < len(sorted_headers):
                end_pos = sorted_headers[i + 1][1]

            section_text = text[start_pos:end_pos]
            core_content.append(section_text)

    if not core_content:
        return ""

    # 5. Combine and perform final cleaning on the extracted content
    final_text = " ".join(core_content)

    # Remove the header text itself from the content
    final_text = re.sub(r'(?i)ITEM\s+\d+[A-Z]?\.?\s*[\w\s\'’]+', '', final_text)

    # Process the text with spaCy
    doc = nlp(final_text)

    # Remove punctuation, stop words, lowercase, stem and lemmatize
    tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop]
    
    # Stem the tokens
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Normalize whitespace and strip
    final_text = " ".join(stemmed_tokens).strip()

    return final_text

def extract_core_business_info_fast(text: str) -> str:
    """
    Extracts everything from the first 'Item 1.' through 'Item 1A.' 
    up until the start of 'Item 2.' in a 10-K filing.
    Cleans, tokenizes, lemmatizes, and stems the output.
    """

    # Remove HTML tags + collapse newlines
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(\n\s*)+\n", "\n", text)

    # Regex: from first "Item 1." up until "Item 2."
    pattern = re.compile(
        r'(?is)'                           # case-insensitive, dot matches newline
        r'ITEM\s*1\.'                      # match "ITEM 1."
        r'(?:(?!ITEM\s*1A\.).)*'           # tempered greedy: anything not followed by "ITEM 1A."
        r'ITEM\s*1A\.',                    # stop right before "ITEM 1A."
    )
    match = pattern.search(text)
    # print(match)
    if not match:
        return ""

    section_text = match.group(0)

    # Drop the section headers themselves
    section_text = re.sub(r"(?i)item\s+1a?\.", " ", section_text)

    # Tokenize with regex
    tokens = re.findall(r"\b\w+\b", section_text.lower())

    # Lemmatize + stem
    processed = [stemmer.stem(lemmatizer.lemmatize(tok)) for tok in tokens]

    return " ".join(processed).strip()

def extract_core_business_info_fast_no_stemming(text: str) -> str:
    """
    Extracts everything from the first 'Item 1.' through 'Item 1A.' 
    up until the start of 'Item 2.' in a 10-K filing.
    Cleans, tokenizes, lemmatizes, and stems the output.
    """

    # Remove HTML tags + collapse newlines
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(\n\s*)+\n", "\n", text)

    # Regex: from first "Item 1." up until "Item 2."
    pattern = re.compile(
        r'(?is)'                           # case-insensitive, dot matches newline
        r'ITEM\s*1\.'                      # match "ITEM 1."
        r'(?:(?!ITEM\s*1A\.).)*'           # tempered greedy: anything not followed by "ITEM 1A."
        r'ITEM\s*1A\.',                    # stop right before "ITEM 1A."
    )
    match = pattern.search(text)
    # print(match)
    if not match:
        return ""

    section_text = match.group(0)

    # Drop the section headers themselves
    section_text = re.sub(r"(?i)item\s+1a?\.", " ", section_text)

    # Tokenize with regex
    tokens = re.findall(r"\b\w+\b", section_text.lower())

    return " ".join(tokens).strip()

def extract_core_business_info_nlp(text: str) -> str:
    """
    Extracts core business information from a 10-K filing using NLP with spaCy.

    This function identifies and extracts the content of "Item 1. Business", 
    "Item 1A. Risk Factors", and "Item 7. Management's Discussion and Analysis",
    which constitute the core narrative of the filing. It is more robust to
    formatting variations than a pure regex approach.

    Args:
        text (str): The raw text from a 10-K filing as a single string.

    Returns:
        str: The concatenated and cleaned text of the core business sections,
             or an empty string if the sections cannot be found.
    """
    # 1. Light pre-cleaning to handle HTML tags and excessive newlines.
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'(\n\s*)+\n', '\n', text) # Consolidate multiple newlines

    # Process the text with spaCy
    doc = nlp(text, disable=["ner", "parser"]) # Disable components we don't need for speed

    # 2. Set up the spaCy Matcher with patterns for section headers.
    # These patterns are more flexible than regex.
    matcher = Matcher(nlp.vocab)

    # Patterns for the sections we want to extract
    pattern_business = [{"LOWER": "item"}, {"IS_DIGIT": True}, {"TEXT": {"IN": [".", ":"]}, "OP": "?"}, {"LOWER": "business"}]
    pattern_risk = [{"LOWER": "item"}, {"IS_DIGIT": True}, {"LOWER": "a"}, {"TEXT": {"IN": [".", ":"]}, "OP": "?"}, {"LOWER": "risk"}, {"LOWER": "factors"}]
    pattern_mda = [{"LOWER": "item"}, {"IS_DIGIT": True}, {"TEXT": {"IN": [".", ":"]}, "OP": "?"}, {"LOWER": "management"}, {"LOWER": "'s"}]

    # Patterns for "boundary" sections to know when to stop extracting
    pattern_properties = [{"LOWER": "item"}, {"IS_DIGIT": True}, {"TEXT": {"IN": [".", ":"]}, "OP": "?"}, {"LOWER": "properties"}]
    pattern_financials = [{"LOWER": "item"}, {"IS_DIGIT": True}, {"TEXT": {"IN": [".", ":"]}, "OP": "?"}, {"LOWER": "financial"}, {"LOWER": "statements"}]
    
    matcher.add("ITEM_1_BUSINESS", [pattern_business])
    matcher.add("ITEM_1A_RISK_FACTORS", [pattern_risk])
    matcher.add("ITEM_2_PROPERTIES", [pattern_properties])
    matcher.add("ITEM_7_MDA", [pattern_mda])
    matcher.add("ITEM_8_FINANCIALS", [pattern_financials])

    # 3. Find all matches in the document
    matches = matcher(doc)
    
    # Store matches with their start/end character positions and label
    # We use a dictionary to ensure we only get the first occurrence of each unique header
    found_headers = {}
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        if label not in found_headers:
            found_headers[label] = doc[start:end].start_char
            
    if not found_headers:
        return "" # No headers found

    # Sort headers by their position in the text
    sorted_headers = sorted(found_headers.items(), key=lambda item: item[1])

    # 4. Extract the content between the desired headers
    core_content = []
    sections_to_extract = ["ITEM_1_BUSINESS", "ITEM_1A_RISK_FACTORS", "ITEM_7_MDA"]
    
    header_map = {label: pos for label, pos in sorted_headers}
    
    for i, (label, start_pos) in enumerate(sorted_headers):
        if label in sections_to_extract:
            # Find the start of the next section to define the end of this one
            end_pos = len(text)
            if i + 1 < len(sorted_headers):
                end_pos = sorted_headers[i+1][1]
            
            # For the last relevant section (MD&A), ensure it stops before financials
            if label == "ITEM_7_MDA" and "ITEM_8_FINANCIALS" in header_map:
                end_pos = min(end_pos, header_map["ITEM_8_FINANCIALS"])

            section_text = text[start_pos:end_pos]
            core_content.append(section_text)

    if not core_content:
        return ""
        
    # 5. Combine and perform final cleaning on the extracted content
    final_text = " ".join(core_content)
    
    # Remove the header text itself from the content
    final_text = re.sub(r'(?i)ITEM\s+\d+[A-Z]?\.?\s*[\w\s\'’]+', '', final_text)
    
    # Normalize whitespace and strip
    final_text = re.sub(r'\s+', ' ', final_text).strip()

    return final_text

def preprocess_10k_dataframe(data: pd.DataFrame, method = extract_core_business_info) -> pd.DataFrame:
    """
    Takes a DataFrame with a 'text' column containing 10-K filing text
    and applies a robust cleaning process that works on the entire text block,
    making it suitable for fine-tuning transformer models.

    This function processes the text in the 'text' column by:
    1.  Slicing the document to isolate the main narrative sections.
    2.  Removing large noisy blocks like the Table of Contents.
    3.  Using regex to strip out metadata, XBRL tags, boilerplate, and URLs.
    4.  Normalizing all whitespace to create a clean, continuous block of text.
    
    A new column, 'cleaned_text', is added to the DataFrame to store the result.

    Args:
        df (pd.DataFrame): A pandas DataFrame which must include a 'text' column
                           containing the raw text of 10-K filings.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'cleaned_text' column.
    """
    if 'text' not in data.columns:
        raise ValueError("Input DataFrame must have a 'text' column.")
    
    # Apply the block-based cleaning function to each entry in the 'text' column
    data['cleaned_text'] = data['text'].apply(method)
    
    return data

# --- Example Usage ---

# Create a sample DataFrame with the text you provided
sample_text = """
 AAR CORP_May 31, 2025 Common Stock, $1.00 par value AIR 0000001750...
 ... (the entire long text from the prompt) ...
 /s/ Marc J. Walfish Director Marc J. Walfish
"""
sample_df = pd.DataFrame({'text': [sample_text]})

# Run the preprocessing function
cleaned_df = preprocess_10k_dataframe(sample_df.copy())

# Check the cleaned text. It should now be well-formed and non-empty.
cleaned_text_output = cleaned_df['cleaned_text'].iloc[0]
print(f"Length of cleaned text: {len(cleaned_text_output)}")
print("\n--- Start of Cleaned Text Snippet ---")
print(cleaned_text_output[:1500])
print("\n--- End of Snippet ---")