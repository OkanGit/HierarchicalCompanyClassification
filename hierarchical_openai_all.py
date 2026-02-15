"""Full implementation of the hierarchical classification framework with modular strategies, dynamic prompt generation, and multi-API support (OpenAI, Gemini, DeepSeek)."""

import os
import json
import re
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import mlflow
import nest_asyncio

from openai import AsyncOpenAI
from google.genai import Client as GeminiClient
from typing import Any

# Apply nest_asyncio for Jupyter compatibility
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 1. Configuration & Setup
# --------------------------------------------------------------------------

# Ensure the relevant API keys are in your environment variables:
# OPENAI_API_KEY for GPT models
# GOOGLE_API_KEY for Gemini models
# DEEPSEEK_API_KEY for DeepSeek models

DEFAULT_MODEL = "gpt-5" 
TEMPERATURE = 1 

# DeepSeek uses an OpenAI-compatible API endpoint
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# --------------------------------------------------------------------------
# NEW: Prompt Hub Function
# --------------------------------------------------------------------------
# REPLACE your existing get_system_prompt function with this one.
# No other changes are needed in your file.

def get_system_prompt(strategy_name: str, **kwargs) -> str:
    """
    Retrieves the system prompt template for a given strategy.
    Dynamically fills in placeholders for strategies that require it.
    """
    # prompts = {
        # "global_multioutput": (
        #     "You are a meticulous financial analyst specializing in the Global Industry Classification Standard (GICS).\n"
        #     "Your task is to analyze a company's 10-K filing summary and assign the correct four-level GICS classification codes.\n\n"
        #     "Follow these steps:\n"
        #     "1. Read the text carefully to understand the company's primary revenue-generating activities.\n"
        #     "2. Determine the broadest Sector code.\n"
        #     "3. Based on the Sector, determine the more specific Group code.\n"
        #     "4. Continue this process for the Industry and Sub-Industry codes.\n"
        #     "5. Format your final answer as a single, valid JSON object.\n\n"
        #     "### Example\n"
        #     "10-K Excerpt: \"Norfolk Southern is an Atlanta-based freight railroad company that owns and operates Norfolk Southern Railway, running roughly 19,200 route miles across 22 states and the District of Columbia to move goods domestically and to/from ports. Its core operations are train transportation, track and equipment maintenance, terminal/interchange services and port connections, hauling merchandise (agriculture, chemicals, metals, automotive), intermodal containers and trailers, and coal for power and export markets. Strategically it emphasizes safe, reliable, capital-intensive rail service and the largest intermodal network in the eastern U.S., targeting manufacturers, distribution centers, power plants, exporters and major shippers across the Southeast, Midwest and East Coast.\"\n"
        #     # CORRECTED: Escaped curly braces for the JSON example
        #     "Correct JSON Output: {{ \"gsector\": \"20\", \"ggroup\": \"2030\", \"gind\": \"203040\", \"gsubind\": \"20304010\" }}\n\n"
        #     "### Your Task\n"
        #     "Now, analyze the following 10-K excerpt and return only the JSON object with the correct codes.\n"
        #     # CORRECTED: Escaped curly braces for the final instruction
        #     "Your response MUST be a single, valid JSON object and nothing else. Your answer should begin with {{ and end with }}."
        # ),
        # "global_multioutput": (
        #     "You are a meticulous financial analyst specializing in GICS.\n"
        #     "Your task: Assign 4-level GICS codes based on the 10-K summary.\n\n"
        #     "CRITICAL INSTRUCTIONS:\n"
        #     "1. Identify the primary revenue-generating activity.\n"
        #     "2. Extract specific 'focus_words' (3-5 phrases) that led to your decision.\n"
        #     "3. Format your final answer as a JSON object.\n\n"
        #     "AVAILABLE CATEGORIES (Code - Name):\n"
        #     "{hierarchy_context}\n\n" # We will inject the names here
        #     "Correct JSON Output Example:\n"
        #     "{{ \"gsector\": \"20\", \"ggroup\": \"2030\", \"gind\": \"203040\", \"gsubind\": \"20304010\", "
        #     "\"focus_words\": [\"freight railroad\", \"route miles\", \"hauling merchandise\"] }}\n\n"
        #     "Return ONLY the JSON object."
        # ),
        # "global_flat": (
        #     "You are a GICS classification expert. Your task is to read a company's 10-K filing excerpt and determine its complete GICS path, outputting it as a single string of codes separated by ' > '.\n\n"
        #     "INSTRUCTIONS:\n"
        #     "- Focus on the company's core business to avoid being misled by descriptions of its partners or secondary activities.\n"
        #     "- The output must be a string containing four numeric codes in hierarchical order.\n"
        #     "- Format: SECTOR_CODE > GROUP_CODE > INDUSTRY_CODE > SUBINDUSTRY_CODE\n\n"
        #     "### Example\n"
        #     "10-K: \"Norfolk Southern is an Atlanta-based freight railroad company that owns and operates Norfolk Southern Railway, running roughly 19,200 route miles across 22 states... Its core operations are train transportation, track and equipment maintenance... hauling merchandise... intermodal containers and trailers, and coal...\"\n"
        #     "Output: 20 > 2030 > 203040 > 20304010\n\n"
        #     "### Common Mistake to Avoid\n"
        #     "A marketing company that serves retailers should be classified under 'Advertising' (50201010), not under a retail-related code. Do not classify a company based on its clients' industries.\n\n"
        #     "Return only the string path. Do not include JSON, code names, or any explanation."
        # ),
        # "local_per_level": (
        #     "You are a specialized GICS classification model focused on a single hierarchical level.\n"
        #     "Your task is to analyze the provided 10-K excerpt and determine the correct GICS code ONLY for the '{level}' level.\n"
        #     "Ignore information that pertains to other, broader, or more specific levels. Focus exclusively on identifying the best fit for the '{level}' level.\n\n"
        #     "### Example for the 'Industry' Level\n"
        #     "10-K: \"Koss Corporation designs, sources and sells stereo headphones and related personal listening accessories, reporting as a single business segment focused on the audio/video home entertainment and communications market... its operations center on product design, sourcing finished headphones from Asian manufacturers and producing components...\"\n"
        #     # CORRECTED: Escaped curly braces for the JSON example
        #     "Output for Level 'gind': {{ \"gind\": \"252010\" }}\n\n"
        #     "Now, for the provided text, return a JSON object with a single key, '\"{level}\"', and its corresponding code as the value.\n"
        #     "Return only the JSON object."
        # ),
        # "top_down_cascade": (
        #     "You are a logical, hierarchical classifier performing a single step in a GICS classification.\n"
        #     "{context_instruction}\n\n"
        #     "### INSTRUCTIONS\n"
        #     "Your task is to classify the following text into exactly ONE of the following valid '{level}' codes. You MUST choose from this list:\n"
        #     "[{options_str}]\n\n"
        #     "Critically evaluate each of the provided options and select the single most relevant code. Do not invent a code or select one not on the list.\n\n"
        #     "### Example\n"
        #     "Context: The parent category is 45.\n"
        #     "Level: ggroup\n"
        #     "Valid Options: [4510, 4520, 4530]\n"
        #     "10-K: \"Genpact is a global provider of technology-enabled business services that transforms and runs clients’ core operations—especially finance, risk, supply chain and industry-specific processes—by combining deep industry expertise, operational excellence and AI/analytics...\"\n"
        #     # CORRECTED: Escaped curly braces for the JSON example
        #     "Output: {{ \"ggroup\": \"4510\" }}\n\n"
        #     "Return a JSON object with the key '\"{level}\"' and your selected code as the value."
        # ),
        # "path_probability": (
        #      "You are a probabilistic GICS classification model.\n"
        #     "{context_instruction}\n\n"
        #     "### INSTRUCTIONS\n"
        #     "Your task is to analyze the 10-K excerpt and assign a probability distribution across the following valid '{level}' codes ONLY. This demonstrates your confidence in each option.\n"
        #     "Valid Codes: [{options_str}]\n\n"
        #     "Follow these steps:\n"
        #     "1. Carefully read the text.\n"
        #     "2. For each code in the list, assess its relevance to the company's core business.\n"
        #     "3. Assign a probability (a float between 0.0 and 1.0) to each code.\n"
        #     "4. CRITICAL: Ensure the sum of all probabilities is exactly 1.0.\n\n"
        #     "### Example\n"
        #     "Context: The parent category is 2020.\n"
        #     "Level: gind\n"
        #     "Valid Options: [202010, 202020]\n"
        #     "10-K: \"Robert Half delivers specialized talent solutions and business consulting...supplies contract and permanent professionals across finance & accounting, technology, marketing & creative, legal and administrative/customer support; and Protiviti, a global consulting and managed-services firm...\"\n"
        #     # CORRECTED: Escaped curly braces for the JSON example
        #     "Output: {{ \"202010\": 0.7, \"202020\": 0.3 }}\n\n"
        #     "Return ONLY the JSON object. The keys must be the codes, and the values must be floats that sum to 1.0."
        # ),
        # "local_per_node_binary": (
        #     "You are a precise, binary GICS node classifier performing a strict verification.\n"
        #     "Your task is to read the 10-K excerpt and decide if the company's PRIMARY business unequivocally belongs to the GICS category code: '{code}'.\n\n"
        #     "### INSTRUCTIONS\n"
        #     "- This is a strict YES or NO decision.\n"
        #     "- Return `true` ONLY if the company's core, revenue-generating activity is a direct and clear match for the code.\n"
        #     "- Return `false` if the company is only tangentially related, serves clients in this sector, or if the description is ambiguous.\n"
        #     "- Be conservative in your judgment.\n\n"
        #     "### Example\n"
        #     "Code: 403010 (Thrifts & Mortgage Finance)\n"
        #     "10-K: \"HG Holdings, through its subsidiaries, operates as a title insurance underwriter and title agency—primarily in Florida—issuing lender and owner title insurance and providing closing, escrow and settlement services for residential and commercial real estate transactions...\"\n"
        #     # CORRECTED: Escaped curly braces for the JSON example
        #     "Output: {{ \"match\": true }}\n\n"
        #     # CORRECTED: Escaped curly braces for the final instruction
        #     "Your response MUST be a JSON object strictly in the format `{{ \"match\": true }}` or `{{ \"match\": false }}`."
        # )
    # }
    
    prompts = {

        # ============================================================
        # GLOBAL MULTI-OUTPUT (REFERENCE STRATEGY)
        # ============================================================
        "global_multioutput": (
            "You are a senior financial analyst and taxonomy expert specializing in GICS.\n\n"

            "OBJECTIVE:\n"
            "Given a 10-K excerpt, determine the correct FOUR-LEVEL GICS classification "
            "(Sector → Group → Industry → Sub-Industry).\n\n"

            "PLANNING STEPS (DO NOT OUTPUT THESE STEPS):\n"
            "1. Identify the company's PRIMARY revenue-generating activity.\n"
            "2. Ignore customers, partners, or end markets unless the company itself operates there.\n"
            "3. Map activity → Sector → Group → Industry → Sub-Industry.\n"
            "4. Internally verify that each deeper level is a valid child of the previous level.\n"
            "5. Extract short textual evidence phrases that justify the decision.\n\n"

            "ERROR AVOIDANCE HEURISTICS:\n"
            "- Do NOT classify based on who the company serves.\n"
            "- Do NOT over-weight minor or secondary business segments.\n"
            "- Prefer the most specific Sub-Industry that directly matches core operations.\n\n"

            "AVAILABLE GICS CODES (Code: Name):\n"
            "{hierarchy_context}\n\n"

            "OUTPUT FORMAT (STRICT):\n"
            "Return ONE valid JSON object with EXACTLY these fields:\n"
            "- gsector (string)\n"
            "- ggroup (string)\n"
            "- gind (string)\n"
            "- gsubind (string)\n"
            "- focus_words (array of 3–5 short phrases copied or paraphrased from the text)\n\n"

            "EXAMPLE:\n"
            "{{\n"
            "  \"gsector\": \"20\",\n"
            "  \"ggroup\": \"2030\",\n"
            "  \"gind\": \"203040\",\n"
            "  \"gsubind\": \"20304010\",\n"
            "  \"focus_words\": [\"freight railroad\", \"rail network\", \"hauling merchandise\"]\n"
            "}}\n\n"

            "Return ONLY the JSON object. No explanations."
        ),

        # ============================================================
        # GLOBAL FLAT (PATH STRING + FOCUS WORDS)
        # ============================================================
        "global_flat": (
            "You are a senior financial analyst and GICS taxonomy expert.\n\n"

            "OBJECTIVE:\n"
            "Analyze the provided 10-K excerpt and determine the full FOUR-LEVEL GICS path "
            "(Sector > Group > Industry > Sub-Industry) as a single serialized string.\n\n"

            "PLANNING STEPS (DO NOT OUTPUT THESE STEPS):\n"
            "1. Identify the company's core business model and primary source of revenue.\n"
            "2. Map the business to the GICS hierarchy starting from the broad Sector down to the Sub-Industry.\n"
            "3. Ensure the final path is logically consistent (each child belongs to the parent).\n"
            "4. Select 3-5 keywords that prove this specific path is the correct one.\n\n"

            "ERROR AVOIDANCE HEURISTICS:\n"
            "- Ensure the path uses the numeric codes, not just names.\n"
            "- Do not confuse 'who they sell to' (the market) with 'what they do' (the industry).\n"
            "- If a company is diversified, choose the path representing the largest revenue contributor.\n\n"

            "AVAILABLE GICS CODES:\n"
            "{hierarchy_context}\n\n"

            "OUTPUT FORMAT (STRICT):\n"
            "Return ONE valid JSON object with EXACTLY these fields:\n"
            "- path (string in format 'CODE > CODE > CODE > CODE')\n"
            "- focus_words (array of 3–5 short phrases from the text)\n\n"

            "EXAMPLE:\n"
            "{{\n"
            "  \"path\": \"20 > 2010 > 201010 > 20101010\",\n"
            "  \"focus_words\": [\"heavy lift aircraft\", \"aerospace manufacturing\", \"defense contracts\"]\n"
            "}}\n\n"

            "Return ONLY the JSON object."
        ),

        # ============================================================
        # LOCAL PER LEVEL (SINGLE LEVEL + FOCUS WORDS)
        # ============================================================
        "local_per_level": (
            "You are a GICS specialist tasked with classifying a company at the '{level}' level ONLY.\n\n"

            "OBJECTIVE:\n"
            "Based on the 10-K excerpt, identify the single most accurate GICS code for the '{level}' tier.\n\n"

            "PLANNING STEPS (DO NOT OUTPUT THESE STEPS):\n"
            "1. Determine the company's primary operational focus.\n"
            "2. Review the list of available codes for the '{level}' tier.\n"
            "3. Select the code that most precisely encompasses the company's core products or services.\n"
            "4. Verify that you are not selecting a code from a different hierarchical level.\n\n"

            "ERROR AVOIDANCE HEURISTICS:\n"
            "- Focus exclusively on the definition of the '{level}' tier.\n"
            "- Ignore secondary business lines that do not drive the majority of value.\n"
            "- Accuracy at this specific level is the priority; do not provide the full path.\n\n"

            "AVAILABLE '{level}' CODES:\n"
            "{hierarchy_context}\n\n"

            "OUTPUT FORMAT (STRICT):\n"
            "Return ONE valid JSON object with EXACTLY these fields:\n"
            "- {level} (string: the chosen code)\n"
            "- focus_words (array of 2–4 phrases justifying this specific level choice)\n\n"

            "EXAMPLE (for level='gind'):\n"
            "{{\n"
            "  \"gind\": \"451030\",\n"
            "  \"focus_words\": [\"software as a service\", \"enterprise cloud platforms\"]\n"
            "}}\n\n"

            "Return ONLY the JSON object."
        ),

        # ============================================================
        # TOP-DOWN CASCADE (CHOICE + FOCUS WORDS)
        # ============================================================
        "top_down_cascade": (
            "You are a GICS auditor performing a targeted classification step.\n\n"

            "OBJECTIVE:\n"
            "Given the company's current classification context: {context_instruction}\n"
            "Select the most appropriate NEXT-LEVEL code ('{level}') from the provided list of children.\n\n"

            "PLANNING STEPS (DO NOT OUTPUT THESE STEPS):\n"
            "1. Analyze the 10-K text for specific evidence related to the provided options.\n"
            "2. Contrast the available choices: Why is one option more accurate than the others?\n"
            "3. Ensure the selection remains consistent with the previously established parent category.\n\n"

            "ERROR AVOIDANCE HEURISTICS:\n"
            "- If the text is ambiguous, select the most traditional or 'core' business option.\n"
            "- Do not choose an option simply because a keyword appears; ensure it represents a primary revenue activity.\n\n"

            "VALID OPTIONS FOR '{level}':\n"
            "[{options_str}]\n\n"

            "OUTPUT FORMAT (STRICT):\n"
            "Return ONE valid JSON object with EXACTLY these fields:\n"
            "- {level} (string: the selected code)\n"
            "- focus_words (array of 2–4 evidence phrases)\n\n"

            "EXAMPLE:\n"
            "{{\n"
            "  \" {level}\": \"101020\",\n"
            "  \"focus_words\": [\"oil and gas exploration\", \"upstream production\"]\n"
            "}}\n\n"

            "Return ONLY the JSON object."
        ),

        # ============================================================
        # PATH PROBABILITY (PROBS + FOCUS WORDS)
        # ============================================================
        "path_probability": (
            "You are a probabilistic GICS classification engine.\n\n"

            "OBJECTIVE:\n"
            "Evaluate the likelihood that the company belongs to the following GICS categories: {level}.\n\n"

            "PLANNING STEPS (DO NOT OUTPUT THESE STEPS):\n"
            "1. Parse the 10-K to identify all revenue-generating segments.\n"
            "2. Compare the business segments against the definitions of the options provided.\n"
            "3. Assign a probability (0.0 to 1.0) to each option based on how well it fits the core business.\n"
            "4. Ensure the sum of all probabilities equals 1.0.\n\n"

            "ERROR AVOIDANCE HEURISTICS:\n"
            "- Be decisive: If one category is clearly the primary, assign it high confidence (>0.8).\n"
            "- Only split probabilities if the company is genuinely a multi-segment conglomerate.\n\n"

            "OPTIONS TO EVALUATE:\n"
            "[{options_str}]\n\n"

            "OUTPUT FORMAT (STRICT):\n"
            "Return ONE valid JSON object with EXACTLY these fields:\n"
            "- probabilities (an object mapping 'code' to 'float probability')\n"
            "- focus_words (array of 3–5 phrases highlighting the most likely candidates)\n\n"

            "EXAMPLE:\n"
            "{{\n"
            "  \"probabilities\": {{ \"351010\": 0.9, \"351020\": 0.1 }},\n"
            "  \"focus_words\": [\"biopharmaceutical research\", \"clinical drug trials\"]\n"
            "}}\n\n"

            "Return ONLY the JSON object."
        ),

        # ============================================================
        # LOCAL PER NODE BINARY (YES/NO + FOCUS WORDS)
        # ============================================================
        "local_per_node_binary": (
            "You are a GICS validation agent.\n\n"

            "OBJECTIVE:\n"
            "Verify with high certainty whether the company in the 10-K excerpt should be classified under GICS code '{code}'.\n\n"

            "DECISION RULES:\n"
            "- Return TRUE if and only if the primary business activity is explicitly described by the definition of GICS code '{code}'.\n"
            "- Return FALSE if the business belongs in a different code, or if this code only describes a minor part of the company.\n\n"

            "PLANNING STEPS (DO NOT OUTPUT THESE STEPS):\n"
            "1. Extract the core activity from the text.\n"
            "2. Compare it strictly against the scope of GICS code '{code}'.\n"
            "3. Identify 'disqualifiers' (activities that would place it in a neighboring code).\n\n"

            "OUTPUT FORMAT (STRICT):\n"
            "Return ONE valid JSON object with EXACTLY these fields:\n"
            "- match (boolean: true or false)\n"
            "- focus_words (array of 2–3 phrases justifying the inclusion or exclusion)\n\n"

            "EXAMPLE:\n"
            "{{\n"
            "  \"match\": true,\n"
            "  \"focus_words\": [\"semiconductor manufacturing\", \"wafer fabrication\"]\n"
            "}}\n\n"

            "Return ONLY the JSON object."
        ),
    }
    
    template = prompts.get(strategy_name, "")
    
    # Safely replace placeholders provided in kwargs
    # This prevents the NameError by checking if the key exists
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        if placeholder in template:
            template = template.replace(placeholder, str(value))
            
    return template

# --------------------------------------------------------------------------
# 2. Hierarchy & Parsing Utilities
# --------------------------------------------------------------------------

def read_gics_hierarchy(gics_csv_path: str, sep=';') -> pd.DataFrame:
    """Reads GICS structure. Assumes columns: gsector, ggroup, gind, gsubind"""
    df = pd.read_csv(gics_csv_path, sep=sep, dtype=str).fillna('')
    # Ensure codes are stripped strings
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df

def build_hierarchy_tree(df: pd.DataFrame, levels: List[str]) -> Dict[str, List[str]]:
    """
    Builds a dictionary: Parent_Code -> [List of Valid Child Codes].
    Used for Top-Down Cascade to constrain model options.
    """
    tree = {'ROOT': sorted(list(set(df[levels[0]].unique()) - {''}))}
    
    for i in range(len(levels) - 1):
        parent_col = levels[i]
        child_col = levels[i+1]
        
        # Get pairs of (parent, child)
        pairs = df[[parent_col, child_col]].drop_duplicates()
        
        for _, row in pairs.iterrows():
            p, c = row[parent_col], row[child_col]
            if p and c:
                if p not in tree:
                    tree[p] = []
                if c not in tree[p]:
                    tree[p].append(c)
    return tree

def build_canonical_path_from_row(row: pd.Series, label_cols: List[str], sep=' > ') -> str:
    parts = []
    for c in label_cols:
        # 1. Convert to string and strip whitespace
        val = str(row[c]).strip()
        
        # 2. Remove trailing '.0' if present (e.g., "10.0" -> "10")
        val = re.sub(r'\.0$', '', val)
        
        # 3. Append only if value exists and isn't "nan"
        if val and val.lower() != 'nan':
            parts.append(val)
            
    return sep.join(parts)

def extract_json_from_text(text: str) -> Dict:
    """Robust JSON extractor."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Regex fallback
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if not m:
        return {}
    js_text = m.group(0).replace("'", '"')
    # Cleanup trailing commas/bad formatting
    js_text = re.sub(r',\s*}', '}', js_text)
    js_text = re.sub(r',\s*\]', ']', js_text)
    try:
        return json.loads(js_text)
    except Exception:
        return {}

def summarize_text_fast(text: str, max_chars=12000) -> str:
    if not isinstance(text, str): return ""
    if len(text) <= max_chars: return text
    third = max_chars // 3
    return text[:third] + "\n...[SNIP]...\n" + text[len(text)//2 : len(text)//2 + third] + "\n...[SNIP]...\n" + text[-third:]

# Add this helper to create a mapping for prompts
def get_gics_mapping(gics_csv_path: str):
    # Assuming CSV has: gsector, gsector_name, ggroup, ggroup_name, etc.
    # If it only has codes, you'll need a CSV that includes the labels.
    df = pd.read_csv(gics_csv_path, sep=';', dtype=str)
    mapping = {}
    for _, row in df.iterrows():
        # Map code -> Name (adjust column names to match your GICS file)
        mapping[row['gsector']] = row.get('gsector_name', 'Unknown Sector')
        mapping[row['ggroup']] = row.get('ggroup_name', 'Unknown Group')
        mapping[row['gind']] = row.get('gind_name', 'Unknown Industry')
        mapping[row['gsubind']] = row.get('gsubind_name', 'Unknown Sub-Industry')
    return mapping
# --------------------------------------------------------------------------
# 3. Async Inference Engine
# --------------------------------------------------------------------------

# async def call_openai(text: str, system_prompt: str, json_mode: bool = True) -> str:
#     """
#     Standard Chat Completion Wrapper.
#     """
#     try:
#         # Inject text into user prompt format provided in request
#         user_content = f"<<<START>>>\n{text}\n<<<END>>>"
        
#         kwargs = {
#             "model": MODEL_NAME,
#             "messages": [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_content}
#             ],
#             "temperature": TEMPERATURE
#         }
        
#         if json_mode:
#             kwargs["response_format"] = {"type": "json_object"}

#         response = await client.chat.completions.create(**kwargs)
#         return response.choices[0].message.content
#     except Exception as e:
#         logger.error(f"API Error: {e}")
#         return "{}"
# ADD THIS ENTIRE BLOCK

# --- NEW: Function to set up the correct API client ---
def setup_client(model_name: str) -> Any:
    """Initializes and returns the correct API client based on the model name."""
    logger.info(f"Setting up client for model: {model_name}")

    if "gpt" in model_name:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        return AsyncOpenAI()

    elif "gemini" in model_name:
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

        # Create async Gemini client
        return GeminiClient(api_key=os.environ["GOOGLE_API_KEY"]).aio

    elif "deepseek" in model_name:
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY environment variable not set.")
        return AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=DEEPSEEK_BASE_URL,
        )

    else:
        raise ValueError(f"Unsupported model name: {model_name}.")

# --- NEW: Specific API call functions ---
async def _call_openai_compatible(client: AsyncOpenAI, model_name: str, system_prompt: str, user_content: str, json_mode: bool) -> str:
    """Handles calls for OpenAI and OpenAI-compatible APIs like DeepSeek."""
    kwargs = { "model": model_name, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}], "temperature": TEMPERATURE }
    if json_mode: kwargs["response_format"] = {"type": "json_object"}
    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

async def _call_gemini(
    client: Any,  # this is GeminiClient().aio
    model_name: str,
    system_prompt: str,
    user_content: str,
    json_mode: bool,
) -> str:
    """Handles async calls for Google Gemini (google-genai)."""

    # Build the prompt (Gemini doesn’t use role messages like OpenAI)
    prompt = f"{system_prompt}\n\n{user_content}"
    if json_mode:
        # Enforce JSON via prompt instructions
        prompt = f"{prompt}\n\nRespond ONLY with valid JSON, no additional text."

    response = await client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "temperature": TEMPERATURE,
        },
    )

    return response.text


# --- REPLACE `call_openai` with this new `call_llm` dispatcher ---
async def call_llm(
    client: Any,
    model_name: str,
    text: str,
    system_prompt: str,
    json_mode: bool = True,
) -> str:
    """Generic LLM wrapper that dispatches to the correct API call function."""
    try:
        user_content = f"<<<START>>>\n{text}\n<<<END>>>"

        if "gpt" in model_name or "deepseek" in model_name:
            return await _call_openai_compatible(
                client, model_name, system_prompt, user_content, json_mode
            )

        elif "gemini" in model_name:
            return await _call_gemini(
                client, model_name, system_prompt, user_content, json_mode
            )

        else:
            logger.error(f"No API call logic for model {model_name}")
            return "{}"

    except Exception as e:
        logger.error(f"API Error with model {model_name}: {e}")
        return "{}"
# --- END OF ADDITION ---
# --------------------------------------------------------------------------
# 4. Strategies (Encapsulated Functions)
# --------------------------------------------------------------------------

async def strategy_global_multioutput(texts: List[str], target_levels: List[str], client: Any, model_name: str, **kwargs) -> List[Dict]:
    """Modified to return a list of DICTS containing path AND focus_words"""
    hierarchy_context = kwargs.get('hierarchy_context', "No context provided.")
    system_prompt = get_system_prompt("global_multioutput", hierarchy_context=hierarchy_context)
    
    logger.info("Running Global Multi-Output Strategy with Context...")
    tasks = [call_llm(client, model_name, t, system_prompt, json_mode=True) for t in texts]
    
    raw_results = []
    batch_size = kwargs.get('batch_size', 5)
    for i in range(0, len(tasks), batch_size):
        raw_results.extend(await asyncio.gather(*tasks[i:i+batch_size]))
        if len(tasks) > batch_size: time.sleep(0.1)
        
    structured_preds = []
    for res in raw_results:
        j = extract_json_from_text(res)
        vals = [str(j.get(lvl, "")) for lvl in target_levels]
        path = " > ".join([v for v in vals if v])
        
        # We return the whole dict so we don't lose focus_words
        structured_preds.append({
            "pred_path": path,
            "focus_words": j.get("focus_words", [])
        })
    return structured_preds

async def strategy_global_flat(texts: List[str], target_levels: List[str], client: Any, model_name: str, **kwargs) -> List[str]:
    system_prompt = get_system_prompt("global_flat")
    logger.info("Running Global Flat (String) Strategy...")
    tasks = [call_llm(client, model_name, t, system_prompt, json_mode=False) for t in texts]
    raw_results = []
    batch_size = kwargs.get('batch_size', 5)
    for i in range(0, len(tasks), batch_size):
        raw_results.extend(await asyncio.gather(*tasks[i:i+batch_size]))
        if len(tasks) > batch_size: time.sleep(0.2)
    structured_preds = []

    for r in raw_results:
        j = extract_json_from_text(r)
        structured_preds.append({
            "pred_path": j.get("path", ""),
            "focus_words": j.get("focus_words", [])
        })

    return structured_preds

async def strategy_local_per_level(texts: List[str], target_levels: List[str], client: Any, model_name: str, **kwargs) -> List[str]:
    logger.info("Running Local Classification Per Level...")
    batch_size = kwargs.get('batch_size', 5)
    level_predictions = {}
    for level in target_levels:
        logger.info(f"  -> Processing Level: {level}")
        system_prompt = get_system_prompt("local_per_level", level=level)
        tasks = [call_llm(client, model_name, t, system_prompt, json_mode=True) for t in texts]
        raw_res = []
        for i in range(0, len(tasks), batch_size):
            raw_res.extend(await asyncio.gather(*tasks[i:i+batch_size]))
            if len(tasks) > batch_size: time.sleep(0.2)
        level_predictions[level] = []
        for r in raw_res:
            parsed = extract_json_from_text(r)
            level_predictions[level].append({
                "code": parsed.get(level, ""),
                "focus_words": parsed.get("focus_words", [])
            })

    preds = []
    for i in range(len(texts)):
        path_parts = []
        focus_words = []

        for lvl in target_levels:
            entry = level_predictions[lvl][i]
            if entry["code"]:
                path_parts.append(str(entry["code"]))
                focus_words.extend(entry["focus_words"])

        preds.append({
            "pred_path": " > ".join(path_parts),
            "focus_words": focus_words
        })

    return preds

async def strategy_top_down_cascade(texts: List[str], target_levels: List[str], client: Any, model_name: str, **kwargs) -> List[str]:
    hierarchy_tree = kwargs.get('hierarchy_tree')
    if not hierarchy_tree: raise ValueError("Hierarchy tree required for Top-Down Cascade")
    logger.info("Running Top-Down Cascade Strategy...")
    batch_size = kwargs.get('batch_size', 5)
    current_paths = [[] for _ in texts]
    current_focus = [[] for _ in texts]

    for i, level in enumerate(target_levels):
        logger.info(f"  -> Cascading Level {i+1}: {level}")
        tasks = []
        for idx, txt in enumerate(texts):
            if i == 0:
                parent, context_instruction = 'ROOT', "Start at the top level (Sector)."
            else:
                parent = current_paths[idx][-1] if current_paths[idx] else ""
                context_instruction = f"The parent category is {parent}."
            valid_options = hierarchy_tree.get(parent, [])
            if not valid_options:
                async def dummy(): return "{}"
                tasks.append(dummy())
                continue
            options_str = ", ".join(valid_options)
            system_prompt = get_system_prompt("top_down_cascade", level=level, context_instruction=context_instruction, options_str=options_str)
            tasks.append(call_llm(client, model_name, txt, system_prompt, json_mode=True))
        level_raw_results = []
        for start_idx in range(0, len(tasks), batch_size):
            batch = tasks[start_idx : start_idx + batch_size]
            level_raw_results.extend(await asyncio.gather(*batch))
            if len(tasks) > batch_size: time.sleep(0.2)
        for idx, res in enumerate(level_raw_results):
            parsed = extract_json_from_text(res)
            val = parsed.get(level, "")
            fw = parsed.get("focus_words", [])

            if val:
                current_paths[idx].append(str(val))
                current_focus[idx].extend(fw)

    return [
        {
            "pred_path": " > ".join(p),
            "focus_words": current_focus[i]
        }
        for i, p in enumerate(current_paths)
    ]


async def strategy_path_probability(texts: List[str], target_levels: List[str], client: Any, model_name: str, **kwargs) -> List[str]:
    hierarchy_tree = kwargs.get('hierarchy_tree')
    
    if not hierarchy_tree: raise ValueError("Hierarchy tree required for Path Probability Strategy")
    logger.info("Running TRUE Path-Based Probability Strategy...")
    batch_size = kwargs.get('batch_size', 5)
    current_paths, current_probs = [[] for _ in texts], [[] for _ in texts]
    current_focus = [[] for _ in texts]

    for level_idx, level in enumerate(target_levels):
        logger.info(f"  -> Probabilistic Level {level_idx+1}: {level}")
        tasks, task_meta = [], []
        for sample_idx, txt in enumerate(texts):
            if level_idx == 0:
                parent, context_instruction = "ROOT", "Start at the top (Sector level)."
            else:
                if len(current_paths[sample_idx]) < level_idx: continue
                parent = current_paths[sample_idx][-1]
                context_instruction = f"The parent category is {parent}."
            valid_options = hierarchy_tree.get(parent, [])
            if not valid_options: continue
            options_str = ", ".join(valid_options)
            system_prompt = get_system_prompt("path_probability", level=level, context_instruction=context_instruction, options_str=options_str)
            tasks.append(call_llm(client, model_name, txt, system_prompt, json_mode=True))
            task_meta.append((sample_idx, valid_options))
        raw_results = []
        for i in range(0, len(tasks), batch_size):
            raw_results.extend(await asyncio.gather(*tasks[i:i+batch_size]))
            if len(tasks) > batch_size: time.sleep(0.2)
        for res, (sample_idx, valid_options) in zip(raw_results, task_meta):
            parsed = extract_json_from_text(res)
            prob_dict = parsed.get("probabilities", {})
            fw = parsed.get("focus_words", [])

            clean_probs = {k: float(v) for k, v in prob_dict.items() if k in valid_options and isinstance(v, (int, float))}
            if not clean_probs: continue
            Z = sum(clean_probs.values())
            if Z > 0:
                for k in clean_probs: clean_probs[k] /= Z
            best_child = max(clean_probs, key=clean_probs.get)
            current_focus[sample_idx].extend(fw)

            current_paths[sample_idx].append(best_child)
            current_probs[sample_idx].append(clean_probs[best_child])
    final_paths = []
    for path in current_paths:
        final_paths.append({
            "pred_path": " > ".join(path) if path else "",
            "focus_words": current_focus[i]
        })

    return final_paths

async def strategy_local_per_node_binary(texts: List[str], target_levels: List[str], client: Any, model_name: str, **kwargs) -> List[str]:
    hierarchy_tree = kwargs.get('hierarchy_tree')
    if not hierarchy_tree: raise ValueError("Hierarchy tree required for Local Per Node Binary Strategy")
    logger.info("Running Local Per Node (Binary) Strategy...")
    batch_size = kwargs.get('batch_size', 5)
    current_paths = [[] for _ in texts]
    current_focus = [[] for _ in texts]

    for i, level in enumerate(target_levels):
        logger.info(f"  -> Binary Check Level {i+1}: {level}")
        tasks, task_metadata = [], []
        for idx, txt in enumerate(texts):
            parent = 'ROOT' if i == 0 else (current_paths[idx][-1] if len(current_paths[idx]) == i else None)
            candidates = hierarchy_tree.get(parent, []) if parent else []
            for code in candidates:
                system_prompt = get_system_prompt("local_per_node_binary", code=code)
                tasks.append(call_llm(client, model_name, txt, system_prompt, json_mode=True))
                task_metadata.append((idx, code))
        if not tasks:
            logger.info("  -> No valid candidates found for this level, skipping.")
            continue
        logger.info(f"  -> Executing {len(tasks)} binary classification calls...")
        binary_results_raw = []
        for start_idx in range(0, len(tasks), batch_size):
            batch = tasks[start_idx : start_idx + batch_size]
            binary_results_raw.extend(await asyncio.gather(*batch))
            if len(tasks) > batch_size: time.sleep(0.2)
        sample_votes = {idx: [] for idx in range(len(texts))}
        for res, (sample_idx, code) in zip(binary_results_raw, task_metadata):
            parsed = extract_json_from_text(res)
            is_match = parsed.get("match", False)
            fw = parsed.get("focus_words", [])

            if isinstance(is_match, str): is_match = is_match.lower() == 'true'
            if is_match:
                sample_votes[sample_idx].append(code)
                current_focus[sample_idx].extend(fw)

        for idx in range(len(texts)):
            votes = sample_votes[idx]
            if votes: current_paths[idx].append(votes[0])
    return [
        {
            "pred_path": " > ".join(p),
            "focus_words": current_focus[i]
        }
        for i, p in enumerate(current_paths)
    ]


STRATEGIES = {
    "global_multioutput": strategy_global_multioutput,
    "global_flat": strategy_global_flat,
    "local_per_level": strategy_local_per_level,
    "top_down_cascade": strategy_top_down_cascade,
    "path_probability": strategy_path_probability,
    "local_per_node_binary": strategy_local_per_node_binary  # <--- ADDED
}

# --------------------------------------------------------------------------
# 5. Metrics (Preserved)
# --------------------------------------------------------------------------
def normalize_path_with_hierarchy(path, hierarchy_tree):
    parts = path.split(" > ")
    valid = []
    
    parent = "ROOT"
    for p in parts:
        if p in hierarchy_tree.get(parent, []):
            valid.append(p)
            parent = p
        else:
            break
    return " > ".join(valid)


def get_ancestors_list(path, sep=" > "):
    parts = [x.strip() for x in path.split(sep) if x.strip()]
    return parts

def lca_depth(true_list, pred_list):
    depth = 0
    for t, p in zip(true_list, pred_list):
        if t == p:
            depth += 1
        else:
            break
    return depth

def hierarchical_metrics(y_true, y_pred):
    hP_list = []
    hR_list = []
    hF_list = []
    depth_list = []  # <--- NEW LIST

    for tpath, ppath in zip(y_true, y_pred):
        T = get_ancestors_list(tpath)
        P = get_ancestors_list(ppath)

        lca = lca_depth(T, P)
        depth_list.append(lca) # <--- Track Depth

        denom_p = len(P)
        denom_t = len(T)

        if denom_p == 0 or denom_t == 0:
            hP_list.append(0.0)
            hR_list.append(0.0)
            hF_list.append(0.0)
            continue

        hP = lca / denom_p
        hR = lca / denom_t

        if (hP + hR) == 0:
            hF = 0
        else:
            hF = 2*hP*hR / (hP + hR)

        hP_list.append(hP)
        hR_list.append(hR)
        hF_list.append(hF)

    return {
        "hier_precision_mean": np.mean(hP_list),
        "hier_recall_mean": np.mean(hR_list),
        "hier_f1_mean": np.mean(hF_list),
        "common_depth_mean": np.mean(depth_list) # <--- NEW METRIC
    }

ERROR_CLASSES = {
    "CLIENT_CONFUSION": "The LLM classified the company based on who its customers are (e.g., classifying a software provider for banks as 'Financials').",
    "SECONDARY_ACTIVITY": "The LLM focused on a minor business segment or secondary revenue stream.",
    "GRANULARITY_ERROR": "The Sector/Group is correct, but the LLM failed to distinguish between specific Industries/Sub-Industries.",
    "VERTICAL_INTEGRATION": "The company operates across the value chain and the LLM picked the wrong stage.",
    "AMBIGUOUS_TEXT": "The 10-K summary lacks enough specific detail to make a distinct choice.",
    "OUTDATED_GICS": "The company belongs to a newer industry category not well captured by standard codes."
}

async def analyze_mistakes_batch(client: Any, model_name: str, mistakes_df: pd.DataFrame) -> List[Dict]:
    """Processes all mistakes in parallel and ensures dictionaries are returned."""
    tasks = []
    
    # We define the internal error categories again to ensure the LLM knows them
    error_context = "\n".join([f"- {k}: {v}" for k, v in ERROR_CLASSES.items()])

    for _, row in mistakes_df.iterrows():
        # Focus words are often a list, let's make sure they are a clean string for the prompt
        focus_words_str = row.get('focus_words', 'N/A')
        
        prompt = f"""
        Analyze this GICS classification error:
        Company Summary: {row['processed_text']}
        
        True GICS: {row['true_path']}
        Predicted GICS: {row['pred_path']}
        LLM Focus Words during decision: {focus_words_str}
        
        Valid Error Classes:
        {error_context}
        
        Identify:
        1. The first level where the error occurred (Sector, Group, Industry, or Sub-Industry).
        2. The most relevant Error Class from the list above.
        3. A brief explanation of the logic failure.

        Return ONLY a JSON object: 
        {{ "error_class": "NAME", "explanation": "...", "first_error_level": "..." }}
        """
        tasks.append(call_llm(client, model_name, prompt, "You are an AI Quality Auditor.", json_mode=True))
    
    # Run all analysis calls in parallel
    raw_responses = await asyncio.gather(*tasks)
    
    # Crucial: Ensure every item is a DICT
    processed_results = []
    for resp in raw_responses:
        parsed = extract_json_from_text(resp)
        if not isinstance(parsed, dict):
            # Fallback if JSON extraction fails completely
            processed_results.append({"error_class": "UNCATEGORIZED", "explanation": "Failed to parse LLM response"})
        else:
            processed_results.append(parsed)
            
    return processed_results

# --------------------------------------------------------------------------
# 6. Main Runner
# --------------------------------------------------------------------------

def run_experiment(
    df: pd.DataFrame,
    text_col: str,
    target_levels: List[str],
    gics_csv_path: str,
    output_dir: str,
    model_name: str = DEFAULT_MODEL,
    strategy_name: str = "global_multioutput",
    test_size: float = 0.2,
    batch_size: int = 5,
    run_name_suffix: str = "",
    with_summary: bool = True
):
    os.makedirs(output_dir, exist_ok=True)
    client = setup_client(model_name)

    # 1. Prepare Data
    eval_df = df.sample(frac=test_size, random_state=42).copy()
    eval_df['processed_text'] = eval_df[text_col].apply(summarize_text_fast) if with_summary else eval_df[text_col]
    
    # 2. Hierarchy Context Injection (Codes + Names)
    # We load the GICS CSV to provide "Semantic Context" to the LLM
    full_hierarchy = read_gics_hierarchy(gics_csv_path)
    hierarchy_tree = build_hierarchy_tree(full_hierarchy, target_levels)
    
    # Create a lookup for Code -> Name (Assuming your CSV has 'label' or 'name' columns)
    # If your CSV columns are different, adjust 'gsubind_name' etc. accordingly
    gics_names = {}
    for _, row in full_hierarchy.iterrows():
        for lvl in target_levels:
            code = row[lvl]
            name = row.get(f"{lvl}_name", "Unknown") # Fallback if names aren't in CSV
            gics_names[code] = name
    
    hierarchy_context = "\n".join([f"{code}: {name}" for code, name in gics_names.items() if code])

    # 3. Inference
    strategy_fn = STRATEGIES[strategy_name]
    loop = asyncio.get_event_loop()
    
    # We pass hierarchy_context to the prompt generator via kwargs
    raw_preds = loop.run_until_complete(
        strategy_fn(
            texts=eval_df['processed_text'].tolist(), 
            target_levels=target_levels, 
            client=client,
            model_name=model_name, 
            batch_size=batch_size, 
            hierarchy_tree=hierarchy_tree,
            hierarchy_context=hierarchy_context # New Injection
        )
    )

    # 4. Parse Results (Including Focus Words)
    # Note: strategy_fn needs to be updated to return focus_words if global_multioutput
    # For now, we assume raw_preds is a list of paths, but we can store focus_words in eval_df
    # eval_df['pred_path'] = raw_preds
    eval_df['true_path'] = eval_df.apply(lambda r: build_canonical_path_from_row(r, target_levels), axis=1)
    eval_df['pred_path'] = [r['pred_path'] for r in raw_preds]
    eval_df['focus_words'] = [str(r.get('focus_words', [])) for r in raw_preds] # Save as string for CSV

    # 5. Hierarchy Level Flags (0 = Correct, 1 = Error)
    def calculate_level_errors(row):
        t_parts = row['true_path'].split(" > ")
        p_parts = row['pred_path'].split(" > ")
        flags = {}
        for i, lvl in enumerate(target_levels):
            t = t_parts[i] if i < len(t_parts) else None
            p = p_parts[i] if i < len(p_parts) else None
            flags[f'err_{lvl}'] = 0 if (t == p and t is not None) else 1
        return pd.Series(flags)

    eval_df = pd.concat([eval_df, eval_df.apply(calculate_level_errors, axis=1)], axis=1)

# --- 6. Error Analysis (Judge LLM) ---
    # Find rows where True Path does not match Predicted Path
    mistake_mask = eval_df['true_path'] != eval_df['pred_path']
    mistakes = eval_df[mistake_mask].copy()

    if not mistakes.empty:
        logger.info(f"Analyzing {len(mistakes)} classification errors...")
        
        # Call the batch function
        analysis_results = loop.run_until_complete(analyze_mistakes_batch(client, model_name, mistakes))
        
        # Safe extraction using .get() with a default value
        # We use a loop or list comprehension now that we are sure analysis_results is a List[Dict]
        eval_df.loc[mistake_mask, 'error_class'] = [r.get('error_class', 'UNKNOWN') for r in analysis_results]
        eval_df.loc[mistake_mask, 'error_explanation'] = [r.get('explanation', 'No explanation provided') for r in analysis_results]
        eval_df.loc[mistake_mask, 'first_error_level'] = [r.get('first_error_level', 'Unknown') for r in analysis_results]
    else:
        logger.info("No classification errors to analyze.")

# --- 7. Hierarchy Level Flags (0 = Correct, 1 = Error) ---
    def generate_level_flags(row):
        t_parts = row['true_path'].split(" > ")
        p_parts = row['pred_path'].split(" > ")
        
        flags = {}
        for i, lvl in enumerate(target_levels):
            # Check if this specific level matches
            true_val = t_parts[i] if i < len(t_parts) else "MISSING_T"
            pred_val = p_parts[i] if i < len(p_parts) else "MISSING_P"
            
            # 0 if correct, 1 if wrong
            flags[f'err_flag_{lvl}'] = 0 if true_val == pred_val else 1
        return pd.Series(flags)

    # Apply the flags to the DataFrame
    level_flags_df = eval_df.apply(generate_level_flags, axis=1)
    eval_df = pd.concat([eval_df, level_flags_df], axis=1)

    # 8. Metrics & MLflow
    clf_report = classification_report(eval_df['true_path'], eval_df['pred_path'], zero_division=0, output_dict=True)
    hier_metrics = hierarchical_metrics(eval_df['true_path'].tolist(), eval_df['pred_path'].tolist())

    mlflow.set_experiment("gics_classification_detailed_analysis")
    with mlflow.start_run(run_name=f"{strategy_name}_{model_name}_{run_name_suffix}"):
        # Log basic params
        mlflow.log_param("strategy", strategy_name)
        mlflow.log_param("model", model_name)
        
        # Log Summary Stats
        mlflow.log_metric("flat_macro_f1", clf_report.get('macro avg', {}).get('f1-score', 0))
        for k, v in hier_metrics.items(): mlflow.log_metric(k, v)
        
        # Log LEVEL ERROR FLAGS (Mean error rate per level)
        for lvl in target_levels:
            error_rate = eval_df[f'err_flag_{lvl}'].mean()
            mlflow.log_metric(f"level_err_rate_{lvl}", error_rate)
            mlflow.log_metric(f"error_rate_{lvl}", eval_df[f"err_{lvl}"].mean())
            
        # Log ERROR CLASS Distributions
        if 'error_class' in eval_df:
            # Drop NaN (correct cases) and get counts
            class_counts = eval_df['error_class'].dropna().value_counts(normalize=True).to_dict()
            for cls_name, freq in class_counts.items():
                mlflow.log_metric(f"err_dist_{cls_name}", freq)

        # Save Artifacts
        csv_path = os.path.join(output_dir, f"detailed_preds_{strategy_name}.csv")
        eval_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        logger.info(f"Experiment complete. Sector Error Rate: {eval_df[f'err_{target_levels[0]}'].mean():.2%}")

    return eval_df

# --------------------------------------------------------------------------
# Example Usage Block
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy data generation for standalone testing
    data = {
        'text': [
            "We drill for oil in the sea.", 
            "We write banking software.", 
            "We sell burgers and fries."
        ] * 10,
        'gsector': ['10', '45', '25'] * 10,
        'ggroup': ['1010', '4510', '2530'] * 10,
        'gind': ['101010', '451020', '253010'] * 10,
        'gsubind': ['10101010', '45102010', '25301010'] * 10
    }
    df = pd.DataFrame(data)
    
    # Create a dummy hierarchy file for the code to work
    with open("gics_hierarchy_dummy.csv", "w") as f:
        f.write("gsector;ggroup;gind;gsubind\n")
        # Add rows corresponding to dummy data
        f.write("10;1010;101010;10101010\n")
        f.write("45;4510;451020;45102010\n")
        f.write("25;2530;253010;25301010\n")

    # Run a specific strategy
    # CHANGE 'strategy_name' to: 
    # 'global_multioutput', 'global_flat', 'local_per_level', 'top_down_cascade', 'path_probability'
    
    run_experiment(
        df=df,
        text_col='text',
        target_levels=['gsector', 'ggroup', 'gind', 'gsubind'],
        gics_csv_path="gics_hierarchy_dummy.csv",
        output_dir="results_experiment",
        strategy_name="top_down_cascade", 
        test_size=0.5,
        batch_size=2
    )