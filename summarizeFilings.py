import pandas as pd
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # For the progress bar

def summarize_filings(df: pd.DataFrame, column_name: str, api_key: str, max_workers: int = 10) -> pd.DataFrame:
    """
    Summarizes a specific column in a dataframe using OpenAI API.
    
    Args:
        df (pd.DataFrame): The dataframe containing the text.
        column_name (str): The name of the column with text to summarize.
        api_key (str): Your OpenAI API key.
        max_workers (int): Number of parallel requests (adjust based on your rate limits).
        
    Returns:
        pd.DataFrame: The original dataframe with a new 'summary' column.
    """
    
    # Initialize OpenAI Client
    client = openai.OpenAI(api_key=api_key)
    
    # Define the System Prompt
    # system_prompt = (
    #     "You are an expert financial analyst. Your task is to summarize company filings. "
    #     "Create a concise summary (2-4 sentences) focusing strictly on the "
    #     "Business Description. Clearly explain: \n"
    #     "1. What the company does (Business Model).\n"
    #     "2. Their key tasks/operations.\n"
    #     "3. Their strategic focus or target market.\n"
    #     "Exclude generic legal jargon."
    # )

    system_prompt = """
    You are given the Business Description section of a companys SEC 10-K filing.

    Your task is to produce a factual, structured summary that preserves information relevant for industry and GICS-style hierarchical classification.

    Follow the structure exactly as specified below.
    Be concise, neutral, and faithful to the filing.
    Do NOT speculate, infer unstated facts, or use marketing language.
    Prioritize revenue-generating activities over descriptive narrative.
    Explicitly distinguish primary business activities from secondary or ancillary ones.

    If information is not stated in the filing, write "Not stated".

    ====================
    STRUCTURED SUMMARY
    ====================

    1. Company Overview
    - Legal company name:
    - Headquarters location:
    - Year founded (if stated):
    - Public listing / ticker (if stated):

    2. Primary Business Activities
    - Core products and/or services:
    - How these are delivered (e.g., manufacturing, licensing, SaaS, retail, services):
    - Primary customer type (B2B, B2C, government, mixed):

    3. Revenue Drivers and Operating Segments
    - Major operating segments (ranked by importance):
    - Segment name:
    - Description of activities:
    - Relative contribution to revenue (primary / secondary / minor or % if stated):

    4. Industry and Market Context
    - Industries the company explicitly states it operates in:
    - End markets served:
    - Position in the value chain (e.g., manufacturer, distributor, platform, service provider):

    5. Geographic Footprint
    - Primary geographic markets:
    - International operations (if any):

    6. Assets, Infrastructure, and Capabilities
    - Key physical assets relevant to operations:
    - Key non-physical assets (e.g., IP, software platforms, proprietary technology):

    7. Secondary or Ancillary Activities
    - Additional business activities not central to revenue:
    - Whether these activities are emerging, declining, or supportive (if stated):

    8. Recent Strategic Changes (if mentioned)
    - Acquisitions, divestitures, restructurings, or material strategy shifts:

    9. Revenue Concentration and Dependencies (if stated)
    - Dependence on specific customers, industries, or contracts:
    - Seasonal or cyclical characteristics:

    10. Explicit Business Classification Signals
    - Keywords or phrases the company uses to describe its business:
    - NAICS, SIC, or other industry classifications mentioned (if any):

    11. Exclusions and Clarifications
    - Activities mentioned but explicitly stated as not core:
    - Clarifications the company provides about what it does not primarily do:

    ====================
    END OF SUMMARY
    ====================
    """

    def get_summary(text):
        """Helper function to call API for a single string."""
        if not text or pd.isna(text):
            return ""
            
        # TRUNCATION: Filings are huge. The business description is usually 
        # in the first few pages. We truncate to 12,000 characters (approx 3k tokens)
        # to save money and stay within context limits.
        truncated_text = str(text)[:12000] 
        
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini", # Highly recommended for cost/speed efficiency
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Summarize this filing text:\n\n{truncated_text}"}
                ],
                # max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    # Copy dataframe to avoid SettingWithCopy warnings
    result_df = df.copy()
    
    # We use a list to store results to maintain order or map back via index
    texts = result_df[column_name].tolist()
    summaries = [None] * len(texts)

    print(f"Starting summarization of {len(texts)} filings using {max_workers} threads...")

    # Parallel Execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a mapping of future -> original index
        future_to_index = {executor.submit(get_summary, text): i for i, text in enumerate(texts)}
        
        # Use tqdm to show a progress bar
        for future in tqdm(as_completed(future_to_index), total=len(texts)):
            index = future_to_index[future]
            try:
                summaries[index] = future.result()
            except Exception as exc:
                summaries[index] = f"Generated Exception: {exc}"

    result_df['summary'] = summaries
    return result_df

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. Setup your API Key
    MY_API_KEY = "sk-..." 

    # 2. Create dummy data for demonstration
    data = {
        'filing_text': [
            "Item 1. Business. Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories. The Company sells its products worldwide...",
            "Item 1. Business. Tesla, Inc. designs, develops, manufactures, sells and leases high-performance fully electric vehicles and energy generation and storage systems..."
        ] * 5 # duplicate to fake a list
    }
    df_filings = pd.DataFrame(data)

    # 3. Run the function
    # Note: For 10k rows, this might take 20-40 minutes depending on rate limits.
    df_processed = summarize_filings(df_filings, 'filing_text', MY_API_KEY, max_workers=5)

    # 4. View results
    print(df_processed[['summary']].head())
    
    # 5. Save to CSV
    # df_processed.to_csv("summarized_filings.csv", index=False)