import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import time

# Custom user agent (SEC requires this for requests)
HEADERS = {"User-Agent": "Your Name your.email@example.com"}

def download_latest_10k(cik: str, save_dir: str = "10k_filings_new") -> str:
    """
    Downloads the latest 10-K report for a given company CIK from SEC EDGAR.

    Parameters:
        cik (str): The company's Central Index Key (CIK).
        save_dir (str): Directory to save the 10-K file.

    Returns:
        str: Path to the saved 10-K file.
    """
    # Normalize CIK (SEC expects 10-digit zero-padded string)
    cik_padded = cik.zfill(10)
    try:
        # Get submissions metadata
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        resp = requests.get(submissions_url, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()

        # Find the latest 10-K filing
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accession_numbers = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])

        tenk_index = None
        for i, form in enumerate(forms):
            if form == "10-K":
                tenk_index = i
                break

        if tenk_index is None:
            print(f"No 10-K filing found for CIK {cik}")
            with open("noCIK.csv", "a") as f:
                f.write(f"{cik}\n")
            return None
        else:
            accession = accession_numbers[tenk_index].replace("-", "")
            primary_doc = primary_docs[tenk_index]

            # Construct the URL for the 10-K filing
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"

            # Download the filing
            filing_resp = requests.get(filing_url, headers=HEADERS)
            filing_resp.raise_for_status()

            # Ensure save directory exists
            os.makedirs(save_dir, exist_ok=True)

            # Save file
            filename = f"{cik}_10-K.html"
            filepath = os.path.join(save_dir, filename)
            with open(filepath, "wb") as f:
                f.write(filing_resp.content)

            print(f"Saved latest 10-K for CIK {cik} to {filepath}")
            return filepath
    except:
        print(f"Error downloading 10-K for CIK {cik}")
        return None
        


fundamentals = pd.read_csv('FundamentalsNANew.csv', dtype={'cik': str}).dropna(subset=['cik']).drop_duplicates(subset=["cik"]).reset_index()
# print(fundamentals)
noCIK = pd.read_csv('noCIK.csv', dtype={'cik': str})
noCIK["cik"] = noCIK["cik"].str.zfill(10)
# print(noCIK)

files = os.listdir("10k_filings_new")  # lists all files and folders
files = pd.DataFrame([f for f in files if os.path.isfile(os.path.join("10k_filings_new", f))], columns=["file"])  # only files
files["file"] = files["file"].str.split("_").str[0].str.zfill(10)

filtered_fundamentals = fundamentals[~fundamentals["cik"].isin(files["file"])]
filtered_fundamentals = filtered_fundamentals[~filtered_fundamentals["cik"].isin(noCIK["cik"])]
print(filtered_fundamentals)

for cik in filtered_fundamentals["cik"].astype(int).astype(str):
    # print(cik)
    download_latest_10k(cik)
    time.sleep(1)