import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import yfinance as yf

def scrape_from_moneycontrol(symbol, num_pages=3):
    """
    Attempts to scrape stock history from MoneyControl.
    Returns a DataFrame or None if scraping fails.
    """
    base_url = f"https://www.moneycontrol.com/stocks/hist_price.php?sc_id={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com"
    }

    all_data = []

    for page in range(1, num_pages + 1):
        url = f"{base_url}&pageno={page}"
        print(f"\nFetching page {page} → {url}")
        try:
            response = requests.get(url, headers=headers)
            print("Status Code:", response.status_code)

            if response.status_code != 200:
                print("⚠️ MoneyControl returned error. Switching to fallback.")
                return None  # fail early

            os.makedirs("data/raw", exist_ok=True)
            with open(f"data/raw/page_{page}.html", "w", encoding="utf-8") as f:
                f.write(response.text)

            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table", {"class": "tbldata14"})

            if not table:
                print("⚠️ Table not found. Skipping.")
                continue

            rows = table.find_all("tr")[1:]
            for row in rows:
                cols = [td.text.strip() for td in row.find_all("td")]
                if len(cols) == 6:
                    all_data.append(cols)

            time.sleep(1)

        except Exception as e:
            print(f"❌ Exception occurred: {e}")
            return None

    if all_data:
        df = pd.DataFrame(all_data, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors='coerce')
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date")
        return df
    else:
        print("❌ No valid rows found.")
        return None

def fetch_from_yfinance(ticker="ITC.NS", start="2020-01-01", end="2023-12-31"):
    """
    Fallback: Downloads stock data using Yahoo Finance via yfinance.
    """
    print(f"\n🔄 Downloading from Yahoo Finance: {ticker}")
    try:
        df = yf.download(ticker, start=start, end=end)
        df.reset_index(inplace=True)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        return df
    except Exception as e:
        print(f"❌ Failed to fetch from yfinance: {e}")
        return None

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Data saved to: {path}")

if __name__ == "__main__":
    final_path = "data/processed/ITC_stock_data.csv"

    df = scrape_from_moneycontrol("ITC")

    if df is None:
        print("\n💡 Falling back to API method...")
        df = fetch_from_yfinance()

    if df is not None:
        save_csv(df, final_path)
    else:
        print("❌ Data retrieval failed completely.")
