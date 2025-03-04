import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from typing import Any, List
from bs4 import BeautifulSoup
import pandas as pd

def build_url(year: str, start_date: str, end_date: str, womens: bool=False) -> str:
    """
    Build the URL for scraping player stats from Bart Torvik's website.

    Args:
        start_date: The start date in 'YYYYMMDD' format.
        end_date: The end date in 'YYYYMMDD' format.

    Returns:
        A formatted URL string.
    """
    if womens:    
        base_url = "https://barttorvik.com/ncaaw/playerstat.php"
    else:
        base_url = "https://barttorvik.com/playerstat.php"

    params = (
        "link=y",
        "sIndex=53",
        "minmin=5",
        f"year={year}",
        f"start={start_date}",
        f"end={end_date}"
    )
    return f"{base_url}?{'&'.join(params)}"


def click_games_column(driver: webdriver.Chrome) -> None:
    """
    Wait for the "Games" column span to be clickable and click it.

    Args:
        driver: The Selenium webdriver instance.
    """
    try:
        games_element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[@class='sname' and @id='3']"))
        )
        games_element.click()
        print("Clicked on 'Games' column successfully.")
        time.sleep(2)  # Allow time for the table to update.
    except Exception as e:
        print("Error clicking 'Games' column:", e)


def click_show_more(driver: webdriver.Chrome, max_clicks: int = 40) -> None:
    """
    Continuously click the "Show 100 more" button until it is no longer clickable or
    the maximum number of clicks is reached.

    Args:
        driver: The Selenium webdriver instance.
        max_clicks: Maximum number of times to click the button.
    """
    clicks = 0
    while clicks < max_clicks:
        try:
            expand_element = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "expand"))
            )
            show_more_button = expand_element.find_element(By.TAG_NAME, "a")
            show_more_button.click()
            print(f"'Show 100 more' clicked ({clicks + 1}/{max_clicks})")
            time.sleep(2)  # Wait for new content to load.
            clicks += 1
        except Exception as e:
            print("No more 'Show 100 more' button found or an error occurred:", e)
            break


def get_html_from_torvik_players(year:str, start_date: str, end_date: str, max_page_expansion_clicks:int=40, womens:str = False) -> str:
    """
    Scrapes player statistics from Bart Torvik's website for a given date range.
    
    The function loads the page, clicks the "Games" column to add it to the dataset,
    repeatedly clicks "Show 100 more" to load all data, and then returns the full HTML.
    
    Args:
        start_date: The start date in 'YYYYMMDD' format.
        end_date: The end date in 'YYYYMMDD' format.
    
    Returns:
        The HTML source of the fully loaded page.
    """
    url: str = build_url(year, start_date, end_date, womens)
    print(f"Loading URL: {url}")

    # Initialize the webdriver (adjust if using a different browser)
    driver: Any = webdriver.Chrome()
    driver.get(url)

    time.sleep(30)

    # Click on the "Games" column to include it in the data.
    click_games_column(driver)

    # Click the "Show 100 more" button repeatedly to load more data.
    click_show_more(driver, max_clicks=max_page_expansion_clicks)

    # Retrieve the complete HTML once all content is loaded.
    html_source: str = driver.page_source
    driver.quit()

    return html_source

def extract_complete_row(row: List[str]) -> List[str]:
    """
    Extracts only the relevant columns from a row based on predefined indices.

    Args:
        row (List[str]): A list of strings representing a row's data.

    Returns:
        List[str]: The filtered row with selected columns.
    """
    selected_indices = {0, 2, 3, 4, 6, 7, 8, 10, 11, 13, 16, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28}
    return [cell for i, cell in enumerate(row) if i in selected_indices]


def parse_table(html_source: str) -> List[List[str]]:
    """
    Parses the HTML source to extract table data.

    Args:
        html_source (str): The raw HTML content.

    Returns:
        List[List[str]]: A list of rows, where each row is a list of cell values.
    """
    soup = BeautifulSoup(html_source, "html.parser")

    # Locate the table by its style attribute
    table = soup.find("table", {"style": "white-space:nowrap;margin:auto;table-layout:fixed"})
    if not table:
        raise ValueError("Table not found!")

    # Locate the <tbody> if it exists, otherwise get all rows directly from <table>
    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")

    extracted_data = []
    for row in rows:
        cells = row.find_all(["td", "th"])  # Some rows may use <th> for data
        row_data = [cell.get_text(strip=True) for cell in cells]
        filtered_row = extract_complete_row(row_data)
        if filtered_row:
            extracted_data.append(filtered_row)

    return extracted_data


def get_data_from_html(html_source: str) -> pd.DataFrame:
    """
    Extracts data from an HTML table and returns it as a pandas DataFrame.

    Args:
        html_source (str): The HTML content of the page.

    Returns:
        pd.DataFrame: The extracted table data as a DataFrame.
    """
    data = parse_table(html_source)

    # Define column headers
    headers = [
        "Rk", "Class", "Height", "Player", "Team", "Conf", "Games", "Min%", "PRPG!", "BPM",
        "ORTG", "USG", "EFG", "TS", "OR", "DR", "AST", "TO", "BLK", "STL", "FTR"
    ]

    return pd.DataFrame(data, columns=headers)


