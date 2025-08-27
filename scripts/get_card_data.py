# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
# ]
# ///
import argparse
import csv
import datetime
import json

import requests

WANTED_FIELDS = [
    "id",
    "setId",
    "number",
    "name",
    "set_code",
    "set_name",
    "rarity",
    "color",
    "type",
    "slug",
    "hp",
    "stage",
    "prev_stage_name",
    "attack",
    "ability",
    "text",
    "weakness",
    "retreat",
    "rule",
]


def fetch_and_write_csv(url: str) -> None:
    try:
        response = requests.get(url)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            print("Warning: Expected JSON but received:", content_type)
            print("Response preview:", response.text[:500])

        data_json = response.json()

        headers = data_json.get("names")
        rows = data_json.get("data")

        if not headers or not rows:
            raise ValueError("JSON structure missing 'names' or 'data' keys.")

        filtered_headers = [h for h in headers if h in WANTED_FIELDS]
        if not filtered_headers:
            print("Warning: None of the WANTED_FIELDS found in response headers.")
            return

        wanted_indexes = [headers.index(h) for h in filtered_headers]

        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        csv_filename = f"../data/cards_{date_str}.csv"

        with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(filtered_headers)

            for row in rows:
                processed_row = []
                for idx in wanted_indexes:
                    item = row[idx]
                    if isinstance(item, (list, dict)):
                        item = json.dumps(item)
                    processed_row.append(item)
                writer.writerow(processed_row)

        print(f"CSV file written successfully to {csv_filename}")

    except Exception as e:
        print("Error fetching or writing data:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cache_index", type=int)
    args = parser.parse_args()

    url = f"https://api.dotgg.gg/cgfw/getcards?game=pokepocket&mode=indexed&cache={args.cache_index}"
    fetch_and_write_csv(url)
