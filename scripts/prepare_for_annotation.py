#!/usr/bin/env python3
"""prepare pokemon cards for rank annotation"""

import argparse
import json
import random
import re
from pathlib import Path


def extract_card_name(line):
    """extract card name from compressed format"""
    match = re.match(r"\[([^\]]+)\]", line)
    return match.group(1) if match else None


def is_pokemon_card(line):
    """check if a compressed card line is a pokemon (not item/support)"""
    # pokemon cards have "Pokemon" in their type description
    return "Pokemon" in line and not line.startswith("[") or "Pokemon" in line


def main():
    parser = argparse.ArgumentParser(
        description="Prepare pokemon cards for rank annotation"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/cards_compressed.txt",
        help="Input compressed text file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/annotated_pokemon.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="scripts/pokemon.txt",
        help="Text file with pokemon names to select (one per line)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    file_path = Path(args.file)

    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return

    if not file_path.exists():
        print(f"Error: File {file_path} not found")
        return

    # read all compressed cards
    all_cards = {}

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                card_name = extract_card_name(line)
                if card_name:
                    all_cards[card_name] = line  # exact name matching

    print(f"Loaded {len(all_cards)} total cards")

    # read pokemon names from file
    requested_names = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                requested_names.append(name)

    print(f"Found {len(requested_names)} pokemon names in {file_path}")

    # find matching cards
    selected_cards = []
    not_found = []
    for name in requested_names:
        if name in all_cards:
            selected_cards.append(all_cards[name])
        else:
            not_found.append(name)

    if not_found:
        print(f"Warning: Could not find cards for: {', '.join(not_found)}")

    print(f"Selected {len(selected_cards)} cards from file")

    # create annotation data
    annotation_data = []
    for card_desc in selected_cards:
        annotation_data.append({"card_desc": card_desc, "card_rank": ""})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotation_data, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(annotation_data)} cards to {output_path}")


if __name__ == "__main__":
    main()
