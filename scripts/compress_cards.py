#!/usr/bin/env python3
"""compress card data from csv to text format"""

import argparse
import csv
import json
from pathlib import Path


def parse_attacks(attack_json):
    """extract attack info from json string"""
    if not attack_json or attack_json == "[]":
        return None
    
    try:
        attacks = json.loads(attack_json)
        if not attacks:
            return None
            
        attack_strs = []
        for atk in attacks:
            info = atk.get("info", "")
            effect = atk.get("effect", "")
            
            # clean up the info string (contains cost and damage)
            attack_str = info
            if effect:
                attack_str += f" - {effect}"
            attack_strs.append(attack_str)
            
        return " | ".join(attack_strs)
    except:
        return None


def parse_abilities(ability_json):
    """extract ability info from json string"""
    if not ability_json or ability_json == "[]":
        return None
    
    try:
        abilities = json.loads(ability_json)
        if not abilities:
            return None
            
        ability_strs = []
        for ab in abilities:
            name = ab.get("name", "")
            effect = ab.get("effect", "")
            ability_str = f"{name}: {effect}" if name else effect
            ability_strs.append(ability_str)
            
        return " | ".join(ability_strs)
    except:
        return None


def compress_card(row):
    """compress card data into single training string"""
    parts = []
    
    # basic info
    parts.append(f"[{row.get('name', '')}]")
    
    # type and stage
    card_type = row.get('type', '')
    stage = row.get('stage', '')
    
    if card_type:
        if stage and stage not in ['Item', 'Support']:
            type_info = f"{stage} {card_type}"
        else:
            type_info = card_type
        parts.append(type_info)
    
    # hp for pokemon
    hp = row.get('hp', '')
    if hp and card_type == 'Pokemon':
        parts.append(f"HP:{hp}")
    
    # weakness and retreat for pokemon
    weakness = row.get('weakness', '')
    if weakness:
        parts.append(f"Weak:{weakness}")
    
    retreat = row.get('retreat', '')
    if retreat:
        parts.append(f"Retreat:{retreat}")
    
    # abilities
    abilities = parse_abilities(row.get('ability'))
    if abilities:
        parts.append(f"Ability: {abilities}")
    
    # attacks
    attacks = parse_attacks(row.get('attack'))
    if attacks:
        parts.append(f"Attacks: {attacks}")
    
    # card text/description
    text = row.get('text', '')
    if text:
        # remove html tags
        text = text.replace('<br />', ' ').replace('<br/>', ' ')
        parts.append(f"Text: {text}")
    
    # rule text
    rule = row.get('rule', '')
    if rule:
        parts.append(f"Rule: {rule}")
    
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Compress card data for model training")
    parser.add_argument("-i", "--input", type=str, default="data/cards_2025-08-26.csv",
                        help="Input CSV file path")
    parser.add_argument("-o", "--output", type=str, default="data/cards_compressed.txt",
                        help="Output text file path")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return
    
    # read csv
    compressed_cards = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            compressed = compress_card(row)
            compressed_cards.append(compressed)
    
    print(f"Loaded {len(compressed_cards)} cards from {input_path}")
    
    # write to output
    with open(output_path, 'w', encoding='utf-8') as f:
        for card in compressed_cards:
            f.write(card + '\n')
    
    print(f"Wrote {len(compressed_cards)} compressed cards to {output_path}")
    
    # show a few examples
    print("\nExample compressed cards:")
    for card in compressed_cards[:3]:
        print(f"  {card[:150]}..." if len(card) > 150 else f"  {card}")


if __name__ == "__main__":
    main()