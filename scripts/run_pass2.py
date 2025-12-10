"""
Run Pass 2 only using existing primitives file
"""

import sys
import json
from pathlib import Path
from pdf_to_gpt import run_pass_2_reason_boundaries, validate_geometry, save_json

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pass2.py <primitives_json_path> [output_path]")
        sys.exit(1)
    
    primitives_path = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not primitives_path.exists():
        print(f"Error: Primitives file not found: {primitives_path}")
        sys.exit(1)
    
    # Load primitives
    print(f"Loading primitives from: {primitives_path}")
    with open(primitives_path, "r", encoding="utf-8") as f:
        primitives = json.load(f)
    
    text_count = len(primitives.get("text", []))
    lines_count = len(primitives.get("lines", []))
    lots_count = len(primitives.get("lots", []))
    print(f"Primitives: {text_count} text blocks, {lines_count} lines, {lots_count} lots")
    print()
    
    # Run Pass 2
    print("Running Pass 2...")
    final_json = run_pass_2_reason_boundaries(primitives, model="gpt-4o")
    
    # Validate
    validated_json = validate_geometry(final_json)
    
    # Save
    if output_path is None:
        output_path = primitives_path.parent.parent / "outputs" / (primitives_path.stem.replace("_primitives", "") + "_pass2.json")
    else:
        output_path = Path(output_path)
    
    save_json(validated_json, str(output_path))
    
    # Summary
    lots_output = len(validated_json.get("lots", []))
    print()
    print("=" * 60)
    print("PASS 2 SUMMARY")
    print("=" * 60)
    print(f"Input lots: {lots_count}")
    print(f"Output lots: {lots_output}")
    if lots_output < lots_count:
        print(f"⚠️  WARNING: {lots_count - lots_output} lots missing!")
    print(f"Output saved to: {output_path}")
    print("=" * 60)

