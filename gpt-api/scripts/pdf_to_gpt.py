"""
PDF to GPT Script - Two-Pass Extraction Architecture
Pass 1: Vision model extracts raw geometric primitives
Pass 2: Reasoning model reconstructs lot boundaries with validation
"""

import os
import json
import sys
import math
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


def get_client():
    """Get OpenAI client, checking for API key."""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_key_here":
        raise ValueError("OPENAI_API_KEY not found in .env file. Please add your OpenAI API key.")
    return OpenAI(api_key=OPENAI_API_KEY)


def upload_pdf(pdf_path, use_chat_completions=True):
    """
    Upload a PDF file to OpenAI.
    
    Args:
        pdf_path: Path to the PDF file
        use_chat_completions: If True, use "user_data" purpose for chat completions
        
    Returns:
        Tuple of (file_id, client)
    """
    client = get_client()
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"Uploading PDF: {pdf_path}")
    with open(pdf_path, "rb") as f:
        purpose = "user_data" if use_chat_completions else "assistants"
        file = client.files.create(
            file=f,
            purpose=purpose
        )
    
    print(f"PDF uploaded successfully. File ID: {file.id}")
    return file.id, client


def load_prompt(prompt_file):
    """
    Load prompt from file.
    
    Args:
        prompt_file: Path to prompt file
        
    Returns:
        Prompt text as string
    """
    script_dir = Path(__file__).parent.parent
    prompt_path = script_dir / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def bearings_to_radians(deg, min, sec):
    """
    Convert bearing (degrees, minutes, seconds) to radians.
    
    Args:
        deg: Degrees (integer)
        min: Minutes (integer)
        sec: Seconds (integer)
        
    Returns:
        Bearing in radians
    """
    total_degrees = deg + min / 60.0 + sec / 3600.0
    return math.radians(total_degrees)


def lot_to_coords(lot):
    """
    Convert a lot's boundaries (bearings + lengths) to coordinate points.
    
    Args:
        lot: Lot dictionary with boundaries array
        
    Returns:
        List of (x, y) coordinate tuples forming the polygon
    """
    coords = [(0.0, 0.0)]  # Start at origin
    x, y = 0.0, 0.0
    
    for boundary in lot.get("boundaries", []):
        length = boundary.get("length", 0.0)
        deg = boundary.get("deg", 0)
        min_val = boundary.get("min", 0)
        sec = boundary.get("sec", 0)
        
        bearing_rad = bearings_to_radians(deg, min_val, sec)
        
        # Calculate delta x and delta y
        # Surveying bearings are measured clockwise from North (0° = North, 90° = East)
        # Convert to standard math coordinates: dx = length * sin(bearing), dy = length * cos(bearing)
        # This gives: 0° (North) → dx=0, dy=length; 90° (East) → dx=length, dy=0
        dx = length * math.sin(bearing_rad)
        dy = length * math.cos(bearing_rad)
        
        x += dx
        y += dy
        coords.append((x, y))
    
    return coords


def polygon_area(coords):
    """
    Calculate polygon area using shoelace formula.
    
    Args:
        coords: List of (x, y) coordinate tuples
        
    Returns:
        Area (positive value)
    """
    if len(coords) < 3:
        return 0.0
    
    area = 0.0
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    
    return abs(area) / 2.0


def validate_lot(lot):
    """
    Validate a single lot's geometry.
    
    Args:
        lot: Lot dictionary with boundaries
        
    Returns:
        Dictionary with validation results:
        - closure_error_m: Distance between start and end points
        - computed_area_m2: Area calculated from polygon
        - area_difference_m2: Difference between computed and labeled area
        - is_valid: Boolean indicating if closure error < 0.20m and area diff < 10%
    """
    if not lot.get("boundaries") or len(lot["boundaries"]) == 0:
        return {
            "closure_error_m": float('inf'),
            "computed_area_m2": 0.0,
            "area_difference_m2": 0.0,
            "is_valid": False
        }
    
    # Convert boundaries to coordinates
    coords = lot_to_coords(lot)
    
    # Calculate closure error (distance from start to end)
    start = coords[0]
    end = coords[-1]
    closure_error_m = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    
    # Calculate area
    computed_area_m2 = polygon_area(coords)
    
    # Compare with labeled area
    labeled_area = lot.get("area_m2", 0.0)
    if labeled_area > 0:
        area_difference_m2 = computed_area_m2 - labeled_area
        area_diff_percent = abs(area_difference_m2 / labeled_area) * 100 if labeled_area > 0 else 100
    else:
        area_difference_m2 = 0.0
        area_diff_percent = 0.0
    
    # Validation criteria: closure error < 0.20m AND area difference < 10%
    is_valid = closure_error_m < 0.20 and area_diff_percent < 10.0
    
    return {
        "closure_error_m": closure_error_m,
        "computed_area_m2": computed_area_m2,
        "area_difference_m2": area_difference_m2,
        "is_valid": is_valid
    }


def validate_geometry(final_json):
    """
    Validate geometry for all lots in the final JSON.
    
    Args:
        final_json: Final JSON dictionary with lots
        
    Returns:
        Validated JSON with computed metrics added to each lot
    """
    validated_lots = []
    
    for lot in final_json.get("lots", []):
        validation = validate_lot(lot)
        
        # Add validation results to lot
        lot_copy = lot.copy()
        lot_copy["computed_area_m2"] = round(validation["computed_area_m2"], 2)
        lot_copy["closure_error_m"] = round(validation["closure_error_m"], 4)
        lot_copy["area_difference_m2"] = round(validation["area_difference_m2"], 2)
        
        # Set confidence based on validation
        if validation["is_valid"]:
            if lot_copy.get("confidence") not in ["high", "medium", "low"]:
                lot_copy["confidence"] = "high"
        else:
            if lot_copy.get("confidence") == "high":
                lot_copy["confidence"] = "medium"
            elif lot_copy.get("confidence") not in ["medium", "low"]:
                lot_copy["confidence"] = "low"
        
        validated_lots.append(lot_copy)
    
    return {"lots": validated_lots}


def run_pass_1_extract_primitives(pdf_path, model="gpt-4o"):
    """
    Pass 1: Extract raw geometric primitives using vision model.
    
    Args:
        pdf_path: Path to PDF file
        model: Vision-capable model (gpt-4o, gpt-4o-mini, gpt-4-turbo)
        
    Returns:
        Dictionary with primitives (text, lines, lots)
    """
    print("=" * 60)
    print("PASS 1: VISION PRIMITIVE EXTRACTION")
    print("=" * 60)
    print(f"Model: {model}")
    
    # Load Pass 1 prompt
    system_prompt = load_prompt("prompts/pass1_vision.txt")
    user_prompt = """Extract ALL raw geometric primitives from this subdivision plan PDF.

CRITICAL: You must extract EVERY piece of text visible on the plan, including:
- All numeric values (lengths/distances like 7.50, 21, 18.10, etc.)
- All bearing values (format: D°M'S" like 200°59'20", 275°58'30", etc.)
- All lot numbers
- All area labels (XXXm² format)
- All street names and labels
- Text at any angle or rotation
- Small text, large text - EVERYTHING

Also extract ALL line segments visible on the plan.

Do NOT skip anything. Be comprehensive and thorough. Missing information in this step will cause errors in the next step.

Return ONLY valid JSON with all extracted primitives."""
    
    # Upload PDF
    file_id, client = upload_pdf(pdf_path, use_chat_completions=True)
    
    try:
        print(f"Sending PDF to {model} for primitive extraction...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "file",
                            "file": {
                                "file_id": file_id
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_completion_tokens=16384
        )
        
        content = response.choices[0].message.content
        print("Pass 1 response received")
        
        # Debug: Print first 500 chars of response
        print(f"Response preview: {content[:500]}...")
        
        # Parse JSON response
        primitives = parse_json_response(content)
        
        # Print usage info
        if response.usage:
            print(f"Tokens used: {response.usage.total_tokens:,} (prompt: {response.usage.prompt_tokens:,}, completion: {response.usage.completion_tokens:,})")
        
        # Warn if extraction seems incomplete
        text_count = len(primitives.get("text", []))
        lines_count = len(primitives.get("lines", []))
        lots_count = len(primitives.get("lots", []))
        print(f"Extracted: {text_count} text blocks, {lines_count} lines, {lots_count} lots")
        
        if text_count < 20:
            print("⚠️  WARNING: Very few text blocks extracted. The PDF may have been incompletely processed.")
        if lines_count < 10:
            print("⚠️  WARNING: Very few lines extracted. The PDF may have been incompletely processed.")
        
        return primitives
        
    finally:
        # Clean up uploaded file
        try:
            client.files.delete(file_id)
            print(f"Cleaned up uploaded file: {file_id}")
        except Exception as e:
            print(f"Warning: Could not delete uploaded file: {e}")


def run_pass_2_reason_boundaries(primitives_json, model="gpt-4o", max_retries=2):
    """
    Pass 2: Reason about geometry and extract lot boundaries.
    
    Args:
        primitives_json: Dictionary with primitives from Pass 1
        model: Stronger reasoning model (gpt-4o, gpt-4-turbo, gpt-4)
        max_retries: Maximum number of retries if validation fails
        
    Returns:
        Dictionary with final lot boundaries
    """
    print("=" * 60)
    print("PASS 2: GEOMETRIC REASONING & LOT BOUNDARY EXTRACTION")
    print("=" * 60)
    print(f"Model: {model}")
    
    # Load Pass 2 prompt
    system_prompt = load_prompt("prompts/pass2_reasoning.txt")
    
    # Convert primitives to JSON string for the prompt
    primitives_str = json.dumps(primitives_json, indent=2)
    
    # Debug: Show what's being passed to Pass 2
    text_count = len(primitives_json.get("text", []))
    lines_count = len(primitives_json.get("lines", []))
    lots_count = len(primitives_json.get("lots", []))
    print(f"Pass 2 input: {text_count} text blocks, {lines_count} lines, {lots_count} lots")
    print(f"Primitives JSON size: {len(primitives_str):,} characters")
    
    # Count lots in primitives
    lots_in_input = len(primitives_json.get("lots", []))
    
    # Get lot numbers for reference
    lot_numbers = [lot.get("lot", "?") for lot in primitives_json.get("lots", [])]
    lot_numbers_str = ", ".join(lot_numbers[:10]) + (f" ... and {len(lot_numbers)-10} more" if len(lot_numbers) > 10 else "")
    
    user_prompt = f"""Using the raw geometric primitives below, reconstruct EACH AND EVERY lot polygon INDEPENDENTLY, assign labels using spatial proximity + orientation, compute bearings, validate polygon closure, compute area, and output the final JSON.

CRITICAL REQUIREMENTS:
1. There are {lots_in_input} lots in the input primitives: {lot_numbers_str}
2. You MUST process ALL {lots_in_input} lots and include them in your output. Do NOT skip any lots.
3. Each lot MUST be processed INDEPENDENTLY using its own centroid coordinates (centroid_x, centroid_y).
4. Each lot MUST have UNIQUE boundaries - do NOT copy boundaries from one lot to another.
5. For each lot, find the polygon that surrounds its specific centroid point.
6. Match text labels to line segments based on proximity to that specific lot's polygon.

PRIMITIVES:
{primitives_str}

Return ONLY valid JSON in the exact format specified in your system prompt. The output must include ALL lots from the input, each with its own unique boundaries."""
    
    client = get_client()
    
    # Store base prompt for retries
    base_user_prompt = user_prompt
    
    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"\nRetry attempt {attempt}/{max_retries} (previous validation failed)...")
            # Reset to base prompt and add correction instruction
            user_prompt = base_user_prompt + "\n\nCRITICAL: Previous attempt had errors. You MUST:\n1. Process EACH lot INDEPENDENTLY using its own centroid coordinates\n2. Do NOT copy boundaries from one lot to another\n3. Each lot must have UNIQUE boundaries based on its specific polygon\n4. Re-evaluate text-to-segment assignments for each lot separately\n5. Ensure closure error < 0.20m and area difference < 10%"
        
        try:
            print(f"Sending primitives to {model} for boundary reasoning...")
            
            response = client.chat.completions.create(
                model=model,
            messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0.1,
                max_completion_tokens=16384
            )
            
            content = response.choices[0].message.content
            print("Pass 2 response received")
            
            # Parse JSON response
            final_json = parse_json_response(content)
            
            # Print usage info
            if response.usage:
                print(f"Tokens used: {response.usage.total_tokens:,} (prompt: {response.usage.prompt_tokens:,}, completion: {response.usage.completion_tokens:,})")
            
            # Validate geometry
            validated_json = validate_geometry(final_json)
            
            # Check if all lots were processed
            lots_output = len(validated_json.get("lots", []))
            lots_input = len(primitives_json.get("lots", []))
            if lots_output < lots_input:
                print(f"⚠️  WARNING: Only {lots_output} lots processed out of {lots_input} lots in input!")
            
            # Check for duplicate boundaries (critical issue - model copying boundaries)
            boundary_signatures = {}
            duplicate_boundaries = False
            for lot in validated_json.get("lots", []):
                boundaries = lot.get("boundaries", [])
                # Create a signature from boundaries (lengths and bearings)
                sig = tuple((b.get("length", 0), b.get("deg", 0), b.get("min", 0), b.get("sec", 0)) for b in boundaries)
                if sig in boundary_signatures:
                    duplicate_boundaries = True
                    print(f"⚠️  ERROR: Lot {lot.get('lot', '?')} has identical boundaries to lot {boundary_signatures[sig]}")
                else:
                    boundary_signatures[sig] = lot.get("lot", "?")
            
            if duplicate_boundaries:
                print(f"⚠️  CRITICAL: Multiple lots have identical boundaries - model is copying instead of processing independently!")
            
            # Check if any lots failed validation
            needs_retry = False
            for lot in validated_json.get("lots", []):
                if lot.get("closure_error_m", 0) > 0.20 or abs(lot.get("area_difference_m2", 0) / max(lot.get("area_m2", 1), 1)) * 100 > 10:
                    needs_retry = True
                    print(f"  Lot {lot.get('lot', '?')}: closure_error={lot.get('closure_error_m', 0):.4f}m, area_diff={lot.get('area_difference_m2', 0):.2f}m²")
                    break
            
            # Retry if duplicate boundaries detected
            if duplicate_boundaries:
                needs_retry = True
                print(f"  Duplicate boundaries detected - will retry to process each lot independently")
            
            # Also retry if not all lots were processed
            if lots_output < lots_input:
                needs_retry = True
                print(f"  Missing {lots_input - lots_output} lots in output - will retry to process all lots")
            
            if not needs_retry or attempt >= max_retries:
                return validated_json
            else:
                print("  Validation failed, will retry...")
                # Don't reset user_prompt - it's already updated with retry instructions
                
        except Exception as e:
            if attempt >= max_retries:
                raise
            print(f"Error on attempt {attempt + 1}: {e}")
            continue
    
    return validated_json


def parse_json_response(response_text):
    """
    Parse JSON from GPT response, handling markdown code blocks if present.
    
    Args:
        response_text: Raw response from GPT
        
    Returns:
        Parsed JSON object
    """
    # Remove markdown code blocks if present
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()
    
    # Try to extract just the JSON object if there's extra content
    try:
        # First, try parsing the whole thing
        data = json.loads(response_text)
        return data
    except json.JSONDecodeError as e:
        # If that fails, try to extract just the JSON portion
        start_idx = response_text.find('{')
        if start_idx != -1:
            # Try to find the matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                try:
                    data = json.loads(json_text)
                    print("⚠️  Extracted JSON from response with extra content")
                    return data
                except json.JSONDecodeError:
                    pass
        
        # If all else fails, show the error
        print(f"Error parsing JSON: {e}")
        print(f"Response text (first 1000 chars): {response_text[:1000]}...")
        raise ValueError(f"Invalid JSON response from GPT: {e}")


def save_json(data, output_path):
    """
    Save JSON data to file.
    
    Args:
        data: JSON object to save
        output_path: Path to save JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON saved to: {output_path}")


def process_pdf(pdf_path, output_dir=None, pass1_model="gpt-4o", pass2_model="gpt-4o"):
    """
    Main function: Two-pass extraction pipeline.
    
    Args:
        pdf_path: Path to input PDF
        output_dir: Directory to save output (default: data/outputs/)
        pass1_model: Model for Pass 1 (vision extraction)
        pass2_model: Model for Pass 2 (reasoning)
        
    Returns:
        Path to saved validated JSON file
    """
    pdf_path = Path(pdf_path)
    
    if output_dir is None:
        # Default to data/outputs relative to script location
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir / "data" / "outputs"
        intermediate_dir = script_dir / "data" / "intermediate"
    else:
        output_dir = Path(output_dir)
        intermediate_dir = output_dir.parent / "intermediate"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("LOT PARCEL CLOSURE - TWO-PASS EXTRACTION")
    print("=" * 60)
    print(f"Input PDF: {pdf_path}")
    print(f"Output directory: {output_dir}")
    print(f"Intermediate directory: {intermediate_dir}")
    print()
    
    try:
        # PASS 1: Extract primitives
        primitives = run_pass_1_extract_primitives(str(pdf_path), model=pass1_model)
        
        # Save primitives
        primitives_filename = pdf_path.stem + "_primitives.json"
        primitives_path = intermediate_dir / primitives_filename
        save_json(primitives, str(primitives_path))
        print(f"✓ Primitives saved to: {primitives_path}")
        print()
        
        # PASS 2: Reason boundaries
        final_json = run_pass_2_reason_boundaries(primitives, model=pass2_model)
        
        # Save final JSON
        json_filename = pdf_path.stem + ".json"
        json_path = output_dir / json_filename
        save_json(final_json, str(json_path))
        print(f"✓ Final JSON saved to: {json_path}")
        print()
        
        # Run local geometry validation
        validated_json = validate_geometry(final_json)
        
        # Save validated JSON
        validated_filename = pdf_path.stem + "_validated.json"
        validated_path = output_dir / validated_filename
        save_json(validated_json, str(validated_path))
        print(f"✓ Validated JSON saved to: {validated_path}")
        print()
        
        # Print validation summary
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        for lot in validated_json.get("lots", []):
            lot_num = lot.get("lot", "?")
            closure = lot.get("closure_error_m", 0)
            area_diff = lot.get("area_difference_m2", 0)
            confidence = lot.get("confidence", "unknown")
            status = "✓" if lot.get("closure_error_m", 0) < 0.20 and abs(area_diff / max(lot.get("area_m2", 1), 1)) * 100 < 10 else "✗"
            print(f"{status} Lot {lot_num}: closure={closure:.4f}m, area_diff={area_diff:.2f}m², confidence={confidence}")
        print("=" * 60)
        print()
        
        return str(validated_path)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_gpt.py <pdf_path> [output_dir] [pass1_model] [pass2_model]")
        print("  pass1_model: Vision model (default: gpt-4o)")
        print("  pass2_model: Reasoning model (default: gpt-4o)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    pass1_model = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else "gpt-4o"
    pass2_model = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else "gpt-4o"
    
    try:
        json_path = process_pdf(pdf_path, output_dir, pass1_model, pass2_model)
        print(f"\n✅ Success! Validated JSON saved to: {json_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
