"""
One-click PDF Processing Pipeline
Processes a PDF through the full workflow: PDF → GPT → JSON → Excel
"""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from pdf_to_gpt import process_pdf as pdf_to_gpt_process
from to_excel import convert_json_to_excel


def process_pdf_pipeline(pdf_path, output_dir=None, model=None, pass1_model="gpt-4o-mini", pass2_model="gpt-4o"):
    """
    Full pipeline: PDF → GPT → JSON → Excel
    
    Args:
        pdf_path: Path to input PDF
        output_dir: Directory to save outputs (default: data/outputs/)
        model: Legacy parameter (ignored, use pass1_model/pass2_model instead)
        pass1_model: Model for Pass 1 vision extraction (default: gpt-4o-mini)
        pass2_model: Model for Pass 2 reasoning (default: gpt-4o)
        
    Returns:
        Tuple of (json_path, excel_path)
    """
    pdf_path = Path(pdf_path)
    
    if output_dir is None:
        # Default to data/outputs relative to script location
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir / "data" / "outputs"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LOT PARCEL CLOSURE PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input PDF: {pdf_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Step 1: PDF → GPT → JSON (two-pass architecture)
    print("Step 1: Extracting lot boundaries from PDF using two-pass GPT extraction...")
    print("-" * 60)
    json_path = pdf_to_gpt_process(str(pdf_path), str(output_dir), pass1_model, pass2_model)
    print()
    
    # Step 2: JSON → Excel
    print("Step 2: Converting JSON to Excel format...")
    print("-" * 60)
    excel_path = convert_json_to_excel(json_path)
    print()
    
    print("=" * 60)
    print("✅ PROCESSING COMPLETE")
    print("=" * 60)
    print(f"JSON output: {json_path}")
    print(f"Excel output: {excel_path}")
    print()
    
    return json_path, excel_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_pdf.py <pdf_path> [output_dir] [pass1_model] [pass2_model]")
        print("  pass1_model: Vision model for Pass 1 (default: gpt-4o-mini)")
        print("  pass2_model: Reasoning model for Pass 2 (default: gpt-4o)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    pass1_model = sys.argv[3] if len(sys.argv) > 3 else "gpt-4o-mini"
    pass2_model = sys.argv[4] if len(sys.argv) > 4 else "gpt-4o"
    
    try:
        json_path, excel_path = process_pdf_pipeline(pdf_path, output_dir, None, pass1_model, pass2_model)
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

