"""
Main CLI Entry Point
Command-line interface for lot parcel closure processing.
"""

import argparse
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from process_pdf import process_pdf_pipeline


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Lot Parcel Closure - Extract lot boundaries from PDF plans using GPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pdf data/pdfs/plan1.pdf
  python main.py --pdf data/pdfs/plan1.pdf --output data/outputs
  python main.py --pdf data/pdfs/plan1.pdf --model gpt-4-turbo-preview
        """
    )
    
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to input PDF file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for JSON and Excel files (default: data/outputs/)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model to use (default: gpt-4.1-mini, options: gpt-4.1-mini, gpt-4-turbo-preview)"
    )
    
    args = parser.parse_args()
    
    # Validate PDF path
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.suffix.lower() == ".pdf":
        print(f"❌ Error: File must be a PDF: {pdf_path}")
        sys.exit(1)
    
    # Process PDF
    try:
        json_path, excel_path = process_pdf_pipeline(
            str(pdf_path),
            args.output,
            args.model
        )
        print(f"\n✅ Successfully processed: {pdf_path.name}")
        return 0
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

