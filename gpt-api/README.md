# Lot Parcel Closure - GPT Edition

Automated extraction of subdivision lot boundaries from PDF plans using OpenAI's GPT API. This tool replaces manual engineering closure table preparation by leveraging GPT's OCR and geometric reasoning capabilities.

## 🎯 Project Purpose

This tool:
- Takes subdivision PDF plans as input
- Sends PDFs directly to OpenAI's ChatGPT API (GPT-4o or GPT-4-turbo-preview)
- GPT performs OCR, text extraction, and geometric reasoning
- GPT outputs structured lot boundary schedules
- Converts schedules to properly formatted Excel files matching engineering closure tables

**No local OCR or fine-tuned models required** - GPT handles everything end-to-end.

## 📂 Project Structure

```
lot-closure-gpt/
├── data/
│   ├── pdfs/          # Input PDFs (place your PDFs here)
│   ├── outputs/       # JSON + Excel outputs (generated here)
├── scripts/
│   ├── pdf_to_gpt.py        # Sends PDF to GPT, gets JSON boundary output
│   ├── to_excel.py          # JSON → Excel converter
│   ├── process_pdf.py       # One-click pipeline: PDF → GPT → Excel
├── .env                    # Your OpenAI API key (create this)
├── requirements.txt
├── README.md
└── main.py                 # CLI entry point
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

**Important:** Never commit your `.env` file to version control. It's already in `.gitignore`.

### 3. Place Your PDF

Copy your subdivision PDF plan to `data/pdfs/`:

```bash
# Example
cp your_plan.pdf data/pdfs/
```

### 4. Run the Pipeline

```bash
python main.py --pdf data/pdfs/your_plan.pdf
```

The system will:
1. Upload the PDF to OpenAI
2. Send it to GPT for analysis
3. Extract lot boundaries, bearings, and distances
4. Generate JSON output
5. Convert to Excel format
6. Save both files to `data/outputs/`

## 📋 Usage Examples

### Basic Usage

```bash
python main.py --pdf data/pdfs/subdivision_plan.pdf
```

### Specify Output Directory

```bash
python main.py --pdf data/pdfs/subdivision_plan.pdf --output custom/output/path
```

### Use Different GPT Model

```bash
python main.py --pdf data/pdfs/subdivision_plan.pdf --model gpt-4-turbo-preview
```

### Run Individual Scripts

You can also run the scripts individually:

```bash
# Step 1: PDF → JSON
python scripts/pdf_to_gpt.py data/pdfs/plan.pdf

# Step 2: JSON → Excel
python scripts/to_excel.py data/outputs/plan.json

# Or run the full pipeline
python scripts/process_pdf.py data/pdfs/plan.pdf
```

## 📊 Output Format

### JSON Output

The JSON output follows this structure:

```json
{
  "lots": [
    {
      "lot": "1002",
      "area_m2": "150",
      "boundaries": [
        {
          "length": 25.0,
          "bearing": "275°58'30\"",
          "deg": 275,
          "min": 58,
          "sec": 30
        },
        {
          "length": 6.0,
          "bearing": "5°37'50\"",
          "deg": 5,
          "min": 37,
          "sec": 50
        }
      ]
    }
  ]
}
```

### Excel Output

The Excel file contains columns:
- **Lot**: Lot number
- **Area**: Lot area in square metres
- **Distance**: Boundary length in metres
- **Bearing**: Bearing in D°M'S" format
- **Degrees**: Bearing degrees
- **Minutes**: Bearing minutes
- **Seconds**: Bearing seconds

Example:

| Lot | Area | Distance | Bearing | Degrees | Minutes | Seconds |
|-----|------|----------|---------|---------|---------|---------|
| 1002 | 150 | 25 | 275°58'30" | 275 | 58 | 30 |
| 1002 | | 6 | 5°37'50" | 5 | 37 | 50 |

## 🔧 How It Works

1. **PDF Upload**: The PDF is uploaded to OpenAI's file storage
2. **GPT Analysis**: GPT-4 analyzes the PDF, performing:
   - OCR (text extraction from images)
   - Geometric reasoning (identifying lot boundaries)
   - Data extraction (bearings, distances, lot numbers)
3. **JSON Generation**: GPT returns structured JSON with all lot boundaries
4. **Excel Conversion**: The JSON is converted to Excel format matching engineering closure tables
5. **Cleanup**: The uploaded PDF file is deleted from OpenAI's storage

## 📝 Requirements

- Python 3.8+
- OpenAI API key with access to GPT-4o or GPT-4-turbo-preview
- Internet connection (for API calls)

## ⚠️ Important Notes

- **API Costs**: Each PDF processing uses OpenAI API credits. GPT-4o is recommended for best results.
- **PDF Quality**: Higher quality PDFs with clear text and diagrams produce better results.
- **File Size**: Large PDFs may take longer to process and cost more.
- **Model Selection**: 
  - `gpt-4.1-mini`: Default model, good balance of accuracy and cost
  - `gpt-4-turbo-preview`: Alternative option for PDF parsing

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"
- Make sure you've created a `.env` file with your API key
- Check that the key is correctly formatted: `OPENAI_API_KEY=sk-...`

### "Invalid JSON response from GPT"
- The PDF might be too complex or unclear
- Try using a different model: `--model gpt-4-turbo-preview`
- Check the PDF quality and ensure it's a valid subdivision plan

### "File not found" errors
- Ensure PDF paths are correct
- Use relative paths from the project root or absolute paths

## 📄 License

This project is for internal use. Ensure compliance with OpenAI's usage policies.

## 🤝 Support

For issues or questions, check:
1. PDF quality and format
2. API key validity
3. OpenAI service status
4. Error messages in the console output

