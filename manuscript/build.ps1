# build.ps1 — Compile LaTeX manuscript for Redundancy Paradox paper
# Usage: .\build.ps1

$ErrorActionPreference = "Stop"

$manuscript_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$sections_dir = "$manuscript_dir\sections"
$output_dir = "$manuscript_dir"
$final_pdf = "$manuscript_dir\manuscript.pdf"

Write-Host "=== Building Redundancy Paradox manuscript ==="

# Step 1: Check dependencies
Write-Host "`n[1/5] Checking dependencies..."
$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
$bibtex = Get-Command bibtex -ErrorAction SilentlyContinue

if (-not $pdflatex) {
    Write-Host "ERROR: pdflatex not found. Install MiKTeX or TeX Live."
    Write-Host "  Windows: https://miktex.org/download"
    Write-Host "  macOS: brew install --cask mactex"
    Write-Host "  Linux: sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra"
    exit 1
}

if (-not $bibtex) {
    Write-Host "ERROR: bibtex not found. Install MiKTeX or TeX Live."
    exit 1
}

Write-Host "  pdflatex: $(pdflatex --version | Select-Object -First 1)"
Write-Host "  bibtex: OK"

# Step 2: Verify sections exist
Write-Host "`n[2/5] Verifying sections..."
$required_sections = @(
    "01_introduction.tex",
    "02_related_work.tex",
    "03_problem.tex",
    "04_synthetic.tex",
    "05_realdata.tex",
    "06_fragility.tex",
    "07_discussion.tex",
    "08_limitations.tex",
    "09_conclusion.tex"
)

$missing = @()
foreach ($section in $required_sections) {
    $path = "$sections_dir\$section"
    if (-not (Test-Path $path)) {
        $missing += $section
    } else {
        Write-Host "  OK: $section"
    }
}

if ($missing.Count -gt 0) {
    Write-Host "ERROR: Missing sections:"
    foreach ($m in $missing) {
        Write-Host "  - $m"
    }
    exit 1
}

# Step 3: Run pdflatex
Write-Host "`n[3/5] Running pdflatex (pass 1)..."
Push-Location $manuscript_dir
& pdflatex -interaction=nonstopmode manuscript.tex 2>&1 | Select-String -Pattern "^!" | ForEach-Object { Write-Host "  WARNING: $_" }

# Step 4: Run bibtex + pdflatex passes
Write-Host "`n[4/5] Running bibtex..."
& bibtex manuscript 2>&1 | Out-Null

Write-Host "  Running pdflatex (pass 2)..."
& pdflatex -interaction=nonstopmode manuscript.tex 2>&1 | Out-Null

Write-Host "  Running pdflatex (pass 3)..."
& pdflatex -interaction=nonstopmode manuscript.tex 2>&1 | Out-Null
Pop-Location

# Step 5: Move final PDF
Write-Host "`n[5/5] Finalizing..."
$pdf_path = "$manuscript_dir\manuscript.pdf"

if (Test-Path $pdf_path) {
    $size = [math]::Round((Get-Item $pdf_path).Length / 1KB, 1)
    Write-Host "  SUCCESS: manuscript.pdf ($size KB)"
    Write-Host "  Location: $pdf_path"
} else {
    Write-Host "  ERROR: manuscript.pdf not generated"
    Write-Host "  Check manuscript.log for errors"
    exit 1
}

# Cleanup aux files
Write-Host "`nCleaning auxiliary files..."
$aux_extensions = @("*.aux", "*.bbl", "*.blg", "*.log", "*.out", "*.toc", "*.fls", "*.fdb_latexmk")
foreach ($ext in $aux_extensions) {
    Remove-Item "$manuscript_dir\$ext" -ErrorAction SilentlyContinue
}
Remove-Item "$sections_dir\*.aux" -ErrorAction SilentlyContinue

Write-Host "`n=== Build complete ==="
Write-Host "PDF: $pdf_path"
