#!/usr/bin/env python
"""Check that expected output files were generated."""

import sys
from pathlib import Path

output_dir = Path('tests/output_all')

if output_dir.exists():
    files = list(output_dir.glob('*.pdf'))
    print(f'Found {len(files)} PDF files:')
    for f in sorted(files):
        size_kb = f.stat().st_size / 1024
        print(f'  - {f.name}: {size_kb:.1f} KB')

    if len(files) >= 5:
        print(f'\n✓ All expected files generated successfully!')
        sys.exit(0)
    else:
        print(f'\n✗ Expected at least 5 PDFs, found {len(files)}')
        sys.exit(1)
else:
    print('✗ Output directory not found')
    sys.exit(1)