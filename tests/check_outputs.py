#!/usr/bin/env python
"""Check that expected output files were generated."""

import sys
import platform
from pathlib import Path

output_dir = Path('tests/output_all')

if output_dir.exists():
    files = list(output_dir.glob('*.pdf'))
    print(f'Found {len(files)} PDF files:')
    for f in sorted(files):
        size_kb = f.stat().st_size / 1024
        print(f'  - {f.name}: {size_kb:.1f} KB')

    if len(files) >= 5:
        # Use ASCII characters on Windows to avoid encoding issues
        if platform.system() == 'Windows':
            print(f'\n[OK] All expected files generated successfully!')
        else:
            print(f'\n✓ All expected files generated successfully!')
        sys.exit(0)
    else:
        # Use ASCII characters on Windows to avoid encoding issues
        if platform.system() == 'Windows':
            print(f'\n[FAIL] Expected at least 5 PDFs, found {len(files)}')
        else:
            print(f'\n✗ Expected at least 5 PDFs, found {len(files)}')
        sys.exit(1)
else:
    # Use ASCII characters on Windows to avoid encoding issues
    if platform.system() == 'Windows':
        print('[FAIL] Output directory not found')
    else:
        print('✗ Output directory not found')
    sys.exit(1)