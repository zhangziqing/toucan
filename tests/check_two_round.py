"""Check strict allocation rejection and wide serialization with a real compiler."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

compiler = str(Path(sys.argv[1]).resolve())
source = Path(__file__).with_name('nanhu_counter.mlir')
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    for budget, expected in ((98304, 0), (18, 1)):
        output = root / str(budget)
        command = [compiler, str(source), '--single-clock-model', '--removeSVDialect',
                   '--region0-cut-target=0.70', '--region0-repartitions=0',
                   '--region1-repartitions=0', '--partSize=1',
                   f'--local-value-budget={budget}', '-gpu', '-o', str(output),
                   '--mlir-print-op-on-diagnostic=false']
        run = subprocess.run(command, capture_output=True, text=True)
        assert run.returncode == expected, run.stdout + run.stderr
        rows = [json.loads(line) for line in (output / 'capacity-two-round.jsonl').read_text().splitlines()]
        assert len(rows) == 2 and {row['region'] for row in rows} == {0, 1}
        design = output / 'GPUSimDesign.wide32.bin'
        if budget == 98304:
            assert design.read_bytes().startswith(b'TCWIDE1\0')
            assert all(row['allocated_bytes'] <= budget for row in rows)
        else:
            assert 'repair and staged fallback disabled' in run.stdout
            assert not design.exists() and any(row['allocated_bytes'] > budget for row in rows)
print('PASS: two original rounds, TCWIDE1 serialization, over-budget rejection without repair')
