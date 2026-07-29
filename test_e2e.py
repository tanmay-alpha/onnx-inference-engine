# E2E test to verify backend fixes
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.chdir('C:/Users/TANMAY/OneDrive/Desktop/onnx-inference-engine/server')
import subprocess
result = subprocess.run(
    ['python', '-m', 'pytest', 'tests/', '-x', '--no-header', '-q'],
    capture_output=True,
    text=True,
    timeout=120,
)
print("STDOUT:", result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print("Exit code:", result.returncode)
