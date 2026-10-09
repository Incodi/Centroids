import os
import sys
import shutil
import logging
import platform
from pathlib import Path
from typing import List, Tuple, Optional

logging.getLogger('torch.distributed.elastic.multiprocessing.redirects').setLevel(logging.ERROR)
logging.getLogger('torch.utils._pytree').setLevel(logging.ERROR)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

IS_TTY = sys.stdout.isatty()

def colorize(level: str, text: str) -> str:
    if not IS_TTY:
        return text

    colors = {
        'ok': '\033[92m',
        'warn': '\033[93m',
        'fail': '\033[91m',
        'reset': '\033[0m',
    }
    color = colors.get(level, '')
    reset = colors.get('reset', '')
    return f"{color}{text}{reset}"

def check_python() -> List[Tuple[str, str, str, str, Optional[str]]]:
    results = []
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version < (3, 8):
        results.append((
            'fail',
            'python',
            version_str,
            'Python version too old (requires 3.8+)',
            'Please upgrade to Python 3.8 or higher'
        ))
    elif version < (3, 9):
        results.append((
            'warn',
            'python',
            version_str,
            'Python version below 3.9 (3.9+ recommended)',
            None
        ))
    else:
        results.append((
            'ok',
            'python',
            version_str,
            'Python version acceptable',
            None
        ))

    system = platform.system()
    machine = platform.machine()
    platform_str = f"{system}-{machine}"

    is_venv = sys.prefix != sys.base_prefix
    venv_str = f" (venv)" if is_venv else " (no venv)"

    if is_venv:
        results.append((
            'ok',
            'venv',
            'yes',
            f'Running in virtual environment{venv_str}',
            None
        ))
    else:
        results.append((
            'warn',
            'venv',
            'no',
            f'Not running in virtual environment{venv_str}',
            'Consider using a virtual environment: python -m venv venv && source venv/bin/activate'
        ))

    return results

def check_hardware() -> List[Tuple[str, str, str, str, Optional[str]]]:
    results = []

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

        results.append((
            'ok',
            'torch',
            'installed',
            f'cuda={cuda_available} mps={mps_available}',
            None
        ))
    except ImportError:
        results.append((
            'fail',
            'torch',
            'not installed',
            'PyTorch not available',
            'pip install torch'
        ))

    if PSUTIL_AVAILABLE:
        try:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            available_gb = mem.available / (1024**3)
            results.append((
                'ok',
                'ram',
                f'{total_gb:.1f}GB total',
                f'{available_gb:.1f}GB available',
                None
            ))
        except Exception:
            pass

    return results

def check_packages() -> List[Tuple[str, str, str, str, Optional[str]]]:
    results = []

    hard_requirements = [
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('sentence_transformers', 'sentence-transformers'),
        ('transformers', 'transformers'),
        ('scipy', 'scipy'),
        ('sklearn', 'scikit-learn'),
        ('umap', 'umap-learn'),
        ('plotly', 'plotly'),
        ('lexicalrichness', 'lexicalrichness'),
        ('vaderSentiment', 'vaderSentiment'),
    ]

    optional_packages = [
        ('orjson', 'orjson', 'optional — faster JSON parsing'),
        ('enchant', 'pyenchant', 'optional — enables OOV metrics'),
        ('spacy', 'spacy', 'optional — enables spaCy metrics'),
        ('leidenalg', 'leidenalg', 'optional — enables Leiden clustering'),
        ('igraph', 'python-igraph', 'optional — required for Leiden clustering'),
        ('nltk', 'nltk', 'optional — NLTK corpora'),
        ('psutil', 'psutil', 'optional — RAM detection'),
    ]

    for module_name, pip_name in hard_requirements:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            results.append((
                'ok',
                module_name,
                version,
                'installed',
                None
            ))
        except ImportError:
            results.append((
                'fail',
                module_name,
                'not installed',
                'required package missing',
                f'pip install {pip_name}'
            ))

    for item in optional_packages:
        if len(item) == 2:
            module_name, pip_name = item
            reason = 'optional'
        else:
            module_name, pip_name, reason = item

        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            results.append((
                'ok',
                module_name,
                version,
                f'installed ({reason})',
                None
            ))
        except ImportError:
            results.append((
                'warn',
                module_name,
                'not installed',
                f'{reason}',
                f'pip install {pip_name}'
            ))

    return results

def check_external_tools() -> List[Tuple[str, str, str, str, Optional[str]]]:
    results = []

    mallet_path = shutil.which('mallet')
    mallet_home = os.environ.get('MALLET_HOME')

    if mallet_path and mallet_home:
        results.append((
            'ok',
            'mallet',
            'installed',
            f'{mallet_path} (MALLET_HOME={mallet_home})',
            None
        ))
    elif mallet_path:
        results.append((
            'warn',
            'mallet',
            'installed',
            f'{mallet_path} (MALLET_HOME not set)',
            'Set MALLET_HOME environment variable'
        ))
    else:
        results.append((
            'warn',
            'mallet',
            'not installed',
            'MALLET not found (optional — enables topic modeling)',
            None
        ))

    try:
        import spacy
        try:
            spacy.load('en_core_web_sm')
            results.append((
                'ok',
                'spacy model',
                'en_core_web_sm',
                'installed',
                None
            ))
        except OSError:
            results.append((
                'warn',
                'spacy model',
                'en_core_web_sm',
                'model not downloaded',
                'python -m spacy download en_core_web_sm'
            ))
    except ImportError:
        pass

    try:
        import enchant
        try:
            enchant.Dict('en_US')
            results.append((
                'ok',
                'enchant',
                'en_US',
                'dictionary available',
                None
            ))
        except Exception:
            system = platform.system()
            if system == 'Darwin':
                fix = 'brew install enchant'
            elif system == 'Linux':
                fix = 'apt install libenchant-2-2'
            else:
                fix = 'Install libenchant for your platform'
            results.append((
                'warn',
                'enchant',
                'en_US',
                'dictionary not available',
                fix
            ))
    except ImportError:
        pass

    return results

def check_models() -> List[Tuple[str, str, str, str, Optional[str]]]:
    results = []

    from atlas_config import EMBEDDING_MODEL_PROFILES, resolve_embedding_model_name

    default_model = resolve_embedding_model_name(None)
    hf_cache = Path.home() / '.cache' / 'huggingface' / 'hub'

    if hf_cache.exists():
        model_dirs = list(hf_cache.glob('models--*'))
        profile = EMBEDDING_MODEL_PROFILES.get(default_model, {})

        if profile:
            context_window = profile.get('context_window', 512)
            chunk_target = profile.get('chunk_target', 384)
            results.append((
                'ok',
                'default model',
                default_model,
                f'context={context_window} chunk={chunk_target}',
                None
            ))

        model_found = any(default_model.replace('/', '--') in str(d) for d in model_dirs)
        if model_found:
            results.append((
                'ok',
                'model cache',
                'cached',
                f'{default_model} found in HF cache',
                None
            ))
        else:
            results.append((
                'warn',
                'model cache',
                'not cached',
                f'{default_model} not in HF cache (will download on first run)',
                None
            ))
    else:
        results.append((
            'warn',
            'model cache',
            'not checked',
            'HF cache directory not found (will download on first run)',
            None
        ))

    return results

def check_dirs() -> List[Tuple[str, str, str, str, Optional[str]]]:
    results = []

    base_dir = Path(__file__).resolve().parents[1]
    input_dir = base_dir / 'data' / 'input'
    cache_dir = base_dir / 'cache'
    metrics_cache_dir = cache_dir / 'metrics'
    video_centroid_cache_dir = cache_dir / 'video_centroids'

    if input_dir.exists():
        text_folders = [d for d in input_dir.iterdir() if d.is_dir()]
        if text_folders:
            results.append((
                'ok',
                'data/input',
                f'{len(text_folders)} text folders',
                'input directory ready',
                None
            ))
        else:
            results.append((
                'warn',
                'data/input',
                'empty',
                'no text folders found',
                'Drop text folders into data/input/ or run atlas_demo.py'
            ))
    else:
        results.append((
            'warn',
            'data/input',
            'not found',
            'input directory missing',
            'mkdir -p data/input'
        ))

    if metrics_cache_dir.exists():
        try:
            probe_file = metrics_cache_dir / '.doctor_probe'
            probe_file.touch()
            probe_file.unlink()
            results.append((
                'ok',
                'cache/metrics',
                'writable',
                'metrics cache directory ready',
                None
            ))
        except Exception:
            results.append((
                'fail',
                'cache/metrics',
                'not writable',
                'cannot write to metrics cache',
                'Check permissions on cache/metrics'
            ))
    else:
        results.append((
            'warn',
            'cache/metrics',
            'not found',
            'metrics cache directory missing (will be created)',
            None
        ))

    if video_centroid_cache_dir.exists():
        try:
            probe_file = video_centroid_cache_dir / '.doctor_probe'
            probe_file.touch()
            probe_file.unlink()
            results.append((
                'ok',
                'cache/video_centroids',
                'writable',
                'video centroid cache directory ready',
                None
            ))
        except Exception:
            results.append((
                'fail',
                'cache/video_centroids',
                'not writable',
                'cannot write to video centroid cache',
                'Check permissions on cache/video_centroids'
            ))
    else:
        results.append((
            'warn',
            'cache/video_centroids',
            'not found',
            'video centroid cache directory missing (will be created)',
            None
        ))

    if metrics_cache_dir.exists():
        cache_files = list(metrics_cache_dir.glob('*.json'))
        if cache_files:
            total_size = sum(f.stat().st_size for f in cache_files) / (1024**2)
            results.append((
                'ok',
                'cache size',
                f'{len(cache_files)} files',
                f'{total_size:.1f}MB cached metrics',
                None
            ))

    return results

def estimate_corpus() -> Optional[List[Tuple[str, str, str, str, Optional[str]]]]:
    results = []

    base_dir = Path(__file__).resolve().parents[1]
    input_dir = base_dir / 'data' / 'input'

    if not input_dir.exists():
        return None

    text_folders = [d for d in input_dir.iterdir() if d.is_dir()]
    if not text_folders:
        return None

    total_files = 0
    total_bytes = 0

    for text_folder in text_folders:
        txt_files_dir = text_folder / 'txt_files'
        if txt_files_dir.exists():
            txt_files = list(txt_files_dir.glob('*.txt'))
            total_files += len(txt_files)
            total_bytes += sum(f.stat().st_size for f in txt_files)

    if total_files == 0:
        return None

    total_tokens = total_bytes // 5
    runtime_estimate = total_tokens * 0.0006

    results.append((
        'ok',
        'corpus',
        f'{len(text_folders)} texts',
        f'{total_files} files, ~{total_tokens} tokens',
        None
    ))

    results.append((
        'ok',
        'runtime estimate',
        f'{runtime_estimate:.0f}s',
        f'~{runtime_estimate/60:.1f} minutes for embeddings (default config)',
        None
    ))

    return results

def print_report(results: List[Tuple[str, str, str, str, Optional[str]]]):
    level_width = 6
    name_width = 20
    value_width = 20

    for level, name, value, detail, fix in results:
        level_colored = colorize(level, level.upper().ljust(level_width))
        name_padded = name.ljust(name_width)
        value_padded = value.ljust(value_width)

        line = f"[{level_colored}] {name_padded} {value_padded}  {detail}"
        print(line)

        if fix:
            print(f"         -> {fix}")

def main():
    print("Atlas Health Check")
    print("=" * 60)
    print()

    all_results = []

    all_results.extend(check_python())
    print("Interpreter & Platform:")
    print_report(all_results)
    print()

    all_results.extend(check_hardware())
    print("Hardware:")
    print_report(check_hardware())
    print()

    all_results.extend(check_packages())
    print("Required Packages:")
    print_report(check_packages())
    print()

    all_results.extend(check_external_tools())
    print("External Tools:")
    print_report(check_external_tools())
    print()

    all_results.extend(check_models())
    print("Model Cache:")
    print_report(check_models())
    print()

    all_results.extend(check_dirs())
    print("Directory Sanity:")
    print_report(check_dirs())
    print()

    corpus_estimate = estimate_corpus()
    if corpus_estimate:
        all_results.extend(corpus_estimate)
        print("Corpus Estimate:")
        print_report(corpus_estimate)
        print()

    print("=" * 60)

    hard_failures = [r for r in all_results if r[0] == 'fail']

    if hard_failures:
        print("Not ready. Fix the red lines above, then re-run.")
        sys.exit(1)
    else:
        print("Ready. Try: python atlas_demo.py")
        sys.exit(0)

if __name__ == '__main__':
    main()
