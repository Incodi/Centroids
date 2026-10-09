import os
import shutil
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from atlas import TextClassifier

def setup_nltk_gutenberg_demo():
    import nltk
    from nltk.corpus import gutenberg

    print("Downloading NLTK Gutenberg corpus...")
    nltk.download('gutenberg', quiet=True)
    nltk.download('punkt', quiet=True)

    base_dir = Path(__file__).resolve().parents[1]
    input_dir = base_dir / 'data' / 'input'

    if input_dir.exists():
        shutil.rmtree(input_dir)

    input_dir.mkdir(parents=True, exist_ok=True)

    # List of Gutenberg texts to use
    file_ids = [
        'austen-emma.txt',
        'austen-persuasion.txt',
        'austen-sense.txt',
        'melville-moby_dick.txt',
        'chesterton-brown.txt',
        'chesterton-thursday.txt',
        'blake-poems.txt',
        'whitman-leaves.txt',
        'milton-paradise.txt',
        'shakespeare-caesar.txt',
        'shakespeare-hamlet.txt',
        'shakespeare-macbeth.txt',
        'bible-kjv.txt',
        'carroll-alice.txt',
        'chesterton-ball.txt'
    ]

    texts = []

    for file_id in file_ids:
        try:
            text = gutenberg.raw(file_id)
            # Create category name from file_id (remove .txt extension)
            category = file_id.replace('.txt', '')
            target_dir = input_dir / category
            txt_files_dir = target_dir / 'txt_files'
            txt_files_dir.mkdir(parents=True, exist_ok=True)

            output_file = txt_files_dir / file_id
            output_file.write_text(text, encoding='utf-8')
            texts.append(category)
            print(f"  Created {category}")
        except Exception as e:
            print(f"  Warning: Could not load {file_id}: {e}")

    return texts

def main():
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / 'demo_output'

    print("Atlas Demo - NLTK Gutenberg Corpus")
    print("=" * 60)
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    texts = setup_nltk_gutenberg_demo()

    print()
    print(f"Demo texts: {len(texts)} individual texts from NLTK Gutenberg corpus")
    print()

    output_csv = str(output_dir / 'demo.csv')
    output_html = str(output_dir / 'demo.html')

    print("Initializing classifier...")
    start_time = time.time()

    classifier = TextClassifier(
        use_cache=False,
        no_metrics=False,
        fast_mode=True,
        cluster_method='hdbscan',
        embedding_model='all-MiniLM-L6-v2',
    )

    print("Processing texts...")
    classifier.run(
        texts,
        output_csv=output_csv,
        output_html=output_html,
        cluster_variants=1,
    )

    elapsed = time.time() - start_time
    print()
    print(f"Demo complete in {elapsed:.0f} seconds.")
    print()

    if classifier.similarity_matrix is not None:
        import numpy as np
        print("Sanity check — nearest neighbor of each text:")
        for i, text_name in enumerate(texts):
            idx = i
            sims = classifier.similarity_matrix[idx].copy()
            sims[idx] = -1
            best_idx = np.argmax(sims)
            best_text = texts[best_idx]
            best_sim = sims[best_idx]
            print(f"  {text_name}      -> {best_text} (cos={best_sim:.2f})")
        print()

    print("What you got:")
    print(f"  - {output_csv}       per-text similarity rankings")
    print(f"  - {output_html}      interactive UMAP + clusters")
    print()
    print("What to look for in the HTML:")
    print("  - Individual texts will cluster by similarity")
    print("  - Similar genres (novels, poetry, drama) should group together")
    print("  - Hover a dot to see its metrics")
    print("  - Use the dropdown to recolor by any single metric")
    print()
    print("Note: The demo uses --fast-mode for speed;")
    print("      your real runs will compute more metrics.")
    print()
    print("Next steps:")
    print("  - Put your own texts in data/input/<name>/txt_files/*.txt")
    print("  - Run: python atlas.py")
    print("  - For a health check: python atlas_doctor.py")
    print()

    print("Opening HTML in browser...")
    webbrowser.open(f'file://{output_html}')

if __name__ == '__main__':
    main()
