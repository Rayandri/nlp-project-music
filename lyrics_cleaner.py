import os
import re

def is_structure_line(line):
    l = line.strip().lower()
    tags = [
        'refrain', 'pré-refrain', 'pre-refrain', 'prerefrain', 'chorus', 'couplet', 'verse', 'instrumental', 'interlude', 'pont'
    ]
    # Balise structurelle en début de ligne, suivie éventuellement de tiret, deux-points, nom, numéro, x2, etc.
    for tag in tags:
        # [Refrain - Ademo + N.O.S] X2, [Kenza Farah – Refrain 2], etc.
        if re.match(r'^[\[\(\{]?[^\]\)\}]*(' + tag + r')[^\]\)\}]*[\]\)\}]?(\s*[x×*]\s*\d+)?\s*$', l):
            return True
    for tag in tags:
        if re.match(r'^[\[\(\{]\s*' + tag + r'.*[\]\)\}]$', l):
            return True
    for tag in tags:
        if tag in l:
            # Si la ligne est très courte par rapport au tag, c'est une balise
            if len(l) <= 3 * len(tag):
                return True
    # Expression régulière robuste pour les variantes
    tag_regex = (
        r'^[\[\(\{\s]*'
        r'(' + '|'.join(tags) + r')'
        r'(\s*\d+)?'
        r'(\s*[x×*]\s*\d+)?'
        r'(\s*\([^)]+\))?'
        r'(\s*[:\-])?'
        r'(\s*(fois|refrains?|x\d+|\d+\s*fois|\d+\s*refrains?))?'
        r'[\]\)\}\s\.\!\?\,\;\-]*$'
    )
    if re.match(tag_regex, l, re.IGNORECASE):
        return True
    # Cas particuliers fréquents
    if re.match(r'^refrain\s*\d*\s*[:\-]$', l):
        return True
    if re.match(r'^refrain\s*[:\-]?\s*x?\d*$', l):
        return True
    if re.match(r'^\(refrain\)$', l):
        return True
    if re.match(r'^refrain\s*\(.*\)$', l):
        return True
    if re.match(r'^refrain\s*\d*\s*$', l):
        return True
    return False

def clean_lyrics_file(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    cleaned = [line for line in lines if not is_structure_line(line)]
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned)

def process_dataset(root):
    all_files = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith('.txt'):
                all_files.append(os.path.join(dirpath, f))
    total = len(all_files)
    print(f"Total files to process: {total}")
    for idx, path in enumerate(all_files, 1):
        clean_lyrics_file(path)
        if total >= 10 and idx % (total // 10) == 0:
            print(f"Progress: {idx}/{total} ({(idx/total)*100:.0f}%)")
    print("Done")

if __name__ == '__main__':
    process_dataset('lyrics_dataset')
