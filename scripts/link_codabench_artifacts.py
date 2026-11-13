"""Link verified NAS artifacts into a Mac/Linux checkout; preserve local backups."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CB = ROOT / 'codabench'
FOLDERS = ('submissions', 'logs/sprint_1106', 'logs/sprint_1109', 'logs/sprint_1110',
           'logs/sprint_1112', 'logs/sprint_1114')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, default=Path(
        os.environ.get('CODABENCH_ARTIFACT_ROOT',
                       '/Volumes/Beast_PC/Projects/codabench/windows-transfer/codabench')))
    NAS = parser.parse_args().artifact_root.resolve()
    state = CB / 'logs/artifact_sync'
    state.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((NAS / 'IMPORT_MANIFEST.json').read_text())
    hashes = {f['path']: f['sha256'] for f in manifest['files']}
    tracked = {p[len('codabench/'):] for p in subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-files', '-z', 'codabench/'], text=True).split('\0') if p}
    backup = state / ('pre-transfer-backup-' + time.strftime('%Y%m%d-%H%M%S'))
    linked = []
    conflicts = []
    replaced_different = []

    def visit(relative):
        local, remote = CB / relative, NAS / relative
        if local.is_symlink():
            if local.resolve() != remote.resolve():
                conflicts.append(relative + ' (existing symlink preserved)')
            return
        if remote.is_dir():
            contains_tracked = any(p.startswith(relative + '/') for p in tracked)
            if not local.exists() and not contains_tracked:
                local.parent.mkdir(parents=True, exist_ok=True)
                local.symlink_to(remote, target_is_directory=True)
                linked.append(relative + '/')
            else:
                local.mkdir(parents=True, exist_ok=True)
                for child in sorted(remote.iterdir()):
                    visit(child.relative_to(NAS).as_posix())
            return
        if relative in tracked:
            return
        if relative not in hashes:
            return
        if local.exists():
            if not local.is_file():
                conflicts.append(relative + ' (unexpected local directory preserved)')
                return
            if sha(local) != hashes[relative]:
                replaced_different.append(relative)
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            assert not saved.exists(), saved
            local.rename(saved)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.symlink_to(remote)
        linked.append(relative)

    for folder in FOLDERS:
        visit(folder)
    result = dict(nas=str(NAS), backup=str(backup), linked_count=len(linked),
                  linked=linked, replaced_different=replaced_different, conflicts=conflicts)
    (state / 'nas_links.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'linked'}, indent=2))
    if conflicts:
        raise RuntimeError('Some local artifacts differ; preserved for explicit review')


if __name__ == '__main__':
    main()
