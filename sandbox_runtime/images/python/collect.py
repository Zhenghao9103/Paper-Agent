"""Image-only artifact stream; archive validation remains on the host service."""

import os
import stat
import sys
import tarfile


def main():
    directory = os.open("/workspace/artifacts", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with os.scandir(directory) as entries, tarfile.open(
            fileobj=sys.stdout.buffer, mode="w|", dereference=False
        ) as archive:
            for entry in entries:
                state = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
                    raise ValueError("unsafe_artifact")
                # Inspect the opened inode too; a path may change after scandir/stat.
                descriptor = os.open(
                    entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
                )
                with os.fdopen(descriptor, "rb") as stream:
                    state = os.fstat(stream.fileno())
                    if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
                        raise ValueError("unsafe_artifact")
                    item = tarfile.TarInfo(f"artifacts/{entry.name}")
                    item.size = state.st_size
                    archive.addfile(item, stream)
    finally:
        os.close(directory)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError):
        print("artifact_path_or_type", file=sys.stderr)
        sys.exit(2)
