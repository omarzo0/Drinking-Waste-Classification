#!/usr/bin/env python3
"""Download the preprocessed Drinking Waste dataset (70/15/15 split) from Google Drive.

Source: https://drive.google.com/file/d/1fzN-rFtcJ9f9bdCXQIHkjQFNUkjH0ccJ/view
Saves to data/dataset_70_15_15.npz (~1.6 GB). Skips the download if the file is
already present and complete.
"""

import argparse
import os
import sys
import urllib.request
import zipfile

FILE_ID = '1fzN-rFtcJ9f9bdCXQIHkjQFNUkjH0ccJ'
URL = f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t'
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'dataset_70_15_15.npz')
EXPECTED_KEYS = {'x_train', 'y_train', 'x_validation', 'y_validation', 'x_test', 'y_test'}


def is_valid(path):
    """An .npz is a zip archive; check it opens and holds the expected arrays."""
    try:
        with zipfile.ZipFile(path) as zf:
            keys = {os.path.splitext(n)[0] for n in zf.namelist()}
        return EXPECTED_KEYS <= keys
    except (zipfile.BadZipFile, OSError):
        return False


def download(url, out):
    tmp = out + '.part'
    with urllib.request.urlopen(url) as resp, open(tmp, 'wb') as fh:
        total = int(resp.headers.get('Content-Length', 0))
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f'\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({100 * done / total:5.1f}%)')
                sys.stdout.flush()
    print()
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.environ.get('DATA_PATH', DEFAULT_OUT))
    ap.add_argument('--force', action='store_true', help='re-download even if the file exists')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not args.force and os.path.exists(args.out) and is_valid(args.out):
        print(f'Dataset already present: {args.out}')
        return

    print(f'Downloading dataset to {args.out}')
    download(URL, args.out)
    if not is_valid(args.out):
        sys.exit('Downloaded file is not a valid dataset archive. '
                 'Check that the Google Drive link is still shared publicly.')
    print('Done.')


if __name__ == '__main__':
    main()
