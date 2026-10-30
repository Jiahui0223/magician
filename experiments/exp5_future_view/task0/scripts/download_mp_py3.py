#!/usr/bin/env python3
# Downloads MP public data release -- reviewed Python 3 port of the official download_mp.py
# (http://kaldir.vc.cit.tum.de/matterport/download_mp.py, sha256 37bc39c84c72c1eafeb96f9290db0410217894b3858ee992cbaafb3a265c09ec,
#  fetched 2026-10-23; raw copy in work_dirs/exp5_future_view/task0/download_logs/official_raw/).
#
# Kept unchanged: release layout (<out_dir>/v1/scans/<scan_id>/<type>.zip), file types, task files, scan list URL,
# the terms-of-use confirmation prompt (interactive, before anything is downloaded), --id / --type / --task_data.
# Changes (EXP5 Task 0):
#   1. Python 3 (print(), urllib.request, input()); stdlib only.
#   2. https BASE_URL (the http URL answers 301 -> https for every file).
#   3. download_file: resumable (<file>.part + HTTP Range), size checked against Content-Length, zip CRC test
#      (zipfile.testzip) and sha256 BEFORE the atomic rename to <type>.zip -> an incomplete archive is never taken as
#      done. The official script renamed a mkstemp temp file without any check.
#   4. Existing <type>.zip files are verified (size + CRC) instead of skipped blindly; a failing one is moved aside
#      to <type>.zip.bad.<time> (never deleted) and downloaded again. Completed files are never deleted.
#   5. --ids / --ids-file (several scans in one call), --retries, JSON-lines log (--log), --verify-only, --extract
#      (zip-slip-checked extraction next to the zip, with a <type>.extracted.json marker).
#   6. One terms-of-use confirmation per call (the official script asked a second time only for --id ALL; kept).
import argparse
import hashlib
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
import zipfile

BASE_URL = 'https://kaldir.vc.cit.tum.de/matterport/'
RELEASE = 'v1/scans'
RELEASE_TASKS = 'v1/tasks/'
RELEASE_SIZE = '1.3TB'
TOS_URL = BASE_URL + 'MP_TOS.pdf'
FILETYPES = [
    'cameras',
    'matterport_camera_intrinsics',
    'matterport_camera_poses',
    'matterport_color_images',
    'matterport_depth_images',
    'matterport_hdr_images',
    'matterport_mesh',
    'matterport_skybox_images',
    'undistorted_camera_parameters',
    'undistorted_color_images',
    'undistorted_depth_images',
    'undistorted_normal_images',
    'house_segmentations',
    'region_segmentations',
    'image_overlap_data',
    'poisson_meshes',
    'sens'
]
TASK_FILES = {
    'keypoint_matching_data': ['keypoint_matching/data.zip'],
    'keypoint_matching_models': ['keypoint_matching/models.zip'],
    'surface_normal_data': ['surface_normal/data_list.zip'],
    'surface_normal_models': ['surface_normal/models.zip'],
    'region_classification_data': ['region_classification/data.zip'],
    'region_classification_models': ['region_classification/models.zip'],
    'semantic_voxel_label_data': ['semantic_voxel_label/data.zip'],
    'semantic_voxel_label_models': ['semantic_voxel_label/models.zip'],
    'minos': ['mp3d_minos.zip'],
    'gibson': ['mp3d_for_gibson.tar.gz'],
    'habitat': ['mp3d_habitat.zip'],
    'pixelsynth': ['mp3d_pixelsynth.zip'],
    'igibson': ['mp3d_for_igibson.zip'],
    'mp360': ['mp3d_360/data_00.zip', 'mp3d_360/data_01.zip', 'mp3d_360/data_02.zip', 'mp3d_360/data_03.zip', 'mp3d_360/data_04.zip', 'mp3d_360/data_05.zip', 'mp3d_360/data_06.zip']
}
CHUNK = 8 * 2 ** 20
LOG = None
RETRIES = 20


def log(event, **kw):
    rec = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'event': event, **kw}
    print(json.dumps(rec), flush=True)
    if LOG:
        with open(LOG, 'a') as f:
            f.write(json.dumps(rec) + '\n')


def urlopen(req, timeout=60):
    return urllib.request.urlopen(req, timeout=timeout)


def get_release_scans(release_file):
    with urlopen(release_file) as r:
        return [line.decode().rstrip('\r\n') for line in r if line.strip()]


def remote_size(url):
    for attempt in range(RETRIES):
        try:
            with urlopen(urllib.request.Request(url, method='HEAD')) as r:
                return int(r.headers['Content-Length'])
        except (urllib.error.URLError, socket.timeout, ConnectionError, TypeError) as e:
            log('head_retry', url=url, attempt=attempt, error=repr(e))
            time.sleep(min(300, 10 * 2 ** attempt))
    raise RuntimeError('HEAD failed: ' + url)


def verify(path, size):
    """(ok, info): exact size, zip CRC test of every member, sha256."""
    got = os.path.getsize(path)
    if got != size:
        return False, {'reason': 'size', 'size': got, 'expected': size}
    if path.endswith('.zip') or path.endswith('.zip.part'):
        try:
            with zipfile.ZipFile(path) as z:
                bad = z.testzip()
                n = len(z.infolist())
        except zipfile.BadZipFile as e:
            return False, {'reason': 'badzip', 'error': repr(e)}
        if bad is not None:
            return False, {'reason': 'crc', 'member': bad}
    else:
        n = None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
    return True, {'size': got, 'members': n, 'sha256': h.hexdigest()}


def download_file(url, out_file, verify_only=False):
    size = remote_size(url)
    if os.path.isfile(out_file):
        ok, info = verify(out_file, size)
        if ok:
            log('exists_ok', file=out_file, **info)
            return True
        aside = out_file + '.bad.' + time.strftime('%Y%m%d%H%M%S')
        os.rename(out_file, aside)
        log('exists_bad_moved_aside', file=out_file, moved_to=aside, **info)
    if verify_only:
        log('missing', file=out_file, remote_size=size)
        return False
    part = out_file + '.part'
    for attempt in range(RETRIES):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > size:
            aside = part + '.bad.' + time.strftime('%Y%m%d%H%M%S')
            os.rename(part, aside)
            log('part_too_large_moved_aside', file=part, moved_to=aside, have=have, size=size)
            have = 0
        if have < size:
            t0, last, got0 = time.time(), time.time(), have
            try:
                req = urllib.request.Request(url, headers={'Range': 'bytes=%d-' % have} if have else {})
                with urlopen(req) as r:
                    if have and r.status != 206:
                        raise RuntimeError('server ignored Range (status %d)' % r.status)
                    with open(part, 'ab') as f:
                        for b in iter(lambda: r.read(CHUNK), b''):
                            f.write(b)
                            have += len(b)
                            if time.time() - last > 60:
                                last = time.time()
                                log('progress', file=out_file, have=have, size=size,
                                    mb_s=round((have - got0) / 2 ** 20 / (last - t0), 2))
                log('transfer_end', file=out_file, have=have, size=size, seconds=round(time.time() - t0, 1),
                    mb_s=round((have - got0) / 2 ** 20 / max(1e-6, time.time() - t0), 2), attempt=attempt)
            except (urllib.error.URLError, socket.timeout, ConnectionError, RuntimeError, OSError) as e:
                log('transfer_error', file=out_file, have=have, size=size, attempt=attempt, error=repr(e))
                time.sleep(min(300, 10 * 2 ** attempt))
                continue
        ok, info = verify(part, size)
        if ok:
            os.replace(part, out_file)
            log('done', file=out_file, url=url, **info)
            return True
        aside = part + '.bad.' + time.strftime('%Y%m%d%H%M%S')
        os.rename(part, aside)
        log('verify_failed_moved_aside', file=part, moved_to=aside, attempt=attempt, **info)
    log('gave_up', file=out_file, url=url)
    return False


def extract(zip_path):
    """Extract next to the zip (archives hold <scan_id>/<type>/...); zip-slip check; marker file."""
    marker = zip_path[:-4] + '.extracted.json'
    if os.path.exists(marker):
        log('extract_skip', file=zip_path)
        return True
    root = os.path.dirname(os.path.dirname(os.path.abspath(zip_path)))   # <out>/v1/scans
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for n in names:
            dst = os.path.abspath(os.path.join(root, n))
            if not dst.startswith(root + os.sep):
                log('extract_refused', file=zip_path, member=n)
                return False
        z.extractall(root)
    with open(marker, 'w') as f:
        json.dump({'zip': os.path.basename(zip_path), 'members': len(names), 'root': root,
                   'time': time.strftime('%Y-%m-%d %H:%M:%S')}, f)
    log('extracted', file=zip_path, members=len(names))
    return True


def download_release(release_scans, out_dir, file_types, verify_only=False, do_extract=False):
    print('Downloading MP release to ' + out_dir + '...')
    for scan_id in release_scans:
        scan_out_dir = os.path.join(out_dir, scan_id)
        download_scan(scan_id, scan_out_dir, file_types, verify_only, do_extract)
    print('Downloaded MP release.')


def download_scan(scan_id, out_dir, file_types, verify_only=False, do_extract=False):
    print('Downloading MP scan ' + scan_id + ' ...')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    ok = True
    for ft in file_types:
        url = BASE_URL + RELEASE + '/' + scan_id + '/' + ft + '.zip'
        out_file = out_dir + '/' + ft + '.zip'
        good = download_file(url, out_file, verify_only)
        if good and do_extract:
            good = extract(out_file)
        ok &= good
    log('scan_done' if ok else 'scan_incomplete', scan=scan_id, types=file_types)
    print('Downloaded scan ' + scan_id)
    return ok


def download_task_data(task_data, out_dir):
    print('Downloading MP task data for ' + str(task_data) + ' ...')
    for task_data_id in task_data:
        if task_data_id in TASK_FILES:
            file = TASK_FILES[task_data_id]
            for filepart in file:
                url = BASE_URL + RELEASE_TASKS + '/' + filepart
                localpath = os.path.join(out_dir, filepart)
                localdir = os.path.dirname(localpath)
                if not os.path.isdir(localdir):
                    os.makedirs(localdir)
                download_file(url, localpath)
                print('Downloaded task data ' + task_data_id)


def main():
    global LOG, RETRIES
    parser = argparse.ArgumentParser(description=
        '''
        Downloads MP public data release (reviewed Python 3 port; see header).
        Example invocation:
          python3 download_mp_py3.py -o base_dir --ids 17DRP5sb8fy --type matterport_mesh --log dl.jsonl
        After download base_dir/v1/scans is populated with scan data, and base_dir/v1/tasks is populated with task data.
        The --type argument is optional (all data types are downloaded if unspecified).
        The --id ALL argument will download all house data. Use --id house_id (or --ids / --ids-file) for specific houses.
        ''',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    parser.add_argument('--task_data', default=[], nargs='+', help='task data files to download. Any of: ' + ','.join(TASK_FILES.keys()))
    parser.add_argument('--id', default='ALL', help='specific scan id to download or ALL to download entire dataset')
    parser.add_argument('--ids', nargs='+', default=None, help='several scan ids')
    parser.add_argument('--ids-file', default=None, help='file with one scan id per line')
    parser.add_argument('--type', nargs='+', help='specific file types to download. Any of: ' + ','.join(FILETYPES))
    parser.add_argument('--log', default=None, help='JSON-lines log file')
    parser.add_argument('--retries', type=int, default=20)
    parser.add_argument('--verify-only', action='store_true', help='only verify existing zips (no download)')
    parser.add_argument('--extract', action='store_true', help='extract each verified zip next to it')
    args = parser.parse_args()
    LOG, RETRIES = args.log, args.retries

    print('By pressing any key to continue you confirm that you have agreed to the MP terms of use as described at:')
    print(TOS_URL)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')
    key = input('')
    log('tos_confirmed', argv=sys.argv[1:])

    release_file = BASE_URL + RELEASE + '.txt'
    release_scans = get_release_scans(release_file)
    file_types = FILETYPES

    # download task data
    if args.task_data:
        if set(args.task_data) & set(TASK_FILES.keys()):  # download task data
            out_dir = os.path.join(args.out_dir, RELEASE_TASKS)
            download_task_data(args.task_data, out_dir)
        else:
            print('ERROR: Unrecognized task data id: ' + str(args.task_data))
        print('Done downloading task_data for ' + str(args.task_data))
        key = input('Press any key to continue on to main dataset download, or CTRL-C to exit.')

    # download specific file types?
    if args.type:
        bad = [t for t in args.type if t not in FILETYPES]
        if bad:
            print('ERROR: Invalid file type: ' + str(bad))
            return 2
        file_types = args.type

    ids = args.ids or ([x.strip() for x in open(args.ids_file) if x.strip()] if args.ids_file else None)
    if ids is None and args.id and args.id not in ('ALL', 'all'):
        ids = [args.id]
    if ids is not None:  # download given scans
        bad = [s for s in ids if s not in release_scans]
        if bad:
            print('ERROR: Invalid scan id: ' + str(bad))
            return 2
        ok = all([download_scan(s, os.path.join(args.out_dir, RELEASE, s), file_types, args.verify_only, args.extract)
                  for s in ids])
        log('all_done' if ok else 'finished_with_errors', n_scans=len(ids), types=file_types)
        return 0 if ok else 1
    elif 'minos' not in args.task_data:  # download entire release
        if len(file_types) == len(FILETYPES):
            print('WARNING: You are downloading the entire MP release which requires ' + RELEASE_SIZE + ' of space.')
        else:
            print('WARNING: You are downloading all MP scans of type ' + ', '.join(file_types))
        print('Existing archives are verified (size + CRC); incomplete .part files are resumed.')
        print('***')
        print('Press any key to continue, or CTRL-C to exit.')
        key = input('')
        out_dir = os.path.join(args.out_dir, RELEASE)
        download_release(release_scans, out_dir, file_types, args.verify_only, args.extract)
    return 0


if __name__ == "__main__":
    sys.exit(main())
