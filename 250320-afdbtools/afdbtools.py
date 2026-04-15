import platform
import sys
import subprocess
from pathlib import Path

import numpy as np
import wfdb

here = Path(__file__).parent.resolve()

records = ('04015', '04043', '04048', '04126', '04746', '04908', '04936', '05091', '05121', '05261',
           '06426', '06453', '06995', '07162', '07859', '07879', '07910', '08215', '08219', '08378',
           '08405', '08434', '08455')

def run_subprocess(cmd, show_stdout=False, show_stderr=True, check=True, cwd=None):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    if show_stdout and r.stdout:
        print(r.stdout.decode(), flush=True)
    if show_stderr and r.stderr:
        print(r.stderr.decode(), flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"Subprocess failed with code {r.returncode}, command: {' '.join(cmd)}")
    return r


def evaluate_test_sequences(afdb_path, sequences, report_path=''):
    """Evaluate test AF sequences

    Expects that the argument sequences is a dictionary with a mapping for each record
    to an array with rows of AF sequences (start and stop times in seconds).

    Writes a text report to the specified report path (defualt is current directory)

    Returns a dictionary with the calculated statistics:
        
        ESe: Event Sensitivity in percent
        E+P: Event PPV in percent
        DSe: Duration Sensitivity in percent
        D+P: Duration PPV in percent

    """
    afdb_path = Path(afdb_path)
    if not afdb_path.exists():
        raise Exception('AFDB path not found')

    annotator = 'test'
    # write test annotations for all records
    for rec in records:
        if rec not in sequences:
            raise Exception(f'Missing AF sequences for record {rec}')
        recpth = afdb_path.joinpath(rec)
        hea = wfdb.rdheader(str(recpth))
        sample, symbol, aux_note = seq2ann(hea, sequences[rec])
        wfdb.wrann(rec, annotator, sample, symbol=symbol, aux_note=aux_note, write_dir=recpth.parent)

    # start with a clean slate
    fn = afdb_path.joinpath('af.out')
    if fn.exists():
        fn.unlink()

    pltf = sys.platform
    if pltf == 'darwin':
        pltf = f'{pltf}-{platform.machine()}'

    # evaluate each record
    for n in records:
        run_subprocess([str(here.joinpath('native', pltf, 'epicmp')), '-r', n, '-a', 'atr',
                        annotator, '-L', '-A', 'af.out'], cwd=afdb_path)

    # generate summary stats
    r = run_subprocess([str(here.joinpath('native', pltf, 'sumstats')), 'af.out'], cwd=afdb_path)
    sumstats = r.stdout.decode()

    # write report
    with open(Path(report_path).joinpath('AF-evaluation.txt'), 'w') as fd:
        fd.write('Atrial fibrillation detection performance\n')
        fd.write(sumstats)

    # gather the gross statistics
    for line in sumstats.splitlines():
        if line.startswith('Gross '):
            stats = dict(zip(['ESe', 'E+P', 'DSe', 'D+P'],
                             [None if i == '-' else float(i) for i in line.split()[1:]]))
            break
    else:
        raise Exception('Gross statistics missing in summary report')

    return stats


def read_signal(afdb_path, rec):
    """Read ECG signal for the specified record

    Returns the sampling rate and an array with the ECG data
    """
    afdb_path = Path(afdb_path)
    if not afdb_path.exists():
        raise Exception('AFDB path not found')

    fn = str(afdb_path.joinpath(rec))
    data, fields = wfdb.rdsamp(str(fn))
    sampling_rate = fields['fs']
    return sampling_rate, data


def read_annotations(afdb_path):
    """Read annotations from all records

    Returns a dictionary with the following keys
        seq: array with rows of AF sequences (start and stop times in seconds), and
        qrs: array with the QRS positions in seconds.
    """

    afdb_path = Path(afdb_path)
    if not afdb_path.exists():
        raise Exception('AFDB path not found')

    annots = {}
    for rec in records:
        fn = str(afdb_path.joinpath(rec))
        hea = wfdb.rdheader(fn)
        atr = wfdb.rdann(fn, 'atr')
        qrs = wfdb.rdann(fn, 'qrs')
        annots[rec] = dict(seq=ann2seq(hea, atr), qrs=qrs.sample / qrs.fs)

    return annots


def ann2seq(hea, atr):
    """Gather AF sequences (start- and end-times in seconds) from WFDB annotations"""
    seq = []
    t = None

    for i, n in enumerate(atr.aux_note):
        if n == '(AFIB':
            # start time of AF sequence in seconds
            t = atr.sample[i] / atr.fs
        elif t is not None:
            # collect start and end time of AF sequence
            seq.append([t, atr.sample[i] / atr.fs])
            t = None

    if t is not None:
        # set AF sequence end time to recording length in seconds
        seq.append([t, hea.sig_len / hea.fs])

    return np.array(seq)


def seq2ann(hea, seq):
    """Convert AF sequences to WFDB format"""
    sample = []
    symbol = []
    aux_note = []
    for i in range(seq.shape[0]):
        r = seq[i, :]
        sample.extend((r * hea.fs).astype(int).tolist())
        symbol.extend(('+', '+'))
        aux_note.extend(('(AFIB', '(N'))
    if not sample:
        sample = np.array([1])
        symbol = ['N']
        aux_note = ['']
    return np.array(sample), symbol, aux_note
