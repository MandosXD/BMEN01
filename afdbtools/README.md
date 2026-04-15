# Evaluate an AF detector using AFDB

The [MIT-BIH Atrial Fibrillation Database][https://physionet.org/content/afdb/1.0.0/] (AFDB) contains 23 records – there are actually 25 records, but we ignore the records 00735 and 03665 that does not have an ECG signal avaiable.

Each individual recording is about 10 hours in duration and contains two ECG signals each sampled at 250 samples per second. All signals contain sequences of atrial fibrillation (AF) rhythm. In total there is about 93 hours of AF rhythm in the recordings.

Download AFDB as a [zip-file](https://physionet.org/static/published-projects/afdb/mit-bih-atrial-fibrillation-database-1.0.0.zip), and unpack the files.

This package contains the afdbtools python module and a bin directory.

The "native" directory contains selected binary files from the [wfdb software package](https://physionet.org/content/wfdb/10.7.0/) that is used in the standardised performance evaluation.

You can use the afdbtools module to run performance evaluation for an AF detector as demonstrated in the following.

## Example use of afdbtools

Make sure you have downloaded the AFDB and installed the python package "wfdb", which is required by afdbtools.

Check your installation in python by importing afdbtools:

    >> import afdbtools

Get the list of records:

    >> afdbtools.records
    ('04015', '04043', '04048', '04126', '04746', '04908', '04936', '05091', '05121', '05261', '06426', '06453', '06995', '07162', '07859', '07879', '07910', '08215', '08219', '08378', '08405', '08434', '08455')

Get samplig rate and ECG data for record 06453:

    >>> fs, sig = afdbtools.read_signal(afdb_path, '06453')
    >>> fs
    250
    >>> sig
    array([[-0.075,  0.2  ],
        [-0.07 ,  0.185],
        [-0.09 ,  0.17 ],
        ...,
        [ 0.07 ,  0.41 ],
        [ 0.075,  0.42 ],
        [ 0.085,  0.42 ]])
    >>> sig.shape
    (8325000, 2)

Get arrays with QRS positions and AF sequences using the path to AFDB as argument:

    >> ann = afdbtools.read_annotations('afdb')

Get QRS positions in seconds for record 04048:

    >>> ann['04048']['qrs']
    array([1.4080000e+00, 2.2760000e+00, 3.1120000e+00, ..., 3.5129552e+04,
        3.5129916e+04, 3.5130788e+04])

Get start- and stop-positions in seconds of AF sequences in record 04048:

    >> ann['04048']['seq']
    array([[ 2226.708,  2323.884],
        [ 2864.44 ,  2896.304],
        [ 8593.892,  8615.988],
        [11995.164, 12012.712],
        [14555.688, 14709.752],
        [16631.908, 16651.976],
        [30726.964, 30744.924],
        [30726.964, 36823.04 ]])

Copy the annotated AF sequences, and run the performance evaluation for these AF sequences (which should be perfect):

    >> seqs = {k:ann[k]['seq'] for k in ann}
    >> afdbtools.evaluate_test_sequences('afdb', seqs)
    {'ESe': 100.0, 'E+P': 100.0, 'DSe': 100.0, 'D+P': 100.0}

The returned dictionary contains the calculated statistics:
    
    ESe: Event Sensitivity in percent
    E+P: Event PPV in percent
    DSe: Duration Sensitivity in percent
    D+P: Duration PPV in percent

Supply a dictionary with an AF sequence array for each record generated using your own AF detector to evaluate the detection performance.

The evaluation function also writes a report "AF-evaluation.txt" in the work directory (or in a directory you specify), that looks like this:

```
Atrial fibrillation detection performance
(AF detection)
Record  TPs   FN  TPp   FP  ESe E+P DSe D+P  Ref duration  Test duration 
 04015    7    0    7    0  100 100 100 100      3:57.236       3:57.236
 04043   82    0   82    0  100 100 100 100   2:11:59.704    2:11:59.708
 04048    7    0    7    0  100 100 100 100      6:00.776       6:00.772
 04126    7    0    7    0  100 100 100 100     18:43.524      18:43.532
 04746    5    0    5    0  100 100 100 100   5:25:53.008    5:25:53.008
 04908    8    0    8    0  100 100 100 100     51:22.452      51:22.452
 04936   36    0   36    0  100 100 100 100   7:22:40.696    7:22:40.696
 05091    8    0    8    0  100 100 100 100      1:26.768       1:26.768
 05121   20    0   20    0  100 100 100 100   6:26:48.944    6:26:48.944
 05261   11    0   11    0  100 100 100 100      7:59.492       7:59.492
 06426   26    0   26    0  100 100 100 100   9:40:13.684    9:40:13.688
 06453    6    0    6    0  100 100 100 100      6:11.280       6:11.284
 06995    6    0    6    0  100 100 100 100   4:44:17.544    4:44:17.544
 07162    1    0    1    0  100 100 100 100  10:08:43.040   10:08:43.040
 07859    1    0    1    0  100 100 100 100  10:08:43.040   10:08:43.040
 07879    2    0    2    0  100 100 100 100   6:10:02.688    6:10:02.688
 07910    5    0    5    0  100 100 100 100   1:37:49.220    1:37:49.220
 08215    2    0    2    0  100 100 100 100   8:14:33.968    8:14:33.968
 08219   39    0   39    0  100 100 100 100   2:12:28.852    2:12:28.852
 08378    5    0    5    0  100 100 100 100   2:08:18.408    2:08:18.408
 08405    2    0    2    0  100 100 100 100   7:23:09.528    7:23:09.528
 08434    3    0    3    0  100 100 100 100     23:43.736      23:43.736
 08455    2    0    2    0  100 100 100 100   7:04:31.024    7:04:31.024
________________________________________________________________________
Sum     291    0  291    0                   92:59:38.612   92:59:38.628
Gross                       100 100 100 100
Average                     100 100 100 100

Summary of results from 23 records
```
