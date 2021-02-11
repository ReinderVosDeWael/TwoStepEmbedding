import os
from pathlib import Path
import nibabel as nib
import numpy as np
from brainspace.utils.parcellation import reduce_by_labels
from raja_dynamic.dynamic import TwoStepEmbedding


def fetch_subject_task_timeseries(subject_dir, parcellations=None, cortex_only=True):
    """Fetches the task (parcellated) timeseries of a subject

    Parameters
    ----------
    subject_dir : str
        Path to an HCP subject directory
    parcellations : str, list, optional
        Files describing the parcellation of the timeseries. If multiple files
        are provided (e.g. left and right hemisphere), then the parcellations
        are concatenated in the order that they're provided in. Defaults to
        None.
    cortex_only : bool, optional
        If true, returns only cortical data, defaults to True.

    Returns
    -------
    numpy.array
        Array containing the timeseries
    """

    task_dir = os.path.join(subject_dir, "MNINonLinear", "Results")
    task_files = [x for x in Path(task_dir).rglob("*tfMRI*MSMAll*dtseries.nii")]
    if len(task_files) != 14:
        ValueError("Could not find all task files.")

    return [load_hcp_timeseries(x, parcellations, cortex_only) for x in task_files]


def load_hcp_timeseries(file, parcellations=None, cortex_only=True):
    """Fetches (parcellated) timeseries of a subject

    Parameters
    ----------
    subject_dir : str
        Path to an HCP subject directory
    parcellations : str, list, optional
        Files describing the parcellation of the timeseries. If multiple files
        are provided (e.g. left and right hemisphere), then the parcellations
        are concatenated in the order that they're provided in. Defaults to
        None.
    cortex_only : bool, optional
        If true, returns only cortical data, defaults to True.

    Returns
    -------
    numpy.array
        Array containing the timeseries
    """

    if isinstance(parcellations, str):
        parcellations = [parcellations]

    cii = nib.load(file)
    timeseries = cii.get_fdata()

    if cortex_only:
        timeseries = timeseries[:, :64984]

    if parcellations is not None:
        parcel_list = [nib.load(x).darrays[0].data for x in parcellations]
        parcellation = np.concatenate(parcel_list, axis=0)
        timeseries = [reduce_by_labels(x, parcellation)[1:] for x in timeseries]

    return timeseries


parcellation_dir = (
    "/data/mica1/03_projects/reinder/micasoft/parcellations/fs_LR-conte69/maps"
)
parcellation_names = ["lh.HCP-MMP1.label.gii", "rh.HCP-MMP1.label.gii"]
parcellation_files = [os.path.join(parcellation_dir, x) for x in parcellation_names]


subject_dirs = ["/media/sdo1/100408"]
timeseries = [
    fetch_subject_task_timeseries(x, parcellation_files) for x in subject_dirs
]
timeseries = [np.atleast_3d(np.concatenate(x)) for x in timeseries]
timeseries = np.concatenate(timeseries, axis=2)
tse = TwoStepEmbedding(kernel="cosine", approach="dm", rng=0, d1=10, d2=10)
tse.fit(timeseries)
