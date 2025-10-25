"""
- Загрузка датасетов через MNE
"""
import mne
from functools import lru_cache

datasets_list = {
    "sample": "Стандартный EEG (auditory & visual)",
    "eegbci": "Motor Imagery BCI (EEG, 64 канала, движения рук/ног)",
}

@lru_cache(maxsize=3)
def load_dataset(name: str):
    """
    Загружает датасет из MNE с кэшированием.
    Возвращает mne.io.raw.
    """
    match name:
        case "sample":
            data_path = mne.datasets.sample.data_path()
            raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
            raw = mne.io.read_raw_fif(raw_fname, preload=True)

        case "eegbci":
            files = mne.datasets.eegbci.load_data(1, [2])
            raw = mne.io.read_raw_edf(files[0], preload=True)

        case _:
            raise ValueError(f"{name} - недоступен или отсутствует")

    return raw