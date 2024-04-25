import numpy as np
import pandas as pd
import anndata as ad
from typing import Literal, Optional, Union
from .detect import Detect_SC
from .align import Align_SC
from .correct import Correct_SC
from .subtyping import Classify_SC


def outlier_detect(
    train: ad.AnnData, test: ad.AnnData, *,
    n_epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    sample_rate: Optional[float] = None,
    mem_dim: Optional[int] = None,
    update_size: Optional[int] = None,
    shrink_thres: Optional[float] = None,
    temperature: Optional[float] = None,
    n_critic: Optional[int] = 3,
    pretrain: Optional[bool] = True,
    GPU: Optional[bool] = None,
    verbose: Optional[bool] = None,
    log_interval: Optional[int] = None,
    random_state: Optional[int] = None,
    weight: Optional[dict] = None,
    **kwargs
):
    parameters = {
        'n_epochs': n_epochs or 600,
        'learning_rate': learning_rate or 1e-5,
        'sample_rate': sample_rate or 1,
        'mem_dim': mem_dim or 1024,
        'update_size': update_size or 64,
        'shrink_thres': shrink_thres or 9e-3,
        'temperature': temperature or 1,
        'n_critic': n_critic or 3,
        'pretrain': pretrain or True,
        'GPU': GPU or True,
        'verbose': verbose or True,
        'log_interval': log_interval or 100,
        'random_state': random_state or 100,
        'weight': weight or None
    }
    model = Detect_SC(**parameters)
    model.fit(train)
    result = model.predict(test)
    return result


def obs_align(
    input: Union[list, ad.AnnData], reference: ad.AnnData, *,
    n_epochs: Optional[int] = None,
    learning_rate: Optional[float] = 1e-3,
    pretrain: Optional[bool] = None,
    GPU: Optional[bool] = None,
    verbose: Optional[bool] = None,
    log_interval: Optional[int] = None,
    weight: Optional[bool] = None,
    random_state: Optional[int] = 100,
    fast: Optional[bool] = None,
    **kwargs
):
    parameters = {
        'n_epochs': n_epochs or 1000,
        'learning_rate': learning_rate or 1e-3,
        'pretrain': pretrain or True,
        'GPU': GPU or True,
        'verbose': verbose or True,
        'log_interval': log_interval or 200,
        'weight': weight or None,
        'random_state': random_state or 100,
        'fast': fast or False
    }
    model = Align_SC(**parameters)
    if isinstance(input, list):
        idx = model.fit_mult(input, reference)
    else:
        idx = model.fit(input, reference)
    return idx


def batch_correct(
    input: Union[list, ad.AnnData], base: ad.AnnData, *,
    idx: Optional[pd.DataFrame] = None,
    full_data: Optional[Union[list, ad.AnnData]] = None,
    n_epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    sample_rate: Optional[float] = None,
    n_critic: Optional[int] = 3,
    pretrain: Optional[bool] = True,
    GPU: Optional[bool] = None,
    verbose: Optional[bool] = None,
    log_interval: Optional[int] = None,
    weight: Optional[bool] = None,
    random_state: Optional[int] = None,
    fast: bool = False,
    include: bool = True,
    **kwargs
):
    parameters = {
        'n_epochs': n_epochs or 1000,
        'learning_rate': learning_rate or 2e-4,
        'sample_rate': sample_rate or 1,
        'n_critic': n_critic or 3,
        'pretrain': pretrain or True,
        'GPU': GPU or True,
        'verbose': verbose or True,
        'log_interval': log_interval or 200,
        'weight': weight or None,
        'random_state': random_state or 100,
        'fast': fast or False,
        'include': include or False
    }
    model = Correct_SC(**parameters)
    adata = model.fit(input, base, idx = idx)

    if full_data is not None:
        adata = model.trans_all(full_data, base)
    return adata


def subtype_detect(
    train: ad.AnnData, outlier: ad.AnnData,
    n_subtypes: Union[int, Literal['auto']],
    z_x: Optional[np.ndarray] = None,
    res_x: Optional[np.ndarray] = None, *,
    n_epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    alpha: Optional[float] = None,
    pretrain: Optional[bool] = None,
    GPU: Optional[bool] = None,
    verbose: Optional[bool] = None,
    log_interval: Optional[int] = 1,
    random_state: Optional[int] = 100,
    detect_para: dict = {},
    **kwargs
):
    detect_para = {
        'n_epochs': 600,
        'learning_rate': 1e-4,
        'sample_rate': 1,
        'mem_dim': 1024,
        'update_size': 64,
        'shrink_thres': 9e-3,
        'temperature': 1,
        'n_critic': 3,
        'pretrain': True,
        'GPU': True,
        'verbose': True,
        'log_interval': 100,
        'random_state': 100,
        'weight': None
    }
    parameters = {
        'n_epochs': n_epochs or 10,
        'learning_rate': learning_rate or 1e-6,
        'weight_decay': weight_decay or 0,
        'alpha': alpha or 1,
        'pretrain': pretrain or True,
        'GPU': GPU or True,
        'verbose': verbose or True,
        'log_interval': log_interval or 1,
        'random_state': random_state or 100,
        'n_subtypes': n_subtypes
    }

    if not all([isinstance(i, np.ndarray) for i in [z_x, res_x]]):
        model = Detect_SC(**detect_para)
        model.fit(train)
        _ = model.predict(outlier)
        z_x = model.z_x.cpu().numpy()
        res_x = model.res_x.cpu().numpy()

    model = Classify_SC(**parameters)
    pred = model.fit(z_x, res_x)

    return pred