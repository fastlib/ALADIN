import os
import warnings

aladin_cache_folder = os.environ.get('aladin_cache')
_HF_REPO_ID = os.environ.get('aladin_hf_repo', 'AUMC/ALADIN')
_DEFAULT_MODEL_FOLDER = os.path.join(os.path.expanduser('~'), '.aladin', 'models')

aladin_model_folder = os.environ.get('aladin_models')


def _build_allow_patterns(modelpaths, use_folds):
    """
    Build a huggingface_hub `allow_patterns` list so snapshot_download only fetches the files
    actually needed to run inference: each modelpath's dataset.json/plans.json, plus the
    checkpoint(s) for the requested fold(s) only (instead of all 5 folds).

    Returns None (meaning "download everything") if `modelpaths` is not given.
    """
    if not modelpaths:
        return None

    patterns = []
    for modelpath in modelpaths:
        patterns.append(f"{modelpath}/dataset.json")
        patterns.append(f"{modelpath}/plans.json")
        if not use_folds:
            # Folds will be auto-detected from whatever is available locally, so we cannot
            # narrow this down upfront.
            patterns.append(f"{modelpath}/fold_*/checkpoint_best.pth")
        else:
            for fold in use_folds:
                fold_name = int(fold) if fold != 'all' else fold
                patterns.append(f"{modelpath}/fold_{fold_name}/checkpoint_best.pth")
    return patterns


def get_model_folder(modelpaths=None, use_folds=None):
    """Return the local folder containing the model weights.

    If `aladin_models` is set, that folder is used as-is (e.g. for offline
    or cluster setups with a pre-staged copy). Otherwise the weights are
    downloaded from the private Hugging Face repo `_HF_REPO_ID` into a
    default cache folder on first use, using the caller's HF credentials
    (env var HF_TOKEN/HUGGING_FACE_HUB_TOKEN or `huggingface-cli login`).

    modelpaths/use_folds: when given, only the dataset.json/plans.json and the checkpoint(s)
    for the requested fold(s) of each modelpath are downloaded, instead of the entire repo
    (all folds of all models). Pass use_folds=None (the default) to download all folds for
    the given modelpaths, e.g. when the fold(s) to use will be auto-detected later.
    """
    global aladin_model_folder

    if aladin_model_folder:
        return aladin_model_folder

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError
    except ImportError as e:
        raise ImportError(
            "aladin_models is not set and huggingface_hub is not installed, so model "
            "weights cannot be auto-downloaded. Install it with `pip install huggingface_hub`, "
            "or set the aladin_models environment variable to a local folder containing the weights."
        ) from e

    allow_patterns = _build_allow_patterns(modelpaths, use_folds)

    if os.path.isdir(_DEFAULT_MODEL_FOLDER) and os.listdir(_DEFAULT_MODEL_FOLDER):
        print(f"aladin_models is not set. Found existing model weights in "
              f"{_DEFAULT_MODEL_FOLDER}, loading from there (missing files, if any, "
              f"will be fetched from Hugging Face repo '{_HF_REPO_ID}') ...")
    else:
        print(f"aladin_models is not set. Downloading model weights from "
              f"Hugging Face repo '{_HF_REPO_ID}' to {_DEFAULT_MODEL_FOLDER} ...")
    try:
        aladin_model_folder = snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=_DEFAULT_MODEL_FOLDER,
            allow_patterns=allow_patterns,
        )
    except (GatedRepoError, RepositoryNotFoundError) as e:
        raise PermissionError(
            f"Could not access the private Hugging Face repo '{_HF_REPO_ID}'. Make sure your "
            "account has been granted access to it, and that you are authenticated: run "
            "`huggingface-cli login`, or set the HF_TOKEN environment variable to a token with "
            "read access. Alternatively, set aladin_models to a local folder with the weights "
            "already in place."
        ) from e
    except HfHubHTTPError as e:
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        if status_code == 401:
            raise PermissionError(
                "Authentication with Hugging Face failed (401 Unauthorized). Run "
                "`huggingface-cli login`, or set the HF_TOKEN environment variable to a token "
                f"with read access to '{_HF_REPO_ID}'. Alternatively, set aladin_models to a "
                "local folder with the weights already in place."
            ) from e
        raise
    return aladin_model_folder
