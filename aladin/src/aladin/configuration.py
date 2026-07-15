import os
import warnings

aladin_cache_folder = os.environ.get('aladin_cache')
if aladin_cache_folder is None:
    warnings.warn("Environment variable 'aladin_cache' is not set. This is not a breaking error, but please set this path if you want to use ALADIN's caching capabilities.")

_HF_REPO_ID = os.environ.get('aladin_hf_repo', 'AUMC/ALADIN')
_DEFAULT_MODEL_FOLDER = os.path.join(os.path.expanduser('~'), '.cache', 'aladin', 'models')

aladin_model_folder = os.environ.get('aladin_models')


def get_model_folder():
    """Return the local folder containing the model weights.

    If `aladin_models` is set, that folder is used as-is (e.g. for offline
    or cluster setups with a pre-staged copy). Otherwise the weights are
    downloaded from the private Hugging Face repo `_HF_REPO_ID` into a
    default cache folder on first use, using the caller's HF credentials
    (env var HF_TOKEN/HUGGING_FACE_HUB_TOKEN or `huggingface-cli login`).
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

    print(f"aladin_models is not set. Downloading model weights from "
          f"Hugging Face repo '{_HF_REPO_ID}' to {_DEFAULT_MODEL_FOLDER} ...")
    try:
        aladin_model_folder = snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=_DEFAULT_MODEL_FOLDER,
            token=True,
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
