import os
import warnings

aladin_cache_folder = os.environ.get('aladin_cache')
_HF_REPO_ID = os.environ.get('aladin_hf_repo', 'fastlib/ALADIN')

aladin_model_folder = os.environ.get('aladin_models')

# Sentinel value for ALADIN(modelpaths=...)/UNetSegmenter(modelpaths=...): instead of an
# explicit list of modelpaths, automatically pick between the pretrained 1-lead and 3-lead
# models on a per-record basis (see select_model_for_leads below).
AUTO_MODELPATHS = "auto"

# The rhythm/logic engine always reasons over lead II specifically (see UNetSegmenter.preprocess,
# which is hardcoded to feed lead II into the symbolic reasoning path regardless of which model
# is used for segmentation), so lead II must always be present.
_RHYTHM_LEAD = "II"
_THREE_LEAD_MODEL_LEADS = {"II", "V1", "V6"}


def select_model_for_leads(available_leads):
    """
    Decide which pretrained model to use for a given set of available ECG leads.

    Policy:
      - Lead II must always be present, since ALADIN's rhythm analysis is always based on it.
        Raises ValueError if it is missing.
      - If lead II is the only lead available, the 1-lead model ("1_lead_model") is used.
      - If more than one lead is available, the 3-lead model ("3_lead_model") is used, but only
        if leads II, V1 and V6 are all present. Otherwise ALADIN falls back to the 1-lead model
        (with a warning), since that is the only other model available.
      - Whenever more than one lead is provided, a warning is always raised to clarify that
        rhythm analysis itself is still based on lead II alone, regardless of which model ends
        up being selected.

    Parameters
    ----------
    available_leads : Iterable[str]
        Lead names available in the ECG, e.g. Record.available_lead_names.

    Returns
    -------
    str
        "1_lead_model" or "3_lead_model" -- the modelpath/folder name (relative to the ALADIN
        model cache) to load.
    """
    leads = set(available_leads)

    if _RHYTHM_LEAD not in leads:
        raise ValueError(
            "Lead II was not found in the provided ECG. ALADIN requires lead II to be present, "
            f"since rhythm analysis is always based on it (got leads: {sorted(leads) if leads else 'none'})."
        )

    if len(leads) == 1:
        # The only available lead has already been confirmed to be lead II above.
        return "1_lead_model"

    # More than one lead is available.
    warnings.warn(
        "More than one lead was provided. Note that rhythm analysis is always based on lead II "
        "only, regardless of which other leads are available."
    )

    if _THREE_LEAD_MODEL_LEADS.issubset(leads):
        return "3_lead_model"

    warnings.warn(
        "The 3-lead model requires leads II, V1 and V6 to all be present. Since they are not "
        "all available, the 1-lead model is still being used; only lead II will be used for "
        "segmentation and analysis."
    )
    return "1_lead_model"


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
    downloaded anonymously (no login/token needed -- `_HF_REPO_ID` is a
    public repo) into huggingface_hub's own default cache
    (~/.cache/huggingface/hub, or wherever HF_HOME/HF_HUB_CACHE point) on
    first use. This is the same cache used by other Hugging Face-based
    libraries, so `huggingface-cli scan-cache` / `delete-cache` and
    cross-project deduplication work as expected.

    modelpaths/use_folds: when given, only the dataset.json/plans.json and the checkpoint(s)
    for the requested fold(s) of each modelpath are downloaded, instead of the entire repo
    (all folds of all models). Pass use_folds=None (the default) to download all folds for
    the given modelpaths, e.g. when the fold(s) to use will be auto-detected later.

    Note: snapshot_download is called on every invocation rather than being memoized after the
    first call, since different (modelpaths, use_folds) combinations need different files
    present locally (e.g. auto mode loading the 1-lead model first, then the 3-lead model).
    huggingface_hub already skips re-downloading files that are cached locally, so repeated
    calls only fetch whatever the current allow_patterns require but weren't fetched before.
    """
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

    print(f"aladin_models is not set. Fetching model weights from Hugging Face repo "
          f"'{_HF_REPO_ID}' (cached under huggingface_hub's default cache; already-cached "
          f"files are reused, only missing ones are downloaded) ...")
    try:
        downloaded_model_folder = snapshot_download(
            repo_id=_HF_REPO_ID,
            allow_patterns=allow_patterns,
            token=False
        )
    except (GatedRepoError, RepositoryNotFoundError) as e:
        # `_HF_REPO_ID` is downloaded anonymously (token=False above), since it's a public repo.
        # These errors mean it could not be found/accessed as such -- e.g. the repo id is wrong,
        # or (if the repo has since been made private/gated) anonymous access no longer works.
        raise PermissionError(
            f"Could not access the Hugging Face repo '{_HF_REPO_ID}'. Check that the repo id "
            "is correct. If it has been made private or gated, anonymous downloads (as used "
            "here) will no longer work -- set aladin_models to a local folder with the weights "
            "already in place instead."
        ) from e
    except HfHubHTTPError as e:
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        if status_code == 401:
            raise PermissionError(
                f"Hugging Face rejected the anonymous download of '{_HF_REPO_ID}' (401 "
                "Unauthorized). This usually means the repo has been made private or gated. "
                "Set aladin_models to a local folder with the weights already in place instead."
            ) from e
        raise
    return downloaded_model_folder
