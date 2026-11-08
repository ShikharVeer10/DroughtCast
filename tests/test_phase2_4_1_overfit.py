from types import SimpleNamespace

import torch

from experiments.run_phase2_4_1_overfit import SITES, select_subset


def _training_subset_fixture():
    metadata = []
    for site in SITES:
        for window in range(4):
            metadata.append({"site": site, "window": window})
    count = len(metadata)
    x = torch.ones(count, 6, 15)
    y = torch.ones(count, 3, 4)
    mask = torch.ones(count, 3, 4, dtype=torch.bool)
    # One invalid candidate per site must never be selected.
    for site_index in range(len(SITES)):
        mask[site_index * 4, 0, 0] = False
    return SimpleNamespace(X=x, Y=y, mask=mask, metadata=metadata)


def test_subset_selection_is_seeded_site_balanced_and_valid_only():
    dataset = _training_subset_fixture()
    first = select_subset(dataset, seed=2401, windows_per_site=2)
    repeat = select_subset(dataset, seed=2401, windows_per_site=2)

    assert first == repeat
    assert len(first) == len(SITES) * 2
    assert len(set(first)) == len(first)
    for site in SITES:
        selected_for_site = [index for index in first if dataset.metadata[index]["site"] == site]
        assert len(selected_for_site) == 2
        assert all(bool(dataset.mask[index].all()) for index in selected_for_site)
        assert all(torch.isfinite(dataset.X[index]).all() for index in selected_for_site)
        assert all(torch.isfinite(dataset.Y[index]).all() for index in selected_for_site)


def test_subset_selection_excludes_nonfinite_inputs_and_targets():
    dataset = _training_subset_fixture()
    for site_index in range(len(SITES)):
        dataset.X[site_index * 4 + 1, 0, 0] = float("nan")
        dataset.Y[site_index * 4 + 2, 0, 0] = float("inf")
    selected = select_subset(dataset, seed=91, windows_per_site=1)

    assert len(selected) == len(SITES)
    assert all(torch.isfinite(dataset.X[index]).all() for index in selected)
    assert all(torch.isfinite(dataset.Y[index]).all() for index in selected)
    assert all(bool(dataset.mask[index].all()) for index in selected)
