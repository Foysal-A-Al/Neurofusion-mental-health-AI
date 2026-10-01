from neurofusion.evaluation import bootstrap_mean_ci, uncertainty_interval


def test_intervals_are_valid():
    mean, lo, hi = bootstrap_mean_ci([0, 1, 1, 0, 1], n_boot=50)
    assert lo <= mean <= hi
    lo2, hi2 = uncertainty_interval(0.7)
    assert 0 <= lo2 <= 0.7 <= hi2 <= 1
