import numpy as np
import pytest
from astropy.table import Table

from gaiamock.dr4 import list_dr4_sources, load_dr4_epoch_astrometry


def _synthetic_dr4_table(n_transits=20, source_ids=(101, 202)):
    rows = []
    # 2017.5 is ~7.5 years after 2010-01-01
    base_ns = 7.5 * 365.25 * 86400.0 * 1e9
    for sid in source_ids:
        for i in range(n_transits):
            t_ns = np.ma.array(
                (base_ns + (i - n_transits / 2) * 30.0 * 86400.0 * 1e9 + np.arange(10) * 4.4e9).astype(np.int64),
                mask=[True] + [False] * 9,
            )
            psi_deg = np.full(10, (i * 37.0) % 360.0)
            al = np.full(10, 1.5 + 0.01 * i)
            al[0] = 999.0  # SkyMapper value that should be skipped
            err = np.full(10, 0.2, dtype=np.float32)
            used = np.ones(10, dtype=bool)
            used[0] = False
            rows.append({
                'source_id': sid,
                'ra0': 150.0,
                'dec0': 20.0,
                'g_mag': 14.0,
                'agis_source_excess_noise': 0.1,
                'obs_time_tcb': t_ns,
                'obs_time_bary_corr': np.float32(1.0e10),
                'scan_pos_angle': psi_deg,
                'parallax_factor_al': np.float32(0.5),
                'centroid_pos_al': al,
                'centroid_pos_error_al': err,
                'used_by_agis_al': used,
                'multipeak': False,
            })
    return Table(rows)


def test_load_dr4_binned_and_unbinned():
    tbl = _synthetic_dr4_table(n_transits=20, source_ids=(101, 202))
    meta = list_dr4_sources(tbl)
    assert [m['source_id'] for m in meta] == [101, 202]

    all_binned = load_dr4_epoch_astrometry(tbl, binned=True)
    assert set(all_binned.keys()) == {101, 202}
    t, psi, pf, obs, err = all_binned[101]
    assert len(t) == 20
    assert np.all(np.diff(t) >= 0)
    # SkyMapper 999.0 was skipped, and 9 AF CCDs with err=0.2 combine to 0.2/3
    np.testing.assert_allclose(obs[0], 1.5, atol=1e-6)
    np.testing.assert_allclose(err[0], 0.2 / 3.0, rtol=1e-5)

    t_u, psi_u, pf_u, obs_u, err_u = load_dr4_epoch_astrometry(tbl, source_id=101, binned=False)
    assert len(t_u) == 20 * 9
    assert np.all(obs_u < 10.0)


def test_load_dr4_missing_source_raises():
    tbl = _synthetic_dr4_table(n_transits=5, source_ids=(101,))
    with pytest.raises(KeyError):
        load_dr4_epoch_astrometry(tbl, source_id=999)
