'''
Loader for ESA Gaia DR4 epoch astrometry files (VOTable XML, FITS, ECSV, Parquet, CSV)
producing the exact 5-tuple `(t_ast_yr, psi, plx_factor, ast_obs, ast_err)` expected by
`gaiamock` orbit fitters and `get_astrometric_likelihoods(..., backend='gpu')`.

Conventions match ESA's `gaia-supdate` and Gaia DR4 draft data model:
- `obs_time_tcb` and `obs_time_bary_corr` are in nanoseconds relative to 2010-01-01T00:00:00 (TCB);
  `t_ast_yr` is returned in Julian years relative to `ref_epoch_jyear` (default 2017.5 for DR4).
- `scan_pos_angle` is in degrees (North through East) -> `psi` in radians, matching `gaiamock`'s
  `sin(psi)` along RA* and `cos(psi)` along Dec.
- `centroid_pos_al` and `centroid_pos_error_al` are in mas; index 0 in the 10-element per-transit
  CCD arrays is SkyMapper (SM) and indices 1..9 are AF1..AF9.
'''
from pathlib import Path
import numpy as np
from astropy.io.votable import parse_single_table
from astropy.table import Table
from astropy.time import Time

_NS_PER_JYEAR = 1e9 * 86400.0 * 365.25
_TCB_REF_JYEAR = float(Time('2010-01-01T00:00:00', format='isot', scale='tcb').jyear)


def _read_table(path_or_table):
    if isinstance(path_or_table, Table):
        return path_or_table
    path = Path(path_or_table)
    suffix = path.suffix.lower()
    if suffix in ('.xml', '.vot') or path.name.lower().endswith(('.xml.gz', '.vot.gz')):
        return parse_single_table(str(path)).to_table()
    return Table.read(str(path))


def _as_1d_float(val):
    arr = np.ma.array(val, dtype=np.float64).filled(np.nan)
    return np.atleast_1d(arr)


def _as_1d_bool(val):
    arr = np.ma.array(val, dtype=bool).filled(False)
    return np.atleast_1d(arr)


def _extract_source(sub, binned=True, skip_skymapper=True, only_used_by_agis=False,
                    drop_multipeak=False, ref_epoch_jyear=2017.5):
    t_list, psi_list, pf_list, obs_list, err_list = [], [], [], [], []
    has_agis = 'used_by_agis_al' in sub.colnames
    has_mp = 'multipeak' in sub.colnames
    t_offset_yr = _TCB_REF_JYEAR - float(ref_epoch_jyear)

    for row in sub:
        if drop_multipeak and has_mp and bool(row['multipeak']):
            continue

        t_ns = _as_1d_float(row['obs_time_tcb'])
        psi_deg = _as_1d_float(row['scan_pos_angle'])
        al_mas = _as_1d_float(row['centroid_pos_al'])
        err_mas = _as_1d_float(row['centroid_pos_error_al'])
        bary_ns = float(np.ma.filled(row['obs_time_bary_corr'], 0.0))
        pf_val = float(np.ma.filled(row['parallax_factor_al'], np.nan))
        if not np.isfinite(pf_val):
            continue

        n_ccd = min(len(t_ns), len(psi_deg), len(al_mas), len(err_mas))
        t_ns, psi_deg, al_mas, err_mas = t_ns[:n_ccd], psi_deg[:n_ccd], al_mas[:n_ccd], err_mas[:n_ccd]

        start = 1 if (skip_skymapper and n_ccd > 1) else 0
        mask = (
            np.isfinite(t_ns) & np.isfinite(psi_deg) & np.isfinite(al_mas)
            & np.isfinite(err_mas) & (err_mas > 0.0)
        )
        if only_used_by_agis and has_agis:
            agis = _as_1d_bool(row['used_by_agis_al'])[:n_ccd]
            mask &= agis
        if start > 0:
            mask[:start] = False

        if not np.any(mask):
            continue

        t_yr = t_offset_yr + (t_ns[mask] + bary_ns) / _NS_PER_JYEAR
        psi_rad = np.deg2rad(psi_deg[mask])
        y = al_mas[mask]
        e = err_mas[mask]

        if binned:
            ivar = 1.0 / (e * e)
            wsum = float(np.sum(ivar))
            t_list.append(float(np.sum(ivar * t_yr) / wsum))
            # circular mean for scan angle across the transit
            psi_list.append(float(np.arctan2(np.sum(ivar * np.sin(psi_rad)), np.sum(ivar * np.cos(psi_rad))) % (2.0 * np.pi)))
            pf_list.append(pf_val)
            obs_list.append(float(np.sum(ivar * y) / wsum))
            err_list.append(float(1.0 / np.sqrt(wsum)))
        else:
            t_list.append(t_yr)
            psi_list.append(psi_rad)
            pf_list.append(np.full(len(t_yr), pf_val, dtype=np.float64))
            obs_list.append(y)
            err_list.append(e)

    if not t_list:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy(), empty.copy(), empty.copy()

    if binned:
        t_ast_yr = np.asarray(t_list, dtype=np.float64)
        psi = np.asarray(psi_list, dtype=np.float64)
        plx_factor = np.asarray(pf_list, dtype=np.float64)
        ast_obs = np.asarray(obs_list, dtype=np.float64)
        ast_err = np.asarray(err_list, dtype=np.float64)
    else:
        t_ast_yr = np.concatenate(t_list)
        psi = np.concatenate(psi_list)
        plx_factor = np.concatenate(pf_list)
        ast_obs = np.concatenate(obs_list)
        ast_err = np.concatenate(err_list)

    order = np.argsort(t_ast_yr)
    return t_ast_yr[order], psi[order], plx_factor[order], ast_obs[order], ast_err[order]


def list_dr4_sources(path_or_table):
    '''Return summary metadata for each unique source_id in a Gaia DR4 epoch astrometry file.'''
    tbl = _read_table(path_or_table)
    sids = np.unique(tbl['source_id'])
    out = []
    for sid in sids:
        sub = tbl[tbl['source_id'] == sid]
        out.append({
            'source_id': int(sid),
            'n_transits': int(len(sub)),
            'g_mag': float(np.nanmedian(np.ma.filled(sub['g_mag'], np.nan))) if 'g_mag' in sub.colnames else np.nan,
            'ra0': float(sub['ra0'][0]) if 'ra0' in sub.colnames else np.nan,
            'dec0': float(sub['dec0'][0]) if 'dec0' in sub.colnames else np.nan,
            'agis_source_excess_noise': float(sub['agis_source_excess_noise'][0]) if 'agis_source_excess_noise' in sub.colnames else np.nan,
        })
    return out


def load_dr4_epoch_astrometry(path_or_table, source_id=None, binned=True, skip_skymapper=True,
                              only_used_by_agis=False, drop_multipeak=False, ref_epoch_jyear=2017.5):
    '''
    Load Gaia DR4 epoch astrometry into `(t_ast_yr, psi, plx_factor, ast_obs, ast_err)`.

    Parameters:
        path_or_table: path to a DR4 VOTable XML, FITS, ECSV, Parquet, or CSV file, or an Astropy Table.
        source_id: optional Gaia source_id to extract. If None and the file contains multiple sources,
            returns a dict `{source_id: (t_ast_yr, psi, plx_factor, ast_obs, ast_err)}`.
        binned: if True (default), inverse-variance-average valid AF1..AF9 CCDs per FOV transit;
            if False, return individual CCD measurements.
        skip_skymapper: if True (default), drop index 0 (SkyMapper SM) from per-transit CCD arrays.
        only_used_by_agis: if True, keep only CCDs with `used_by_agis_al == True` (default False,
            since AGIS single-star outlier rejection can flag valid high-amplitude orbital transits).
        drop_multipeak: if True, drop transits flagged `multipeak == True`.
        ref_epoch_jyear: reference epoch in Julian years (default 2017.5 for Gaia DR4).
    '''
    tbl = _read_table(path_or_table)
    kwargs = dict(
        binned=binned, skip_skymapper=skip_skymapper, only_used_by_agis=only_used_by_agis,
        drop_multipeak=drop_multipeak, ref_epoch_jyear=ref_epoch_jyear,
    )
    if source_id is not None:
        sub = tbl[tbl['source_id'] == int(source_id)]
        if len(sub) == 0:
            raise KeyError(f'source_id {source_id} not found in table.')
        return _extract_source(sub, **kwargs)

    sids = np.unique(tbl['source_id'])
    if len(sids) == 1:
        return _extract_source(tbl, **kwargs)
    return {int(sid): _extract_source(tbl[tbl['source_id'] == sid], **kwargs) for sid in sids}
