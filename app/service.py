from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location


APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
ML_DIR = REPO_DIR / "ml"
DATASET_FILE = ML_DIR / "data" / "processed" / "features_dataset.csv"
MODELS_DIR = ML_DIR / "models"
RESULTS_DIR = ML_DIR / "data" / "results"

ALL_YEARS = [2021, 2022, 2023, 2024, 2025]

# Urutan fitur harus sama dengan fitur saat training model METEO.
METEO_FEATURES = [
    "lat", "lon", "month", "day_of_year",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "TEMP", "RH", "PRECIP",
]

# Konversi GHI harian (kWh/m²) menjadi iradiansi rata-rata (W/m²).
EFFECTIVE_SUN_HOURS = 10.0

PV_MODULES = {
    "generic_mono_400W": {
        "name": "Monocrystalline 400 Wp",
        "watt": 400,
        "area_m2": 2.0,
    },
    "generic_mono_550W": {
        "name": "Monocrystalline 550 Wp",
        "watt": 550,
        "area_m2": 2.6,
    },
    "generic_poly_330W": {
        "name": "Polycrystalline 330 Wp",
        "watt": 330,
        "area_m2": 2.0,
    },
}

DEFAULT_PARAMS = {
    "capex_per_kwp": 15_000_000.0,
    "tarif_rp_kwh": 1_444.70,
    "lifetime_thn": 25,
    "discount_rate": 0.06,
    "degradasi": 0.005,
    "performance_ratio": 0.80,
    "self_consumption": 0.65,
    "opex_rate": 0.01,
    "eskalasi_tarif": 0.03,
    "tilt": 10.0,
    "azimuth": 0.0,
    "albedo": 0.2,
}

USABLE_FRACTION = 0.70


@lru_cache(maxsize=1)
def _select_best_model() -> dict:
    """
    Memilih model produksi berdasarkan Test RMSE terendah
    pada skenario METEO.
    """
    candidates = []

    for results_path in RESULTS_DIR.glob("*_results.json"):
        try:
            with open(results_path, encoding="utf-8") as f:
                data = json.load(f)

            algo = data.get(
                "algorithm",
                results_path.stem.split("_")[0],
            )
            meteo = data["scenarios"]["METEO"]
            rmse = float(meteo["metrics_test"]["RMSE"])

            candidates.append({
                "algorithm": algo,
                "rmse": rmse,
            })

        except (KeyError, ValueError, json.JSONDecodeError):
            continue

    # Model dengan RMSE test terendah digunakan sebagai model produksi.
    if not candidates:
        best = {"algorithm": "lightgbm", "rmse": float("nan")}
    else:
        best = min(candidates, key=lambda c: c["rmse"])

    algo = best["algorithm"]
    model_pkl = MODELS_DIR / f"{algo}_METEO.pkl"

    if not model_pkl.exists():
        raise FileNotFoundError(
            f"Model terlatih tidak ditemukan: {model_pkl}"
        )

    estimator = joblib.load(model_pkl)

    return {
        "algorithm": algo,
        "test_rmse": best["rmse"],
        "estimator": estimator,
        "model_file": model_pkl.name,
    }


def get_active_model_info() -> dict:
    info = _select_best_model()

    return {
        "algorithm": info["algorithm"],
        "test_rmse": info["test_rmse"],
        "model_file": info["model_file"],
    }


@lru_cache(maxsize=1)
def _load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_FILE, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    return df


@lru_cache(maxsize=1)
def get_grids() -> list[dict]:
    df = _load_dataset()

    g = (
        df.groupby("location_id")[["lat", "lon"]]
        .first()
        .reset_index()
        .sort_values(["lat", "lon"])
    )

    return [
        {
            "id": r.location_id,
            "lat": float(r.lat),
            "lon": float(r.lon),
        }
        for r in g.itertuples(index=False)
    ]


def get_modules() -> dict:
    return PV_MODULES


def _nearest_grid(lat: float, lon: float) -> dict:
    # Data meteorologi diambil dari grid terdekat dengan lokasi user.
    grids = get_grids()

    return min(
        grids,
        key=lambda g: (g["lat"] - lat) ** 2 + (g["lon"] - lon) ** 2,
    )


def estimate_system(
    area_m2: float,
    module_key: str,
    custom: dict | None = None,
) -> dict:

    if custom:
        module = {"name": "Custom Module", **custom}
    else:
        module = PV_MODULES.get(
            module_key,
            PV_MODULES["generic_mono_400W"],
        )

    # Hanya 70% luas atap dianggap tersedia untuk pemasangan panel.
    usable_area = area_m2 * USABLE_FRACTION

    n_panels = (
        int(usable_area / module["area_m2"])
        if module["area_m2"] > 0
        else 0
    )

    capacity_kwp = n_panels * module["watt"] / 1000.0

    return {
        "module": module,
        "usable_area_m2": round(usable_area, 1),
        "n_panels": n_panels,
        "capacity_kwp": round(capacity_kwp, 2),
    }


def _predict_ghi_all_years(
    lat: float,
    lon: float,
) -> tuple[pd.DataFrame, dict]:

    grid = _nearest_grid(lat, lon)
    df = _load_dataset()

    site = (
        df[df["location_id"] == grid["id"]]
        .sort_values("date")
        .copy()
    )

    month = site["date"].dt.month
    doy = site["date"].dt.dayofyear

    feats = pd.DataFrame({
        "lat": lat,
        "lon": lon,
        "month": month.values,
        "day_of_year": doy.values,
        "month_sin": np.sin(2 * np.pi * month.values / 12.0),
        "month_cos": np.cos(2 * np.pi * month.values / 12.0),
        "doy_sin": np.sin(2 * np.pi * doy.values / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy.values / 365.25),
        "TEMP": site["TEMP"].values,
        "RH": site["RH"].values,
        "PRECIP": site["PRECIP"].values,
    })[METEO_FEATURES]

    estimator = _select_best_model()["estimator"]

    # Inferensi menggunakan model terlatih tanpa retraining.
    ghi_pred = estimator.predict(feats)

    # GHI secara fisik tidak boleh bernilai negatif.
    ghi_pred = np.clip(
        np.asarray(ghi_pred, dtype=float),
        0,
        None,
    )

    result = site[["date", "year"]].copy().reset_index(drop=True)
    result["GHI_pred"] = ghi_pred

    return result, grid


FORECAST_YEAR = 2026


def _forecast_ghi_2026(
    lat: float,
    lon: float,
) -> tuple[pd.DataFrame, dict]:

    grid = _nearest_grid(lat, lon)
    df = _load_dataset()

    site = df[df["location_id"] == grid["id"]].copy()
    site["doy"] = site["date"].dt.dayofyear

    # Covariate 2026 dibentuk dari rata-rata meteorologi 2021–2025
    # untuk setiap hari dalam setahun.
    clim = site.groupby("doy")[["TEMP", "RH", "PRECIP"]].mean()
    clim_fallback = clim.mean()

    dates = pd.date_range(
        f"{FORECAST_YEAR}-01-01",
        f"{FORECAST_YEAR}-12-31",
        freq="D",
    )
    doy = dates.dayofyear.to_numpy()

    def _clim_series(col: str) -> np.ndarray:
        vals = clim[col].reindex(doy).to_numpy()
        return np.where(
            np.isnan(vals),
            float(clim_fallback[col]),
            vals,
        )

    month = dates.month.to_numpy()

    feats = pd.DataFrame({
        "lat": lat,
        "lon": lon,
        "month": month,
        "day_of_year": doy,
        "month_sin": np.sin(2 * np.pi * month / 12.0),
        "month_cos": np.cos(2 * np.pi * month / 12.0),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "TEMP": _clim_series("TEMP"),
        "RH": _clim_series("RH"),
        "PRECIP": _clim_series("PRECIP"),
    })[METEO_FEATURES]

    estimator = _select_best_model()["estimator"]

    # Forecast GHI 2026 dilakukan menggunakan model terlatih.
    ghi_pred = estimator.predict(feats)
    ghi_pred = np.clip(
        np.asarray(ghi_pred, dtype=float),
        0,
        None,
    )

    result = pd.DataFrame({
        "date": dates,
        "GHI_pred": ghi_pred,
    })

    return result, grid


def _decompose_day(
    date,
    ghi_kwh: float,
    site: Location,
) -> dict | None:

    noon = pd.Timestamp(date).replace(
        hour=12,
        minute=0,
    )
    times = pd.DatetimeIndex(
        [noon],
        tz="Asia/Jakarta",
    )

    solpos = site.get_solarposition(times)
    zenith = float(solpos["apparent_zenith"].iloc[0])
    sol_azimuth = float(solpos["azimuth"].iloc[0])

    if zenith >= 90 or ghi_kwh <= 0:
        return None

    doy = pd.Timestamp(date).dayofyear
    dni_extra = float(
        pvlib.irradiance.get_extra_radiation(doy)
    )
    cos_z = np.cos(np.radians(zenith))

    ghi_wm2 = ghi_kwh * 1000 / EFFECTIVE_SUN_HOURS

    # Clearness index menentukan fraksi radiasi difus.
    kt = float(
        np.clip(
            ghi_wm2 / (dni_extra * max(cos_z, 0.05)),
            0,
            1.0,
        )
    )

    # Erbs digunakan untuk memisahkan GHI menjadi DHI dan DNI.
    if kt <= 0.22:
        kd = 1.0 - 0.09 * kt
    elif kt <= 0.80:
        kd = (
            0.9511
            - 0.1604 * kt
            + 4.388 * kt**2
            - 16.638 * kt**3
            + 12.336 * kt**4
        )
    else:
        kd = 0.165

    dhi_wm2 = kd * ghi_wm2
    dni_wm2 = max(
        (ghi_wm2 - dhi_wm2) / max(cos_z, 0.05),
        0,
    )

    return {
        "zenith": zenith,
        "sol_azimuth": sol_azimuth,
        "dni_extra": dni_extra,
        "ghi_wm2": ghi_wm2,
        "dhi_wm2": dhi_wm2,
        "dni_wm2": dni_wm2,
    }


def _transpose(
    comp: dict,
    tilt: float,
    azimuth: float,
    albedo: float,
) -> float:

    try:
        # Perez mentransposisikan DNI, DHI, dan GHI menjadi POA.
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            solar_zenith=comp["zenith"],
            solar_azimuth=comp["sol_azimuth"],
            dni=comp["dni_wm2"],
            ghi=comp["ghi_wm2"],
            dhi=comp["dhi_wm2"],
            dni_extra=comp["dni_extra"],
            model="perez",
            albedo=albedo,
        )

        poa_wm2 = max(
            float(np.asarray(poa["poa_global"]).item()),
            0,
        )

    except Exception:
        poa_wm2 = comp["ghi_wm2"]

    return poa_wm2 * EFFECTIVE_SUN_HOURS / 1000.0


def _poa_daily(
    dates,
    ghi_daily,
    lat,
    lon,
    tilt,
    azimuth,
    albedo,
) -> np.ndarray:

    site = Location(
        lat,
        lon,
        tz="Asia/Jakarta",
    )

    out = []

    for date, ghi_kwh in zip(dates, ghi_daily):
        comp = _decompose_day(
            date,
            ghi_kwh,
            site,
        )

        if comp is None:
            out.append(0.0)
            continue

        out.append(
            _transpose(
                comp,
                tilt,
                azimuth,
                albedo,
            )
        )

    return np.array(out)


def recommend_orientation(
    lat: float,
    lon: float,
    albedo: float = 0.2,
) -> dict:

    # Panel diarahkan ke ekuator berdasarkan hemisfer lokasi.
    if lat < 0:
        azimuth = 0.0
        azimuth_label = "Utara (menghadap ekuator)"
    else:
        azimuth = 180.0
        azimuth_label = "Selatan (menghadap ekuator)"

    ghi_data, _ = _predict_ghi_all_years(lat, lon)

    # Tahun tengah digunakan sebagai tahun representatif untuk optimasi.
    mid_year = ALL_YEARS[len(ALL_YEARS) // 2]
    yd = ghi_data[ghi_data["year"] == mid_year]

    if len(yd) == 0:
        yd = ghi_data

    site = Location(
        lat,
        lon,
        tz="Asia/Jakarta",
    )

    # Dekomposisi cukup dilakukan sekali karena tidak bergantung pada tilt.
    comps = [
        _decompose_day(d, g, site)
        for d, g in zip(
            yd["date"].values,
            yd["GHI_pred"].values,
        )
    ]

    candidate_tilts = range(0, 31, 2)
    best_tilt, best_poa = 0, -1.0

    for tilt in candidate_tilts:
        # Tilt dengan POA tahunan terbesar dipilih sebagai rekomendasi.
        poa_annual = sum(
            _transpose(
                c,
                float(tilt),
                azimuth,
                albedo,
            )
            for c in comps
            if c is not None
        )

        if poa_annual > best_poa:
            best_poa = poa_annual
            best_tilt = tilt

    return {
        "tilt": float(best_tilt),
        "azimuth": azimuth,
        "azimuth_label": azimuth_label,
        "method": "POA-maximizing scan (Perez) + equator-facing azimuth",
        "poa_annual_kwh_m2": round(best_poa, 1),
    }


def _npv(
    rate: float,
    cashflows: np.ndarray,
) -> float:

    cf = np.asarray(cashflows, dtype=float)
    t = np.arange(len(cf))

    return float(
        np.sum(cf / (1 + rate) ** t)
    )


def run_analysis(
    lat: float,
    lon: float,
    params: dict,
) -> dict:

    p = {**DEFAULT_PARAMS, **params}

    capacity = float(p["capacity_kwp"])
    capex_total = capacity * float(p["capex_per_kwp"])
    opex_annual = float(p["opex_rate"]) * capex_total
    tariff = float(p["tarif_rp_kwh"])
    lifetime = int(p["lifetime_thn"])
    discount = float(p["discount_rate"])
    degrad = float(p["degradasi"])
    pr = float(p["performance_ratio"])
    self_cons = float(p["self_consumption"])
    escalation = float(p["eskalasi_tarif"])
    tilt = float(p["tilt"])
    azimuth_val = float(p["azimuth"])
    albedo = float(p["albedo"])

    model_info = _select_best_model()

    # Forecast GHI 2026 menjadi basis produksi energi Year-1.
    forecast, grid = _forecast_ghi_2026(lat, lon)

    # GHI ditransposisikan menjadi POA sesuai orientasi panel.
    poa_daily = _poa_daily(
        forecast["date"].values,
        forecast["GHI_pred"].values,
        lat,
        lon,
        tilt,
        azimuth_val,
        albedo,
    )

    ghi_annual = float(forecast["GHI_pred"].sum())
    poa_annual = float(poa_daily.sum())

    # AEP = kapasitas × POA × performance ratio.
    aep_year1 = capacity * poa_annual * pr

    ghi_profile_values = [
        round(float(v), 3)
        for v in forecast["GHI_pred"].to_numpy()
    ]

    ghi_profile_labels = [
        d.strftime("%Y-%m-%d")
        for d in forecast["date"]
    ]

    e_year1 = aep_year1

    specific_yield = (
        e_year1 / capacity
        if capacity > 0
        else 0
    )

    years_arr = np.arange(1, lifetime + 1)

    # Produksi energi menurun sesuai degradasi panel.
    energy = e_year1 * (1 - degrad) ** (years_arr - 1)

    # Tarif listrik meningkat sesuai eskalasi tahunan.
    tariff_year = tariff * (1 + escalation) ** (years_arr - 1)

    # Penghematan berasal dari energi yang dikonsumsi sendiri.
    savings = energy * self_cons * tariff_year

    net_cf = savings - opex_annual

    # CAPEX menjadi arus kas awal pada tahun ke-0.
    cashflows = np.concatenate([
        [-capex_total],
        net_cf,
    ])

    # NPV positif digunakan sebagai indikator proyek layak.
    project_npv = _npv(
        discount,
        cashflows,
    )

    return {
        "algorithm": model_info["algorithm"].upper(),
        "model_file": model_info["model_file"],
        "params": p,
        "capex_total": capex_total,
        "lokasi": {
            "lat": lat,
            "lon": lon,
            "grid_id": grid["id"],
            "area_m2": round(
                float(p.get("area_m2", 0)),
                1,
            ),
            "usable_area_m2": round(
                float(p.get("usable_area_m2", 0)),
                1,
            ),
        },
        "orientasi": {
            "tilt": round(tilt, 1),
            "azimuth": round(azimuth_val, 1),
        },
        "pv_module": p.get("pv_module_name", "Custom"),
        "n_panels": int(p.get("n_panels", 0)),
        "capacity_kwp": round(capacity, 2),
        "energi": {
            "ghi_annual_kwh_m2": round(ghi_annual, 1),
            "aep_kwh": round(e_year1, 0),
            "specific_yield_kwh_kwp": round(
                specific_yield,
                0,
            ),
        },
        "ghi_profile": {
            "labels": ghi_profile_labels,
            "values": ghi_profile_values,
            "annual_kwh_m2": round(ghi_annual, 1),
        },
        "npv": round(project_npv, 0),
        "layak": bool(project_npv > 0),
    }