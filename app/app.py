from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from service import (
    DEFAULT_PARAMS,
    PV_MODULES,
    estimate_system,
    get_grids,
    recommend_orientation,
    run_analysis,
)

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# TODO: pindahkan ke environment variable sebelum deploy ke production
app.secret_key = "solarfit-dev-key"


def _get_float(data: dict, name: str, default: float) -> float:
    """Ambil nilai float dari payload, fallback ke default jika kosong/tidak valid."""
    raw = str(data.get(name, "")).strip()

    # Field kosong dianggap belum diisi, gunakan default
    if raw == "":
        return float(default)

    try:
        return float(raw)
    except ValueError:
        # Input tidak bisa dikonversi ke float
        return float(default)


@app.route("/")
def landing():
    """Landing page — penjelasan tool, input/output, cara pakai."""
    return render_template("landing.html")


@app.route("/app")
def analyze():
    """Halaman utama analisis: peta interaktif + panel konfigurasi + hasil."""
    return render_template(
        "index.html",
        grids=get_grids(),
        defaults=DEFAULT_PARAMS,
        modules=PV_MODULES,
    )


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """
    Rekomendasi orientasi panel (tilt & azimuth) untuk lokasi terpilih.
    Dipanggil otomatis setelah user selesai menggambar atap.
    """
    data = request.get_json(silent=True) or {}

    # Validasi koordinat sebelum menjalankan rekomendasi orientasi
    try:
        lat = float(data["lat"])
        lon = float(data["lon"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Koordinat tidak valid."}), 400

    rec = recommend_orientation(lat, lon)
    return jsonify(rec)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Jalankan pipeline analisis lengkap dan kembalikan hasil sebagai JSON,
    untuk dirender di bawah panel konfigurasi pada halaman yang sama.
    """
    data = request.get_json(silent=True) or {}

    # Validasi koordinat lokasi yang dikirim dari peta
    try:
        lat = float(data["lat"])
        lon = float(data["lon"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Silakan gambar area atap pada peta terlebih dahulu."}), 400

    # Validasi luas area atap sebelum melakukan estimasi sistem
    area_m2 = _get_float(data, "area_m2", 0.0)
    if area_m2 <= 0:
        return jsonify({"error": "Gambar area atap pada peta untuk mengestimasi ukuran sistem."}), 400

    # Estimasi kapasitas sistem PV berdasarkan luas atap dan spesifikasi modul
    module_key = data.get("pv_module", "generic_mono_400W")

    if module_key == "custom":
        # Ambil spesifikasi modul kustom dari payload
        custom_watt = _get_float(data, "custom_watt", 400.0)
        custom_area = _get_float(data, "custom_area", 2.0)
        custom_name = str(data.get("custom_name", "")).strip() or "Modul Kustom"

        if custom_watt <= 0 or custom_area <= 0:
            return jsonify({"error": "Daya panel dan luas panel harus lebih dari 0 untuk modul kustom."}), 400

        system = estimate_system(
            area_m2,
            module_key="generic_mono_400W",  # fallback key, tidak dipakai saat custom
            custom={"watt": custom_watt, "area_m2": custom_area, "name": custom_name},
        )
    else:
        system = estimate_system(area_m2, module_key)

    # Pastikan area mencukupi untuk memasang minimal satu panel PV
    if system["capacity_kwp"] <= 0:
        return jsonify({"error": "Area yang digambar terlalu kecil untuk modul PV manapun."}), 400

    # Susun parameter analisis dan gunakan nilai default untuk input yang kosong
    params = {
        "capacity_kwp":      system["capacity_kwp"],
        "capex_per_kwp":     _get_float(data, "capex_per_kwp",     DEFAULT_PARAMS["capex_per_kwp"]),
        "tarif_rp_kwh":      _get_float(data, "tarif_rp_kwh",      DEFAULT_PARAMS["tarif_rp_kwh"]),
        "lifetime_thn":      int(_get_float(data, "lifetime_thn",  DEFAULT_PARAMS["lifetime_thn"])),
        "discount_rate":     _get_float(data, "discount_rate",     DEFAULT_PARAMS["discount_rate"]),
        "degradasi":         _get_float(data, "degradasi",         DEFAULT_PARAMS["degradasi"]),
        "performance_ratio": _get_float(data, "performance_ratio", DEFAULT_PARAMS["performance_ratio"]),
        "self_consumption":  _get_float(data, "self_consumption",  DEFAULT_PARAMS["self_consumption"]),
        "opex_rate":         _get_float(data, "opex_rate",         DEFAULT_PARAMS["opex_rate"]),
        "eskalasi_tarif":    _get_float(data, "eskalasi_tarif",    DEFAULT_PARAMS["eskalasi_tarif"]),
        "tilt":              _get_float(data, "tilt",              DEFAULT_PARAMS["tilt"]),
        "azimuth":           _get_float(data, "azimuth",           DEFAULT_PARAMS["azimuth"]),
        "area_m2":           area_m2,
        "usable_area_m2":    system["usable_area_m2"],
        "pv_module_name":    system["module"]["name"],
        "n_panels":          system["n_panels"],
    }

    # Jalankan pipeline TEA berdasarkan parameter yang telah divalidasi
    result = run_analysis(lat, lon, params)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)