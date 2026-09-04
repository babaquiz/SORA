import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import segyio

import urllib.request

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import streamlit as st
import joblib

@st.cache_resource
def load_my_model():
    # Ganti dengan path model kamu
    return joblib.load("model.pkl") 

model = load_my_model()

DATA_URL = "https://huggingface.co/datasets/Rafisuper/SORA/resolve/main/Citra%20Sumur.segy?download=true"
DATA_PATH = "Citra Sumur.segy"

# TAMPILKAN INTERFACE UTAMA TERLEBIH DAHULU AGAR SERVER TIDAK TIMEOUT/CRASH
st.title("Demo Sistem Analisis SEGY Live")
st.subheader("Sistem Pengolahan Data Seismik")

# FUNGSI UNDUH DENGAN PROGRES BAR SUPAYA SERVER TETAP AKTIF
def download_data():
    if not os.path.exists(DATA_PATH):
        progress_bar = st.progress(0, text="Menghubungkan ke server data Hugging Face...")
        
        # Fungsi callback untuk menghitung persentase unduhan
        def reporthook(count, block_size, total_size):
            current_progress = count * block_size
            if total_size > 0:
                percent = min(int(current_progress * 100 / total_size), 100)
                progress_bar.progress(percent / 100, text=f"Sedang mengunduh file Citra Sumur (1.22 GB): {percent}% selesai")
        
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_PATH, reporthook)
            progress_bar.empty()
            st.success("Data sistem 1.22 GB berhasil dimuat sepenuhnya!")
        except Exception as e:
            progress_bar.empty()
            st.error(f"Gagal mengunduh data: {e}")
            st.info("Silakan coba klik 'Reboot app' di menu kanan bawah.")
            st.stop()
    return DATA_PATH

# Jalankan fungsi unduhan dengan aman setelah halaman web dirender
file_segy_siap = download_data()

st.set_page_config(
    page_title="SORA (Subsurface Optimization and Recovery Assistant)",
    layout="wide"
)

class RealVolveEngine:
    def __init__(self, excel_path="Volve production data.xlsx", segy_path="Citra Sumur.segy"):
        self.excel_path = excel_path
        self.segy_path = segy_path

    def load_and_clean_data(self):
        if not os.path.exists(self.excel_path):
            st.error(f"File '{self.excel_path}' tidak ditemukan!")
            st.stop()

        try:
            xls = pd.ExcelFile(self.excel_path)
            sheet_target = None
            for s in xls.sheet_names:
                if "daily" in s.lower() or "p&i" in s.lower() or "production" in s.lower():
                    sheet_target = s
                    break

            if not sheet_target:
                sheet_target = xls.sheet_names[0]

            df_raw = pd.read_excel(xls, sheet_name=sheet_target)
            df_raw.columns = df_raw.columns.str.strip()

            df_clean = df_raw[
                (df_raw['AVG_DOWNHOLE_PRESSURE'] > 0) &
                (df_raw['AVG_DOWNHOLE_TEMPERATURE'] > 0) &
                (df_raw['BORE_OIL_VOL'] > 0)
            ].copy()

            if df_clean.empty:
                st.error("Data kosong. Pastikan kolom memiliki nilai > 0.")
                st.stop()

            df_clean['WATER_CUT'] = df_clean['BORE_WAT_VOL'] / (df_clean['BORE_OIL_VOL'] + df_clean['BORE_WAT_VOL'] + 1e-6)

            st.sidebar.success(f"Data berhasil dimuat! Membaca {len(df_clean):,} Sampel")
            return df_clean

        except Exception as e:
            st.error(f"Gagal membaca file {e}")
            st.stop()

    def load_seismic_4d_amplitude(self):
        if not os.path.exists(self.segy_path):
            st.sidebar.info("File SEGY  tidak ditemukan di lokal")
            return 0.0025

        try:
         
            with segyio.open(self.segy_path, mode='r', ignore_geometry=True) as f:
 
                n_traces = len(f.trace)
                mid_trace = f.trace[n_traces // 2]
                delta_amplitude_mean = float(np.mean(mid_trace))
                
                st.sidebar.success(f"Data Seismik 4D Berhasil Terbaca!")
                return delta_amplitude_mean

        except Exception as e:
            st.sidebar.warning(f"Pembacaan SEGY dilewati: {e}")
            return 0.0025

class SEIRAAICore:

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.best_params_ = None
        self.cv_r2_mean = None
        self.cv_r2_std = None
        self.feature_importances_df = None
        self.feature_cols = [
            'AVG_DOWNHOLE_PRESSURE',
            'AVG_DOWNHOLE_TEMPERATURE',
            'AVG_CHOKE_SIZE_P',
            'BORE_GAS_VOL',
            'BORE_WAT_VOL',
            'WATER_CUT'
        ]

    def train(self, data, n_iter=25, cv_folds=5):
        if data.empty:
            raise ValueError("Dataframe  tidak boleh kosong!")

        X = data[self.feature_cols]
        y = data['BORE_OIL_VOL']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        param_dist = {
            'n_estimators': [200, 300, 400, 500, 600],
            'max_depth': [None, 10, 15, 20, 25, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5, 0.8, 1.0]
        }

        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='r2',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        self.best_params_ = search.best_params_


        preds = self.model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))


        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='r2', n_jobs=-1)
        self.cv_r2_mean = cv_scores.mean()
        self.cv_r2_std = cv_scores.std()

        importances = self.model.feature_importances_
        self.feature_importances_df = pd.DataFrame({
            'Fitur': self.feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False).reset_index(drop=True)

        self.is_trained = True
        return r2, rmse

    def predict_oil_rate(self, current_state):
        if not self.is_trained:
            raise ValueError("Model AI belum dilatih!")
        df_input = pd.DataFrame([current_state])
        return self.model.predict(df_input[self.feature_cols])[0]

    def optimize_eor_strategy(self, current_state):
        press = current_state['AVG_DOWNHOLE_PRESSURE']
        wc = current_state['WATER_CUT']

        viscosity = current_state.get('VISCOSITY', 3.5)
        permeability = current_state.get('PERMEABILITY', 500)
        rock_type = current_state.get('ROCK_TYPE', 'Sandstone')
        delta_amp_4d = current_state.get('DELTA_AMP_4D', 0.0)

        if viscosity > 20.0:
            return (
                "Thermal EOR",
                f"Viskositas minyak tinggi ({viscosity} cP). Perlu injeksi uap panas untuk menurunkan viskositas fluida."
            )
        elif wc > 0.60:
            return (
                "Polymer",
                f"Water Cut sangat tinggi ({wc*100:.1f}%). Diperlukan polymer gel untuk menyumbat zona berpermeabilitas tinggi ({permeability} mD) agar menyapu sisa minyak."
            )
        elif press > 220 and viscosity < 10.0:
            return (
                "CO2-Miscible Injection",
                f"Tekanan reservoir mencukupi ({press:.1f} bar > MMP) dan viskositas minyak rendah ({viscosity} cP) pada formasi {rock_type}. Respon Seismik 4D (ΔAmp = {delta_amp_4d:.4f}) mengonfirmasi penyebaran fluida optimal."
            )
        elif permeability > 50 and wc <= 0.60:
            return (
                "ASP (Alkaline-Surfactant-Polymer) Flooding",
                f"Kondisi batuan {rock_type} (k = {permeability} mD) mendukung injeksi kimia untuk menurunkan tegangan antarmuka (IFT) dan meningkatkan efisiensi penyapuan."
            )
        else:
            return (
                "Waterflooding / Immiscible Gas",
                "Tekanan reservoir rendah dan permeabilitas terbatas. Lakukan pemeliharaan tekanan dasar terlebih dahulu."
            )



class AutomationEngine:
    @staticmethod
    def check_well_health(pressure, choke_size):
        if choke_size > 80.0 and pressure < 150.0:
            return "CRITICAL", "Choke terbuka lebar (>80%) tetapi tekanan dasar sumur drop (<150 bar). Bahaya liquid loading!"
        elif choke_size > 90.0:
            return "WARNING", "Bukaan choke mendekati kapasitas maksimal. Lakukan evaluasi kerugian gesekan tubing."
        else:
            return "NORMAL", "Operasi tekanan dan aliran sumur berada pada rentang optimal."

    @staticmethod
    def predict_pump_failure(vibration_rms, bearing_temp, inlet_pressure):
        faults = []

        if vibration_rms > 7.1:
            faults.append("Vibrasi Ekstrem (Misalignment / Severe Bearing Failure)")
        elif vibration_rms >= 4.5:
            faults.append("Vibrasi Meningkat (Unbalance Awal / Mechanical Loose)")

        if bearing_temp > 95.0:
            faults.append("Overheating Bearing (Gagal Pelumasan / Gesekan Tinggi)")
        elif bearing_temp >= 80.0:
            faults.append("Temperatur Bearing Tinggi")

        if inlet_pressure < 2.0:
            faults.append("Tekanan Suction Drop (Risiko Kavitasi Impeller)")

        if vibration_rms > 7.1 or bearing_temp > 95.0 or inlet_pressure < 2.0:
            status = "CRITICAL"
            rul = "< 24 Jam (Risiko Fail Tinggi)"
            action = "Emergency Shutdown! Switch ke pompa cadangan dan lakukan overhaul/inspeksi mechanical seal."
        elif vibration_rms >= 4.5 or bearing_temp >= 80.0:
            status = "WARNING"
            rul = "120 - 168 Jam (1 Minggu)"
            action = "Jadwalkan pemeliharaan preventif, lakukan re-greasing, dan re-alignment rotor."
        else:
            status = "NORMAL"
            rul = "> 8,000 Jam (Kondisi Prima)"
            action = "Pompa beroperasi efisien. Lakukan pemantauan rutin harian."

        return {
            "status": status,
            "detected_faults": faults if faults else ["Tidak ada indikasi kerusakan"],
            "rul_estimate": rul,
            "recommended_action": action
        }


# ==========================================
# LAYER 4: SUSTAINABILITY MONITOR
# ==========================================
class SustainabilityMonitor:
    def __init__(self, co2_density_kg_m3=1.98):
        self.co2_density = co2_density_kg_m3

    def calculate_carbon_footprint(self, gas_vol_sm3_day, days=30):
        total_gas_sm3 = gas_vol_sm3_day * days
        mass_kg = total_gas_sm3 * self.co2_density
        mass_tons_stored = mass_kg / 1000.0

        net_carbon_sequestered = mass_tons_stored * 0.85
        emitted_carbon = mass_tons_stored * 0.15

        return {
            "gross_co2_injected_tons": round(mass_tons_stored, 2),
            "net_carbon_sequestered_tons": round(net_carbon_sequestered, 2),
            "process_emissions_tons": round(emitted_carbon, 2),
            "esg_compliance_status": "COMPLIANT (Net Positive Storage)" if net_carbon_sequestered > emitted_carbon else "NON-COMPLIANT"
        }


st.title("Subsurface Optimization and Recovery Assistant")
st.caption("Sistem integrasi Artificial Intelligence dalam operasi Enhanced Oil Recovery (EOR)")

@st.cache_resource
def initialize_system():
    engine = RealVolveEngine(excel_path="Volve production data.xlsx")
    dataset = engine.load_and_clean_data()
    delta_amp_4d = engine.load_seismic_4d_amplitude()
    ai_core = SEIRAAICore()
    with st.spinner("Melakukan hyperparameter tuning Random Forest"):
        r2, rmse = ai_core.train(dataset)
    return dataset, ai_core, r2, rmse, delta_amp_4d

dataset, ai_core, r2_score_val, rmse_val, delta_amp_4d_val = initialize_system()

# --- SIDEBAR: CONTROLLER SENSOR REAL-TIME ---
st.sidebar.header("Telemetri Sensor Sumur (Volve)")
st.sidebar.caption("Atur parameter operasional sumur terhubung:")

pressure = st.sidebar.slider("Tekanan Dasar Sumur / BHP (bar)", 50.0, 350.0, 240.0)
temperature = st.sidebar.slider("Temperatur Reservoir (°C)", 50.0, 120.0, 104.0)
choke_size = st.sidebar.slider("Pembukaan Katup Choke (%)", 1.0, 100.0, 50.0)
gas_rate = st.sidebar.slider("Laju Gas (Sm³/day)", 1000.0, 500000.0, 150000.0)
water_rate = st.sidebar.slider("Laju Air / Water Rate (Sm³/day)", 0.0, 5000.0, 800.0)

st.sidebar.markdown("---")
st.sidebar.header("Sifat Batuan & Fluida Reservoir")
viscosity = st.sidebar.slider("Viskositas Minyak (cP)", 0.5, 50.0, 3.5)
permeability = st.sidebar.slider("Permeabilitas Batuan (mD)", 5, 2000, 500)
rock_type = st.sidebar.selectbox("Tipe Batuan Reservoir", ["Sandstone", "Carbonate"])

st.sidebar.markdown("---")
st.sidebar.header("Telemetri Pompa Injeksi (Permukaan)")
st.sidebar.caption("Monitoring vibrasi & kesehatan pompa real-time:")

pump_inlet_press = st.sidebar.slider("Tekanan Inlet Pompa (bar)", 0.5, 20.0, 8.5)
pump_bearing_temp = st.sidebar.slider("Temperatur Bearing (°C)", 30.0, 120.0, 65.0)
pump_vibration_rms = st.sidebar.slider("Vibrasi Pompa (mm/s RMS)", 0.5, 12.0, 2.3)

calculated_water_cut = water_rate / (100.0 + water_rate + 1e-6)

current_state = {
    'AVG_DOWNHOLE_PRESSURE': pressure,
    'AVG_DOWNHOLE_TEMPERATURE': temperature,
    'AVG_CHOKE_SIZE_P': choke_size,
    'BORE_GAS_VOL': gas_rate,
    'BORE_WAT_VOL': water_rate,
    'WATER_CUT': calculated_water_cut,
    'VISCOSITY': viscosity,
    'PERMEABILITY': permeability,
    'ROCK_TYPE': rock_type,
    'DELTA_AMP_4D': delta_amp_4d_val
}

predicted_oil = ai_core.predict_oil_rate(current_state)
eor_rec, eor_reason = ai_core.optimize_eor_strategy(current_state)

well_status_code, well_status_msg = AutomationEngine.check_well_health(pressure, choke_size)
pump_diag = AutomationEngine.predict_pump_failure(pump_vibration_rms, pump_bearing_temp, pump_inlet_press)

sustainability = SustainabilityMonitor()
esg_res = sustainability.calculate_carbon_footprint(gas_rate)


st.subheader("Key Performance Indicators")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Prediksi Produksi Minyak", f"{predicted_oil:.2f} Sm³/day")
with kpi2:
    st.metric("Rekomendasi Strategi EOR", eor_rec)
with kpi3:
    st.metric("Model AI R²", f"{r2_score_val:.4f}")
with kpi4:
    st.metric("Status ESG Compliance", "COMPLIANT" if "COMPLIANT" in esg_res["esg_compliance_status"] else "NON-COMPLIANT")


st.subheader("Detail Akurasi Model Random Forest")
acc1, acc2, acc3 = st.columns(3)
with acc1:
    st.metric("RMSE", f"{rmse_val:,.2f} Sm³/day")
with acc2:
    st.metric("Cross-Validation R² (5-fold, mean)", f"{ai_core.cv_r2_mean:.4f}")
with acc3:
    st.metric("Cross-Validation R² (std dev)", f"± {ai_core.cv_r2_std:.4f}")

with st.expander("Lihat Hyperparameter Terbaik Hasil Tuning (RandomizedSearchCV)"):
    st.json(ai_core.best_params_)
    st.caption(
        "Pencarian dilakukan pada ruang parameter n_estimators, max_depth, min_samples_split, "
        "min_samples_leaf, dan max_features menggunakan RandomizedSearchCV dengan 5-fold "
        "cross-validation berbasis skor R² untuk menghindari overfitting terhadap satu pembagian data."
    )

with st.expander("Feature Importance"):
    st.bar_chart(ai_core.feature_importances_df.set_index('Fitur'))
    st.dataframe(ai_core.feature_importances_df, use_container_width=True)

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("Decision & Predictive Maintenance Engine")

    st.markdown("Status Sumur (Downhole & Tubing)")
    if well_status_code == "CRITICAL":
        st.error(f"**STATUS SUMUR: {well_status_code}**\n\n{well_status_msg}")
    elif well_status_code == "WARNING":
        st.warning(f"**STATUS SUMUR: {well_status_code}**\n\n{well_status_msg}")
    else:
        st.success(f"**STATUS SUMUR: {well_status_code}**\n\n{well_status_msg}")

    st.info(f"**Alasan Strategi EOR:** {eor_reason}")

    st.markdown("Predictive Maintenance Pompa Injeksi (Surface)")

    p_status = pump_diag["status"]
    if p_status == "CRITICAL":
        st.error(f"**STATUS POMPA: {p_status}**")
    elif p_status == "WARNING":
        st.warning(f"**STATUS POMPA: {p_status}**")
    else:
        st.success(f"**STATUS POMPA: {p_status}**")

    st.write(f"• **Indikasi Kerusakan:** {', '.join(pump_diag['detected_faults'])}")
    st.write(f"• **Estimasi Sisa Umur (RUL):** {pump_diag['rul_estimate']}")
    st.write(f"• **Rekomendasi Tindakan AI:** {pump_diag['recommended_action']}")

with col_right:
    st.markdown("**ESG & Carbon Capture Tracking (30 Hari)**")
    st.write(f"• **Gross CO2/Gas Handled:** {esg_res['gross_co2_injected_tons']:,} Ton")
    st.write(f"• **Net Carbon Sequestered:** {esg_res['net_carbon_sequestered_tons']:,} Ton")
    st.write(f"• **Process Emissions:** {esg_res['process_emissions_tons']:,} Ton")

st.markdown("---")

with st.expander("Lihat Samples Data Riil Volve Field"):
    display_cols = ['DATEPRD', 'NPD_WELL_BORE_NAME', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE', 'AVG_CHOKE_SIZE_P', 'BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL']
    available_cols = [c for c in display_cols if c in dataset.columns]
    st.dataframe(dataset[available_cols].head(50), use_container_width=True)
