import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Config ──
st.set_page_config(
    page_title="Customer Propensity Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800; color: #E74C3C;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem; color: #7F8C8D; text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 1.2rem; text-align: center; color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value { font-size: 2rem; font-weight: 800; }
    .metric-label { font-size: 0.85rem; opacity: 0.9; }
    .result-box-buy {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 12px; padding: 1.5rem; text-align: center; color: white;
    }
    .result-box-nobuy {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 12px; padding: 1.5rem; text-align: center; color: white;
    }
    .result-title { font-size: 1.6rem; font-weight: 800; }
    .result-prob  { font-size: 3rem; font-weight: 900; }
    .seg-badge {
        display: inline-block; padding: 0.3rem 1rem; border-radius: 20px;
        font-weight: 700; font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 20px; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ──
@st.cache_resource
def load_model():
    model_path = 'model_propensity.pkl'
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

artifacts = load_model()


# ── Feature Engineering (sama persis dengan notebook) ──
def add_engineered_features(df):
    df = df.copy()
    df['purchase_intent_score'] = (
        df['basket_add_detail'] + df['basket_add_list'] +
        df['saw_checkout']      + df['basket_icon_click'] + df['sign_in']
    )
    df['engagement_score'] = (
        df['list_size_dropdown']       + df['checked_delivery_detail'] +
        df['image_picker']             + df['sort_by'] + df['promo_banner_click']
    )
    df['research_score'] = (
        df['detail_wishlist_add']     + df['checked_delivery_detail'] +
        df['checked_returns_detail']  + df['saw_sizecharts'] + df['saw_delivery']
    )
    base_cols = [c for c in df.columns if c not in
                 ['purchase_intent_score','engagement_score','research_score']]
    binary_cols = [c for c in base_cols if df[c].isin([0,1]).all()]
    df['total_interactions'] = df[binary_cols].sum(axis=1)
    return df


def get_segment_color(seg):
    colors = {'Low': '#3498DB', 'Medium': '#F39C12', 'High': '#E67E22', 'Very High': '#E74C3C'}
    return colors.get(seg, '#95A5A6')

def get_segment(prob):
    if prob < 0.2:   return 'Low'
    elif prob < 0.5: return 'Medium'
    elif prob < 0.75: return 'High'
    else:            return 'Very High'


# ── SIDEBAR ──
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)
    st.markdown("### 🛒 Customer Propensity")
    st.markdown("---")

    if artifacts:
        st.markdown("#### 📊 Model Info")
        metrics = artifacts.get('metrics', {})
        st.success(f"**Model**: {artifacts.get('model_name','XGBoost')}")
        st.info(f"**ROC-AUC**: {metrics.get('ROC-AUC', 0):.4f}")
        st.info(f"**F1-Score**: {metrics.get('F1-Score', 0):.4f}")
        st.info(f"**Precision**: {metrics.get('Precision', 0):.4f}")
        st.info(f"**Recall**: {metrics.get('Recall', 0):.4f}")
        st.markdown("---")

    st.markdown("#### 🎯 Tentang App")
    st.markdown("""
    App ini memprediksi kemungkinan seorang customer akan melakukan **pembelian** berdasarkan perilaku browsing mereka di e-commerce.
    
    **Fitur yang dianalisis:**
    - Aktivitas keranjang belanja
    - Perilaku pencarian & filter
    - Tipe perangkat & lokasi
    - Status pengguna (baru/lama)
    """)
    st.markdown("---")
    st.caption("Built with ❤️ using XGBoost + Streamlit")


# ── HEADER ──
st.markdown('<p class="main-title">🛒 Customer Propensity to Purchase</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Prediksi kemungkinan customer melakukan pembelian berdasarkan perilaku browsing</p>', unsafe_allow_html=True)

if not artifacts:
    st.error("⚠️ Model tidak ditemukan! Pastikan file `model_propensity.pkl` ada di direktori yang sama dengan `app.py`.")
    st.stop()


# ── TABS ──
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Manual", "📂 Prediksi Batch (CSV)", "📊 Model Performance"])


# ═══════════════════════════════════════════
# TAB 1: Manual Prediction
# ═══════════════════════════════════════════
with tab1:
    st.markdown("### 🔮 Prediksi Satu Customer")
    st.markdown("Masukkan aktivitas browsing customer, lalu klik **Prediksi** untuk melihat hasilnya.")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("#### 🛍️ Aktivitas Keranjang & Checkout")
        basket_icon_click    = st.toggle("Klik ikon keranjang (basket_icon_click)",     value=False)
        basket_add_list      = st.toggle("Tambah ke list dari keranjang (basket_add_list)", value=False)
        basket_add_detail    = st.toggle("Tambah ke keranjang dari detail (basket_add_detail)", value=False)
        saw_checkout         = st.toggle("Melihat halaman checkout (saw_checkout)",     value=False)
        closed_minibasket    = st.toggle("Menutup mini basket (closed_minibasket_click)", value=False)

        st.markdown("#### 🔍 Aktivitas Pencarian & Filter")
        sort_by              = st.toggle("Menggunakan sort/filter (sort_by)",           value=False)
        image_picker         = st.toggle("Menggunakan image picker (image_picker)",     value=False)
        list_size_dropdown   = st.toggle("Memilih ukuran dari list (list_size_dropdown)", value=False)
        promo_banner_click   = st.toggle("Klik promo banner (promo_banner_click)",      value=False)

        st.markdown("#### 💳 Akun & Sign In")
        sign_in              = st.toggle("Login / Sign in (sign_in)",                   value=False)
        account_page_click   = st.toggle("Klik halaman akun (account_page_click)",      value=False)
        saw_account_upgrade  = st.toggle("Melihat halaman upgrade akun (saw_account_upgrade)", value=False)

    with col_r:
        st.markdown("#### 📦 Informasi Pengiriman & Detail")
        checked_delivery     = st.toggle("Cek detail pengiriman (checked_delivery_detail)", value=False)
        checked_returns      = st.toggle("Cek detail retur (checked_returns_detail)",  value=False)
        saw_delivery         = st.toggle("Melihat halaman delivery (saw_delivery)",    value=False)
        saw_sizecharts       = st.toggle("Melihat size chart (saw_sizecharts)",        value=False)

        st.markdown("#### ❤️ Wishlist & Halaman")
        detail_wishlist_add  = st.toggle("Tambah ke wishlist (detail_wishlist_add)",   value=False)
        saw_homepage         = st.toggle("Melihat homepage (saw_homepage)",            value=False)

        st.markdown("#### 📱 Perangkat")
        device_type = st.radio("Tipe Perangkat", ["Mobile", "Computer", "Tablet"],
                                index=0, horizontal=True)
        device_mobile   = 1 if device_type == "Mobile"   else 0
        device_computer = 1 if device_type == "Computer" else 0
        device_tablet   = 1 if device_type == "Tablet"   else 0

        st.markdown("#### 🌍 Profil Customer")
        returning_user  = st.toggle("Returning User (bukan customer baru)", value=False)
        loc_uk          = st.toggle("Lokasi UK",                            value=True)

    st.markdown("---")
    predict_btn = st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True)

    if predict_btn:
        input_data = {
            'basket_icon_click':        int(basket_icon_click),
            'basket_add_list':          int(basket_add_list),
            'basket_add_detail':        int(basket_add_detail),
            'sort_by':                  int(sort_by),
            'image_picker':             int(image_picker),
            'account_page_click':       int(account_page_click),
            'promo_banner_click':       int(promo_banner_click),
            'detail_wishlist_add':      int(detail_wishlist_add),
            'list_size_dropdown':       int(list_size_dropdown),
            'closed_minibasket_click':  int(closed_minibasket),
            'checked_delivery_detail':  int(checked_delivery),
            'checked_returns_detail':   int(checked_returns),
            'sign_in':                  int(sign_in),
            'saw_checkout':             int(saw_checkout),
            'saw_sizecharts':           int(saw_sizecharts),
            'saw_delivery':             int(saw_delivery),
            'saw_account_upgrade':      int(saw_account_upgrade),
            'saw_homepage':             int(saw_homepage),
            'device_mobile':            device_mobile,
            'device_computer':          device_computer,
            'device_tablet':            device_tablet,
            'returning_user':           int(returning_user),
            'loc_uk':                   int(loc_uk),
        }

        input_df = pd.DataFrame([input_data])
        input_eng = add_engineered_features(input_df)

        model    = artifacts['model']
        feat_col = artifacts['feature_cols']

        proba     = model.predict_proba(input_eng[feat_col])[0][1]
        pred      = int(proba >= 0.5)
        segment   = get_segment(proba)
        seg_color = get_segment_color(segment)

        st.markdown("---")
        st.markdown("### 🎯 Hasil Prediksi")

        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            if pred == 1:
                st.markdown(f"""
                <div class="result-box-buy">
                    <div class="result-title">✅ AKAN MEMBELI</div>
                    <div class="result-prob">{proba*100:.1f}%</div>
                    <div>Probabilitas Pembelian</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box-nobuy">
                    <div class="result-title">❌ TIDAK MEMBELI</div>
                    <div class="result-prob">{proba*100:.1f}%</div>
                    <div>Probabilitas Pembelian</div>
                </div>""", unsafe_allow_html=True)

        with col_res2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Propensity Segment</div>
                <div class="metric-value" style="color:white">{segment}</div>
                <div style="margin-top:8px">
                    <span class="seg-badge" style="background:rgba(255,255,255,0.2)">{segment} Propensity</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with col_res3:
            actions_taken = sum([
                int(basket_icon_click), int(basket_add_list), int(basket_add_detail),
                int(saw_checkout), int(sign_in), int(checked_delivery),
                int(detail_wishlist_add), int(sort_by), int(promo_banner_click)
            ])
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%)">
                <div class="metric-label">Total Aksi Dilakukan</div>
                <div class="metric-value">{actions_taken}</div>
                <div>dari 9 aksi utama</div>
            </div>""", unsafe_allow_html=True)

        # Gauge chart
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        theta = np.linspace(np.pi, 0, 200)
        for t_start, t_end, color in [
            (np.pi, np.pi*0.75, '#3498DB'),
            (np.pi*0.75, np.pi*0.5, '#F39C12'),
            (np.pi*0.5, np.pi*0.25, '#E67E22'),
            (np.pi*0.25, 0, '#E74C3C')
        ]:
            t = np.linspace(t_start, t_end, 50)
            ax.fill_between(np.cos(t), np.sin(t)*0.6, np.sin(t), alpha=0.85, color=color)
        needle_angle = np.pi * (1 - proba)
        ax.annotate('', xy=(np.cos(needle_angle)*0.75, np.sin(needle_angle)*0.75),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='white', lw=3))
        ax.text(0, -0.2, f'{proba*100:.1f}%', ha='center', va='center',
                fontsize=22, fontweight='bold', color='white')
        ax.text(0, -0.45, 'Probabilitas Pembelian', ha='center', color='#BDC3C7', fontsize=9)
        for label, pos in [('Low\n0-20%', -1.05), ('Medium\n20-50%', -0.35),
                            ('High\n50-75%', 0.35), ('Very High\n75-100%', 1.0)]:
            ax.text(pos, -0.15, label, ha='center', color='#ECF0F1', fontsize=7)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.6, 1.1)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title('Purchase Propensity Gauge', color='white', fontsize=12, fontweight='bold', pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Feature summary
        with st.expander("🔎 Detail Fitur Input"):
            eng_vals = input_eng[['purchase_intent_score','engagement_score','research_score','total_interactions']].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Purchase Intent", int(eng_vals['purchase_intent_score']))
            c2.metric("Engagement Score", int(eng_vals['engagement_score']))
            c3.metric("Research Score", int(eng_vals['research_score']))
            c4.metric("Total Interactions", int(eng_vals['total_interactions']))
            st.dataframe(input_df, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 2: Batch Prediction
# ═══════════════════════════════════════════
with tab2:
    st.markdown("### 📂 Prediksi Batch — Upload CSV")
    st.markdown("Upload file CSV yang berisi data customer. Kolom harus sama dengan dataset training (tanpa kolom `ordered`).")

    uploaded = st.file_uploader("Upload CSV", type=['csv'], key="batch_upload")

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success(f"✅ File berhasil diupload: **{uploaded.name}** ({len(batch_df):,} baris)")

            feature_base = [
                'basket_icon_click','basket_add_list','basket_add_detail','sort_by',
                'image_picker','account_page_click','promo_banner_click','detail_wishlist_add',
                'list_size_dropdown','closed_minibasket_click','checked_delivery_detail',
                'checked_returns_detail','sign_in','saw_checkout','saw_sizecharts',
                'saw_delivery','saw_account_upgrade','saw_homepage',
                'device_mobile','device_computer','device_tablet','returning_user','loc_uk'
            ]

            missing_cols = [c for c in feature_base if c not in batch_df.columns]
            if missing_cols:
                st.error(f"❌ Kolom tidak lengkap. Kolom yang kurang: `{missing_cols}`")
            else:
                with st.spinner("🔄 Memproses prediksi..."):
                    X_batch     = batch_df[feature_base]
                    X_batch_eng = add_engineered_features(X_batch)
                    feat_col    = artifacts['feature_cols']
                    model       = artifacts['model']

                    probas = model.predict_proba(X_batch_eng[feat_col])[:, 1]
                    preds  = (probas >= 0.5).astype(int)
                    segs   = pd.cut(probas, bins=[0,0.2,0.5,0.75,1.0],
                                    labels=['Low','Medium','High','Very High'])

                    result_batch = batch_df.copy()
                    if 'UserID' not in result_batch.columns:
                        result_batch.insert(0, 'UserID', [f'USER_{i+1:06d}' for i in range(len(result_batch))])
                    result_batch['predicted_ordered']    = preds
                    result_batch['purchase_probability'] = probas.round(4)
                    result_batch['propensity_segment']   = segs

                st.success(f"✅ Prediksi selesai untuk {len(result_batch):,} customer!")

                # Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Customer",    f"{len(result_batch):,}")
                c2.metric("Prediksi Akan Beli", f"{preds.sum():,}")
                c3.metric("Conversion Rate",   f"{preds.mean()*100:.2f}%")
                c4.metric("Avg Probability",   f"{probas.mean()*100:.2f}%")

                # Segment distribution chart
                seg_dist = pd.Series(segs).value_counts().reindex(['Low','Medium','High','Very High'])
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                seg_colors = ['#3498DB','#F39C12','#E67E22','#E74C3C']
                bars = axes[0].bar(seg_dist.index, seg_dist.values, color=seg_colors, edgecolor='white', width=0.6)
                axes[0].set_title('Propensity Segment Distribution', fontsize=12, fontweight='bold')
                axes[0].set_ylabel('Jumlah Customer')
                axes[0].spines[['top','right']].set_visible(False)
                for bar, v in zip(bars, seg_dist.values):
                    if not np.isnan(v):
                        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(seg_dist.dropna())*0.01,
                                     f'{int(v):,}', ha='center', fontweight='bold')
                axes[1].hist(probas, bins=50, color='#E74C3C', edgecolor='white', alpha=0.8)
                axes[1].set_title('Distribusi Probabilitas Pembelian', fontsize=12, fontweight='bold')
                axes[1].set_xlabel('Probabilitas'); axes[1].set_ylabel('Frekuensi')
                axes[1].spines[['top','right']].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Preview & Download
                st.markdown("#### 📋 Preview Hasil (10 baris pertama)")
                display_cols = ['UserID','predicted_ordered','purchase_probability','propensity_segment']
                st.dataframe(result_batch[display_cols].head(10), use_container_width=True)

                csv_out = result_batch.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Hasil Prediksi (CSV)",
                    data=csv_out,
                    file_name="hasil_prediksi_batch.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("👆 Upload file CSV untuk memulai prediksi batch.")
        with st.expander("📌 Format CSV yang dibutuhkan"):
            sample_cols = [
                'basket_icon_click','basket_add_list','basket_add_detail','sort_by',
                'image_picker','account_page_click','promo_banner_click','detail_wishlist_add',
                'list_size_dropdown','closed_minibasket_click','checked_delivery_detail',
                'checked_returns_detail','sign_in','saw_checkout','saw_sizecharts',
                'saw_delivery','saw_account_upgrade','saw_homepage',
                'device_mobile','device_computer','device_tablet','returning_user','loc_uk'
            ]
            sample_df = pd.DataFrame([[0]*len(sample_cols), [1]*len(sample_cols)], columns=sample_cols)
            st.dataframe(sample_df, use_container_width=True)
            st.caption("Semua nilai harus binary (0 atau 1). Boleh ada kolom `UserID` sebagai identifikasi.")


# ═══════════════════════════════════════════
# TAB 3: Model Performance
# ═══════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Model Performance Dashboard")

    metrics = artifacts.get('metrics', {})
    model_name = artifacts.get('model_name', 'XGBoost')

    # Metric cards
    st.markdown(f"#### 🏆 Best Model: **{model_name}**")
    c1, c2, c3, c4, c5 = st.columns(5)
    metric_display = [
        ("ROC-AUC",       "ROC-AUC",       "🎯"),
        ("Avg Precision", "Avg Precision",  "📈"),
        ("F1-Score",      "F1-Score",       "⚖️"),
        ("Precision",     "Precision",      "🎪"),
        ("Recall",        "Recall",         "🔍"),
    ]
    for col, (key, label, icon) in zip([c1,c2,c3,c4,c5], metric_display):
        val = metrics.get(key, 0)
        col.metric(f"{icon} {label}", f"{val:.4f}")

    st.markdown("---")

    # Load & display all saved plots
    plot_files = [
        ('plot_target_distribution.png', 'Distribusi Target Variable'),
        ('plot_feature_correlation.png', 'Korelasi Fitur vs Target'),
        ('plot_conversion_rate.png',     'Conversion Rate per Aksi'),
        ('plot_group_analysis.png',      'Analisis Group (Device, Lokasi, User Type)'),
        ('plot_smote_comparison.png',    'Sebelum vs Sesudah SMOTE'),
        ('plot_model_comparison.png',    'Perbandingan Model & ROC Curve'),
        ('plot_precision_recall.png',    'Precision-Recall Curve'),
        ('plot_confusion_matrices.png',  'Confusion Matrix Semua Model'),
        ('plot_feature_importance.png',  'Feature Importance XGBoost & RF'),
        ('plot_heatmap_correlation.png', 'Heatmap Korelasi Antar Fitur'),
        ('plot_prediction_results.png',  'Distribusi Hasil Prediksi Testing'),
    ]

    for fpath, title in plot_files:
        if os.path.exists(fpath):
            st.markdown(f"#### 📊 {title}")
            st.image(fpath, use_container_width=True)
            st.markdown("---")
        else:
            st.warning(f"⚠️ Plot tidak ditemukan: {fpath}")

    # Feature info
    with st.expander("🔧 Fitur yang Digunakan Model"):
        feat_col = artifacts.get('feature_cols', [])
        original = [f for f in feat_col if f not in
                    ['purchase_intent_score','engagement_score','research_score','total_interactions']]
        engineered = [f for f in feat_col if f in
                      ['purchase_intent_score','engagement_score','research_score','total_interactions']]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original Features (23)**")
            for f in original: st.markdown(f"• `{f}`")
        with c2:
            st.markdown("**Engineered Features (4)**")
            for f in engineered: st.markdown(f"• `{f}`")
