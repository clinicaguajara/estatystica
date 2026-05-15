import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import signal

from utils.dataframe_state import select_active_dataframe
from utils.design import load_css


INDEX_TIME_OPTION = "Indice das linhas"


def infer_time_column(numeric_columns: list[str]) -> str:
    preferred_terms = ("time", "tempo", "timestamp")
    for column in numeric_columns:
        column_lower = str(column).lower()
        if any(term in column_lower for term in preferred_terms):
            return column
    return INDEX_TIME_OPTION


def prepare_signal_frame(
    df: pd.DataFrame,
    time_column: str,
    channel_columns: list[str],
    *,
    start_value: float,
    end_value: float,
    max_points: int | None,
) -> pd.DataFrame:
    if time_column == INDEX_TIME_OPTION:
        work_df = df[channel_columns].copy()
        work_df.insert(0, INDEX_TIME_OPTION, np.arange(len(work_df), dtype=float))
        time_column = INDEX_TIME_OPTION
    else:
        work_df = df[[time_column, *channel_columns]].copy()

    work_df[time_column] = pd.to_numeric(work_df[time_column], errors="coerce")
    for channel in channel_columns:
        work_df[channel] = pd.to_numeric(work_df[channel], errors="coerce")

    work_df = work_df.dropna(subset=[time_column])
    work_df = work_df[
        (work_df[time_column] >= start_value)
        & (work_df[time_column] <= end_value)
    ]

    if max_points is not None and len(work_df) > max_points:
        step = int(np.ceil(len(work_df) / max_points))
        work_df = work_df.iloc[::step, :]

    return work_df


def plot_signals(
    signal_df: pd.DataFrame,
    time_column: str,
    channel_columns: list[str],
    *,
    stacked: bool,
) -> go.Figure:
    fig = go.Figure()
    x_values = signal_df[time_column]

    if stacked:
        channel_ranges = [
            signal_df[channel].max() - signal_df[channel].min()
            for channel in channel_columns
        ]
        offset_step = max([value for value in channel_ranges if value > 0] or [1.0])
    else:
        offset_step = 0.0

    for index, channel in enumerate(channel_columns):
        y_values = signal_df[channel]
        if stacked:
            y_values = y_values + (index * offset_step)

        fig.add_trace(
            go.Scattergl(
                x=x_values,
                y=y_values,
                mode="lines",
                name=channel,
                line={"width": 1},
            )
        )

    fig.update_layout(
        height=560,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis_title=time_column,
        yaxis_title="Amplitude" if not stacked else "Amplitude + deslocamento",
    )
    fig.update_xaxes(rangeslider={"visible": True})
    return fig


def infer_sampling_rate(time_values: pd.Series) -> float | None:
    values = pd.to_numeric(time_values, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size < 2:
        return None

    diffs = np.diff(np.sort(values))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None

    median_step = float(np.median(diffs))
    if median_step <= 0:
        return None
    return 1.0 / median_step


def downsample_values(values: np.ndarray, max_samples: int) -> tuple[np.ndarray, int]:
    if values.size <= max_samples:
        return values, 1
    step = int(np.ceil(values.size / max_samples))
    return values[::step], step


def plot_power_spectrum(frequencies: np.ndarray, power: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=power,
            mode="lines",
            name="PSD",
            line={"width": 1.5},
        )
    )
    fig.update_layout(
        height=380,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        hovermode="x unified",
        xaxis_title="Frequencia (Hz)",
        yaxis_title="Densidade de potencia",
    )
    return fig


def plot_spectrogram(
    frequencies: np.ndarray,
    times: np.ndarray,
    spectrogram_power: np.ndarray,
) -> go.Figure:
    power_db = 10 * np.log10(np.maximum(spectrogram_power, np.finfo(float).tiny))
    fig = go.Figure(
        data=go.Heatmap(
            x=times,
            y=frequencies,
            z=power_db,
            colorscale="Viridis",
            colorbar={"title": "dB"},
        )
    )
    fig.update_layout(
        height=480,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis_title="Tempo (s)",
        yaxis_title="Frequencia (Hz)",
    )
    return fig


def compute_band_power(
    frequencies: np.ndarray,
    power: np.ndarray,
    bands: list[tuple[str, float, float]],
) -> pd.DataFrame:
    total_power = float(np.trapz(power, frequencies))
    rows = []
    for name, low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        if mask.sum() < 2:
            band_power = 0.0
        else:
            band_power = float(np.trapz(power[mask], frequencies[mask]))

        rows.append(
            {
                "Banda": name,
                "Intervalo (Hz)": f"{low:g}-{high:g}",
                "Potencia": band_power,
                "Potencia relativa": band_power / total_power if total_power > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


load_css()

st.title("Sinais Temporais")
st.caption("Visualizacao de sinais multicanais no tempo.")

selected_df_name, df = select_active_dataframe(
    state_key="selected_df_name",
    label="Selecione o dataframe para analise:",
    widget_key="signals_selected_df",
)

if not isinstance(df, pd.DataFrame):
    st.error("O objeto selecionado nao e um DataFrame valido.")
    st.stop()

numeric_columns = df.select_dtypes(include="number").columns.tolist()
if not numeric_columns:
    st.warning("Este dataframe nao possui colunas numericas para visualizacao.")
    st.stop()

default_time_column = infer_time_column(numeric_columns)
time_options = [INDEX_TIME_OPTION, *numeric_columns]
default_time_index = time_options.index(default_time_column)

st.write(f"**Dataframe:** {selected_df_name} | **Dimensoes:** {df.shape[0]} x {df.shape[1]}")

time_column = st.selectbox(
    "Eixo do tempo:",
    time_options,
    index=default_time_index,
    key="signals_time_column",
)

channel_options = [
    column
    for column in numeric_columns
    if column != time_column
]

if not channel_options:
    st.warning("Selecione um dataframe com pelo menos uma coluna numerica alem do tempo.")
    st.stop()

default_channels = channel_options[: min(4, len(channel_options))]
channel_columns = st.multiselect(
    "Canais:",
    channel_options,
    default=default_channels,
    key="signals_channels",
)

if not channel_columns:
    st.info("Selecione pelo menos um canal para visualizar.")
    st.stop()

if time_column == INDEX_TIME_OPTION:
    time_values = pd.Series(np.arange(len(df), dtype=float), name=INDEX_TIME_OPTION)
else:
    time_values = pd.to_numeric(df[time_column], errors="coerce").dropna()

if time_values.empty:
    st.warning("O eixo de tempo selecionado nao possui valores numericos validos.")
    st.stop()

min_time = float(time_values.min())
max_time = float(time_values.max())

col_start, col_end, col_points = st.columns([1, 1, 1])
with col_start:
    start_value = st.number_input(
        "Inicio:",
        value=min_time,
        min_value=min_time,
        max_value=max_time,
        key="signals_start_value",
    )
with col_end:
    end_value = st.number_input(
        "Fim:",
        value=max_time,
        min_value=min_time,
        max_value=max_time,
        key="signals_end_value",
    )
with col_points:
    max_points = st.number_input(
        "Max. pontos:",
        min_value=500,
        max_value=50000,
        value=10000,
        step=500,
        format="%d",
        key="signals_max_points",
    )

if start_value >= end_value:
    st.warning("O inicio precisa ser menor que o fim.")
    st.stop()

stacked = st.toggle(
    "Empilhar canais",
    value=True,
    key="signals_stacked_channels",
)

signal_df = prepare_signal_frame(
    df,
    time_column,
    channel_columns,
    start_value=float(start_value),
    end_value=float(end_value),
    max_points=int(max_points),
)

if signal_df.empty:
    st.warning("Nenhum ponto encontrado no intervalo selecionado.")
    st.stop()

plot_time_column = time_column if time_column != INDEX_TIME_OPTION else INDEX_TIME_OPTION
fig = plot_signals(
    signal_df,
    plot_time_column,
    channel_columns,
    stacked=stacked,
)

st.plotly_chart(fig, use_container_width=True)
st.caption(f"{len(signal_df):,} pontos renderizados por canal.".replace(",", "."))

st.divider()
st.write("### Analise de potencia e espectrograma")

spectral_channel = st.selectbox(
    "Canal para analise espectral:",
    channel_columns,
    key="signals_spectral_channel",
)

inferred_fs = infer_sampling_rate(time_values) if time_column != INDEX_TIME_OPTION else None
default_fs = float(inferred_fs) if inferred_fs else 1000.0

spectral_col_fs, spectral_col_limit = st.columns([1, 1])
with spectral_col_fs:
    sampling_rate = st.number_input(
        "Taxa de amostragem (Hz):",
        min_value=0.001,
        value=default_fs,
        step=1.0,
        format="%.3f",
        key="signals_sampling_rate",
    )
with spectral_col_limit:
    max_spectral_samples = st.number_input(
        "Max. amostras espectrais:",
        min_value=1000,
        max_value=500000,
        value=200000,
        step=1000,
        format="%d",
        key="signals_max_spectral_samples",
    )

spectral_df = prepare_signal_frame(
    df,
    time_column,
    [spectral_channel],
    start_value=float(start_value),
    end_value=float(end_value),
    max_points=None,
).dropna(subset=[spectral_channel])

if spectral_df.empty:
    st.warning("Nenhum valor valido encontrado para a analise espectral.")
    st.stop()

raw_values = spectral_df[spectral_channel].to_numpy(dtype=float)
raw_values = raw_values[np.isfinite(raw_values)]
if raw_values.size < 8:
    st.warning("A analise espectral precisa de pelo menos 8 amostras validas.")
    st.stop()

signal_values, downsample_step = downsample_values(
    raw_values,
    int(max_spectral_samples),
)
effective_fs = float(sampling_rate) / downsample_step
nyquist = effective_fs / 2.0

if downsample_step > 1:
    st.caption(
        f"Sinal reamostrado por passo {downsample_step} para calculo espectral; "
        f"taxa efetiva: {effective_fs:.3f} Hz."
    )

detrended_values = signal.detrend(signal_values)
default_nperseg = min(2048, detrended_values.size)
min_nperseg = min(64, detrended_values.size)
max_nperseg = max(min_nperseg, detrended_values.size)

nperseg = st.slider(
    "Tamanho da janela espectral:",
    min_value=int(min_nperseg),
    max_value=int(max_nperseg),
    value=int(default_nperseg),
    step=1,
    key="signals_spectral_nperseg",
)

frequencies, power = signal.welch(
    detrended_values,
    fs=effective_fs,
    nperseg=int(nperseg),
    noverlap=int(nperseg // 2),
    scaling="density",
)

max_frequency = st.slider(
    "Frequencia maxima exibida (Hz):",
    min_value=0.0,
    max_value=float(nyquist),
    value=float(min(150.0, nyquist)),
    step=max(float(nyquist) / 500.0, 0.1),
    key="signals_max_frequency",
)

freq_mask = frequencies <= max_frequency
st.plotly_chart(
    plot_power_spectrum(frequencies[freq_mask], power[freq_mask]),
    use_container_width=True,
)

bands = [
    ("Delta", 0.5, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta", 13.0, 30.0),
    ("Gamma baixa", 30.0, 80.0),
    ("Gamma alta", 80.0, 150.0),
]
valid_bands = [
    (name, low, min(high, nyquist))
    for name, low, high in bands
    if low < nyquist
]
band_power_df = compute_band_power(frequencies, power, valid_bands)

st.write("#### Potencia por banda")
st.dataframe(
    band_power_df.style.format(
        {
            "Potencia": "{:.6g}",
            "Potencia relativa": "{:.2%}",
        }
    ),
    use_container_width=True,
)

spectrogram_frequencies, spectrogram_times, spectrogram_power = signal.spectrogram(
    detrended_values,
    fs=effective_fs,
    nperseg=int(nperseg),
    noverlap=int(nperseg // 2),
    scaling="density",
    mode="psd",
)

spectrogram_mask = spectrogram_frequencies <= max_frequency
time_offset = float(start_value) if time_column != INDEX_TIME_OPTION else 0.0
st.write("#### Espectrograma")
st.plotly_chart(
    plot_spectrogram(
        spectrogram_frequencies[spectrogram_mask],
        spectrogram_times + time_offset,
        spectrogram_power[spectrogram_mask, :],
    ),
    use_container_width=True,
)
st.caption(
    "Analise espectral calculada com scipy.signal.welch e scipy.signal.spectrogram."
)
