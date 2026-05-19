from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.design import load_css


st.set_page_config(
    page_title="Cerebro 3D",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_css()


Hemisphere = Literal["left", "right"]
MeshKind = Literal["cortex", "ellipsoid", "cylinder", "tube"]


@dataclass(frozen=True)
class BrainRegion:
    name: str
    category: str
    color: str
    function: str
    note: str
    kind: MeshKind
    hemispheres: tuple[Hemisphere, ...] = ("left", "right")
    theta_range: tuple[float, float] | None = None
    phi_range: tuple[float, float] | None = None
    center: tuple[float, float, float] | None = None
    radii: tuple[float, float, float] | None = None
    z_range: tuple[float, float] | None = None
    pair_offset: float | None = None
    curve: str | None = None
    tube_radius: float = 0.06


REGIONS: tuple[BrainRegion, ...] = (
    BrainRegion(
        name="Lobo frontal",
        category="Lobos e estruturas",
        color="#E15759",
        function="Planejamento, tomada de decisão, controle motor voluntário e linguagem expressiva.",
        note="Fica na porção anterior do cérebro.",
        kind="cortex",
        theta_range=(0.00, 1.02),
        phi_range=(-0.48, 1.02),
    ),
    BrainRegion(
        name="Lobo parietal",
        category="Lobos e estruturas",
        color="#4E79A7",
        function="Integração sensorial, orientação espacial e percepção corporal.",
        note="Ocupa a porção superior e posterior ao sulco central.",
        kind="cortex",
        theta_range=(1.02, 2.15),
        phi_range=(0.00, 1.03),
    ),
    BrainRegion(
        name="Lobo temporal",
        category="Lobos e estruturas",
        color="#F28E2B",
        function="Audição, memória, reconhecimento de objetos e aspectos da linguagem.",
        note="Fica nas laterais inferiores dos hemisférios.",
        kind="cortex",
        theta_range=(0.66, 2.55),
        phi_range=(-0.88, 0.02),
    ),
    BrainRegion(
        name="Lobo occipital",
        category="Lobos e estruturas",
        color="#59A14F",
        function="Processamento visual inicial e integração de informações visuais.",
        note="Fica na parte posterior do cérebro.",
        kind="cortex",
        theta_range=(2.08, math.pi),
        phi_range=(-0.35, 0.88),
    ),
    BrainRegion(
        name="Cerebelo",
        category="Lobos e estruturas",
        color="#B07AA1",
        function="Coordenação motora, equilíbrio, precisão de movimentos e aprendizagem motora.",
        note="Estrutura posterior e inferior, atrás do tronco encefálico.",
        kind="ellipsoid",
        hemispheres=("left", "right"),
        center=(0.0, -1.18, -0.67),
        radii=(0.92, 0.46, 0.34),
    ),
    BrainRegion(
        name="Tronco encefálico",
        category="Lobos e estruturas",
        color="#9C755F",
        function="Regulação de funções vitais, vias motoras e sensoriais, sono e vigília.",
        note="Conecta encéfalo e medula espinal.",
        kind="cylinder",
        hemispheres=("left", "right"),
        center=(0.0, -0.34, -1.04),
        radii=(0.18, 0.22, 0.0),
        z_range=(-1.52, -0.55),
    ),
    BrainRegion(
        name="Núcleos da rafe",
        category="Lobos e estruturas",
        color="#E6C229",
        function="Conjunto de núcleos serotoninérgicos envolvidos em modulação de humor, sono, dor e estado de alerta.",
        note="Representados de forma esquemática como uma coluna na linha média do tronco encefálico.",
        kind="cylinder",
        hemispheres=("left", "right"),
        center=(0.0, -0.34, -0.98),
        radii=(0.055, 0.07, 0.0),
        z_range=(-1.43, -0.44),
    ),
    BrainRegion(
        name="Autorreceptores 5-HT1A (núcleos da rafe)",
        category="Slide 1 - Monoaminérgica",
        color="#E6C229",
        function="Autorreceptores somatodendríticos nos núcleos da rafe diminuem as taxas de disparo por feedback negativo.",
        note="A serotonina é sintetizada por neurônios do núcleo da rafe, no mesencéfalo, a partir de L-triptofano.",
        kind="cylinder",
        hemispheres=("left", "right"),
        center=(0.0, -0.34, -0.98),
        radii=(0.07, 0.09, 0.0),
        z_range=(-1.43, -0.44),
    ),
    BrainRegion(
        name="Hipocampo",
        category="Áreas límbicas",
        color="#76B7B2",
        function="Memória episódica, navegação espacial, contexto emocional e aprendizagem.",
        note="Representação bilateral esquemática em arco na região temporal medial.",
        kind="tube",
        curve="hippocampus",
        tube_radius=0.075,
    ),
    BrainRegion(
        name="Amígdala",
        category="Áreas límbicas",
        color="#E15759",
        function="Saliência emocional, medo, ameaça, valência afetiva e aprendizagem emocional.",
        note="Representação bilateral simplificada anterior ao hipocampo.",
        kind="ellipsoid",
        center=(0.0, 0.34, -0.34),
        radii=(0.16, 0.18, 0.14),
        pair_offset=0.50,
    ),
    BrainRegion(
        name="Hipotálamo",
        category="Áreas límbicas",
        color="#9C755F",
        function="Integra respostas autonômicas, endócrinas, motivacionais e homeostáticas.",
        note="Representado próximo à linha média, abaixo do tálamo.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.05, -0.56),
        radii=(0.23, 0.18, 0.13),
    ),
    BrainRegion(
        name="Septo lateral",
        category="Áreas límbicas",
        color="#59A14F",
        function="Modulação de recompensa, ansiedade, ritmos hipocampais e comportamento social.",
        note="Representação esquemática próxima à linha média anterior.",
        kind="ellipsoid",
        center=(0.0, 0.35, -0.16),
        radii=(0.08, 0.13, 0.10),
        pair_offset=0.16,
    ),
    BrainRegion(
        name="Núcleo accumbens",
        category="Áreas límbicas",
        color="#F28E2B",
        function="Componente límbico-estriatal associado a motivação, recompensa e integração afetiva.",
        note="Representação bilateral simplificada na região ventral anterior.",
        kind="ellipsoid",
        center=(0.0, 0.52, -0.36),
        radii=(0.12, 0.14, 0.10),
        pair_offset=0.30,
    ),
    BrainRegion(
        name="Tálamo anterior",
        category="Áreas límbicas",
        color="#4E79A7",
        function="Componente de circuitos límbicos relacionados a memória e integração com o cíngulo.",
        note="Representação bilateral simplificada próxima à linha média.",
        kind="ellipsoid",
        center=(0.0, -0.02, -0.12),
        radii=(0.15, 0.18, 0.12),
        pair_offset=0.22,
    ),
    BrainRegion(
        name="Corpos mamilares",
        category="Áreas límbicas",
        color="#FF9DA7",
        function="Estruturas hipotalâmicas associadas a circuitos de memória.",
        note="Representação esquemática inferior e medial.",
        kind="ellipsoid",
        center=(0.0, 0.08, -0.72),
        radii=(0.07, 0.08, 0.06),
        pair_offset=0.10,
    ),
    BrainRegion(
        name="Fórnix",
        category="Áreas límbicas",
        color="#BAB0AC",
        function="Feixe de conexão entre hipocampo, corpos mamilares e circuitos límbicos.",
        note="Representado como feixe curvo que sai do hipocampo em direção à linha média.",
        kind="tube",
        curve="fornix",
        tube_radius=0.035,
    ),
    BrainRegion(
        name="Córtex cingulado",
        category="Áreas límbicas",
        color="#EDC948",
        function="Integra emoção, atenção, controle cognitivo, dor e monitoramento de conflito.",
        note="Representado como arco medial superior acima de estruturas límbicas profundas.",
        kind="tube",
        curve="cingulate",
        tube_radius=0.052,
    ),
    BrainRegion(
        name="Córtex parahipocampal",
        category="Áreas límbicas",
        color="#86BCB6",
        function="Região límbica cortical próxima ao hipocampo, associada a memória contextual.",
        note="Representação curva abaixo e medial ao hipocampo.",
        kind="tube",
        curve="parahippocampal",
        tube_radius=0.055,
    ),
    BrainRegion(
        name="Córtex entorrinal",
        category="Áreas límbicas",
        color="#B07AA1",
        function="Interface entre neocórtex e hipocampo, importante para memória e navegação.",
        note="Representação curta na região temporal medial anterior.",
        kind="tube",
        curve="entorhinal",
        tube_radius=0.045,
    ),
    BrainRegion(
        name="Área tegmental ventral (VTA)",
        category="Slide 2 - Dopaminérgica",
        color="#E6C229",
        function="Origem esquemática da via dopaminérgica mesolímbica.",
        note="Representação didática no mesencéfalo.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, -0.48, -0.72),
        radii=(0.16, 0.12, 0.10),
    ),
    BrainRegion(
        name="Núcleo accumbens (NAc)",
        category="Slide 2 - Dopaminérgica",
        color="#F28E2B",
        function="Alvo mesolímbico relacionado a depressão, ansiedade, recompensa e BDNF.",
        note="Representação bilateral simplificada.",
        kind="ellipsoid",
        center=(0.0, 0.52, -0.36),
        radii=(0.13, 0.15, 0.10),
        pair_offset=0.30,
    ),
    BrainRegion(
        name="Via VTA-NAc",
        category="Slide 2 - Dopaminérgica",
        color="#E6C229",
        function="Projeção dopaminérgica mesolímbica esquemática.",
        note="Conexão didática entre VTA e núcleo accumbens.",
        kind="tube",
        curve="vta_accumbens",
        tube_radius=0.028,
    ),
    BrainRegion(
        name="Via VTA-amígdala",
        category="Slide 2 - Dopaminérgica",
        color="#E6C229",
        function="Projeção dopaminérgica mesolímbica esquemática.",
        note="Conexão didática entre VTA e amígdala.",
        kind="tube",
        curve="vta_amygdala",
        tube_radius=0.026,
    ),
    BrainRegion(
        name="Via VTA-hipocampo",
        category="Slide 2 - Dopaminérgica",
        color="#E6C229",
        function="Projeção dopaminérgica mesolímbica esquemática.",
        note="Conexão didática entre VTA e hipocampo.",
        kind="tube",
        curve="vta_hippocampus",
        tube_radius=0.026,
    ),
    BrainRegion(
        name="Via VTA-PFC",
        category="Slide 2 - Dopaminérgica",
        color="#E6C229",
        function="Projeção dopaminérgica mesolímbica esquemática.",
        note="Conexão didática entre VTA e córtex pré-frontal.",
        kind="tube",
        curve="vta_pfc",
        tube_radius=0.026,
    ),
    BrainRegion(
        name="D1-like na amígdala",
        category="Slide 2 - Dopaminérgica",
        color="#E15759",
        function="Representação do papel de receptores D1-like na amígdala.",
        note="Sobreposição didática na amígdala.",
        kind="ellipsoid",
        center=(0.0, 0.34, -0.34),
        radii=(0.18, 0.20, 0.15),
        pair_offset=0.50,
    ),
    BrainRegion(
        name="D2-like PFC-amígdala",
        category="Slide 2 - Dopaminérgica",
        color="#4E79A7",
        function="Representação de terminais pré-sinápticos do PFC e efeito GABAérgico na amígdala.",
        note="Conexão didática entre PFC e amígdala.",
        kind="tube",
        curve="pfc_amygdala",
        tube_radius=0.032,
    ),
    BrainRegion(
        name="Complexo D1-D2 no núcleo accumbens",
        category="Slide 2 - Dopaminérgica",
        color="#B07AA1",
        function="Representação de complexos heteroméricos D1-D2 no núcleo accumbens.",
        note="Sobreposição didática no núcleo accumbens.",
        kind="ellipsoid",
        center=(0.0, 0.52, -0.36),
        radii=(0.17, 0.19, 0.13),
        pair_offset=0.30,
    ),
    BrainRegion(
        name="Complexo D1-D2 no caudado/estriado",
        category="Slide 2 - Dopaminérgica",
        color="#B07AA1",
        function="Representação de maior densidade de D1-D2 em caudado/estriado descrita no artigo.",
        note="Representação bilateral simplificada.",
        kind="ellipsoid",
        center=(0.0, 0.02, -0.10),
        radii=(0.18, 0.32, 0.13),
        pair_offset=0.36,
    ),
    BrainRegion(
        name="BDNF/TrkB no núcleo accumbens",
        category="Slide 2 - Dopaminérgica",
        color="#D62728",
        function="Representação do efeito depressor em regiões mesolímbicas.",
        note="Sobreposição didática no núcleo accumbens.",
        kind="ellipsoid",
        center=(0.0, 0.52, -0.36),
        radii=(0.19, 0.21, 0.15),
        pair_offset=0.30,
    ),
    BrainRegion(
        name="BDNF/TrkB no hipocampo",
        category="Slide 2 - Dopaminérgica",
        color="#59A14F",
        function="Representação do efeito antidepressivo no hipocampo descrito no artigo.",
        note="Sobreposição didática no hipocampo.",
        kind="tube",
        curve="hippocampus",
        tube_radius=0.052,
    ),
    BrainRegion(
        name="Córtex pré-frontal (PFC)",
        category="Slide 2 - Dopaminérgica",
        color="#4E79A7",
        function="Região citada no artigo em terminais pré-sinápticos e BDNF/TrkB.",
        note="Representação cortical anterior.",
        kind="cortex",
        theta_range=(0.10, 0.72),
        phi_range=(0.00, 0.78),
    ),
    BrainRegion(
        name="BDNF/TrkB no córtex pré-frontal",
        category="Slide 2 - Dopaminérgica",
        color="#59A14F",
        function="Representação do efeito antidepressivo no PFC descrito no artigo.",
        note="Sobreposição didática no córtex pré-frontal.",
        kind="cortex",
        theta_range=(0.12, 0.70),
        phi_range=(0.02, 0.76),
    ),
    BrainRegion(
        name="Neurônios adrenérgicos centrais",
        category="Slide 3 - Noradrenérgica",
        color="#E6C229",
        function="Nó esquemático do sistema noradrenérgico central.",
        note="Representação didática, sem detalhar subnúcleos anatômicos.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, -0.56, -0.60),
        radii=(0.15, 0.12, 0.10),
    ),
    BrainRegion(
        name="Via noradrenérgica para neurônios pós-sinápticos",
        category="Slide 3 - Noradrenérgica",
        color="#E6C229",
        function="Projeção esquemática do sistema noradrenérgico central.",
        note="Conexão didática para neurônios pós-sinápticos.",
        kind="tube",
        curve="na_postsynaptic",
        tube_radius=0.026,
    ),
    BrainRegion(
        name="α2-autorreceptores pré-sinápticos",
        category="Slide 3 - Noradrenérgica",
        color="#F28E2B",
        function="Autorreceptores α2 em membranas pré-sinápticas de neurônios adrenérgicos.",
        note="Representação sobre o nó noradrenérgico central.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, -0.56, -0.60),
        radii=(0.21, 0.17, 0.13),
    ),
    BrainRegion(
        name="α2-heterorreceptores dopaminérgicos",
        category="Slide 3 - Noradrenérgica",
        color="#4E79A7",
        function="Heterorreceptores α2 em neurônios dopaminérgicos.",
        note="Representação didática em população dopaminérgica.",
        kind="ellipsoid",
        center=(0.0, -0.32, -0.54),
        radii=(0.12, 0.10, 0.09),
        pair_offset=0.28,
    ),
    BrainRegion(
        name="α2-heterorreceptores serotoninérgicos",
        category="Slide 3 - Noradrenérgica",
        color="#4E79A7",
        function="Heterorreceptores α2 em neurônios serotoninérgicos.",
        note="Representação didática na região serotoninérgica.",
        kind="cylinder",
        hemispheres=("left", "right"),
        center=(0.0, -0.34, -0.98),
        radii=(0.06, 0.08, 0.0),
        z_range=(-1.38, -0.52),
    ),
    BrainRegion(
        name="α2-heterorreceptores glutamatérgicos",
        category="Slide 3 - Noradrenérgica",
        color="#4E79A7",
        function="Heterorreceptores α2 em neurônios glutamatérgicos.",
        note="Representação cortical esquemática.",
        kind="cortex",
        theta_range=(0.40, 1.75),
        phi_range=(0.02, 0.80),
    ),
    BrainRegion(
        name="Via NA-dopaminérgica",
        category="Slide 3 - Noradrenérgica",
        color="#4E79A7",
        function="Conexão esquemática para heterorreceptores em neurônios dopaminérgicos.",
        note="Representação didática.",
        kind="tube",
        curve="na_dopaminergic",
        tube_radius=0.022,
    ),
    BrainRegion(
        name="Via NA-serotoninérgica",
        category="Slide 3 - Noradrenérgica",
        color="#4E79A7",
        function="Conexão esquemática para heterorreceptores em neurônios serotoninérgicos.",
        note="Representação didática.",
        kind="tube",
        curve="na_serotonergic",
        tube_radius=0.022,
    ),
    BrainRegion(
        name="Via NA-glutamatérgica",
        category="Slide 3 - Noradrenérgica",
        color="#4E79A7",
        function="Conexão esquemática para heterorreceptores em neurônios glutamatérgicos.",
        note="Representação didática.",
        kind="tube",
        curve="na_glutamatergic",
        tube_radius=0.022,
    ),
    BrainRegion(
        name="Ativação α2 pós-sináptica",
        category="Slide 3 - Noradrenérgica",
        color="#59A14F",
        function="Modulação pós-sináptica da excitabilidade neuronal por canais iônicos.",
        note="Representação cortical esquemática.",
        kind="cortex",
        theta_range=(0.70, 2.05),
        phi_range=(-0.10, 0.72),
    ),
    BrainRegion(
        name="Subtipos α2A/α2B/α2C",
        category="Slide 3 - Noradrenérgica",
        color="#B07AA1",
        function="Bloco esquemático dos subtipos α2A, α2B e α2C.",
        note="Representação conceitual, não anatômica.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.16, 0.48),
        radii=(0.30, 0.16, 0.11),
    ),
    BrainRegion(
        name="Sítio de ligação α2A/α2B",
        category="Slide 3 - Noradrenérgica",
        color="#B07AA1",
        function="Bloco esquemático do sítio de ligação descrito no texto.",
        note="Representação conceitual, não anatômica.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.50, 0.58),
        radii=(0.22, 0.12, 0.09),
    ),
    BrainRegion(
        name="Transportador SERT",
        category="Slide 4 - SERT/NET",
        color="#E15759",
        function="Transportador de serotonina.",
        note="Representação conceitual em região serotoninérgica.",
        kind="cylinder",
        hemispheres=("left", "right"),
        center=(0.0, -0.34, -0.98),
        radii=(0.07, 0.09, 0.0),
        z_range=(-1.38, -0.52),
    ),
    BrainRegion(
        name="Transportador NET",
        category="Slide 4 - SERT/NET",
        color="#4E79A7",
        function="Transportador de noradrenalina.",
        note="Representação conceitual no sistema noradrenérgico.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, -0.56, -0.60),
        radii=(0.22, 0.17, 0.13),
    ),
    BrainRegion(
        name="Gradiente Na+/Cl-",
        category="Slide 4 - SERT/NET",
        color="#E6C229",
        function="Força de transporte dependente de Na+ e Cl-.",
        note="Representação conceitual da força de recaptação.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, -0.18, 0.62),
        radii=(0.24, 0.14, 0.10),
    ),
    BrainRegion(
        name="Via 5-HT para centros emocionais e somáticos",
        category="Slide 4 - SERT/NET",
        color="#E15759",
        function="Projeções serotoninérgicas para centros emocionais e somáticos.",
        note="Representação didática.",
        kind="tube",
        curve="sert_limbic",
        tube_radius=0.024,
    ),
    BrainRegion(
        name="Via NA para córtex frontal",
        category="Slide 4 - SERT/NET",
        color="#4E79A7",
        function="Projeção noradrenérgica para córtex frontal.",
        note="Representação didática.",
        kind="tube",
        curve="net_frontal",
        tube_radius=0.024,
    ),
    BrainRegion(
        name="Via NA para amígdala",
        category="Slide 4 - SERT/NET",
        color="#4E79A7",
        function="Projeção noradrenérgica para amígdala.",
        note="Representação didática.",
        kind="tube",
        curve="net_amygdala",
        tube_radius=0.024,
    ),
    BrainRegion(
        name="Via NA para hipocampo",
        category="Slide 4 - SERT/NET",
        color="#4E79A7",
        function="Projeção noradrenérgica para hipocampo.",
        note="Representação didática.",
        kind="tube",
        curve="net_hippocampus",
        tube_radius=0.024,
    ),
    BrainRegion(
        name="Via NA para hipotálamo",
        category="Slide 4 - SERT/NET",
        color="#4E79A7",
        function="Projeção noradrenérgica para hipotálamo.",
        note="Representação didática.",
        kind="tube",
        curve="net_hypothalamus",
        tube_radius=0.024,
    ),
    BrainRegion(
        name="Neuroplasticidade hipocampal e BDNF",
        category="Slide 4 - SERT/NET",
        color="#59A14F",
        function="Representação de proliferação hipocampal e BDNF citadas no texto.",
        note="Sobreposição didática no hipocampo.",
        kind="tube",
        curve="hippocampus",
        tube_radius=0.060,
    ),
    BrainRegion(
        name="Bolsões S1/S2 de SERT/NET",
        category="Slide 4 - SERT/NET",
        color="#B07AA1",
        function="Representação conceitual dos bolsões S1/S2.",
        note="Bloco molecular conceitual, não anatômico.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.34, 0.58),
        radii=(0.30, 0.16, 0.11),
    ),
    BrainRegion(
        name="SERT - Asp-98 no Sítio A",
        category="Slide 4 - SERT/NET",
        color="#B07AA1",
        function="Representação conceitual da seletividade do transportador 5-HT.",
        note="Bloco molecular conceitual, não anatômico.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.66, 0.66),
        radii=(0.22, 0.12, 0.09),
    ),
    BrainRegion(
        name="Regiões mesocorticolímbicas NMDA",
        category="Slide 5 - Glutamatérgica",
        color="#E6C229",
        function="Regiões mesocorticolímbicas com alta densidade de receptores NMDA.",
        note="Representação cortical-límbica esquemática.",
        kind="cortex",
        theta_range=(0.24, 2.18),
        phi_range=(-0.28, 0.88),
    ),
    BrainRegion(
        name="Receptor NMDA - canal de Ca2+",
        category="Slide 5 - Glutamatérgica",
        color="#4E79A7",
        function="Receptor ionotrópico controlado por ligantes e condutor de corrente de cálcio.",
        note="Bloco molecular conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.10, 0.62),
        radii=(0.28, 0.16, 0.12),
    ),
    BrainRegion(
        name="Subunidades GluN1/GluN2/GluN3",
        category="Slide 5 - Glutamatérgica",
        color="#4E79A7",
        function="Heterotetrâmeros de subunidades NMDA.",
        note="Bloco molecular conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.48, 0.66),
        radii=(0.24, 0.13, 0.10),
    ),
    BrainRegion(
        name="Antagonistas NMDA - ketamina e análogos",
        category="Slide 5 - Glutamatérgica",
        color="#E15759",
        function="Bloqueio/antagonismo do receptor NMDA.",
        note="Bloco farmacológico conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, -0.24, 0.74),
        radii=(0.28, 0.14, 0.10),
    ),
    BrainRegion(
        name="Hipocampo ventral",
        category="Slide 5 - Glutamatérgica",
        color="#76B7B2",
        function="Região citada para bloqueio do NMDA após dissociação do glutamato.",
        note="Representação bilateral esquemática.",
        kind="tube",
        curve="hippocampus",
        tube_radius=0.062,
    ),
    BrainRegion(
        name="Córtex pré-frontal medial NMDA",
        category="Slide 5 - Glutamatérgica",
        color="#76B7B2",
        function="Região citada para interneurônios GABAérgicos e desinibição glutamatérgica.",
        note="Representação cortical anterior.",
        kind="cortex",
        theta_range=(0.10, 0.72),
        phi_range=(0.06, 0.82),
    ),
    BrainRegion(
        name="Interneurônios GABAérgicos mPFC",
        category="Slide 5 - Glutamatérgica",
        color="#F28E2B",
        function="Alvo de inibição NMDA no córtex pré-frontal medial.",
        note="Representação cortical esquemática.",
        kind="cortex",
        theta_range=(0.34, 0.78),
        phi_range=(0.16, 0.62),
    ),
    BrainRegion(
        name="Neurônios piramidais glutamatérgicos",
        category="Slide 5 - Glutamatérgica",
        color="#59A14F",
        function="Neurônios glutamatérgicos desinibidos após bloqueio de NMDA em interneurônios GABAérgicos.",
        note="Representação cortical esquemática.",
        kind="cortex",
        theta_range=(0.72, 1.18),
        phi_range=(0.16, 0.72),
    ),
    BrainRegion(
        name="Pico de disparo glutamatérgico",
        category="Slide 5 - Glutamatérgica",
        color="#59A14F",
        function="Pico de disparo glutamatérgico citado no mecanismo.",
        note="Conexão cortical esquemática.",
        kind="tube",
        curve="gaba_pyramidal",
        tube_radius=0.026,
    ),
    BrainRegion(
        name="BDNF/TrkB no córtex límbico",
        category="Slide 5 - Glutamatérgica",
        color="#B07AA1",
        function="Expressão de BDNF mRNA e receptor trkB após bloqueio de NMDA.",
        note="Representação límbico-cortical esquemática.",
        kind="cortex",
        theta_range=(0.62, 1.90),
        phi_range=(0.58, 1.02),
    ),
    BrainRegion(
        name="Lavanda/linalol no NMDA",
        category="Slide 5 - Glutamatérgica",
        color="#B07AA1",
        function="Afinidade/inibição relacionada ao receptor NMDA descrita no texto.",
        note="Bloco farmacológico conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, -0.58, 0.68),
        radii=(0.26, 0.14, 0.10),
    ),
    BrainRegion(
        name="Docking amitriptilina GluN1/GluN2B",
        category="Slide 5 - Glutamatérgica",
        color="#9C755F",
        function="Interação molecular descrita para amitriptilina e NMDAr.",
        note="Bloco molecular conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.82, 0.72),
        radii=(0.28, 0.14, 0.10),
    ),
    BrainRegion(
        name="Citocinas inflamatórias TNF-α/IL-1β/IL-6",
        category="Slide 6 - Imuno-inflamatória",
        color="#E15759",
        function="Mediadores inflamatórios elevados em depressão e ansiedade.",
        note="Bloco conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, -0.34, 0.68),
        radii=(0.32, 0.16, 0.11),
    ),
    BrainRegion(
        name="Citocina anti-inflamatória IL-10",
        category="Slide 6 - Imuno-inflamatória",
        color="#59A14F",
        function="Citocina anti-inflamatória citada no trecho.",
        note="Bloco conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, -0.62, 0.50),
        radii=(0.22, 0.12, 0.09),
    ),
    BrainRegion(
        name="Ativação serotoninérgica em células imunes",
        category="Slide 6 - Imuno-inflamatória",
        color="#4E79A7",
        function="Mecanismo proposto para controle de citocinas por antidepressivos.",
        note="Bloco conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, -0.02, 0.56),
        radii=(0.28, 0.14, 0.10),
    ),
    BrainRegion(
        name="cAMP-PKA-CREB",
        category="Slide 6 - Imuno-inflamatória",
        color="#4E79A7",
        function="Via intracelular relacionada à redução de citocinas pró-inflamatórias.",
        note="Bloco conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.30, 0.62),
        radii=(0.25, 0.13, 0.09),
    ),
    BrainRegion(
        name="TNF-α/TNFR1-TNFR2",
        category="Slide 6 - Imuno-inflamatória",
        color="#E15759",
        function="Ligação de TNF-α a TNFR1/TNFR2.",
        note="Bloco conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.62, 0.62),
        radii=(0.28, 0.14, 0.10),
    ),
    BrainRegion(
        name="Eixo HPA",
        category="Slide 6 - Imuno-inflamatória",
        color="#F28E2B",
        function="Eixo hipotálamo-hipófise-adrenal ativado por TNF-α.",
        note="Representação no hipotálamo.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.05, -0.56),
        radii=(0.28, 0.22, 0.15),
    ),
    BrainRegion(
        name="Atividade do transportador 5-HT",
        category="Slide 6 - Imuno-inflamatória",
        color="#E15759",
        function="Aumento da atividade do transportador 5-HT induzido por TNF-α.",
        note="Representação no sistema serotoninérgico.",
        kind="cylinder",
        hemispheres=("left", "right"),
        center=(0.0, -0.34, -0.98),
        radii=(0.07, 0.09, 0.0),
        z_range=(-1.38, -0.52),
    ),
    BrainRegion(
        name="Indoleamina 2,3-dioxigenase e triptofano",
        category="Slide 6 - Imuno-inflamatória",
        color="#F28E2B",
        function="Diminuição de triptofano por estimulação de indoleamina 2,3-dioxigenase.",
        note="Bloco conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.92, 0.54),
        radii=(0.30, 0.14, 0.10),
    ),
    BrainRegion(
        name="Microglia e plasticidade sináptica",
        category="Slide 6 - Imuno-inflamatória",
        color="#9C755F",
        function="Ativação de microglia e redução de marcadores de plasticidade sináptica.",
        note="Representação cortical-límbica esquemática.",
        kind="cortex",
        theta_range=(0.52, 2.05),
        phi_range=(0.20, 0.92),
    ),
    BrainRegion(
        name="Via TNF-α TLR-4/NF-κB",
        category="Slide 6 - Imuno-inflamatória",
        color="#B07AA1",
        function="Via inflamatória bloqueada por etanercepte no texto.",
        note="Bloco conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 1.22, 0.38),
        radii=(0.28, 0.14, 0.10),
    ),
    BrainRegion(
        name="Anti-TNF-α etanercepte/infliximabe",
        category="Slide 6 - Imuno-inflamatória",
        color="#59A14F",
        function="Agentes anti-TNF-α citados no texto.",
        note="Bloco farmacológico conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 1.44, 0.18),
        radii=(0.30, 0.14, 0.10),
    ),
    BrainRegion(
        name="LPS/endotoxina",
        category="Slide 6 - Imuno-inflamatória",
        color="#E15759",
        function="Indutor de inflamação e liberação de IL-6/IL-1β.",
        note="Bloco conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, -0.96, 0.34),
        radii=(0.24, 0.12, 0.09),
    ),
    BrainRegion(
        name="IL-6/IL-1β e comportamento",
        category="Slide 6 - Imuno-inflamatória",
        color="#E15759",
        function="Influência sobre humor, anedonia, exploração, retraimento social e memória.",
        note="Representação cortical-límbica esquemática.",
        kind="cortex",
        theta_range=(0.30, 2.20),
        phi_range=(-0.20, 0.70),
    ),
    BrainRegion(
        name="LTP: serotonina, plasticidade e HPA",
        category="Slide 6 - Imuno-inflamatória",
        color="#F28E2B",
        function="Alterações que influenciam potenciação de longo prazo.",
        note="Representação conceitual cortical-límbica.",
        kind="cortex",
        theta_range=(0.62, 2.10),
        phi_range=(0.44, 1.02),
    ),
    BrainRegion(
        name="Sistema endocanabinoide CB1 no estriado",
        category="Slide 6 - Imuno-inflamatória",
        color="#4E79A7",
        function="Interferência de citocinas no sistema endocanabinoide CB1.",
        note="Representação bilateral no estriado.",
        kind="ellipsoid",
        center=(0.0, 0.02, -0.10),
        radii=(0.18, 0.32, 0.13),
        pair_offset=0.36,
    ),
    BrainRegion(
        name="Sinapses GABA no estriado",
        category="Slide 6 - Imuno-inflamatória",
        color="#4E79A7",
        function="Sinapses GABA no estriado ligadas a efeitos ansiogênicos e depressivos.",
        note="Representação bilateral no estriado.",
        kind="ellipsoid",
        center=(0.0, 0.02, -0.10),
        radii=(0.14, 0.26, 0.10),
        pair_offset=0.36,
    ),
    BrainRegion(
        name="nNOS/sGC no hipotálamo",
        category="Slide 7 - NO/cGMP",
        color="#E15759",
        function="Níveis elevados de nNOS e sGC em modelos de estresse.",
        note="Representação no hipotálamo.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.05, -0.56),
        radii=(0.28, 0.22, 0.15),
    ),
    BrainRegion(
        name="nNOS/sGC no hipocampo",
        category="Slide 7 - NO/cGMP",
        color="#E15759",
        function="Níveis elevados de nNOS e sGC em modelos de estresse.",
        note="Representação no hipocampo.",
        kind="tube",
        curve="hippocampus",
        tube_radius=0.062,
    ),
    BrainRegion(
        name="nNOS/sGC no córtex pré-frontal",
        category="Slide 7 - NO/cGMP",
        color="#E15759",
        function="Níveis elevados de nNOS e sGC em modelos de estresse.",
        note="Representação cortical anterior.",
        kind="cortex",
        theta_range=(0.10, 0.72),
        phi_range=(0.06, 0.82),
    ),
    BrainRegion(
        name="Isoformas NOS: nNOS/eNOS/iNOS",
        category="Slide 7 - NO/cGMP",
        color="#4E79A7",
        function="Três isoformas funcionais da nitric oxide synthase.",
        note="Bloco conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, -0.42, 0.64),
        radii=(0.30, 0.14, 0.10),
    ),
    BrainRegion(
        name="L-arginina -> NO por nNOS",
        category="Slide 7 - NO/cGMP",
        color="#4E79A7",
        function="Produção de NO a partir de L-arginina regulada por Ca2+.",
        note="Bloco conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, -0.08, 0.62),
        radii=(0.30, 0.14, 0.10),
    ),
    BrainRegion(
        name="NO excessivo: estresse oxidativo/neuroinflamação",
        category="Slide 7 - NO/cGMP",
        color="#E15759",
        function="Excesso de NO associado à piora de ansiedade e depressão.",
        note="Representação cortical-límbica.",
        kind="cortex",
        theta_range=(0.30, 2.20),
        phi_range=(-0.20, 0.80),
    ),
    BrainRegion(
        name="Barreira hematoencefálica e células periféricas",
        category="Slide 7 - NO/cGMP",
        color="#F28E2B",
        function="Dilatação/permeabilidade da barreira e migração de células inflamatórias.",
        note="Faixa cortical externa esquemática.",
        kind="cortex",
        theta_range=(0.05, 2.55),
        phi_range=(0.86, 1.16),
    ),
    BrainRegion(
        name="O2- + NO -> ONOO-",
        category="Slide 7 - NO/cGMP",
        color="#B07AA1",
        function="Formação de peroxinitrito.",
        note="Bloco molecular conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.30, 0.64),
        radii=(0.28, 0.14, 0.10),
    ),
    BrainRegion(
        name="Dano oxidativo por peroxinitrito",
        category="Slide 7 - NO/cGMP",
        color="#B07AA1",
        function="Peroxidação lipídica, dano ao DNA, disfunção proteica e depleção antioxidante.",
        note="Representação cortical-límbica.",
        kind="cortex",
        theta_range=(0.56, 2.30),
        phi_range=(0.26, 0.94),
    ),
    BrainRegion(
        name="RyR1/Ca2+ e radicais livres",
        category="Slide 7 - NO/cGMP",
        color="#F28E2B",
        function="NO ativa RyR1 e aumenta Ca2+ citosólico.",
        note="Bloco conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.62, 0.58),
        radii=(0.25, 0.13, 0.09),
    ),
    BrainRegion(
        name="Exocitose de glutamato e NMDA",
        category="Slide 7 - NO/cGMP",
        color="#59A14F",
        function="Aumento de exocitose de glutamato e excitotoxicidade por NMDA.",
        note="Representação cortical.",
        kind="cortex",
        theta_range=(0.64, 1.80),
        phi_range=(0.04, 0.72),
    ),
    BrainRegion(
        name="NMDA hiperativo: Ca2+ -> nNOS -> NO",
        category="Slide 7 - NO/cGMP",
        color="#59A14F",
        function="Fluxo pós-sináptico de Ca2+, superestimulação de nNOS e aumento de NO.",
        note="Conexão conceitual.",
        kind="tube",
        curve="nmda_nnos_no",
        tube_radius=0.026,
    ),
    BrainRegion(
        name="NO -> sGC -> cGMP",
        category="Slide 7 - NO/cGMP",
        color="#4E79A7",
        function="NO ativa sGC, convertendo GTP em cGMP.",
        note="Bloco conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 0.92, 0.50),
        radii=(0.24, 0.12, 0.09),
    ),
    BrainRegion(
        name="Via L-arginina-NO-cGMP",
        category="Slide 7 - NO/cGMP",
        color="#4E79A7",
        function="Via modulada por nNOS e sGC.",
        note="Conexão conceitual.",
        kind="tube",
        curve="arginine_no_cgmp",
        tube_radius=0.026,
    ),
    BrainRegion(
        name="Inibição nNOS/sGC",
        category="Slide 7 - NO/cGMP",
        color="#59A14F",
        function="Alvos valiosos para estratégias terapêuticas.",
        note="Bloco terapêutico conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 1.20, 0.32),
        radii=(0.26, 0.13, 0.09),
    ),
    BrainRegion(
        name="Docking: L-NAME, 7NA, 7NI e azul de metileno",
        category="Slide 7 - NO/cGMP",
        color="#B07AA1",
        function="Ensaio de docking com inibidores de nNOS.",
        note="Bloco molecular conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, 1.42, 0.12),
        radii=(0.34, 0.15, 0.10),
    ),
    BrainRegion(
        name="7NI: efeito antidepressivo e proteção serotoninérgica",
        category="Slide 7 - NO/cGMP",
        color="#59A14F",
        function="Inibidor seletivo de nNOS com efeito antidepressivo em camundongos.",
        note="Representação no hipocampo/serotonina.",
        kind="tube",
        curve="hippocampus",
        tube_radius=0.048,
    ),
    BrainRegion(
        name="Linalol/THL: NO e NMDA",
        category="Slide 7 - NO/cGMP",
        color="#9C755F",
        function="Inibição da formação de NO e bloqueio de NMDA.",
        note="Bloco natural/terapêutico conceitual.",
        kind="ellipsoid",
        hemispheres=(),
        center=(0.0, -0.76, 0.46),
        radii=(0.28, 0.14, 0.10),
    ),
    BrainRegion(
        name="Córtex pré-frontal medial (5-HT1A)",
        category="Receptores 5-HT1A (córtex)",
        color="#4E79A7",
        function="Área cortical com expressão/ligação relevante de 5-HT1A, associada a regulação emocional e controle top-down.",
        note="5-HT1A é tratado aqui como subtipo de receptor serotoninérgico, não como mapa quantitativo de densidade.",
        kind="cortex",
        theta_range=(0.12, 0.78),
        phi_range=(0.08, 0.78),
    ),
    BrainRegion(
        name="Córtex orbitofrontal (5-HT1A)",
        category="Receptores 5-HT1A (córtex)",
        color="#4E79A7",
        function="Área frontal ventral relacionada a valoração, decisão afetiva e regulação de respostas emocionais.",
        note="Representação esquemática de área cortical associada a sinal serotoninérgico 5-HT1A.",
        kind="cortex",
        theta_range=(0.08, 0.52),
        phi_range=(-0.58, -0.20),
    ),
    BrainRegion(
        name="Córtex cingulado (5-HT1A)",
        category="Receptores 5-HT1A (córtex)",
        color="#4E79A7",
        function="Região paralímbica com expressão relevante de 5-HT1A, ligada a emoção, dor e controle cognitivo.",
        note="Inclui a representação didática do cingulado como área cortical/paralímbica.",
        kind="cortex",
        theta_range=(0.64, 1.90),
        phi_range=(0.74, 1.08),
    ),
    BrainRegion(
        name="Córtex insular (5-HT1A)",
        category="Receptores 5-HT1A (córtex)",
        color="#4E79A7",
        function="Área associada a interocepção, saliência emocional e integração corpo-emoção.",
        note="Estudos relatam expressão de mRNA de 5-HT1A em neurônios piramidais do córtex insular.",
        kind="cortex",
        theta_range=(0.78, 1.34),
        phi_range=(-0.30, 0.20),
    ),
    BrainRegion(
        name="Córtex temporal/entorrinal (5-HT1A)",
        category="Receptores 5-HT1A (córtex)",
        color="#4E79A7",
        function="Áreas temporais mediais/paralímbicas relacionadas a memória e integração com hipocampo.",
        note="Inclui referência didática ao córtex entorrinal, frequentemente citado em distribuição 5-HT1A.",
        kind="cortex",
        theta_range=(1.62, 2.24),
        phi_range=(-0.74, -0.42),
    ),
    BrainRegion(
        name="Córtex occipital (5-HT1A)",
        category="Receptores 5-HT1A (córtex)",
        color="#59A14F",
        function="Área visual onde há relatos de expressão cortical de 5-HT1A em camadas superficiais.",
        note="Representação didática; não indica predominância funcional sobre regiões límbicas.",
        kind="cortex",
        theta_range=(2.42, math.pi),
        phi_range=(-0.16, 0.48),
    ),
    BrainRegion(
        name="Córtex motor primário",
        category="Áreas funcionais",
        color="#D62728",
        function="Inicia comandos para movimentos voluntários, com organização somatotópica.",
        note="Faixa anterior ao sulco central.",
        kind="cortex",
        theta_range=(0.93, 1.07),
        phi_range=(-0.08, 0.92),
    ),
    BrainRegion(
        name="Córtex somatossensorial primário",
        category="Áreas funcionais",
        color="#1F77B4",
        function="Recebe tato, pressão, dor, temperatura e propriocepção do corpo.",
        note="Faixa posterior ao sulco central.",
        kind="cortex",
        theta_range=(1.12, 1.27),
        phi_range=(-0.08, 0.92),
    ),
    BrainRegion(
        name="Área visual primária",
        category="Áreas funcionais",
        color="#2CA02C",
        function="Primeiro processamento cortical das informações visuais.",
        note="Localizada no lobo occipital.",
        kind="cortex",
        theta_range=(2.55, math.pi),
        phi_range=(-0.06, 0.48),
    ),
    BrainRegion(
        name="Área auditiva primária",
        category="Áreas funcionais",
        color="#9467BD",
        function="Processamento inicial de frequências, intensidade e padrões sonoros.",
        note="Localizada na região superior do lobo temporal.",
        kind="cortex",
        theta_range=(1.16, 1.58),
        phi_range=(-0.62, -0.34),
    ),
    BrainRegion(
        name="Área de Broca",
        category="Áreas funcionais",
        color="#FF7F0E",
        function="Planejamento motor da fala e produção da linguagem.",
        note="Representada no hemisfério esquerdo, na região frontal inferior.",
        kind="cortex",
        hemispheres=("left",),
        theta_range=(0.52, 0.82),
        phi_range=(-0.49, -0.18),
    ),
    BrainRegion(
        name="Área de Wernicke",
        category="Áreas funcionais",
        color="#17BECF",
        function="Compreensão da linguagem falada e escrita.",
        note="Representada no hemisfério esquerdo, próxima ao temporal posterior.",
        kind="cortex",
        hemispheres=("left",),
        theta_range=(1.83, 2.24),
        phi_range=(-0.54, -0.25),
    ),
)


SLIDE_LABELS = [
    "1.1.1 - Rota monoaminérgica: receptores 5-HT1A, 5-HT2A e 5-HT2C",
    "1.1.5 - Receptores dopaminérgicos D1 e D2",
    "1.1.6 - Receptores α2A e α2B",
    "1.1.7 - Transportadores de recaptação de noradrenalina e serotonina",
    "1.1.4 - Via glutamatérgica: receptor NMDA",
    "1.1.3 - Via inflamatória: TNF-α e IL-6",
    "1.1.2 - Via sGC/nNOS",
]

SLIDE_1_RAPHE_NAMES = {
    "Autorreceptores 5-HT1A (núcleos da rafe)",
}

SLIDE_1_LIMBIC_NAMES = {
    "Hipocampo",
    "Amígdala",
    "Hipotálamo",
    "Septo lateral",
    "Núcleo accumbens",
    "Tálamo anterior",
    "Corpos mamilares",
    "Fórnix",
    "Córtex cingulado",
    "Córtex parahipocampal",
    "Córtex entorrinal",
}

SLIDE_1_CORTICAL_NAMES = {
    "Córtex pré-frontal medial (5-HT1A)",
    "Córtex orbitofrontal (5-HT1A)",
    "Córtex cingulado (5-HT1A)",
    "Córtex insular (5-HT1A)",
    "Córtex temporal/entorrinal (5-HT1A)",
}

SLIDE_1_CONTEXT_NAMES = {
    "Tronco encefálico",
}

SLIDE_1_GROUPS = {
    "Autorreceptores 5-HT1A - núcleos da rafe": SLIDE_1_RAPHE_NAMES,
    "Área límbica - heterorreceptores 5-HT1A": SLIDE_1_LIMBIC_NAMES,
    "Zonas corticais - heterorreceptores 5-HT1A": SLIDE_1_CORTICAL_NAMES,
}

SLIDE_2_MESOLIMBIC_NAMES = {
    "Área tegmental ventral (VTA)",
    "Núcleo accumbens (NAc)",
    "Via VTA-NAc",
    "Via VTA-amígdala",
    "Via VTA-hipocampo",
    "Via VTA-PFC",
    "Córtex pré-frontal (PFC)",
}

SLIDE_2_D1_NAMES = {
    "D1-like na amígdala",
}

SLIDE_2_D2_NAMES = {
    "D2-like PFC-amígdala",
}

SLIDE_2_D1_D2_NAMES = {
    "Complexo D1-D2 no núcleo accumbens",
    "Complexo D1-D2 no caudado/estriado",
}

SLIDE_2_BDNF_NAMES = {
    "BDNF/TrkB no núcleo accumbens",
    "BDNF/TrkB no hipocampo",
    "BDNF/TrkB no córtex pré-frontal",
}

SLIDE_2_CONTEXT_NAMES = {
    "Tronco encefálico",
}

SLIDE_2_GROUPS = {
    "Sistema dopaminérgico mesolímbico": SLIDE_2_MESOLIMBIC_NAMES,
    "Receptores D1-like": SLIDE_2_D1_NAMES,
    "Receptores D2-like": SLIDE_2_D2_NAMES,
    "Complexo heteromérico D1-D2": SLIDE_2_D1_D2_NAMES,
    "BDNF/TrkB por região": SLIDE_2_BDNF_NAMES,
}

SLIDE_3_NA_SYSTEM_NAMES = {
    "Neurônios adrenérgicos centrais",
    "Via noradrenérgica para neurônios pós-sinápticos",
}

SLIDE_3_AUTO_NAMES = {
    "α2-autorreceptores pré-sinápticos",
}

SLIDE_3_HETERO_NAMES = {
    "α2-heterorreceptores dopaminérgicos",
    "α2-heterorreceptores serotoninérgicos",
    "α2-heterorreceptores glutamatérgicos",
    "Via NA-dopaminérgica",
    "Via NA-serotoninérgica",
    "Via NA-glutamatérgica",
}

SLIDE_3_POST_NAMES = {
    "Ativação α2 pós-sináptica",
}

SLIDE_3_SUBTYPE_NAMES = {
    "Subtipos α2A/α2B/α2C",
    "Sítio de ligação α2A/α2B",
}

SLIDE_3_CONTEXT_NAMES = {
    "Tronco encefálico",
}

SLIDE_3_GROUPS = {
    "Sistema noradrenérgico central": SLIDE_3_NA_SYSTEM_NAMES,
    "α2-autorreceptores": SLIDE_3_AUTO_NAMES,
    "α2-heterorreceptores": SLIDE_3_HETERO_NAMES,
    "Ativação pós-sináptica": SLIDE_3_POST_NAMES,
    "Subtipos e sítio de ligação": SLIDE_3_SUBTYPE_NAMES,
}

SLIDE_4_TRANSPORTER_NAMES = {
    "Transportador SERT",
    "Transportador NET",
}

SLIDE_4_ION_NAMES = {
    "Gradiente Na+/Cl-",
}

SLIDE_4_PROJECTION_NAMES = {
    "Via 5-HT para centros emocionais e somáticos",
    "Via NA para córtex frontal",
    "Via NA para amígdala",
    "Via NA para hipocampo",
    "Via NA para hipotálamo",
}

SLIDE_4_BDNF_NAMES = {
    "Neuroplasticidade hipocampal e BDNF",
}

SLIDE_4_BINDING_NAMES = {
    "Bolsões S1/S2 de SERT/NET",
    "SERT - Asp-98 no Sítio A",
}

SLIDE_4_CONTEXT_NAMES = {
    "Tronco encefálico",
}

SLIDE_4_GROUPS = {
    "Transportadores SERT/NET": SLIDE_4_TRANSPORTER_NAMES,
    "Dependência Na+/Cl-": SLIDE_4_ION_NAMES,
    "Projeções 5-HT/NA": SLIDE_4_PROJECTION_NAMES,
    "Neuroplasticidade hipocampal/BDNF": SLIDE_4_BDNF_NAMES,
    "Bolsões S1/S2 e seletividade": SLIDE_4_BINDING_NAMES,
}

SLIDE_5_NMDA_SYSTEM_NAMES = {
    "Regiões mesocorticolímbicas NMDA",
    "Receptor NMDA - canal de Ca2+",
    "Subunidades GluN1/GluN2/GluN3",
}

SLIDE_5_ANTAGONIST_NAMES = {
    "Antagonistas NMDA - ketamina e análogos",
    "Hipocampo ventral",
    "Córtex pré-frontal medial NMDA",
}

SLIDE_5_DISINHIBITION_NAMES = {
    "Interneurônios GABAérgicos mPFC",
    "Neurônios piramidais glutamatérgicos",
    "Pico de disparo glutamatérgico",
}

SLIDE_5_BDNF_NAMES = {
    "BDNF/TrkB no córtex límbico",
}

SLIDE_5_NATURAL_NAMES = {
    "Lavanda/linalol no NMDA",
}

SLIDE_5_DOCKING_NAMES = {
    "Docking amitriptilina GluN1/GluN2B",
}

SLIDE_5_CONTEXT_NAMES = {
    "Tronco encefálico",
}

SLIDE_5_GROUPS = {
    "NMDA em regiões mesocorticolímbicas": SLIDE_5_NMDA_SYSTEM_NAMES,
    "Antagonistas NMDA": SLIDE_5_ANTAGONIST_NAMES,
    "Desinibição glutamatérgica mPFC": SLIDE_5_DISINHIBITION_NAMES,
    "BDNF/TrkB e neurogênese": SLIDE_5_BDNF_NAMES,
    "Lavanda/linalol": SLIDE_5_NATURAL_NAMES,
    "Docking amitriptilina GluN1/GluN2B": SLIDE_5_DOCKING_NAMES,
}

SLIDE_6_CYTOKINE_NAMES = {
    "Citocinas inflamatórias TNF-α/IL-1β/IL-6",
    "Citocina anti-inflamatória IL-10",
}

SLIDE_6_ANTIDEPRESSANT_NAMES = {
    "Ativação serotoninérgica em células imunes",
    "cAMP-PKA-CREB",
}

SLIDE_6_TNF_NAMES = {
    "TNF-α/TNFR1-TNFR2",
    "Eixo HPA",
    "Atividade do transportador 5-HT",
    "Indoleamina 2,3-dioxigenase e triptofano",
    "Microglia e plasticidade sináptica",
}

SLIDE_6_ANTI_TNF_NAMES = {
    "Via TNF-α TLR-4/NF-κB",
    "Anti-TNF-α etanercepte/infliximabe",
}

SLIDE_6_LPS_NAMES = {
    "LPS/endotoxina",
    "IL-6/IL-1β e comportamento",
    "LTP: serotonina, plasticidade e HPA",
}

SLIDE_6_CB1_NAMES = {
    "Sistema endocanabinoide CB1 no estriado",
    "Sinapses GABA no estriado",
}

SLIDE_6_CONTEXT_NAMES = {
    "Tronco encefálico",
}

SLIDE_6_GROUPS = {
    "Citocinas pró/anti-inflamatórias": SLIDE_6_CYTOKINE_NAMES,
    "SSRIs/tricíclicos e cAMP-PKA-CREB": SLIDE_6_ANTIDEPRESSANT_NAMES,
    "TNF-α: HPA, SERT, triptofano e microglia": SLIDE_6_TNF_NAMES,
    "Bloqueio TNF-α/TLR-4/NF-κB": SLIDE_6_ANTI_TNF_NAMES,
    "LPS, IL-6/IL-1β e LTP": SLIDE_6_LPS_NAMES,
    "CB1/GABA no estriado": SLIDE_6_CB1_NAMES,
}

SLIDE_7_NNOS_SGC_NAMES = {
    "nNOS/sGC no hipotálamo",
    "nNOS/sGC no hipocampo",
    "nNOS/sGC no córtex pré-frontal",
}

SLIDE_7_NOS_NO_NAMES = {
    "Isoformas NOS: nNOS/eNOS/iNOS",
    "L-arginina -> NO por nNOS",
    "NO excessivo: estresse oxidativo/neuroinflamação",
}

SLIDE_7_INFLAMMATION_NAMES = {
    "Barreira hematoencefálica e células periféricas",
    "O2- + NO -> ONOO-",
    "Dano oxidativo por peroxinitrito",
}

SLIDE_7_NMDA_NAMES = {
    "RyR1/Ca2+ e radicais livres",
    "Exocitose de glutamato e NMDA",
    "NMDA hiperativo: Ca2+ -> nNOS -> NO",
}

SLIDE_7_CGMP_NAMES = {
    "NO -> sGC -> cGMP",
    "Via L-arginina-NO-cGMP",
}

SLIDE_7_INHIBITOR_NAMES = {
    "Inibição nNOS/sGC",
    "Docking: L-NAME, 7NA, 7NI e azul de metileno",
    "7NI: efeito antidepressivo e proteção serotoninérgica",
}

SLIDE_7_LINALOOL_NAMES = {
    "Linalol/THL: NO e NMDA",
}

SLIDE_7_CONTEXT_NAMES = {
    "Tronco encefálico",
}

SLIDE_7_GROUPS = {
    "nNOS/sGC em estresse": SLIDE_7_NNOS_SGC_NAMES,
    "NOS, L-arginina e NO": SLIDE_7_NOS_NO_NAMES,
    "Neuroinflamação e peroxinitrito": SLIDE_7_INFLAMMATION_NAMES,
    "Ca2+, glutamato e NMDA": SLIDE_7_NMDA_NAMES,
    "NO-sGC-cGMP": SLIDE_7_CGMP_NAMES,
    "Inibidores nNOS/sGC": SLIDE_7_INHIBITOR_NAMES,
    "Linalol/THL": SLIDE_7_LINALOOL_NAMES,
}

PRESENTATION_REGION_NAMES = (
    SLIDE_1_RAPHE_NAMES
    | SLIDE_1_LIMBIC_NAMES
    | SLIDE_1_CORTICAL_NAMES
    | SLIDE_2_MESOLIMBIC_NAMES
    | SLIDE_2_D1_NAMES
    | SLIDE_2_D2_NAMES
    | SLIDE_2_D1_D2_NAMES
    | SLIDE_2_BDNF_NAMES
    | SLIDE_3_NA_SYSTEM_NAMES
    | SLIDE_3_AUTO_NAMES
    | SLIDE_3_HETERO_NAMES
    | SLIDE_3_POST_NAMES
    | SLIDE_3_SUBTYPE_NAMES
    | SLIDE_4_TRANSPORTER_NAMES
    | SLIDE_4_ION_NAMES
    | SLIDE_4_PROJECTION_NAMES
    | SLIDE_4_BDNF_NAMES
    | SLIDE_4_BINDING_NAMES
    | SLIDE_5_NMDA_SYSTEM_NAMES
    | SLIDE_5_ANTAGONIST_NAMES
    | SLIDE_5_DISINHIBITION_NAMES
    | SLIDE_5_BDNF_NAMES
    | SLIDE_5_NATURAL_NAMES
    | SLIDE_5_DOCKING_NAMES
    | SLIDE_6_CYTOKINE_NAMES
    | SLIDE_6_ANTIDEPRESSANT_NAMES
    | SLIDE_6_TNF_NAMES
    | SLIDE_6_ANTI_TNF_NAMES
    | SLIDE_6_LPS_NAMES
    | SLIDE_6_CB1_NAMES
    | SLIDE_7_NNOS_SGC_NAMES
    | SLIDE_7_NOS_NO_NAMES
    | SLIDE_7_INFLAMMATION_NAMES
    | SLIDE_7_NMDA_NAMES
    | SLIDE_7_CGMP_NAMES
    | SLIDE_7_INHIBITOR_NAMES
    | SLIDE_7_LINALOOL_NAMES
)


def _wrap_hover_text(title: str, *paragraphs: str, width: int = 78) -> str:
    wrapped_paragraphs = [
        "<br>".join(
            textwrap.wrap(
                paragraph,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
        for paragraph in paragraphs
    ]
    return "<br><br>".join([title, *wrapped_paragraphs])


RAPHE_HOVER_TEXT = _wrap_hover_text(
    "Autorreceptores 5-HT1A - núcleos da rafe",
    (
        "São autorreceptores somatodendríticos localizados nos núcleos da rafe. "
        "Segundo o artigo, os receptores 5-HT são metabotrópicos, acoplados a proteínas Gi, "
        "levando à redução de cAMP, inativação de canais de cálcio e ativação de canais de potássio, "
        "culminando em inibição da atividade neuronal. Os autorreceptores reduzem as taxas de disparo "
        "por feedback negativo, limitando a liberação."
    ),
)

HETERO_HOVER_TEXT = _wrap_hover_text(
    "Heterorreceptores 5-HT1A - áreas límbicas e corticais",
    (
        "São heterorreceptores pós-sinápticos encontrados em áreas centrais diversas "
        "inervadas por projeções serotoninérgicas, em neurônios piramidais e interneurônios GABAérgicos. "
        "Nesta apresentação, essas áreas foram agrupadas visualmente como área límbica e zonas corticais, "
        "sem acrescentar mecanismos além do trecho do artigo."
    ),
)

MESOLIMBIC_DOPAMINE_HOVER_TEXT = _wrap_hover_text(
    "Sistema dopaminérgico mesolímbico",
    (
        "Em modelos animais, mecanismos envolvendo o sistema dopaminérgico mesolímbico "
        "são relacionados a comportamentos do tipo depressão e ansiedade."
    ),
    (
        "A via é sinalizada pela ativação de receptores dopaminérgicos metabotrópicos "
        "acoplados à proteína G (GPCR), subdivididos em D1-like (D1 e D5, Gs, aumento de cAMP) "
        "e D2-like (D2, D3 e D4, Gi, redução de cAMP)."
    ),
)

D1_HOVER_TEXT = _wrap_hover_text(
    "Receptores D1-like",
    (
        "O receptor D1, quando ativado, medeia comportamento relacionado a ansiedade, depressão, "
        "recompensa e memória, além de desencadear respostas imunes e neuroinflamatórias."
    ),
    (
        "Seu mecanismo pode ser estimulatório ou inibitório, dependendo da região ativada. "
        "Em células da amígdala, a inibição de interneurônios GABAérgicos por receptores D1 "
        "em grupos para-capsulares aumenta angiogênese e medo; em regiões basolaterais, em "
        "neurônios de projeção, o efeito é excitatório."
    ),
)

D2_HOVER_TEXT = _wrap_hover_text(
    "Receptores D2-like",
    (
        "Como receptores D2 pré-sinápticos, são observados principalmente em terminais "
        "pré-sinápticos do córtex pré-frontal. Sua ativação exerce efeito supressivo "
        "GABAérgico sobre grupos de interneurônios positivos para parvalbumina em células da amígdala."
    ),
    (
        "A administração local de agonistas D1 ou D2 na amígdala induz efeitos do tipo ansiedade "
        "e depressão, e também efeitos ansiolíticos e antidepressivos."
    ),
)

D1_D2_HOVER_TEXT = _wrap_hover_text(
    "Complexo heteromérico D1-D2",
    (
        "Complexos heteroméricos dopaminérgicos, como D1-D2, apresentam distribuição discreta, "
        "e suas funções neurofisiológicas podem diferir dos receptores constituintes individuais."
    ),
    (
        "O artigo relata maior concentração de BDNF no núcleo accumbens em pacientes deprimidos "
        "e ansiosos, resultante do aumento da expressão de D1-D2. Também descreve maior densidade "
        "de complexos D1-D2 no núcleo caudado de primatas não humanos e no estriado de roedores fêmeas."
    ),
)

BDNF_HOVER_TEXT = _wrap_hover_text(
    "BDNF/TrkB por região",
    (
        "A sinalização BDNF/TrkB apresenta efeitos diferentes, dependendo da região cerebral, "
        "na fisiopatologia da ansiedade e da depressão."
    ),
    (
        "Em regiões mesolímbicas, sua função pode ser depressora; em regiões como hipocampo (HC) "
        "e córtex pré-frontal (PFC), seu efeito é antidepressivo."
    ),
)

NA_SYSTEM_HOVER_TEXT = _wrap_hover_text(
    "Sistema noradrenérgico central",
    (
        "O sistema noradrenérgico central tem sido implicado em muitos transtornos neuropsiquiátricos."
    ),
    (
        "Os α-adrenorreceptores são acoplados a proteína G regulatória e classificados nos subtipos "
        "α1A, α1B, α1D, α2A, α2B e α2C."
    ),
)

ALPHA2_AUTO_HOVER_TEXT = _wrap_hover_text(
    "α2-autorreceptores",
    (
        "Os α2-autorreceptores estão localizados nas membranas pré-sinápticas de neurônios adrenérgicos "
        "e regulam a liberação de noradrenalina e adrenalina."
    ),
    (
        "Os receptores α2 são ligados a proteína G do tipo Gi/o; quando ativados, regulam a atividade "
        "intracelular da adenilil ciclase, modulam negativamente a abertura de canais de Ca2+ e bombas "
        "Na+/H+-ATPase, e modulam positivamente a abertura de canais de K+, limitando a liberação de catecolaminas."
    ),
)

ALPHA2_HETERO_HOVER_TEXT = _wrap_hover_text(
    "α2-heterorreceptores",
    (
        "Os α2-heterorreceptores estão localizados em membranas pré-sinápticas de neurônios "
        "dopaminérgicos, serotoninérgicos e glutamatérgicos."
    ),
)

ALPHA2_POST_HOVER_TEXT = _wrap_hover_text(
    "Ativação α2 pós-sináptica",
    (
        "Em neurônios pós-sinápticos, a ativação α2 modula a excitabilidade neuronal por regulação "
        "de canais iônicos, incluindo modulação direta de canais retificadores de potássio e modulação "
        "indireta de canais ativados por hiperpolarização."
    ),
)

ALPHA2_SUBTYPE_HOVER_TEXT = _wrap_hover_text(
    "Subtipos α2A/α2B/α2C e sítio de ligação",
    (
        "A maioria das funções fisiológicas e farmacológicas dos α2-adrenorreceptores é atribuída ao "
        "subtipo α2A. Estudos farmacológicos sugeriram que apenas α2A mediaria efeitos analgésicos e "
        "ansiolíticos, mas as evidências apontam contribuição do subtipo α2B."
    ),
    (
        "O subtipo α2A é responsável por funções clássicas como hipotensão, sedação, hipotermia e efeitos "
        "anestésicos e analgésicos. O subtipo α2B compartilha características de complexos acoplados a Gi/o, "
        "e seu desempenho é mediado por óxido nítrico."
    ),
    (
        "Em estudos in silico, α2A apresenta maior seletividade contra ligantes com substituintes fenil "
        "volumosos, enquanto α2B parece menos sensível ao tamanho do ligante por ter abertura maior. "
        "Na posição 5.31, α2A apresenta ácido glutâmico e α2B apresenta lisina."
    ),
    (
        "No sítio de ligação α2A, substâncias com anéis imidazólicos se ligam entre as hélices 3 e 6 "
        "do bolso hidrofóbico, com resíduos Phe 6.52, Phe 6.53 e Val 3.33, além de ligação de hidrogênio "
        "com Tyr 6.55 e interação eletrostática com Asp 3.32."
    ),
)

SERT_NET_HOVER_TEXT = _wrap_hover_text(
    "Transportadores SERT/NET",
    (
        "Serotonina e noradrenalina são neurotransmissores que podem ser recapturados pelos "
        "transportadores de serotonina (SERT) e noradrenalina (NET)."
    ),
    (
        "O papel desses transportadores é regular a concentração de neurotransmissores na fenda "
        "sináptica e, consequentemente, as respostas desencadeadas."
    ),
)

SERT_NET_ION_HOVER_TEXT = _wrap_hover_text(
    "Dependência Na+/Cl-",
    (
        "A recaptação por SERT e NET ocorre por um mecanismo dependente de Na+ e Cl- extracelulares, "
        "que constituem a força motriz do transporte."
    ),
)

SERT_NET_PROJECTION_HOVER_TEXT = _wrap_hover_text(
    "Projeções 5-HT/NA",
    (
        "A noradrenalina é produzida por neurônios noradrenérgicos que emergem de corpos celulares "
        "no locus coeruleus e projetam para diferentes regiões do SNC, incluindo córtex frontal, "
        "amígdala, hipocampo, hipotálamo e medula espinal."
    ),
    (
        "No sistema límbico, essa via participa de emoção, apetite, resposta à dor, prazer, "
        "satisfação sexual, agressão e cognição, processos comprometidos em pacientes com ansiedade "
        "ou depressão. De modo geral, centros emocionais e somáticos recebem projeções das vias NE e 5-HT."
    ),
)

SERT_NET_BDNF_HOVER_TEXT = _wrap_hover_text(
    "Neuroplasticidade hipocampal e BDNF",
    (
        "A 5-HT é importante na neuroquímica da depressão e da ansiedade. Antidepressivos como SSRIs "
        "induzem aumento da proliferação de células hipocampais e aumento da expressão de proteínas "
        "relacionadas à neuroplasticidade, como BDNF."
    ),
)

SERT_NET_BINDING_HOVER_TEXT = _wrap_hover_text(
    "Bolsões S1/S2 e seletividade",
    (
        "Apesar de inibidores de SERT e NET serem usados na farmacoterapia de ansiedade e depressão, "
        "seus mecanismos moleculares de ligação, inibição e seletividade ainda não estão totalmente elucidados."
    ),
    (
        "Análises mutacionais sugerem ligação em regiões chamadas bolsões S1 e S2, com base em estruturas "
        "cristalográficas de LeuT, em conformação de substrato ocluída voltada para fora da célula."
    ),
    (
        "Fármacos como (S)-citalopram apresentam conformação potencial de ligação no fundo do bolsão S1, "
        "ocupando seus três subsítios."
    ),
    (
        "Estudos de docking molecular sugerem base para seletividade do transportador 5-HT: a presença "
        "do aminoácido Asp-98, com excesso de carga negativa no Sítio A da unidade S1. A ligação formada "
        "é diretamente iônica com os grupos amino positivos da 5-HT e de inibidores de SERT."
    ),
    (
        "Estudos com linalol e β-pireno indicaram respostas parcialmente devidas à ativação da via "
        "serotoninérgica e à modulação da função adrenérgica por estimulação de receptores noradrenérgicos. "
        "Ainda não há dados conclusivos sobre a atividade desses compostos como inibidores da recaptação de 5-HT e NA."
    ),
)

NMDA_SYSTEM_HOVER_TEXT = _wrap_hover_text(
    "Via glutamatérgica - receptor NMDA",
    (
        "A modulação da via glutamatérgica é alvo de estudos sobre ansiolíticos e antidepressivos. "
        "O glutamato é um neurotransmissor excitatório que atua em receptores ionotrópicos e metabotrópicos."
    ),
    (
        "O receptor NMDA é um receptor iônico controlado por ligantes, condutor de corrente de cálcio, "
        "encontrado em alta densidade em regiões mesocorticolímbicas."
    ),
    (
        "Ele medeia respostas fisiológicas relacionadas a função sináptica, plasticidade, aprendizagem, "
        "memória, depressão, ansiedade e medo."
    ),
    (
        "Sua estrutura cristalográfica é composta por heterotetrâmeros das subunidades GluN1 e "
        "GluN2A/B/C/D, ou GluN3A/B. Domínios carboxiterminais e aminoterminais são importantes para "
        "atividade regulatória."
    ),
)

NMDA_ANTAGONIST_HOVER_TEXT = _wrap_hover_text(
    "Antagonistas NMDA",
    (
        "Estudos pré-clínicos mostraram que antagonistas do receptor NMDA, como ketamina e análogos "
        "como fenciclidina e dizocilpina, atuam diminuindo a atividade do canal, reduzindo influxo de Ca2+ "
        "e mediando correntes."
    ),
    (
        "Em doses subanestésicas, a ketamina atua na fisiopatologia de transtornos psiquiátricos e "
        "atinge tecido cerebral em concentrações micromolares."
    ),
    (
        "Agentes que bloqueiam o receptor NMDA após a dissociação do glutamato em regiões como "
        "hipocampo ventral e córtex pré-frontal podem afetar comportamento emocional."
    ),
    (
        "Estudos com camundongos knockout mostraram que deleção de subunidades do receptor e administração "
        "intra-hipocampal de fármacos podem melhorar ansiedade e depressão."
    ),
)

NMDA_DISINHIBITION_HOVER_TEXT = _wrap_hover_text(
    "Desinibição glutamatérgica no mPFC",
    (
        "Dados clínicos sugerem que antagonistas podem estar relacionados à restauração de plasticidade "
        "cerebral maladaptada por inibição de receptores NMDA em interneurônios GABAérgicos no córtex "
        "pré-frontal medial."
    ),
    (
        "Esse processo resulta em desinibição de neurônios piramidais glutamatérgicos e pico de disparo "
        "glutamatérgico."
    ),
)

NMDA_BDNF_HOVER_TEXT = _wrap_hover_text(
    "BDNF/TrkB e neurogênese",
    (
        "O bloqueio de receptores NMDA por antagonistas como memantina pode induzir maior expressão de "
        "BDNF mRNA e de seu receptor trkB no córtex límbico de ratos."
    ),
    (
        "Isso melhora a função neurogênica, apontada no texto como uma base fisiopatológica da ansiedade "
        "e da depressão."
    ),
)

NMDA_NATURAL_HOVER_TEXT = _wrap_hover_text(
    "Lavanda/linalol no receptor NMDA",
    (
        "O óleo essencial de lavanda, usado em aromaterapia para transtornos psiquiátricos, apresenta alta "
        "composição de linalol e acetato de linalila, precursores de THL."
    ),
    (
        "Investigações demonstraram alta afinidade dose-dependente ao receptor glutamatérgico NMDA, com "
        "IC50 de 0,04 μl/mL para óleo de lavanda."
    ),
    (
        "Estudos mostraram que o linalol inibe receptores NMDA, influenciando preferência condicionada por "
        "local induzida por morfina, aquisição e reintegração, fatores predisponentes de tolerância e "
        "dependência de opioides."
    ),
)

NMDA_DOCKING_HOVER_TEXT = _wrap_hover_text(
    "Docking amitriptilina GluN1/GluN2B",
    (
        "Estudos de docking molecular com amitriptilina e NMDAr em GluN1/GluN2B revelaram posição "
        "energeticamente estável influenciada pela pose tricíclica, estabilizada em forma de V."
    ),
    (
        "Simulações de dinâmica molecular revelaram que o hidrogênio da amina da amitriptilina interage "
        "com o oxigênio da asparagina N612 de uma subunidade GluN2B a 1,7 Å."
    ),
    (
        "A estabilização no canal é favorecida por interações hidrofóbicas com aminoácidos alifáticos "
        "sem carga nas hélices M3 de todas as subunidades NMDAr."
    ),
    (
        "Os aminoácidos responsáveis pela interação na posição V são GluN1 V642, GluN1 T646, "
        "GluN2B L640 e GluN2B T644."
    ),
    (
        "Receptores contendo subunidades GluN2B respondem por cerca de 80% das correntes despolarizantes "
        "em neurônios corticais em culturas primárias."
    ),
)

IMMUNE_CYTOKINE_HOVER_TEXT = _wrap_hover_text(
    "Citocinas pró/anti-inflamatórias",
    (
        "Nas últimas décadas, foi mostrado em pacientes com depressão e transtornos de ansiedade que há "
        "correlações diretas entre disfunção imune e aumento de mediadores inflamatórios, como TNF-α, "
        "IL-1β e IL-6."
    ),
    (
        "Citocinas são proteínas ou polipeptídeos capazes de modular condições inflamatórias e imunológicas. "
        "Ao atingir o SNC, participam da regulação neurofisiológica da plasticidade neural e de circuitos "
        "que envolvem metabolismo de neurotransmissores e hormônios."
    ),
)

IMMUNE_ANTIDEPRESSANT_HOVER_TEXT = _wrap_hover_text(
    "SSRIs/tricíclicos e cAMP-PKA-CREB",
    (
        "Inibidores de recaptação de serotonina (SSRIs) e antidepressivos tricíclicos reduzem níveis de "
        "citocinas pró-inflamatórias e aumentam níveis de citocinas anti-inflamatórias, como IL-10."
    ),
    (
        "Postula-se que isso ocorra por aumento da ativação de receptores serotoninérgicos em células imunes, "
        "controlando a dinâmica de liberação de citocinas pró- e anti-inflamatórias."
    ),
    (
        "Outro mecanismo parece envolver aumento da produção de cAMP, ativação de PKA e aumento da produção "
        "de CREB, o que diminui a produção de citocinas pró-inflamatórias."
    ),
)

IMMUNE_TNF_HOVER_TEXT = _wrap_hover_text(
    "TNF-α: HPA, SERT, triptofano e microglia",
    (
        "TNF-α é uma citocina pleiotrópica com papel importante na regulação de funções cognitivas em "
        "transtornos de ansiedade e depressão."
    ),
    (
        "Seus efeitos biológicos ocorrem por ligação aos receptores TNFR1 e TNFR2. TNF-α ativa o eixo HPA, "
        "aumenta a atividade do transportador 5-HT e estimula indoleamina 2,3-dioxigenase, diminuindo a "
        "concentração de triptofano."
    ),
    (
        "A molécula também está envolvida na ativação da microglia cerebral, observada por reduções de "
        "marcadores de plasticidade sináptica e aumento de eventos neurodegenerativos."
    ),
)

IMMUNE_ANTI_TNF_HOVER_TEXT = _wrap_hover_text(
    "Bloqueio TNF-α/TLR-4/NF-κB",
    (
        "Em camundongos db/db, bloquear a via de sinalização TNF-α TLR-4/NF-κB com agentes como etanercepte "
        "limitou condições inflamatórias e melhorou comportamentos do tipo depressão e ansiedade."
    ),
    (
        "Anticorpos monoclonais anti-TNF-α, como infliximabe, diminuíram imobilidade no teste de nado forçado, "
        "anedonia no teste de preferência por sacarose e ansiedade no labirinto em cruz elevado."
    ),
)

IMMUNE_LPS_HOVER_TEXT = _wrap_hover_text(
    "LPS, IL-6/IL-1β e LTP",
    (
        "A administração central ou periférica de lipopolissacarídeo, endotoxina derivada de bactérias "
        "gram-negativas, promove inflamação e aumenta a liberação de citocinas inflamatórias como IL-6 e IL-1β."
    ),
    (
        "Disfunções na comunicação entre sistema imune e cérebro podem se relacionar a distúrbios do SNC "
        "mediados por citocinas."
    ),
    (
        "Em roedores, aumentos de IL-6 e IL-1β podem influenciar o humor, com manifestações semelhantes "
        "à ansiedade e depressão, incluindo anedonia, menor capacidade exploratória, retraimento social "
        "e efeitos de memória."
    ),
    (
        "O mecanismo exato ainda é pouco compreendido, mas acredita-se que alterações no metabolismo da "
        "serotonina, na plasticidade neural e no eixo HPA influenciem a potenciação de longo prazo."
    ),
)

IMMUNE_CB1_HOVER_TEXT = _wrap_hover_text(
    "CB1/GABA no estriado",
    (
        "Mecanismos adicionais de interferência de citocinas foram descobertos no sistema endocanabinoide CB1."
    ),
    (
        "Esses mecanismos contribuem para efeitos ansiogênicos e depressivos por meio de sinapses GABA "
        "no estriado."
    ),
)

NO_NNOS_SGC_HOVER_TEXT = _wrap_hover_text(
    "nNOS/sGC em estresse",
    (
        "A ativação das enzimas sGC e nNOS é considerada uma via molecular importante para piora do humor "
        "em pacientes com transtornos comportamentais."
    ),
    (
        "Em modelos animais de estresse, níveis elevados de nNOS e sGC foram encontrados no hipotálamo, "
        "hipocampo e córtex pré-frontal."
    ),
)

NO_NOS_NO_HOVER_TEXT = _wrap_hover_text(
    "NOS, L-arginina e NO",
    (
        "Em mamíferos há três isoformas funcionais de nitric oxide synthase: nNOS neuronal, eNOS endotelial "
        "e iNOS induzível."
    ),
    (
        "A nNOS foi inicialmente descrita em neurônios, mas também pode ser encontrada em musculatura estriada "
        "esquelética, epitélio brônquico e traqueia. Ela é regulada por níveis de Ca2+."
    ),
    (
        "Quando estimulada, a nNOS produz óxido nítrico a partir de L-arginina. Em excesso, o NO é associado "
        "à piora de ansiedade e depressão por estimular estresse oxidativo e neuroinflamação."
    ),
)

NO_INFLAMMATION_HOVER_TEXT = _wrap_hover_text(
    "Neuroinflamação e peroxinitrito",
    (
        "Como mediador pró-inflamatório, NO é fator-chave na iniciação de eventos que aumentam a atividade "
        "neuroinflamatória, principalmente por promover dilatação e maior permeabilidade da barreira "
        "hematoencefálica."
    ),
    (
        "Isso favorece a migração de células inflamatórias periféricas para o tecido cerebral."
    ),
    (
        "Em situações de estresse, células neuronais geram NO e ânion superóxido, que podem se unir para "
        "formar peroxinitrito. O peroxinitrito é um oxidante poderoso capaz de iniciar peroxidação lipídica, "
        "danificar DNA, causar disfunção proteica e depletar antioxidantes lipossolúveis."
    ),
)

NO_NMDA_HOVER_TEXT = _wrap_hover_text(
    "Ca2+, glutamato e NMDA",
    (
        "O NO também pode ativar receptores de rianodina tipo 1 no retículo endoplasmático e aumentar a "
        "concentração de Ca2+ citosólico, resultando em mais radicais livres e maior exocitose de glutamato."
    ),
    (
        "Isso leva à excitotoxicidade mediada pela ativação excessiva de receptores NMDA."
    ),
    (
        "Em neurônios pós-sinápticos, receptores NMDA hiperativos causam influxo anormal de Ca2+, "
        "superestimulação de nNOS e aumento da síntese de NO."
    ),
)

NO_CGMP_HOVER_TEXT = _wrap_hover_text(
    "NO-sGC-cGMP",
    (
        "NO tem como alvo a sGC, enzima responsável por converter GTP em cGMP."
    ),
    (
        "A via L-arginina-NO-cGMP, modulada por nNOS e sGC, é uma importante via de sinalização provavelmente "
        "envolvida na regulação de processos comportamentais, cognitivos e emocionais, incluindo ansiedade e depressão."
    ),
)

NO_INHIBITOR_HOVER_TEXT = _wrap_hover_text(
    "Inibidores nNOS/sGC",
    (
        "Estudos indicam que nNOS e sGC, por inibição, são alvos valiosos e fornecem estratégias efetivas "
        "para terapia desses transtornos comportamentais."
    ),
    (
        "Um ensaio de docking molecular investigou afinidade de ligação e atividade antidepressiva de "
        "ésteres N'-metileno de omega-Nitro-L-arginina, L-NAME, 7NA, 7NI e azul de metileno com nNOS."
    ),
    (
        "Foi observado que o acoplamento é favorecido por ligação de hidrogênio inibidor-alvo, formando "
        "complexos estáveis em estados próximos de equilíbrio."
    ),
    (
        "O uso de inibidores seletivos de nNOS, como 7NI, resultou em efeitos antidepressivos em camundongos, "
        "com diminuição da imobilidade no teste de suspensão pela cauda e efeito protetor em terminações "
        "serotoninérgicas hipocampais."
    ),
)

NO_LINALOOL_HOVER_TEXT = _wrap_hover_text(
    "Linalol/THL: NO e NMDA",
    (
        "Como possível mecanismo de atividade antinociceptiva de monoterpenos, o linalol inibe a formação "
        "de NO e causa bloqueio de receptores NMDA."
    ),
    (
        "Foi sugerido que seu precursor, tetrahidrolinalol, atua especificamente inibindo nNOS, modulando "
        "mecanismos neurais relacionados à fisiopatologia da ansiedade e da depressão."
    ),
)


def _grid_triangles(row_count: int, column_count: int) -> tuple[list[int], list[int], list[int]]:
    i_values: list[int] = []
    j_values: list[int] = []
    k_values: list[int] = []

    for row in range(row_count - 1):
        for column in range(column_count - 1):
            current = row * column_count + column
            right = current + 1
            below = current + column_count
            below_right = below + 1

            i_values.extend([current, right])
            j_values.extend([below, below])
            k_values.extend([right, below_right])

    return i_values, j_values, k_values


def _cortex_coordinates(theta: np.ndarray, phi: np.ndarray, hemisphere: Hemisphere) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side = -1 if hemisphere == "left" else 1
    lateral_axis = 1.05
    anterior_axis = 1.55
    superior_axis = 1.00
    midline_gap = 0.08

    x_values = side * (midline_gap + lateral_axis * np.sin(theta) * np.cos(phi))
    y_values = anterior_axis * np.cos(theta) * np.cos(phi)
    z_values = 0.15 + superior_axis * np.sin(phi)
    return x_values, y_values, z_values


def _cortex_mesh(region: BrainRegion, hemisphere: Hemisphere) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], list[int]]:
    if region.theta_range is None or region.phi_range is None:
        raise ValueError(f"Região cortical sem intervalos angulares: {region.name}")

    theta = np.linspace(region.theta_range[0], region.theta_range[1], 24)
    phi = np.linspace(region.phi_range[0], region.phi_range[1], 18)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    x_values, y_values, z_values = _cortex_coordinates(theta_grid, phi_grid, hemisphere)
    i_values, j_values, k_values = _grid_triangles(phi.size, theta.size)
    return x_values.ravel(), y_values.ravel(), z_values.ravel(), i_values, j_values, k_values


def _shifted_center(region: BrainRegion, hemisphere: Hemisphere | None) -> tuple[float, float, float]:
    if region.center is None:
        raise ValueError(f"Região sem centro definido: {region.name}")

    center_x, center_y, center_z = region.center
    if hemisphere is not None and region.pair_offset is not None:
        side = -1 if hemisphere == "left" else 1
        center_x = side * region.pair_offset
    return center_x, center_y, center_z


def _ellipsoid_mesh(region: BrainRegion, hemisphere: Hemisphere | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], list[int]]:
    if region.center is None or region.radii is None:
        raise ValueError(f"Elipsoide sem centro ou raios: {region.name}")

    u_values = np.linspace(0, 2 * math.pi, 42)
    v_values = np.linspace(-math.pi / 2, math.pi / 2, 22)
    u_grid, v_grid = np.meshgrid(u_values, v_values)

    center_x, center_y, center_z = _shifted_center(region, hemisphere)
    radius_x, radius_y, radius_z = region.radii
    x_values = center_x + radius_x * np.cos(v_grid) * np.cos(u_grid)
    y_values = center_y + radius_y * np.cos(v_grid) * np.sin(u_grid)
    z_values = center_z + radius_z * np.sin(v_grid)
    i_values, j_values, k_values = _grid_triangles(v_values.size, u_values.size)
    return x_values.ravel(), y_values.ravel(), z_values.ravel(), i_values, j_values, k_values


def _cylinder_mesh(region: BrainRegion, hemisphere: Hemisphere | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], list[int]]:
    if region.center is None or region.radii is None or region.z_range is None:
        raise ValueError(f"Cilindro sem parâmetros suficientes: {region.name}")

    theta_values = np.linspace(0, 2 * math.pi, 34)
    z_values = np.linspace(region.z_range[0], region.z_range[1], 18)
    theta_grid, z_grid = np.meshgrid(theta_values, z_values)

    center_x, center_y, _ = _shifted_center(region, hemisphere)
    radius_x, radius_y, _ = region.radii
    x_values = center_x + radius_x * np.cos(theta_grid)
    y_values = center_y + radius_y * np.sin(theta_grid)
    i_values, j_values, k_values = _grid_triangles(z_values.size, theta_values.size)
    return x_values.ravel(), y_values.ravel(), z_grid.ravel(), i_values, j_values, k_values


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _tube_path(region: BrainRegion, hemisphere: Hemisphere | None) -> np.ndarray:
    if region.curve is None:
        raise ValueError(f"Tubo sem curva definida: {region.name}")

    side = -1 if hemisphere == "left" else 1
    t = np.linspace(0.0, 1.0, 54)
    curved_segments = {
        "vta_accumbens": ((side * 0.04, -0.48, -0.72), (side * 0.30, 0.52, -0.36), 0.18),
        "vta_amygdala": ((side * 0.04, -0.48, -0.72), (side * 0.50, 0.34, -0.34), 0.14),
        "vta_hippocampus": ((side * 0.04, -0.48, -0.72), (side * 0.48, -0.50, -0.30), 0.24),
        "vta_pfc": ((side * 0.04, -0.48, -0.72), (side * 0.48, 1.18, 0.28), 0.30),
        "pfc_amygdala": ((side * 0.52, 1.06, 0.24), (side * 0.50, 0.34, -0.34), 0.12),
        "na_postsynaptic": ((side * 0.04, -0.56, -0.60), (side * 0.56, 0.78, 0.26), 0.26),
        "na_dopaminergic": ((side * 0.04, -0.56, -0.60), (side * 0.28, -0.32, -0.54), 0.08),
        "na_serotonergic": ((side * 0.04, -0.56, -0.60), (side * 0.06, -0.34, -0.98), 0.10),
        "na_glutamatergic": ((side * 0.04, -0.56, -0.60), (side * 0.72, 0.68, 0.42), 0.30),
        "sert_limbic": ((side * 0.06, -0.34, -0.98), (side * 0.50, 0.10, -0.28), 0.22),
        "net_frontal": ((side * 0.04, -0.56, -0.60), (side * 0.50, 1.20, 0.30), 0.34),
        "net_amygdala": ((side * 0.04, -0.56, -0.60), (side * 0.50, 0.34, -0.34), 0.18),
        "net_hippocampus": ((side * 0.04, -0.56, -0.60), (side * 0.48, -0.50, -0.30), 0.22),
        "net_hypothalamus": ((side * 0.04, -0.56, -0.60), (side * 0.10, 0.05, -0.56), 0.10),
        "gaba_pyramidal": ((side * 0.46, 0.92, 0.34), (side * 0.78, 0.64, 0.44), 0.08),
        "nmda_nnos_no": ((side * 0.64, 0.68, 0.38), (side * 0.12, -0.04, 0.60), 0.18),
        "arginine_no_cgmp": ((side * 0.04, -0.08, 0.62), (side * 0.04, 0.92, 0.50), 0.22),
    }

    if region.curve in curved_segments:
        start, end, arc_height = curved_segments[region.curve]
        start_arr = np.array(start)
        end_arr = np.array(end)
        points = (1.0 - t)[:, None] * start_arr + t[:, None] * end_arr
        points[:, 2] = points[:, 2] + arc_height * np.sin(math.pi * t)
        return points
    if region.curve == "hippocampus":
        x_values = side * (0.48 + 0.10 * np.cos(math.pi * t))
        y_values = 0.34 - 0.98 * t
        z_values = -0.44 + 0.26 * np.sin(math.pi * t)
    elif region.curve == "parahippocampal":
        x_values = side * (0.58 + 0.04 * np.cos(math.pi * t))
        y_values = 0.28 - 0.94 * t
        z_values = -0.64 + 0.16 * np.sin(math.pi * t)
    elif region.curve == "entorhinal":
        x_values = side * (0.54 + 0.04 * np.cos(math.pi * t))
        y_values = 0.42 - 0.34 * t
        z_values = -0.55 + 0.08 * np.sin(math.pi * t)
    elif region.curve == "fornix":
        x_values = side * (0.42 * (1.0 - t) ** 1.25 + 0.06)
        y_values = -0.50 + 0.70 * t
        z_values = -0.30 + 0.42 * np.sin(math.pi * t) - 0.12 * t
    elif region.curve == "cingulate":
        x_values = side * np.full_like(t, 0.16)
        y_values = 0.70 - 1.45 * t
        z_values = 0.68 + 0.28 * np.sin(math.pi * t)
    else:
        raise ValueError(f"Curva desconhecida para tubo: {region.curve}")

    return np.column_stack([x_values, y_values, z_values])


def _tube_mesh(region: BrainRegion, hemisphere: Hemisphere | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], list[int]]:
    points = _tube_path(region, hemisphere)
    path_count = points.shape[0]
    radial_count = 18
    angles = np.linspace(0, 2 * math.pi, radial_count)

    tangents = np.gradient(points, axis=0)
    tangents = _normalize_vectors(tangents)
    reference = np.tile(np.array([0.0, 0.0, 1.0]), (path_count, 1))
    parallel = np.abs(np.sum(tangents * reference, axis=1)) > 0.92
    reference[parallel] = np.array([1.0, 0.0, 0.0])

    normal_a = _normalize_vectors(np.cross(tangents, reference))
    normal_b = _normalize_vectors(np.cross(tangents, normal_a))

    cos_values = np.cos(angles)[None, :]
    sin_values = np.sin(angles)[None, :]
    x_values = points[:, 0, None] + region.tube_radius * (
        cos_values * normal_a[:, 0, None] + sin_values * normal_b[:, 0, None]
    )
    y_values = points[:, 1, None] + region.tube_radius * (
        cos_values * normal_a[:, 1, None] + sin_values * normal_b[:, 1, None]
    )
    z_values = points[:, 2, None] + region.tube_radius * (
        cos_values * normal_a[:, 2, None] + sin_values * normal_b[:, 2, None]
    )

    i_values, j_values, k_values = _grid_triangles(path_count, radial_count)
    return x_values.ravel(), y_values.ravel(), z_values.ravel(), i_values, j_values, k_values


def _hover_text_for_region(region: BrainRegion) -> str:
    if region.name in SLIDE_7_NNOS_SGC_NAMES:
        return NO_NNOS_SGC_HOVER_TEXT
    if region.name in SLIDE_7_NOS_NO_NAMES:
        return NO_NOS_NO_HOVER_TEXT
    if region.name in SLIDE_7_INFLAMMATION_NAMES:
        return NO_INFLAMMATION_HOVER_TEXT
    if region.name in SLIDE_7_NMDA_NAMES:
        return NO_NMDA_HOVER_TEXT
    if region.name in SLIDE_7_CGMP_NAMES:
        return NO_CGMP_HOVER_TEXT
    if region.name in SLIDE_7_INHIBITOR_NAMES:
        return NO_INHIBITOR_HOVER_TEXT
    if region.name in SLIDE_7_LINALOOL_NAMES:
        return NO_LINALOOL_HOVER_TEXT
    if region.name in SLIDE_6_CYTOKINE_NAMES:
        return IMMUNE_CYTOKINE_HOVER_TEXT
    if region.name in SLIDE_6_ANTIDEPRESSANT_NAMES:
        return IMMUNE_ANTIDEPRESSANT_HOVER_TEXT
    if region.name in SLIDE_6_TNF_NAMES:
        return IMMUNE_TNF_HOVER_TEXT
    if region.name in SLIDE_6_ANTI_TNF_NAMES:
        return IMMUNE_ANTI_TNF_HOVER_TEXT
    if region.name in SLIDE_6_LPS_NAMES:
        return IMMUNE_LPS_HOVER_TEXT
    if region.name in SLIDE_6_CB1_NAMES:
        return IMMUNE_CB1_HOVER_TEXT
    if region.name in SLIDE_5_NMDA_SYSTEM_NAMES:
        return NMDA_SYSTEM_HOVER_TEXT
    if region.name in SLIDE_5_ANTAGONIST_NAMES:
        return NMDA_ANTAGONIST_HOVER_TEXT
    if region.name in SLIDE_5_DISINHIBITION_NAMES:
        return NMDA_DISINHIBITION_HOVER_TEXT
    if region.name in SLIDE_5_BDNF_NAMES:
        return NMDA_BDNF_HOVER_TEXT
    if region.name in SLIDE_5_NATURAL_NAMES:
        return NMDA_NATURAL_HOVER_TEXT
    if region.name in SLIDE_5_DOCKING_NAMES:
        return NMDA_DOCKING_HOVER_TEXT
    if region.name in SLIDE_4_TRANSPORTER_NAMES:
        return SERT_NET_HOVER_TEXT
    if region.name in SLIDE_4_ION_NAMES:
        return SERT_NET_ION_HOVER_TEXT
    if region.name in SLIDE_4_PROJECTION_NAMES:
        return SERT_NET_PROJECTION_HOVER_TEXT
    if region.name in SLIDE_4_BDNF_NAMES:
        return SERT_NET_BDNF_HOVER_TEXT
    if region.name in SLIDE_4_BINDING_NAMES:
        return SERT_NET_BINDING_HOVER_TEXT
    if region.name in SLIDE_3_NA_SYSTEM_NAMES:
        return NA_SYSTEM_HOVER_TEXT
    if region.name in SLIDE_3_AUTO_NAMES:
        return ALPHA2_AUTO_HOVER_TEXT
    if region.name in SLIDE_3_HETERO_NAMES:
        return ALPHA2_HETERO_HOVER_TEXT
    if region.name in SLIDE_3_POST_NAMES:
        return ALPHA2_POST_HOVER_TEXT
    if region.name in SLIDE_3_SUBTYPE_NAMES:
        return ALPHA2_SUBTYPE_HOVER_TEXT
    if region.name in SLIDE_2_MESOLIMBIC_NAMES:
        return MESOLIMBIC_DOPAMINE_HOVER_TEXT
    if region.name in SLIDE_2_D1_NAMES:
        return D1_HOVER_TEXT
    if region.name in SLIDE_2_D2_NAMES:
        return D2_HOVER_TEXT
    if region.name in SLIDE_2_D1_D2_NAMES:
        return D1_D2_HOVER_TEXT
    if region.name in SLIDE_2_BDNF_NAMES:
        return BDNF_HOVER_TEXT
    if region.name in SLIDE_1_RAPHE_NAMES:
        return RAPHE_HOVER_TEXT
    if region.name in SLIDE_1_LIMBIC_NAMES or region.name in SLIDE_1_CORTICAL_NAMES:
        return HETERO_HOVER_TEXT
    return region.function


def _region_mesh(region: BrainRegion, hemisphere: Hemisphere | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], list[int]]:
    if region.kind == "cortex":
        if hemisphere is None:
            raise ValueError(f"Região cortical sem hemisfério: {region.name}")
        return _cortex_mesh(region, hemisphere)
    if region.kind == "ellipsoid":
        return _ellipsoid_mesh(region, hemisphere)
    if region.kind == "tube":
        if hemisphere is None:
            raise ValueError(f"Tubo sem hemisfério: {region.name}")
        return _tube_mesh(region, hemisphere)
    return _cylinder_mesh(region, hemisphere)


def _trace_for_region(
    region: BrainRegion,
    *,
    hemisphere: Hemisphere | None,
    opacity: float,
    show_legend: bool,
    legend_group: str,
) -> go.Mesh3d:
    x_values, y_values, z_values, i_values, j_values, k_values = _region_mesh(region, hemisphere)
    hemisphere_label = ""
    if hemisphere == "left":
        hemisphere_label = "<br>Hemisfério esquerdo"
    elif hemisphere == "right":
        hemisphere_label = "<br>Hemisfério direito"
    hover_text = _hover_text_for_region(region)

    return go.Mesh3d(
        x=x_values,
        y=y_values,
        z=z_values,
        i=i_values,
        j=j_values,
        k=k_values,
        name=region.name,
        color=region.color,
        opacity=opacity,
        flatshading=False,
        hovertemplate=(
            f"<b>{region.name}</b>"
            f"{hemisphere_label}"
            f"<br>{hover_text}"
            "<extra></extra>"
        ),
        hoverlabel={"align": "left"},
        legendgroup=legend_group,
        showlegend=show_legend,
        lighting={"ambient": 0.58, "diffuse": 0.78, "specular": 0.22, "roughness": 0.75},
        lightposition={"x": 2, "y": 2.5, "z": 3},
    )


def _allowed_hemispheres(region: BrainRegion, hemisphere_choice: str) -> tuple[Hemisphere | None, ...]:
    if region.kind not in ("cortex", "tube") and region.pair_offset is None:
        return (None,)
    if not region.hemispheres:
        return (None,)
    if hemisphere_choice == "Esquerdo":
        requested: tuple[Hemisphere, ...] = ("left",)
    elif hemisphere_choice == "Direito":
        requested = ("right",)
    else:
        requested = ("left", "right")

    return tuple(hemisphere for hemisphere in requested if hemisphere in region.hemispheres)


def _opacity_for_region(region: BrainRegion, selected_names: set[str], *, context: bool) -> float:
    if context:
        if region.name == "Tronco encefálico":
            return 0.20
        return 0.10
    if region.name in PRESENTATION_REGION_NAMES:
        if not selected_names:
            return 0.16
        if region.name in selected_names:
            return 0.92
        return 0.08
    if not selected_names:
        return 0.82 if region.kind == "cortex" else 0.76
    if region.name in selected_names:
        return 0.92
    return 0.12


def build_brain_figure(
    visible_regions: list[BrainRegion],
    context_regions: list[BrainRegion],
    selected_names: set[str],
    hemisphere_choice: str,
) -> go.Figure:
    fig = go.Figure()

    for region in context_regions:
        for index, hemisphere in enumerate(_allowed_hemispheres(region, hemisphere_choice)):
            fig.add_trace(
                _trace_for_region(
                    region,
                    hemisphere=hemisphere,
                    opacity=_opacity_for_region(region, selected_names, context=True),
                    show_legend=index == 0,
                    legend_group=f"context-{region.name}",
                )
            )

    for region in visible_regions:
        for index, hemisphere in enumerate(_allowed_hemispheres(region, hemisphere_choice)):
            fig.add_trace(
                _trace_for_region(
                    region,
                    hemisphere=hemisphere,
                    opacity=_opacity_for_region(region, selected_names, context=False),
                    show_legend=index == 0,
                    legend_group=region.name,
                )
            )

    fig.update_layout(
        height=690,
        margin={"l": 0, "r": 0, "t": 12, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.01,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 11},
        },
        scene={
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.65, "y": 1.65, "z": 1.05}},
            "xaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "yaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "zaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    return fig


def regions_for_layer(layer: str) -> list[BrainRegion]:
    if layer == "Tudo":
        return list(REGIONS)
    return [region for region in REGIONS if region.category == layer]


def regions_for_names(region_names: set[str]) -> list[BrainRegion]:
    return [region for region in REGIONS if region.name in region_names]


def region_names_for_slide_1_groups(group_names: list[str]) -> set[str]:
    region_names: set[str] = set()
    for group_name in group_names:
        region_names.update(SLIDE_1_GROUPS[group_name])
    return region_names


def region_names_for_slide_2_groups(group_names: list[str]) -> set[str]:
    region_names: set[str] = set()
    for group_name in group_names:
        region_names.update(SLIDE_2_GROUPS[group_name])
    return region_names


def region_names_for_slide_3_groups(group_names: list[str]) -> set[str]:
    region_names: set[str] = set()
    for group_name in group_names:
        region_names.update(SLIDE_3_GROUPS[group_name])
    return region_names


def region_names_for_slide_4_groups(group_names: list[str]) -> set[str]:
    region_names: set[str] = set()
    for group_name in group_names:
        region_names.update(SLIDE_4_GROUPS[group_name])
    return region_names


def region_names_for_slide_5_groups(group_names: list[str]) -> set[str]:
    region_names: set[str] = set()
    for group_name in group_names:
        region_names.update(SLIDE_5_GROUPS[group_name])
    return region_names


def region_names_for_slide_6_groups(group_names: list[str]) -> set[str]:
    region_names: set[str] = set()
    for group_name in group_names:
        region_names.update(SLIDE_6_GROUPS[group_name])
    return region_names


def region_names_for_slide_7_groups(group_names: list[str]) -> set[str]:
    region_names: set[str] = set()
    for group_name in group_names:
        region_names.update(SLIDE_7_GROUPS[group_name])
    return region_names


def render_slide_notes(slide: str) -> None:
    if slide.startswith("1.1.1"):
        st.subheader("Rota monoaminérgica")
        st.caption("Receptores 5-hidroxitriptamina 1A, 2A e 2C | PDB: 6A93, 6BQH")
        st.write(
            "A serotonina, ou 5-hidroxitriptamina (5-HT), é um neurotransmissor da classe das monoaminas. "
            "Ela é sintetizada por neurônios do núcleo da rafe, no mesencéfalo, a partir do aminoácido L-triptofano. "
            "No texto do artigo, a 5-HT é relacionada à depressão e à ansiedade por participar de circuitos de "
            "emoções, memória, sono e regulação térmica."
        )
        st.write(
            "A família de receptores 5-HT é classificada em 14 subtipos segundo funções farmacológicas e "
            "mecanismos de transdução, incluindo 5-HT2A, 5-HT2B e 5-HT2C. A isoforma 5-HT1A é uma das mais "
            "abundantes no cérebro e aparece em duas populações distintas."
        )
        st.markdown("**Autorreceptores 5-HT1A - núcleos da rafe**")
        st.write(
            "São autorreceptores somatodendríticos localizados nos núcleos da rafe. Segundo o artigo, "
            "os receptores 5-HT são metabotrópicos, acoplados a proteínas Gi, levando à redução de cAMP, "
            "inativação de canais de cálcio e ativação de canais de potássio, culminando em inibição da "
            "atividade neuronal. Os autorreceptores reduzem as taxas de disparo por feedback negativo, "
            "limitando a liberação."
        )
        st.caption("No modelo 3D, os núcleos da rafe aparecem dentro do tronco encefálico, usado apenas como referência anatômica.")
        st.markdown("**Heterorreceptores 5-HT1A - áreas límbicas e corticais**")
        st.write(
            "São heterorreceptores pós-sinápticos encontrados em áreas centrais diversas inervadas por "
            "projeções serotoninérgicas, em neurônios piramidais e interneurônios GABAérgicos. Nesta "
            "apresentação, essas áreas foram agrupadas visualmente como área límbica e zonas corticais, "
            "sem acrescentar mecanismos além do trecho do artigo."
        )
        st.markdown("**Agrupamento visual**")
        st.write(
            "Área límbica: hipocampo, amígdala, hipotálamo, septo lateral, núcleo accumbens, "
            "tálamo anterior, corpos mamilares, fórnix, córtex cingulado, córtex parahipocampal "
            "e córtex entorrinal."
        )
        st.write(
            "Zonas corticais: córtex pré-frontal medial, orbitofrontal, cingulado, insular "
            "e temporal/entorrinal."
        )
        st.caption(
            "O tópico 1.1.1 destaca dois papéis conceituais do 5-HT1A: autorreceptores na rafe e "
            "heterorreceptores em áreas límbicas/corticais."
        )
        return

    st.info("Estrutura reservada para a próxima teoria. O visualizador 3D já está preparado para receber novos recortes.")


def render_region_notes(selected_names: set[str], visible_regions: list[BrainRegion]) -> None:
    selected_regions = [region for region in visible_regions if region.name in selected_names]

    if not selected_regions:
        st.info("Selecione uma ou mais áreas para ver o resumo didático.")
        return

    st.subheader("Resumo")
    for region in selected_regions:
        st.markdown(f"**{region.name}**")
        st.write(region.function)
        st.caption(region.note)


st.title("Mecanismos Neurais da Ansiedade e Depressão")
st.caption(
    "Visualização 3D esquemática para explorar teorias e circuitos cerebrais. "
    "As formas são simplificadas e não substituem um atlas anatômico clínico. "
)

control_column, viewer_column = st.columns([0.18, 0.82], gap="large")

with control_column:
    slide = st.selectbox(
        "Slide",
        SLIDE_LABELS,
        index=0,
    )

    hemisphere_choice = st.radio(
        "Hemisfério",
        ["Ambos", "Esquerdo", "Direito"],
        horizontal=False,
    )

    if slide.startswith("1.1.1"):
        selected_groups = st.multiselect(
            "Estruturas",
            list(SLIDE_1_GROUPS.keys()),
            default=[],
        )
        visible_regions = regions_for_names(
            SLIDE_1_RAPHE_NAMES | SLIDE_1_LIMBIC_NAMES | SLIDE_1_CORTICAL_NAMES
        )
        selected_names = region_names_for_slide_1_groups(selected_groups)
        context_regions = regions_for_names(SLIDE_1_CONTEXT_NAMES)
    elif slide.startswith("1.1.5"):
        selected_groups = st.multiselect(
            "Estruturas",
            list(SLIDE_2_GROUPS.keys()),
            default=[],
        )
        visible_regions = regions_for_names(
            SLIDE_2_MESOLIMBIC_NAMES
            | SLIDE_2_D1_NAMES
            | SLIDE_2_D2_NAMES
            | SLIDE_2_D1_D2_NAMES
            | SLIDE_2_BDNF_NAMES
        )
        selected_names = region_names_for_slide_2_groups(selected_groups)
        context_regions = regions_for_names(SLIDE_2_CONTEXT_NAMES)
    elif slide.startswith("1.1.6"):
        selected_groups = st.multiselect(
            "Estruturas",
            list(SLIDE_3_GROUPS.keys()),
            default=[],
        )
        visible_regions = regions_for_names(
            SLIDE_3_NA_SYSTEM_NAMES
            | SLIDE_3_AUTO_NAMES
            | SLIDE_3_HETERO_NAMES
            | SLIDE_3_POST_NAMES
            | SLIDE_3_SUBTYPE_NAMES
        )
        selected_names = region_names_for_slide_3_groups(selected_groups)
        context_regions = regions_for_names(SLIDE_3_CONTEXT_NAMES)
    elif slide.startswith("1.1.7"):
        selected_groups = st.multiselect(
            "Estruturas",
            list(SLIDE_4_GROUPS.keys()),
            default=[],
        )
        visible_regions = regions_for_names(
            SLIDE_4_TRANSPORTER_NAMES
            | SLIDE_4_ION_NAMES
            | SLIDE_4_PROJECTION_NAMES
            | SLIDE_4_BDNF_NAMES
            | SLIDE_4_BINDING_NAMES
        )
        selected_names = region_names_for_slide_4_groups(selected_groups)
        context_regions = regions_for_names(SLIDE_4_CONTEXT_NAMES)
    elif slide.startswith("1.1.4"):
        selected_groups = st.multiselect(
            "Estruturas",
            list(SLIDE_5_GROUPS.keys()),
            default=[],
        )
        visible_regions = regions_for_names(
            SLIDE_5_NMDA_SYSTEM_NAMES
            | SLIDE_5_ANTAGONIST_NAMES
            | SLIDE_5_DISINHIBITION_NAMES
            | SLIDE_5_BDNF_NAMES
            | SLIDE_5_NATURAL_NAMES
            | SLIDE_5_DOCKING_NAMES
        )
        selected_names = region_names_for_slide_5_groups(selected_groups)
        context_regions = regions_for_names(SLIDE_5_CONTEXT_NAMES)
    elif slide.startswith("1.1.3"):
        selected_groups = st.multiselect(
            "Estruturas",
            list(SLIDE_6_GROUPS.keys()),
            default=[],
        )
        visible_regions = regions_for_names(
            SLIDE_6_CYTOKINE_NAMES
            | SLIDE_6_ANTIDEPRESSANT_NAMES
            | SLIDE_6_TNF_NAMES
            | SLIDE_6_ANTI_TNF_NAMES
            | SLIDE_6_LPS_NAMES
            | SLIDE_6_CB1_NAMES
        )
        selected_names = region_names_for_slide_6_groups(selected_groups)
        context_regions = regions_for_names(SLIDE_6_CONTEXT_NAMES)
    elif slide.startswith("1.1.2"):
        selected_groups = st.multiselect(
            "Estruturas",
            list(SLIDE_7_GROUPS.keys()),
            default=[],
        )
        visible_regions = regions_for_names(
            SLIDE_7_NNOS_SGC_NAMES
            | SLIDE_7_NOS_NO_NAMES
            | SLIDE_7_INFLAMMATION_NAMES
            | SLIDE_7_NMDA_NAMES
            | SLIDE_7_CGMP_NAMES
            | SLIDE_7_INHIBITOR_NAMES
            | SLIDE_7_LINALOOL_NAMES
        )
        selected_names = region_names_for_slide_7_groups(selected_groups)
        context_regions = regions_for_names(SLIDE_7_CONTEXT_NAMES)
    else:
        visible_regions = []
        selected_names = set()
        context_regions = []
        st.info("Slide em preparo.")

with viewer_column:
    figure = build_brain_figure(
        visible_regions=visible_regions,
        context_regions=context_regions,
        selected_names=selected_names,
        hemisphere_choice=hemisphere_choice,
    )
    st.plotly_chart(figure, width="stretch", theme=None)
