# Pesquisa de Mestrado — Visão Computacional e Rastreamento do Olhar (IME-USP)

![Python](https://img.shields.io/badge/Python-3-blue?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?logo=cplusplus&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4-5C3EE8?logo=opencv&logoColor=white)
![Intel RealSense](https://img.shields.io/badge/Intel-RealSense-0071C5?logo=intel&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ORB--SLAM2-2496ED?logo=docker&logoColor=white)

Código da pesquisa de mestrado em Ciência da Computação no **Instituto de
Matemática e Estatística da Universidade de São Paulo (IME-USP)**, com bolsa
**CAPES**, sob orientação do **Prof. Dr. Carlos Hitoshi Morimoto**.

A pesquisa trata do **rastreamento do olhar em ambientes naturais**,
combinando câmeras RGB-D, visão computacional em tempo real e um eye tracker
remoto com **iluminação estroboscópica diferencial** (LEDs infravermelhos
sincronizados com a câmera), para segmentar a pupila e o reflexo corneano
(*glint*) e estimar para onde o usuário está olhando.

**Publicação:** *"A Cascade Image-Processing Pipeline for Real-Time Pupil and
Glint Segmentation in Remote Stroboscopic Differential Lighting"* —
submetido ao **ETRA 2027** (em revisão por pares).

## Mapa da pesquisa

Este repositório reúne a **fase inicial** (captura RGB-D, calibração e
extração de características) e o **sistema final da dissertação**: rastreador
em tempo real, pipeline de segmentação, ground truth e firmware. As versões
anteriores de cada etapa estão em repositórios próprios:

| Repositório | Conteúdo |
|---|---|
| **Pesquisa-USP** (este) | Captura RGB-D, calibração e extração de características (fase inicial); rastreador ocular em tempo real, pipeline de segmentação de pupila e glints, ground truth com amostragem estratificada e firmware do estroboscópio (sistema final) |
| [Differential-Stroboscopic-Eye-Tracker](https://github.com/hard747/Differential-Stroboscopic-Eye-Tracker) | Eye tracker com luz estroboscópica diferencial: sincronização câmera–LEDs, detecção do centro da pupila e dos glints em vídeo |
| [Segmentacao-da-Pupila](https://github.com/hard747/Segmentacao-da-Pupila) | Pipeline de segmentação da pupila e dos glints, calibração do olhar e predição do ponto de fixação, com diagramas UML da arquitetura |
| [Pupil-Segmentation-from-SDL](https://github.com/hard747/Pupil-Segmentation-from-SDL) | Implementação em C++ (CMake) da detecção de pupila e glint, integrando algoritmos clássicos da literatura (ElSe, ExCuSe, Świrski) para comparação |
| [control-luces-estroboscopicas](https://github.com/hard747/control-luces-estroboscopicas) | Firmware Arduino para o controle dos LEDs infravermelhos sincronizado com o obturador da câmera, e scripts de captura |

## Estrutura deste repositório

```
Pesquisa-USP/
├── 01-captura-rgbd/
│   ├── realsense/        # Intel RealSense F200/SR300 (Python + C++)
│   │   └── cpp/
│   ├── kinect/           # Kinect v1 via libfreenect (C++)
│   └── ps-eye/           # Câmera PlayStation Eye: foto, sequência e vídeo
├── 02-calibracao-camera/ # Calibração intrínseca com tabuleiro de xadrez
│   └── imagens/
├── 03-processamento-imagens/  # ORB, Harris, Shi-Tomasi, Canny, DoG, painéis RGB-D
├── 04-interfaces-graficas/    # Captura e processamento com PyQt5 e Tkinter
├── 05-slam/                   # Ambiente Docker com ORB-SLAM2 + RealSense (GPU)
├── 06-rastreador-tempo-real/  # Sistema final: hardware, detector, calibração e UI em processos
├── 07-segmentacao-pupila-glint/  # Detector v4 (Kalman, subpixel) e pipelines dos 15 participantes
├── 08-ground-truth-e-avaliacao/  # Amostragem estratificada, anotação manual e avaliação
├── 09-firmware-arduino/       # Sincronização câmera–LEDs infravermelhos (estroboscópio)
├── 10-coleta-de-dados/        # Aplicação de coleta: fixações e perseguições suaves
├── dados/
│   ├── exemplos/         # Imagens RGB e de profundidade de exemplo
│   ├── sequencia_ps_eye/ # 50 frames capturados com a PS Eye
│   └── videos/           # Vídeos RGB, em cinza, de profundidade e processado
```

## Módulos

### 01 · Captura RGB-D

- **Intel RealSense:** captura de cor e profundidade com `pyrealsense2`,
  inspeção dos sensores e perfis de stream do dispositivo, e versões em C++
  com `librealsense2`. Inclui a versão com a API legada `librealsense` 1.x,
  necessária para os modelos F200 e SR300.
- **Kinect v1:** captura e visualização de RGB e profundidade com
  `libfreenect` e OpenCV.
- **PlayStation Eye:** identificação da câmera pelo ID USB (`1415:2000`) via
  `pyudev`, captura de foto, de sequências de frames e de vídeo.

### 02 · Calibração de câmera
Calibração intrínseca pelo método de Zhang (tabuleiro de xadrez de 11×8 casas, 10×7 cantos internos): estima
a matriz da câmera (fx, fy, cx, cy) e os coeficientes de distorção (k1, k2,
p1, p2, k3), e exporta um YAML no formato de configuração do ORB-SLAM2.

### 03 · Processamento de imagens
- Detectores de pontos de interesse: **ORB**, **Harris** (implementado passo a
  passo, a partir dos gradientes de Sobel e da matriz de autocorrelação) e
  **Shi-Tomasi**.
- **Canny** sobre mapas de profundidade e **Diferença de Gaussianas (DoG)**.
- Painéis sincronizados RGB + profundidade + ORB + Canny, para imagens e vídeo.
- Extração do canal de profundidade de imagens RGB-D com **Open3D**.

### 04 · Interfaces gráficas
Aplicações em **PyQt5** e **Tkinter** para captura em tempo real, gravação
simultânea de vídeo colorido e em cinza (com `QThread`) e processamento
posterior dos vídeos RGB e de profundidade.

### 05 · SLAM
`docker-compose.yml` com um ambiente **ORB-SLAM2** pronto para câmeras
RealSense, com acesso à GPU NVIDIA e à interface gráfica (X11). É a base para
localizar a câmera de cena no ambiente 3D e projetar o olhar sobre ele.

```bash
cd 05-slam
xhost +local:docker
docker compose up -d
docker exec -it orb-slam-container bash
```

### 06 · Rastreador ocular em tempo real
Sistema final da dissertação, organizado em **4 processos** coordenados por um
orquestrador (`rt_orchestrator.py`):
- **Hardware** (`rt_hardware.py`): captura com a PS3 Eye, controle do
  estroboscópio via Arduino e sincronização ativa pelo método da "faixa preta",
  com monitoramento de falhas de sincronia.
- **Detector** (`rt_detector.py`): classificação de quadros claros/escuros,
  segmentação da pupila por watershed, seleção dos glints e vetor
  pupila–glint com filtro EMA.
- **Engine** (`rt_engine.py`): calibração por regressão polinomial Ridge e
  estimação do olhar na tela, com relatório final (CSV, estatísticas e heatmap).
- **UI** (`rt_ui.py`): interface de calibração e visualização.

A pasta `analise_perda_quadros/` traz as medições de perda de quadros usadas
para validar a sincronização.

### 07 · Segmentação de pupila e glints
- `pupil_glint_detector_v4.py`: detector com **filtro de Kalman** (estado
  `[x, y, r, vx, vy]`), busca anular a partir da predição, *gate* de
  Mahalanobis para rejeitar falhas e distinção entre sacada e outlier.
- `pipeline_15_participantes.py` e `pipeline_completo.py`: processamento de ponta
  a ponta por participante — segmentação, calibração (Ridge polinomial grau 2,
  9 alvos de treino e 26 de teste), filtro EMA e exportação dos resultados.
- `segmentacao_pupila_glint.py`: versão interativa com validação de piscadas e
  gravação de vídeo de saída.

### 08 · Ground truth e avaliação
Ferramentas para medir a **acurácia real** do detector contra anotações
manuais, e não apenas contra verificações internas:
- `sample_frames.py`: amostragem **estratificada** (participante × sessão ×
  quadro claro/escuro), com limiar por sessão (Otsu) e exclusão de piscadas.
- `refine_seeds.py`: centraliza o recorte de anotação por diferença temporal.
- `annotate.py`: anotação com zoom e precisão subpixel do centro da pupila e
  dos glints.
- `eval_v4_large_gt.py`: avaliação do detector v4 contra o ground truth,
  separada por quadros claros e escuros.

### 09 · Firmware Arduino
- `strobe/`: alternância dos LEDs infravermelhos on-axis/off-axis sincronizada
  com o obturador da câmera por interrupções.
- `clock/`: gerador de clock para sincronização externa.
- `teste_strobo/`: teste simples de alternância dos LEDs.

### 10 · Coleta de dados
`experimento_fixacoes_perseguicoes.py` é a aplicação usada nos experimentos com
os participantes. Ela apresenta, em tela cheia (1920×1080), alvos em uma grade
de 7×5 pontos para **fixações** e alvos em movimento para **perseguições
suaves**, enquanto a câmera PS3 Eye grava os olhos. Para cada sessão, grava os
quadros e um CSV com a posição do alvo em cada instante, usados depois para
calibrar e avaliar o rastreador.

## O que não está publicado

- **Dados dos participantes:** as imagens dos olhos, os vídeos e as anotações
  coletados nos experimentos não são publicados, conforme o Termo de
  Consentimento Livre e Esclarecido (TCLE) e a proteção de dados pessoais. No
  código, os participantes aparecem apenas com códigos anônimos (P01, P02, …).
- **Material do laboratório:** parte do código, das imagens e dos resultados
  foi desenvolvida em projetos do laboratório do IME-USP, como os experimentos
  de reconstrução 3D (COLMAP, NeRF e 3D Gaussian Splatting), SLAM e ROS. Esse
  material pertence ao laboratório e não pode ser publicado.

## Como executar

```bash
pip install -r requirements.txt

# Calibração com as imagens incluídas (gera calibracion.yaml)
python 02-calibracao-camera/calibracao_tabuleiro.py

# Detecção de cantos de Harris na imagem de exemplo
python 03-processamento-imagens/harris_cantos.py
```

Os scripts de `01-captura-rgbd` e `04-interfaces-graficas` precisam das
câmeras conectadas e foram desenvolvidos em **Linux (Ubuntu)**. Os programas
em C++ dependem de OpenCV, `librealsense2`/`librealsense` ou `libfreenect`.

## Hardware utilizado

| Dispositivo | Uso |
|---|---|
| Intel RealSense F200 / SR300 | Imagens RGB e de profundidade |
| Microsoft Kinect v1 | Imagens RGB e de profundidade |
| Sony PlayStation Eye | Câmera de alta taxa de quadros para captura de vídeo |
| Arduino + LEDs infravermelhos | Iluminação estroboscópica sincronizada ([repositório](https://github.com/hard747/control-luces-estroboscopicas)) |

## Autor

**Harre Bams Ayma Aranda** — Mestre em Ciência da Computação (IME-USP),
Bacharel em Matemática (UNMSM).

[LinkedIn](https://www.linkedin.com/in/harre-bams-ayma-aranda-a71a12236/) ·
[GitHub](https://github.com/hard747) ·
harre@alumni.usp.br
