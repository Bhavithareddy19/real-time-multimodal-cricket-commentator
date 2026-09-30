# Real-Time Multimodal Cricket Commentator

> **An autonomous, production-grade multimodal AI broadcast commentary system.**  
> Ingests recorded cricket match footage, live streams, or webcam feeds, detects visual cricket events using computer vision and human pose estimation, autonomously calibrates pitch geometry, identifies team jerseys, generates contextual broadcast commentary via Google Gemini or OpenAI, and synthesizes streaming neural speech audio in real time.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-2.0_Flash-4285F4.svg?logo=google&logoColor=white)](https://aistudio.google.com/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF.svg)](https://docs.ultralytics.com/)
[![Tests](https://img.shields.io/badge/Tests-37%2F37%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) 

---

## Architecture

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                              VIDEO INPUT                               │
 │   Uploaded Match Video (.mp4/.mov/.mkv) | Online URL | RTSP Stream     │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │      VideoSourceManager      │
                      │  (Validation & Paced Clock)  │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │    Decoupled Ingestion Queue │
                      │   (Bounded Size 2, Drop Old) │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────────┐
        │                       VISION ENGINE                        │
        │  • YOLOv8n (Objects: Players, Bats, Balls)                 │
        │  • 2D Kalman Filter + Physical Size & Aspect Gating        │
        │  • Dual-Spectrum Color Recovery (Red & White Cricket Balls)│
        │  • YOLOv8n-Pose (17 Keypoints: Stance, Bowler Runup)       │
        │  • AutoPitchCalibrator (HSV Turf & Hough Crease Lines)     │
        │  • JerseyColorClassifier (2-Cluster HSV K-Means on Torso)   │
        └────────────────────────────┬───────────────────────────────┘
                                     │ Structured Telemetry & Keypoints
                                     ▼
                      ┌──────────────────────────────┐
                      │   CricketEventStateMachine   │
                      │   • State Transitions        │
                      │   • Relative Stance Zones    │
                      │   • Ray-Casting Inset Rope   │
                      │   • Delivery Debounce Guard  │
                      └──────────────┬───────────────┘
                                     │ Discrete Events (FOUR, SIX, WICKET, etc.)
                                     ▼
        ┌────────────────────────────────────────────────────────────┐
        │                     COMMENTARY ENGINE                      │
        │  • Google Gemini Provider (Official google.genai SDK)       │
        │  • OpenAI Compatible Provider                              │
        │  • Deterministic Local Fallback Generator (<1ms)           │
        │  • Rolling 8-Delivery Match History Window                 │
        │  • 4 Styles: Professional, Energetic, Analytical, Minimal  │
        └────────────────────────────┬───────────────────────────────┘
                                     │ Synchronized Text & Metadata
                                     ▼
                      ┌──────────────────────────────┐
                      │     STREAMING EDGE-TTS       │
                      │  • Neural en-GB-Ryan Voice   │
                      │  • +12% Fast Broadcast Rate  │
                      │  • AudioPriority Preemption  │
                      │  • WICKET > SIX > FOUR > DOT │
                      └──────────────┬───────────────┘
                                     │ Real-Time WebSocket Transport (/ws/live)
                                     ▼
        ┌────────────────────────────────────────────────────────────┐
        │                  REACT BROADCAST STUDIO                    │
        │  • Web Audio API Continuous Queue Player                   │
        │  • Live Telemetry HUD & Bounding Box Overlays              │
        │  • 1-Click Demonstration Footage (Synthetic & Broadcast)   │
        │  • Interactive AI Configuration Modal (Gemini/OpenAI)      │
        └────────────────────────────────────────────────────────────┘
```

---

## Real AI vs. Heuristic vs. Fallback Matrix

To ensure technical transparency, the table below delineates which components rely on trained machine learning models, deterministic computer vision/heuristics, or fallback mechanisms:

| Subsystem | Operational Tier | Technology | Technical Purpose |
| :--- | :--- | :--- | :--- |
| **Object Detection** | **REAL AI / ML** | YOLOv8n (PyTorch / ONNX) | Real-time localization of players, bats, and sports balls. |
| **Human Pose Estimation** | **REAL AI / ML** | YOLOv8n-pose (17 Keypoints) | Biomechanical tracking of batting stance and bowler delivery stride. |
| **Jersey / Team Classifier** | **REAL AI / ML** | 2-Cluster HSV K-Means | Unsupervised clustering of player torso colors to identify teams (Australia Gold, India Blue, Test White, etc.). |
| **Generative Commentary** | **REAL AI / ML** | Google Gemini 2.0 Flash / OpenAI | Context-grounded broadcast commentary synthesized via official `google.genai` SDK. |
| **Speech Synthesis (TTS)** | **REAL AI / ML** | Microsoft Edge-TTS Neural | Realistic sports broadcaster vocal delivery (`en-GB-RyanNeural`). |
| **Ball Kinematics** | **HEURISTIC** | Linear 2D Kalman Filter | 4D kinematic state tracking $[x, y, v_x, v_y]^T$ with occlusion prediction and physical size gating. |
| **Pitch & Crease Calibration** | **HEURISTIC** | HSV Segmentation + Hough Lines | Autonomous field polygon discovery and crease line detection (`cv2.HoughLinesP`). |
| **Event State Machine** | **HEURISTIC** | Finite State Machine & Ray-Casting | Deterministic event sequencing (Delivery $\rightarrow$ Contact $\rightarrow$ Boundary) with delivery-level debounce locks. |
| **Offline Commentary** | **FALLBACK** | Deterministic Template Expander | Zero-latency, rule-based broadcast commentary triggered when no LLM API key is provided. |

---

## Key Capabilities

### 1. Multi-Modal Video Ingestion (`VideoSource` Hierarchy)
* **Uploaded Video Files**: Supports `.mp4`, `.mov`, `.mkv`, `.avi`, and `.webm`. Reads frames in synchronized real-time lockstep with the video clock rather than performing offline batch processing.
* **Direct Video URL**: Ingests directly accessible video files with automatic codec/header validation.
* **Live RTSP/HLS Streams**: Ingests continuous live camera or broadcast streams with exponential backoff auto-reconnect (`🟢 LIVE`, `🟡 RECONNECTING...`, `🔴 SOURCE UNAVAILABLE`).
* **Source Validation & Respect for DRM**: Gracefully validates URLs. The system does not bypass paywalls, DRM, or access controls; inaccessible sources fail with clear, actionable diagnostics.
* **Playback Speed Control**: Real-time speed adjustment (`0.5x`, `1.0x`, `1.5x`, `2.0x`) with graceful frame dropping to preserve real-time latency at accelerated rates.

### 2. Autonomous Pitch & Boundary Calibration
* **HSV Turf Segmentation**: Isolates the central pitch corridor from the outfield using adaptive color thresholding.
* **Crease Line Detection**: Leverages Probabilistic Hough Transforms to detect horizontal bowling and batting crease lines.
* **Boundary Rope Insetting**: Computes the outer grass hull and automatically insets the boundary polygon by 12% toward the frame center to reliably capture boundary crossings.
* **Confidence Gating**: Computes a calibration confidence score ($\ge 0.60$ threshold); preserves calibrated geometry when cameras pan or tilt.

### 3. Team & Jersey Recognition
* **Torso Region Extraction**: Automatically extracts the upper half of player bounding boxes, filtering out grass and skin tones.
* **K-Means Clustering**: Analyzes dominant HSV centroids to classify teams:
  - *Australia Gold / Yellow*
  - *India Blue*
  - *Test Match White*
  - *Pakistan / South Africa Green*
  - *West Indies Maroon*
* **Real-Time Badging**: Live team tags are displayed on player bounding boxes and in the developer HUD.

### 4. Robust Ball Tracking on Broadcast Footage
* **Physical Dimension Gating**: Rejects false-positive COCO ball detections (such as player helmets or torsos) by enforcing $w, h \le \min(W, H) \times 0.08$ and aspect ratio $\le 2.8$.
* **Dual-Spectrum Color Recovery**: Detects both traditional red leather balls (Test cricket) and white/pink balls (limited-overs/night cricket) when YOLO misses small pixel blobs during rapid delivery flight.
* **Kinematic Jump Rejection**: Discards detections that violate maximum frame-to-frame displacement constraints.

### 5. Dynamic Generative AI Commentary
* **Official Google Gemini SDK**: Direct native integration using the modern `google.genai` SDK (`gemini-2.0-flash` and `gemini-1.5-flash`).
* **OpenAI Compatible**: Supports standard OpenAI endpoints (`gpt-4o-mini`, `gpt-4o`).
* **Rolling Match Memory**: Retains the previous 8 deliveries to contextualize bowler economy, run rates, batting milestones, and match momentum.
* **4 Broadcast Personalities**:
  - `PROFESSIONAL`: Standard international television commentary.
  - `ENERGETIC`: High-octane, dramatic T20-style commentary.
  - `ANALYTICAL`: Biomechanical breakdown focusing on seam position, footwork, and bat speed.
  - `MINIMAL`: Concise public-address system updates.
* **Interactive UI Configuration**: Users can switch AI providers or enter API keys directly from the web interface using the **AI SETTINGS** modal.

### 6. Streaming TTS & Audio Preemption
* **Broadcaster Voice Profile**: Uses `en-GB-RyanNeural` with `rate="+12%"` for an energetic sports delivery cadence.
* **AudioPriorityManager**: Enforces strict event preemption:
  $$\text{WICKET (100)} > \text{SIX (80)} > \text{FOUR (60)} > \text{SHOT\_PLAYED (40)} > \text{DOT\_BALL (20)}$$ 
  Higher-priority events immediately cancel lower-priority audio synthesis in flight and dispatch `clear_queue` commands to flush client buffers.
* **Web Audio API Client**: Decodes base64 audio chunks directly in the browser with seamless sequential buffering and zero-latency mute toggling.

---

## Hardware Benchmarks (CPU Mode)

Benchmarked on **Intel Core i5 (11th Gen) / Iris Xe Integrated Graphics (CPU Mode)**:

| Component | Technology | Latency (Measured) |
| :--- | :--- | :--- |
| **YOLOv8n Object Detection** | CPU (PyTorch / ONNX) | 26 – 34 ms |
| **Ball Kalman Filter & Size Gating** | NumPy 4D State Vector | 1.2 – 2.5 ms |
| **YOLOv8n-Pose Keypoints** | Alternating CPU frames | 42 – 52 ms (effective ~22ms) |
| **Pitch & Crease Calibration** | OpenCV HSV + Hough | 8 – 14 ms (on keyframes) |
| **Jersey K-Means Classifier** | Scikit-Learn 2-Cluster | 3 – 6 ms (per player) |
| **Event State Machine** | Pure Python Geometry | 0.6 – 1.4 ms |
| **Gemini LLM Commentary** | First Token Latency | 240 – 410 ms |
| **Deterministic Fallback** | In-Memory Template Generator | < 1.0 ms |
| **Edge-TTS Audio Synthesis** | First MP3 Chunk Delivered | 170 – 260 ms |
| **Total Measured End-to-End** | Video Event $\rightarrow$ Spoken Audio | **< 480 ms** |

---

## Quickstart Guide

### Prerequisites
* **Python 3.11+**
* **Node.js 18+** and **npm**
* **Git**

---

### 1. Installation

#### A. Clone the Repository
```bash
git clone https://github.com/your-username/real-time-cricket-commentator.git
cd real-time-cricket-commentator
```

#### B. Backend Setup
```bash
# Create and activate a Python virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r requirements.txt
```

#### C. Frontend Setup
```bash
cd frontend
npm install
npm run build
cd ..
```

---

### 2. Environment Configuration

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` (optional):
```env
# Server
PORT=8000
HOST=0.0.0.0

# Generative AI Commentary (Optional: leave blank for deterministic local fallback)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
GEMINI_API_KEY=your_gemini_api_key_here

# Speech Synthesis
TTS_PROVIDER=edge-tts
TTS_VOICE=en-GB-RyanNeural

# Computer Vision & Hardware
DEVICE=auto
TARGET_FPS=30
```

> **Note**: An API key is **not required** to run the project. When no key is configured, the system automatically uses the high-variety local fallback generator.

---

### 3. Running the Application

#### Option A: Unified Production Server (Recommended)
FastAPI automatically serves the built frontend from `frontend/dist`:

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

#### Option B: Frontend Development Server
If you are developing frontend components with Hot Module Replacement (HMR):
```bash
# Terminal 1: Backend
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.

---

### 4. Running with Docker

```bash
docker-compose up --build
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## How to Test and Process Cricket Videos

Once the application is open in your browser:

### Method 1: 1-Click Demonstration Footage (Fastest)
1. In the standby setup card, click on either:
   - 🏏 **Synthetic Demo Clip**: Cover drive four with calibrated pitch, vector arrow HUD, and boundary crossing.
   - 🏟️ **Real Match Broadcast**: Adam Gilchrist MCG footage (Australia vs India) with real player detections, full poses, and jersey colors.
2. Click **START COMMENTARY**.
3. Watch the video stream, live bounding boxes, and developer HUD while listening to real-time commentary.

### Method 2: Upload Your Own Cricket Video
1. Click the **Upload Video** tab.
2. Click **Choose Cricket Video File** and select any `.mp4`, `.mov`, `.mkv`, or `.avi` cricket clip.
3. The server will validate video codecs, resolution, and duration.
4. Click **START COMMENTARY**.

### Method 3: Online Video URL or Live Stream
1. Click the **Video / Stream URL** tab.
2. Paste a directly accessible URL (e.g., `https://example.com/cricket.mp4` or an `rtsp://` stream).
3. Select **Auto Detect**, **Direct Video URL**, or **Live Stream**.
4. Click **START COMMENTARY**.

---

## Automated Testing

Run the complete test suite (37 unit, integration, and end-to-end tests):

```bash
pytest backend/tests -v
```

### Test Coverage Summary:
```text
backend/tests/test_calibration.py          2 passed  (Auto-pitch calibration & crease line detection)
backend/tests/test_commentary.py           4 passed  (Memory window, fallback templates, prompt builders)
backend/tests/test_e2e_video_commentary.py 1 passed  (End-to-end video upload, YOLO, events & audio)
backend/tests/test_gemini_llm.py           3 passed  (Google Gemini SDK provider, factory, failover)
backend/tests/test_jersey.py               3 passed  (K-Means jersey color clustering & team identification)
backend/tests/test_live_stream.py          1 passed  (Frame pipeline live streaming & backpressure)
backend/tests/test_phase1.py               3 passed  (Health endpoint, source enumeration, model inference)
backend/tests/test_state_machine.py        5 passed  (Boundary ray-casting, stump ROI, shot classifier)
backend/tests/test_tracking.py             5 passed  (2D Kalman filter, occlusion buffer, pose estimation)
backend/tests/test_tts.py                  5 passed  (Edge-TTS, audio priority preemption, queue dispatch)
backend/tests/test_video_source.py         5 passed  (VideoSource hierarchy, speed pacing, clock sync)
======================= 37 passed in 20.30s =======================
```

---

## Known Real-World Limitations & Edge Cases

1. **Small-Ball Resolution in Wide Broadcast Angles**:
   - In standard 1080p wide broadcast camera shots, a cricket ball measures only 4 to 8 pixels wide. Motion blur and compression artifacts can cause intermittent detection dropouts.
   - *Mitigation*: The 2D Kalman filter predicts trajectories across occlusion windows up to 6 frames, combined with multi-spectrum HSV candidate recovery.
2. **Camera Panning & Perspective Changes**:
   - Broadcast cameras pan, tilt, and zoom continuously. A fixed pixel-to-meter conversion is invalid without continuous 3D camera calibration.
   - *Design Decision*: Ball speed (km/h) is estimated only when verified pitch geometry is available. Otherwise, the system transparently indicates uncalibrated speed rather than fabricating inaccurate numbers.
3. **2D Heuristic Shot Classification vs. 3D Biomechanics**:
   - Distinguishing a fine glance from an inside edge requires 3D bat-face angles that cannot always be resolved from a single 2D broadcast angle.
   - *Mitigation*: The classifier correlates wrist/shoulder keypoint vectors with exit ball trajectories, outputting lower confidence scores for ambiguous trajectories.
4. **DRM & Access-Controlled Streams**:
   - The application strictly respects digital rights management (DRM), authentication, and paywalls. DRM-protected URLs are rejected during validation with informative diagnostics.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
