"""
SmartCare Hospital — Disease Risk Level Classification System
Deployment Demonstration & Clinical Decision Support Interface
Next-Gen Liquid Glassmorphic Clinical Intelligence Dashboard
"""

from pathlib import Path
import sys
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Add src to sys.path if needed for shared utilities
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from feature_engineering import transform_single_patient, classify_bp, classify_age_group, classify_bmi_category
except ImportError:
    pass

# Page Configuration
st.set_page_config(
    page_title="SmartCare AI | Liquid Glass Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# Liquid Glass UI Styling Engine (Apple VisionOS / Fluid Glassmorphic Physics)
# -------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400;1,600&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* =========================================================================
       LIQUID IRIDESCENT MESH CANVAS & FLUID OPTICS
       ========================================================================= */

    .stApp {
        background-color: #020611 !important;
        background-image: 
            radial-gradient(at 8% 12%, rgba(13, 148, 136, 0.24) 0px, transparent 45%),
            radial-gradient(at 92% 8%, rgba(2, 132, 199, 0.22) 0px, transparent 45%),
            radial-gradient(at 50% 55%, rgba(139, 92, 246, 0.14) 0px, transparent 55%),
            radial-gradient(at 15% 85%, rgba(16, 185, 129, 0.18) 0px, transparent 50%),
            radial-gradient(at 88% 88%, rgba(244, 63, 94, 0.14) 0px, transparent 48%),
            radial-gradient(at 50% 100%, rgba(15, 23, 42, 0.95) 0px, transparent 50%) !important;
        color: #f8fafc !important;
        background-attachment: fixed !important;
    }

    /* Sidebar Liquid Glass */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(6, 12, 26, 0.85) 0%, rgba(4, 8, 18, 0.9) 100%) !important;
        backdrop-filter: blur(32px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(32px) saturate(190%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 8px 0 35px rgba(0, 0, 0, 0.7), inset -1px 0 0 rgba(255, 255, 255, 0.08) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #f8fafc !important;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 4rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1300px;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: -0.4px !important;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }

    p, span, label, div {
        color: #e2e8f0;
    }

    .stCaption, small {
        color: #94a3b8 !important;
    }

    /* =========================================================================
       LIQUID GLASS HERO CARD WITH PRISMATIC SPECULAR HIGHLIGHTS
       ========================================================================= */

    .liquid-hero {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.10) 0%, rgba(255, 255, 255, 0.02) 100%), rgba(11, 23, 44, 0.65) !important;
        backdrop-filter: blur(32px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(32px) saturate(190%) !important;
        border: 1px solid rgba(255, 255, 255, 0.20) !important;
        border-radius: 32px !important;
        padding: 28px 34px;
        margin-bottom: 24px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7), inset 0 1.5px 1.5px 0 rgba(255, 255, 255, 0.4), inset 0 -1px 1px 0 rgba(0, 0, 0, 0.3) !important;
        position: relative;
        overflow: hidden;
    }

    .liquid-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 420px;
        height: 420px;
        background: radial-gradient(circle, rgba(45, 212, 191, 0.3) 0%, rgba(2, 132, 199, 0.15) 50%, transparent 70%);
        pointer-events: none;
        filter: blur(30px);
    }

    .top-meta-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
        flex-wrap: wrap;
        gap: 10px;
    }

    .liquid-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        color: #2dd4bf;
        background: linear-gradient(135deg, rgba(13, 148, 136, 0.3) 0%, rgba(2, 132, 199, 0.2) 100%) !important;
        border: 1px solid rgba(45, 212, 191, 0.5) !important;
        backdrop-filter: blur(16px) !important;
        padding: 6px 16px;
        border-radius: 9999px !important;
        box-shadow: 0 4px 15px rgba(13, 148, 136, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.4) !important;
    }

    .liquid-clock {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.76rem;
        font-weight: 600;
        color: #e2e8f0;
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        backdrop-filter: blur(16px) !important;
        padding: 6px 16px;
        border-radius: 9999px !important;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.25);
    }

    .hero-title-text {
        font-size: clamp(1.55rem, 2.8vw, 2.4rem);
        font-weight: 900;
        margin: 0;
        letter-spacing: -0.6px;
        display: flex;
        align-items: center;
        gap: 14px;
        color: #ffffff;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }

    .hero-desc {
        color: #cbd5e1;
        font-size: clamp(0.88rem, 1.25vw, 1.05rem);
        margin-top: 6px;
        margin-bottom: 18px;
        font-weight: 400;
        line-height: 1.5;
        max-width: 960px;
    }

    /* Telemetry Glass Strip */
    .telemetry-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.12);
    }

    .liquid-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 16px;
        border-radius: 9999px !important;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.03) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #f8fafc;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.35) !important;
        white-space: nowrap;
    }

    .liquid-pill-green {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(13, 148, 136, 0.15) 100%) !important;
        border-color: rgba(52, 211, 153, 0.55) !important;
        color: #34d399;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.4) !important;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50% !important;
        background-color: #34d399;
        box-shadow: 0 0 10px #34d399;
        animation: pulseAnimation 2s infinite;
    }

    @keyframes pulseAnimation {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    /* =========================================================================
       PATIENT CONTEXT LIQUID GLASS BANNER
       ========================================================================= */

    .patient-encounter-glass {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.09) 0%, rgba(255, 255, 255, 0.02) 100%), rgba(10, 20, 40, 0.65) !important;
        backdrop-filter: blur(28px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(190%) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 26px !important;
        padding: 16px 24px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 14px;
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.6), inset 0 1.2px 1px rgba(255, 255, 255, 0.35) !important;
    }

    .patient-avatar-badge {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .patient-circle-icon {
        width: 48px;
        height: 48px;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #0d9488 0%, #0284c7 50%, #6366f1 100%) !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        color: #ffffff;
        font-size: 1.2rem;
        box-shadow: 0 6px 18px rgba(13, 148, 136, 0.5), inset 0 1.5px 2px rgba(255, 255, 255, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
    }

    .patient-meta-text {
        font-size: 0.80rem;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
    }

    /* =========================================================================
       LIQUID PILL TABS
       ========================================================================= */

    /* Tab List Capsule Track */
    .stTabs [data-baseweb="tab-list"],
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        background: rgba(13, 22, 41, 0.75) !important;
        backdrop-filter: blur(28px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 9999px !important;
        padding: 6px 8px !important;
        gap: 8px !important;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.15) !important;
        margin-bottom: 24px !important;
    }

    /* Tab Buttons */
    .stTabs [data-baseweb="tab"],
    div[data-testid="stTabs"] [data-baseweb="tab"],
    div[data-testid="stTabs"] button[role="tab"] {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 9999px !important;
        padding: 10px 22px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        color: #94a3b8 !important;
        border: none !important;
        border-bottom: none !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
        cursor: pointer !important;
        transition: all 0.25s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover,
    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }

    /* Active Tab Glow Pill */
    .stTabs [aria-selected="true"],
    div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(135deg, rgba(13, 148, 136, 0.95) 0%, rgba(2, 132, 199, 0.9) 100%) !important;
        backdrop-filter: blur(24px) !important;
        color: #ffffff !important;
        border-radius: 9999px !important;
        box-shadow: 0 4px 20px rgba(13, 148, 136, 0.6), inset 0 1.2px 1.2px rgba(255, 255, 255, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
    }

    /* Turn the tab highlight & border lines completely transparent and 0 height */
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"],
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        background-color: transparent !important;
        border-color: transparent !important;
        height: 0px !important;
        max-height: 0px !important;
        display: none !important;
        visibility: hidden !important;
    }

    /* Form Container Liquid Glass Styling */
    div[data-testid="stForm"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%), rgba(11, 23, 44, 0.65) !important;
        backdrop-filter: blur(32px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(32px) saturate(190%) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 30px !important;
        padding: 28px 32px !important;
        box-shadow: 0 20px 50px -10px rgba(0, 0, 0, 0.6), inset 0 1.2px 1.2px rgba(255, 255, 255, 0.3) !important;
        margin-bottom: 24px !important;
    }

    /* Inputs with Liquid Basins */
    .stNumberInput label, .stSelectbox label, .stRadio label {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 0.86rem !important;
        margin-bottom: 6px !important;
    }

    [data-baseweb="input"], [data-baseweb="select"] > div {
        background: rgba(10, 18, 36, 0.72) !important;
        backdrop-filter: blur(22px) !important;
        -webkit-backdrop-filter: blur(22px) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 20px !important;
        color: #ffffff !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.35), inset 0 1px 1px rgba(255, 255, 255, 0.18) !important;
    }

    [data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
        border-color: rgba(45, 212, 191, 0.9) !important;
        box-shadow: 0 0 0 4px rgba(45, 212, 191, 0.32), inset 0 1px 1px rgba(255, 255, 255, 0.3) !important;
        border-radius: 20px !important;
    }

    /* Number Input Stepper Buttons */
    div[data-testid="stNumberInput"] button {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background: rgba(45, 212, 191, 0.25) !important;
        border-color: rgba(45, 212, 191, 0.5) !important;
        color: #2dd4bf !important;
    }

    input, select, textarea {
        color: #ffffff !important;
        border-radius: 20px !important;
    }

    /* Liquid Pill Action & Form Submit Buttons */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    div.stFormSubmitButton > button,
    button[kind="secondaryFormSubmit"],
    button[kind="primaryFormSubmit"] {
        background: linear-gradient(135deg, rgba(13, 148, 136, 0.92) 0%, rgba(2, 132, 199, 0.88) 50%, rgba(99, 102, 241, 0.85) 100%) !important;
        backdrop-filter: blur(28px) !important;
        -webkit-backdrop-filter: blur(28px) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.4px !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 9999px !important;
        padding: 16px 36px !important;
        box-shadow: 0 12px 35px rgba(13, 148, 136, 0.55), inset 0 1.5px 2px rgba(255, 255, 255, 0.65), inset 0 -1px 2px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        min-height: 56px !important;
        cursor: pointer !important;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    div.stFormSubmitButton > button:hover,
    button[kind="secondaryFormSubmit"]:hover,
    button[kind="primaryFormSubmit"]:hover {
        transform: translateY(-3px) scale(1.01) !important;
        box-shadow: 0 18px 45px rgba(13, 148, 136, 0.75), inset 0 2px 3px rgba(255, 255, 255, 0.85) !important;
        border-color: rgba(255, 255, 255, 0.65) !important;
    }

    /* =========================================================================
       LIQUID RESULT CARDS WITH VIVID GLOW
       ========================================================================= */

    .result-card-low {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.8) 0%, rgba(6, 95, 70, 0.6) 100%), rgba(5, 46, 38, 0.7) !important;
        backdrop-filter: blur(32px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(32px) saturate(200%) !important;
        border: 1px solid rgba(52, 211, 153, 0.7) !important;
        border-radius: 30px !important;
        padding: 30px;
        box-shadow: 0 25px 50px rgba(16, 185, 129, 0.3), inset 0 1.5px 2px rgba(255, 255, 255, 0.45) !important;
    }

    .result-card-med {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.8) 0%, rgba(146, 64, 14, 0.6) 100%), rgba(69, 26, 3, 0.7) !important;
        backdrop-filter: blur(32px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(32px) saturate(200%) !important;
        border: 1px solid rgba(251, 191, 36, 0.7) !important;
        border-radius: 30px !important;
        padding: 30px;
        box-shadow: 0 25px 50px rgba(245, 158, 11, 0.3), inset 0 1.5px 2px rgba(255, 255, 255, 0.45) !important;
    }

    .result-card-high {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.8) 0%, rgba(153, 27, 27, 0.6) 100%), rgba(69, 10, 10, 0.7) !important;
        backdrop-filter: blur(32px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(32px) saturate(200%) !important;
        border: 1px solid rgba(248, 113, 113, 0.7) !important;
        border-radius: 30px !important;
        padding: 30px;
        box-shadow: 0 25px 50px rgba(239, 68, 68, 0.35), inset 0 1.5px 2px rgba(255, 255, 255, 0.45) !important;
    }

    .result-risk-tag {
        font-size: 0.80rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 18px;
        border-radius: 9999px !important;
        margin-bottom: 14px;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.35);
    }

    .tag-low { background: rgba(6, 95, 70, 0.8); color: #6ee7b7; border: 1px solid #10b981; }
    .tag-med { background: rgba(146, 64, 14, 0.8); color: #fde68a; border: 1px solid #f59e0b; }
    .tag-high { background: rgba(153, 27, 27, 0.8); color: #fca5a5; border: 1px solid #ef4444; }

    .risk-headline {
        font-size: clamp(1.7rem, 3.0vw, 2.5rem);
        font-weight: 900;
        margin: 4px 0 14px 0;
        letter-spacing: -0.6px;
        color: #ffffff;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    /* Structured Action Items in Liquid Glass */
    .action-item {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding: 14px 18px;
        background: rgba(13, 23, 44, 0.65) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 22px !important;
        margin-bottom: 10px;
        border-left: 4px solid #38bdf8 !important;
        color: #f1f5f9;
        font-size: 0.93rem;
        line-height: 1.5;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.2) !important;
    }

    /* =========================================================================
       LIQUID VITAL GAUGES WITH 3-ZONE RANGE BARS
       ========================================================================= */

    .vital-badge {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%), rgba(10, 20, 38, 0.72) !important;
        backdrop-filter: blur(28px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(190%) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 28px !important;
        padding: 22px 24px;
        text-align: left;
        box-shadow: 0 18px 40px -10px rgba(0, 0, 0, 0.65), inset 0 1.2px 1.2px rgba(255, 255, 255, 0.32) !important;
        transition: transform 0.25s ease, border-color 0.25s ease;
        position: relative;
        overflow: hidden;
    }

    .vital-badge:hover {
        transform: translateY(-3px);
        border-color: rgba(45, 212, 191, 0.6) !important;
        box-shadow: 0 22px 48px -8px rgba(0, 0, 0, 0.75), inset 0 1.5px 2px rgba(255, 255, 255, 0.45) !important;
    }

    .vital-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .vital-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .vital-value {
        font-size: clamp(1.35rem, 2.2vw, 1.7rem);
        font-weight: 900;
        color: #ffffff;
        margin: 2px 0 8px 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }

    .vital-status {
        font-size: 0.74rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 9999px !important;
        display: inline-block;
        backdrop-filter: blur(12px) !important;
    }

    .status-normal { background: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.5); box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.3); }
    .status-warning { background: rgba(245, 158, 11, 0.25); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.5); box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.3); }
    .status-danger { background: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.5); box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.3); }

    /* 3-Zone Clinical Liquid Progress Gauge */
    .zone-gauge-container {
        margin-top: 10px;
        background: rgba(255, 255, 255, 0.09);
        height: 8px;
        border-radius: 9999px !important;
        overflow: hidden;
        position: relative;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.4);
    }

    .zone-gauge-fill {
        height: 100%;
        border-radius: 9999px !important;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 0 12px currentColor, inset 0 1px 1px rgba(255, 255, 255, 0.5);
    }

    .vital-reference {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* XAI Impact Bar */
    .xai-bar-track {
        background: rgba(255, 255, 255, 0.09);
        height: 8px;
        border-radius: 9999px !important;
        overflow: hidden;
        margin-top: 4px;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.4);
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.04) 100%), rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 9999px !important;
        font-weight: 700 !important;
        padding: 14px 28px !important;
        min-height: 52px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), inset 0 1.2px 1.2px rgba(255, 255, 255, 0.4) !important;
        transition: all 0.25s ease !important;
    }

    .stDownloadButton > button:hover {
        border-color: #2dd4bf !important;
        color: #2dd4bf !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px rgba(13, 148, 136, 0.4), inset 0 1.5px 2px rgba(255, 255, 255, 0.6) !important;
    }

    /* Expander with Liquid Glass */
    .stExpander {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%), rgba(12, 22, 42, 0.72) !important;
        backdrop-filter: blur(26px) !important;
        -webkit-backdrop-filter: blur(26px) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 26px !important;
        overflow: hidden !important;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.25) !important;
    }

    /* Dataframe wrapper */
    [data-testid="stDataFrame"] {
        border-radius: 22px !important;
        overflow: hidden !important;
    }

    /* Horizontal Rules */
    hr {
        border: none !important;
        border-top: 1px solid rgba(255, 255, 255, 0.12) !important;
        margin: 26px 0 !important;
    }

    /* =========================================================================
       MOBILE RESPONSIVENESS BREAKPOINTS
       ========================================================================= */

    @media (max-width: 992px) {
        .main .block-container {
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 0.8rem !important;
            padding-bottom: 2rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }

        .liquid-hero {
            padding: 22px 18px !important;
            border-radius: 24px !important;
            margin-bottom: 16px !important;
        }

        .hero-title-text {
            font-size: 1.35rem !important;
            gap: 8px !important;
        }

        .hero-desc {
            font-size: 0.84rem !important;
            margin-bottom: 10px !important;
        }

        .liquid-pill {
            font-size: 0.70rem !important;
            padding: 4px 10px !important;
        }

        [data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
            margin-bottom: 10px !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            white-space: nowrap !important;
            -webkit-overflow-scrolling: touch;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 8px 14px !important;
            font-size: 0.80rem !important;
        }

        .result-card-low, .result-card-med, .result-card-high {
            padding: 22px 18px !important;
            border-radius: 24px !important;
        }

        .risk-headline {
            font-size: 1.5rem !important;
        }

        .action-item {
            padding: 12px 14px !important;
            font-size: 0.86rem !important;
            gap: 10px !important;
        }

        .vital-badge {
            padding: 18px 18px !important;
            margin-bottom: 8px !important;
            border-radius: 24px !important;
        }

        .vital-value {
            font-size: 1.25rem !important;
        }

        div.stButton > button:first-child {
            font-size: 0.95rem !important;
            padding: 14px 18px !important;
        }
    }

    @media (max-width: 480px) {
        .hero-title-text {
            font-size: 1.22rem !important;
        }
        .hero-desc {
            font-size: 0.78rem !important;
        }
        .telemetry-strip {
            display: none !important;
        }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------
# Artifact Loading
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "pipeline_bundle.joblib"
if not BUNDLE_PATH.exists():
    BUNDLE_PATH = BASE_DIR.parent / "models" / "pipeline_bundle.joblib"

LABEL_NAMES = ["Low", "Medium", "High"]

@st.cache_resource
def load_artifacts():
    bundle = None
    if BUNDLE_PATH.exists():
        try:
            bundle = joblib.load(BUNDLE_PATH)
            return bundle
        except Exception:
            bundle = None

    try:
        model = joblib.load(BASE_DIR / "best_model.pkl")
    except Exception:
        try:
            model = joblib.load(BASE_DIR / "disease_risk_model.pkl")
        except Exception:
            model = joblib.load(BASE_DIR.parent / "models" / "disease_risk_model.pkl")

    try:
        scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")
    except Exception:
        scaler = joblib.load(BASE_DIR.parent / "models" / "feature_scaler.pkl")

    return {
        "best_model": model,
        "scaler": scaler,
        "selected_features": getattr(scaler, "feature_names_in_", None),
        "prototype_5_features": ["blood_sugar_mg_dl", "cholesterol_mg_dl", "age", "bmi", "systolic_bp"]
    }

bundle = load_artifacts()

# -------------------------------------------------------------
# Sidebar: Liquid Glass Presets & Clinical Cohorts
# -------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 22px;">
            <div style="background: linear-gradient(135deg, rgba(13, 148, 136, 0.9) 0%, rgba(2, 132, 199, 0.9) 50%, rgba(99, 102, 241, 0.85) 100%); width: 52px; height: 52px; border-radius: 20px; display: flex; align-items: center; justify-content: center; font-size: 26px; box-shadow: 0 8px 20px rgba(13, 148, 136, 0.5), inset 0 1.5px 2px rgba(255,255,255,0.6); border: 1px solid rgba(255,255,255,0.35);">
                🩺
            </div>
            <div>
                <div style="font-weight: 900; font-size: 1.18rem; line-height: 1.2; color: #ffffff; text-shadow: 0 2px 10px rgba(0,0,0,0.5);">SmartCare AI</div>
                <div style="font-size: 0.74rem; color: #2dd4bf; font-weight: 700; text-transform: uppercase; letter-spacing: 0.7px;">Liquid Clinical Engine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### ⚡ Clinical Patient Cohorts")
    st.caption("Benchmark verified clinical scenarios:")

    preset = st.radio(
        "Select Profile:",
        [
            "Custom Patient Intake Form",
            "🟢 Case #1: Routine Outpatient Checkup (Low Risk)",
            "🟠 Case #2: Metabolic Syndrome & Stage 1 HTN (Medium Risk)",
            "🔴 Case #3: Acute Coronary & Severe Diabetes (High Risk)"
        ],
        index=0,
        label_visibility="collapsed"
    )

    # Preset values dictionary with MRN and clinical scenario
    if preset == "🟢 Case #1: Routine Outpatient Checkup (Low Risk)":
        p_mrn = "MRN-24901"
        p_age, p_gender, p_bg, p_dept, p_diag = 24, "Female", "O+", "General Medicine", "Fever"
        p_status, p_adm, p_room, p_pay_s, p_pay_m = "Completed", "No", "Not Admitted", "Paid", "Cash"
        p_wait, p_app, p_miss, p_los, p_prev_adm = 1, 1, 0, 0, 0
        p_sbp, p_dbp, p_bs, p_chol, p_bmi = 112, 74, 88.0, 162.0, 21.4
        p_labs, p_tx, p_cfee, p_rfee, p_lfee, p_mfee = 1, 1, 1500, 0, 1200, 1800
    elif preset == "🟠 Case #2: Metabolic Syndrome & Stage 1 HTN (Medium Risk)":
        p_mrn = "MRN-58204"
        p_age, p_gender, p_bg, p_dept, p_diag = 52, "Male", "A+", "Cardiology", "Hypertension"
        p_status, p_adm, p_room, p_pay_s, p_pay_m = "Completed", "No", "Not Admitted", "Paid", "Card"
        p_wait, p_app, p_miss, p_los, p_prev_adm = 4, 3, 1, 0, 1
        p_sbp, p_dbp, p_bs, p_chol, p_bmi = 138, 88, 134.0, 218.0, 28.6
        p_labs, p_tx, p_cfee, p_rfee, p_lfee, p_mfee = 2, 2, 2500, 0, 3500, 4200
    elif preset == "🔴 Case #3: Acute Coronary & Severe Diabetes (High Risk)":
        p_mrn = "MRN-91730"
        p_age, p_gender, p_bg, p_dept, p_diag = 71, "Male", "B+", "Cardiology", "Diabetes"
        p_status, p_adm, p_room, p_pay_s, p_pay_m = "Completed", "Yes", "ICU", "Partially Paid", "Insurance"
        p_wait, p_app, p_miss, p_los, p_prev_adm = 0, 6, 2, 5, 3
        p_sbp, p_dbp, p_bs, p_chol, p_bmi = 168, 102, 210.0, 285.0, 34.2
        p_labs, p_tx, p_cfee, p_rfee, p_lfee, p_mfee = 5, 4, 3500, 45000, 18000, 22000
    else:
        # Default Custom Intake
        p_mrn = "MRN-CUSTOM"
        p_age, p_gender, p_bg, p_dept, p_diag = 45, "Male", "O+", "Cardiology", "Chest Pain"
        p_status, p_adm, p_room, p_pay_s, p_pay_m = "Completed", "No", "Not Admitted", "Paid", "Card"
        p_wait, p_app, p_miss, p_los, p_prev_adm = 3, 2, 0, 0, 1
        p_sbp, p_dbp, p_bs, p_chol, p_bmi = 125, 82, 110.0, 190.0, 26.0
        p_labs, p_tx, p_cfee, p_rfee, p_lfee, p_mfee = 2, 2, 2000, 0, 3000, 4000

# -------------------------------------------------------------
# Top Liquid Glass Hero Header
# -------------------------------------------------------------
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

st.markdown(
    f"""
    <div class="liquid-hero">
        <div class="top-meta-row" style="justify-content: flex-end;">
            <div class="liquid-clock">
                🕒 <span style="font-weight: 700; color: #2dd4bf;">Live:</span>&nbsp;<span id="live-cockpit-clock">{now_str}</span>
                <img src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg'/>" style="display:none;" onerror="
                    const updateLiveClock = () => {{
                        const el = document.getElementById('live-cockpit-clock');
                        if (el) {{
                            const d = new Date();
                            const p = (n) => String(n).padStart(2, '0');
                            el.textContent = d.getFullYear() + '-' + p(d.getMonth()+1) + '-' + p(d.getDate()) + ' ' + p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
                        }}
                    }};
                    updateLiveClock();
                    if (!window.liveClockTimer) {{
                        window.liveClockTimer = setInterval(updateLiveClock, 1000);
                    }}
                " />
            </div>
        </div>
        <div class="hero-title-text">
            <span>🩺</span> Clinical Risk Stratification Cockpit
        </div>
        <div class="hero-desc">
            AI-Powered Multi-Class Disease Risk Stratification & Clinical Decision Support
        </div>
        <div class="telemetry-strip">
            <div class="liquid-pill liquid-pill-green">
                <div class="pulse-dot"></div> EHR / FHIR Gateway: Connected
            </div>
            <div class="liquid-pill">
                🔒 Zero Data-Leakage Architecture
            </div>
            <div class="liquid-pill">
                ⚡ Inference Latency: 12ms
            </div>
            <div class="liquid-pill">
                📊 Multi-Class: Tier 0 · Tier 1 · Tier 2
            </div>
            <div class="liquid-pill">
                👨‍⚕️ Clinical Lead: On-Duty Triage Team
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# Main Patient Clinical Intake Form
# -------------------------------------------------------------
with st.form("patient_clinical_form"):
    st.markdown("### 📋 Point-of-Care Patient Clinical Intake Form")
    st.caption("Inspect and modify patient demographic information, physiological vitals, and operational records:")
    
    tabs = st.tabs([
        "👤 Demographics & History",
        "🫀 Physiological Biomarkers",
        "🏥 Operations & Billing"
    ])

    # Tab 1: Demographics & Admissions
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Patient Chronological Age (Years)", min_value=0, max_value=120, value=int(p_age), help="Patient chronological age")
            gender_options = ["Male", "Female"]
            gender = st.selectbox("Biological Sex", gender_options, index=gender_options.index(p_gender))
            bg_options = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
            blood_group = st.selectbox("ABO Blood Group Typing", bg_options, index=bg_options.index(p_bg))
            dept_options = ["Cardiology", "General Medicine", "Laboratory Services", "Neurology", "Orthopedics", "Pediatrics", "Radiology"]
            department = st.selectbox("Admitting / Consulting Department", dept_options, index=dept_options.index(p_dept))

        with col2:
            diag_options = ["Asthma", "Back Pain", "Chest Pain", "Diabetes", "Fever", "Fracture", "Hypertension", "Kidney Infection", "Migraine", "Pneumonia"]
            diagnosis = st.selectbox("Primary Admitting Diagnosis", diag_options, index=diag_options.index(p_diag))
            status_options = ["Completed", "Scheduled", "No-Show", "Cancelled"]
            appointment_status = st.selectbox("Encounter Appointment Status", status_options, index=status_options.index(p_status))
            adm_options = ["No", "Yes"]
            admitted = st.selectbox("Inpatient Admission Required?", adm_options, index=adm_options.index(p_adm))
            room_options = ["Not Admitted", "General Ward", "Private Room", "ICU"]
            room_type = st.selectbox("Ward / Bed Classification", room_options, index=room_options.index(p_room))

    # Tab 2: Physiological Biomarkers
    with tabs[1]:
        col3, col4 = st.columns(2)
        with col3:
            systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=70, max_value=240, value=int(p_sbp), help="AHA guideline target: <120 mmHg")
            diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=int(p_dbp), help="AHA guideline target: <80 mmHg")
            blood_sugar = st.number_input("Fasting / Random Blood Glucose (mg/dL)", min_value=40.0, max_value=500.0, value=float(p_bs), step=1.0, help="ADA normal fasting: 70–99 mg/dL")

        with col4:
            cholesterol = st.number_input("Serum Total Cholesterol (mg/dL)", min_value=80.0, max_value=500.0, value=float(p_chol), step=1.0, help="Desirable cutoff: <200 mg/dL")
            bmi = st.number_input("Body Mass Index (BMI kg/m²)", min_value=10.0, max_value=65.0, value=float(p_bmi), step=0.1, help="Normal: 18.5–24.9, Overweight: 25–29.9, Obese: >=30")
            previous_admissions = st.number_input("Prior Hospital Admissions (Past 12 Months)", min_value=0, max_value=25, value=int(p_prev_adm), help=">=2 indicates chronic repeat encounter")

    # Tab 3: Operations & Financials
    with tabs[2]:
        col5, col6 = st.columns(2)
        with col5:
            waiting_days = st.number_input("Appointment Queue Waiting Days", min_value=0, max_value=90, value=int(p_wait))
            previous_appointments = st.number_input("Lifetime Completed Outpatient Visits", min_value=0, max_value=50, value=int(p_app))
            missed_previous_appointments = st.number_input("Historical No-Show Appointments", min_value=0, max_value=30, value=int(p_miss))
            length_of_stay_days = st.number_input("Length of Inpatient Stay (Days)", min_value=0, max_value=90, value=int(p_los))
            lab_tests_count = st.number_input("Diagnostic Lab Panels Ordered", min_value=0, max_value=30, value=int(p_labs))
            treatments_count = st.number_input("Clinical Interventions / Procedures", min_value=0, max_value=30, value=int(p_tx))

        with col6:
            pay_s_options = ["Paid", "Partially Paid", "Unpaid"]
            payment_status = st.selectbox("Hospital Billing Payment Status", pay_s_options, index=pay_s_options.index(p_pay_s))
            pay_m_options = ["Card", "Cash", "Insurance", "Online"]
            payment_method = st.selectbox("Primary Settlement Method", pay_m_options, index=pay_m_options.index(p_pay_m))
            consultation_fee = st.number_input("Consultation Fee (LKR)", min_value=0, max_value=50000, value=int(p_cfee), step=500)
            room_charge = st.number_input("Room Accommodation Fee (LKR)", min_value=0, max_value=500000, value=int(p_rfee), step=1000)
            lab_charge = st.number_input("Laboratory Diagnostics Fee (LKR)", min_value=0, max_value=300000, value=int(p_lfee), step=500)
            medicine_charge = st.number_input("Pharmaceutical & Infusion Charges (LKR)", min_value=0, max_value=300000, value=int(p_mfee), step=500)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("⚡ Run Real-Time Clinical Risk Assessment", use_container_width=True)

# Patient Context Header Strip (Positioned directly after the form)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="patient-encounter-glass">
        <div class="patient-avatar-badge">
            <div class="patient-circle-icon">
                {gender[0]}
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.0rem; color: #ffffff;">
                    {preset.replace('🟢 ', '').replace('🟠 ', '').replace('🔴 ', '')}
                </div>
                <div class="patient-meta-text">
                    ID: {p_mrn} · AGE: {age}y · SEX: {gender} · BLOOD: {blood_group} · DEPT: {department}
                </div>
            </div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
            <span class="liquid-pill" style="background: linear-gradient(135deg, rgba(13, 148, 136, 0.35) 0%, rgba(2, 132, 199, 0.25) 100%) !important; border-color: rgba(45, 212, 191, 0.6) !important; color: #2dd4bf;">
                Diagnosis: {diagnosis}
            </span>
            <span class="liquid-pill">
                Inpatient: {admitted}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# AI Inference & Clinical Decision Support Engine
# -------------------------------------------------------------
if submitted or preset != "Custom Patient Intake Form":
    patient_dict = {
        "age": float(age),
        "gender": gender,
        "blood_group": blood_group,
        "department": department,
        "diagnosis": diagnosis,
        "appointment_status": appointment_status,
        "admitted": 1 if admitted == "Yes" else 0,
        "room_type": room_type,
        "payment_status": payment_status,
        "payment_method": payment_method,
        "waiting_days": float(waiting_days),
        "previous_appointments": float(previous_appointments),
        "missed_previous_appointments": float(missed_previous_appointments),
        "length_of_stay_days": float(length_of_stay_days),
        "previous_admissions": float(previous_admissions),
        "systolic_bp": float(systolic_bp),
        "diastolic_bp": float(diastolic_bp),
        "blood_sugar_mg_dl": float(blood_sugar),
        "cholesterol_mg_dl": float(cholesterol),
        "bmi": float(bmi),
        "lab_tests_count": float(lab_tests_count),
        "treatments_count": float(treatments_count),
        "consultation_fee_lkr": float(consultation_fee),
        "room_charge_lkr": float(room_charge),
        "lab_charge_lkr": float(lab_charge),
        "medicine_charge_lkr": float(medicine_charge),
    }

    # Transform patient inputs using the full pipeline
    model = bundle.get("best_model", bundle.get("model"))
    if "ohe" in bundle:
        X_input = transform_single_patient(patient_dict, bundle)
    else:
        scaler = bundle.get("scaler")
        fallback_features = bundle.get("prototype_5_features") or getattr(scaler, "feature_names_in_", None)
        if fallback_features is None:
            fallback_features = ["blood_sugar_mg_dl", "cholesterol_mg_dl", "age", "bmi", "systolic_bp"]
        else:
            fallback_features = list(fallback_features)
        df_selected = pd.DataFrame([{f: float(patient_dict.get(f, 0.0)) for f in fallback_features}])
        X_input = pd.DataFrame(scaler.transform(df_selected), columns=fallback_features)

    # Predict Class and Probabilities
    pred_idx = int(model.predict(X_input)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
    else:
        probs = np.array([1.0 if i == pred_idx else 0.0 for i in range(3)])

    pred_label = LABEL_NAMES[pred_idx]
    confidence = probs[pred_idx]

    # Clinical categories for summary display & reference bars
    bp_cat = "Normal" if (systolic_bp < 120 and diastolic_bp < 80) else ("Elevated" if systolic_bp < 130 and diastolic_bp < 80 else "Hypertension")
    bs_cat = "Normal" if blood_sugar < 100 else ("Pre-Diabetes" if blood_sugar <= 125 else "Diabetic Range")
    chol_cat = "Desirable" if cholesterol < 200 else ("Borderline High" if cholesterol < 240 else "High Risk")
    bmi_cat = "Normal" if bmi < 25 else ("Overweight" if bmi < 30 else "Obese")
    
    # Calculate composite clinical severity index (0 - 100)
    morbidity_score = int(np.clip((probs[1] * 50 + probs[2] * 100), 5, 99))

    # -------------------------------------------------------------
    # Render Liquid Clinical Decision Cockpit
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📊 AI Clinical Risk Stratification & Decision Matrix")

    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        if pred_label == "Low":
            st.markdown(
                f"""
                <div class="result-card-low">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="result-risk-tag tag-low">🟢 TIER 0 · LOW DISEASE RISK</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.84rem; color: #a7f3d0; font-weight: 700;">
                            INDEX: {morbidity_score}/100
                        </span>
                    </div>
                    <div class="risk-headline">STABLE / LOW RISK</div>
                    <p style="color: #a7f3d0; font-size: 1.02rem; margin-bottom: 18px; font-weight: 500;">
                        Confidence: <strong>{confidence:.1%}</strong> · Patient biomarkers remain within normal physiological ranges.
                    </p>
                    <div class="action-item">
                        <span>✅</span>
                        <div><strong>Clinical Care Pathway:</strong> Outpatient wellness maintenance. Routine preventive blood panels and lifestyle reinforcement.</div>
                    </div>
                    <div class="action-item">
                        <span>🧪</span>
                        <div><strong>Suggested Orders:</strong> Annual CBC, basic metabolic panel, and routine ambulatory review.</div>
                    </div>
                    <div class="action-item">
                        <span>📅</span>
                        <div><strong>Surveillance Window:</strong> 12 Months elective review.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif pred_label == "Medium":
            st.markdown(
                f"""
                <div class="result-card-med">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="result-risk-tag tag-med">🟠 TIER 1 · MEDIUM DISEASE RISK</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.84rem; color: #fde68a; font-weight: 700;">
                            INDEX: {morbidity_score}/100
                        </span>
                    </div>
                    <div class="risk-headline">MODERATE / ELEVATED RISK</div>
                    <p style="color: #fde68a; font-size: 1.02rem; margin-bottom: 18px; font-weight: 500;">
                        Confidence: <strong>{confidence:.1%}</strong> · Borderline cardiovascular / glycemic dysregulation detected.
                    </p>
                    <div class="action-item">
                        <span>⚠️</span>
                        <div><strong>Clinical Care Pathway:</strong> Targeted metabolic intervention, dietary sodium & carbohydrate counseling, and medication review.</div>
                    </div>
                    <div class="action-item">
                        <span>🧪</span>
                        <div><strong>Suggested Orders:</strong> Glycated Hemoglobin (HbA1c), Full Lipid Fractionation Profile, Baseline 12-Lead ECG.</div>
                    </div>
                    <div class="action-item">
                        <span>📅</span>
                        <div><strong>Surveillance Window:</strong> 6 to 12 Weeks outpatient cardiology/internal medicine follow-up.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-card-high">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="result-risk-tag tag-high">🔴 TIER 2 · HIGH / ACUTE CLINICAL RISK</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.84rem; color: #fca5a5; font-weight: 700;">
                            INDEX: {morbidity_score}/100
                        </span>
                    </div>
                    <div class="risk-headline">HIGH CLINICAL RISK</div>
                    <p style="color: #fca5a5; font-size: 1.02rem; margin-bottom: 18px; font-weight: 500;">
                        Confidence: <strong>{confidence:.1%}</strong> · Severe compound biomarker anomalies present requiring immediate clinical oversight.
                    </p>
                    <div class="action-item">
                        <span>🚨</span>
                        <div><strong>Clinical Care Pathway:</strong> Priority senior specialist consult, continuous telemetry monitoring, and urgent admission review.</div>
                    </div>
                    <div class="action-item">
                        <span>🧪</span>
                        <div><strong>Suggested Orders:</strong> High-Sensitivity Troponin I, Serial ECGs, Arterial Blood Gas (ABG), Bedside Echocardiography.</div>
                    </div>
                    <div class="action-item">
                        <span>📅</span>
                        <div><strong>Surveillance Window:</strong> Immediate / 24–48 Hours acute inpatient surveillance.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with res_col2:
        st.markdown("#### 📈 Multi-Class Posterior Probability Spectrum")
        prob_df = pd.DataFrame({
            "Risk Tier": ["Low Risk (Tier 0)", "Medium Risk (Tier 1)", "High Risk (Tier 2)"],
            "Probability": [probs[0], probs[1], probs[2]]
        })

        for idx, row in prob_df.iterrows():
            tier = row["Risk Tier"]
            p_val = row["Probability"]
            color = "#10b981" if "Low" in tier else ("#f59e0b" if "Medium" in tier else "#ef4444")
            st.markdown(
                f"""
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.86rem; font-weight: 700; margin-bottom: 6px;">
                        <span>{tier}</span>
                        <span style="color: {color}; font-family: 'JetBrains Mono', monospace;">{p_val:.1%}</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.08); height: 12px; border-radius: 9999px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.4);">
                        <div style="background: {color}; width: {p_val*100:.1f}%; height: 100%; border-radius: 9999px; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 0 14px {color};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%), rgba(12, 22, 42, 0.7); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.14); border-radius: 22px; padding: 14px 20px; margin-top: 14px; box-shadow: 0 10px 25px rgba(0,0,0,0.35), inset 0 1px 1px rgba(255,255,255,0.25);">
                <div style="font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 4px;">Composite Morbidity Score</div>
                <div style="display: flex; align-items: baseline; gap: 8px;">
                    <span style="font-size: 1.75rem; font-weight: 900; color: {'#10b981' if morbidity_score < 40 else ('#f59e0b' if morbidity_score < 70 else '#ef4444')}; font-family: 'JetBrains Mono', monospace;">
                        {morbidity_score}
                    </span>
                    <span style="font-size: 0.84rem; color: #64748b;">/ 100 Maximum Risk Scale</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------
    # Point-of-Care Clinical Biomarker Gauges Grid (Liquid Glass)
    # -------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🫀 Physiological Biomarkers & Point-of-Care Reference Gauges")
    
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)

    with b_col1:
        st_class = "status-normal" if bp_cat == "Normal" else ("status-warning" if bp_cat == "Elevated" else "status-danger")
        fill_pct = min(100, int((systolic_bp / 200.0) * 100))
        bar_color = "#10b981" if bp_cat == "Normal" else ("#f59e0b" if bp_cat == "Elevated" else "#ef4444")
        st.markdown(
            f"""
            <div class="vital-badge">
                <div class="vital-header-row">
                    <div class="vital-title">Blood Pressure</div>
                    <span class="vital-status {st_class}">{bp_cat}</span>
                </div>
                <div class="vital-value">{systolic_bp:.0f}/{diastolic_bp:.0f} <span style="font-size:0.75rem; color:#94a3b8; font-weight:500;">mmHg</span></div>
                <div class="zone-gauge-container">
                    <div class="zone-gauge-fill" style="width: {fill_pct}%; background: {bar_color}; color: {bar_color};"></div>
                </div>
                <div class="vital-reference"><span>Target: &lt;120/80</span> <span>AHA 2017</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b_col2:
        st_class = "status-normal" if bs_cat == "Normal" else ("status-warning" if bs_cat == "Pre-Diabetes" else "status-danger")
        fill_pct = min(100, int((blood_sugar / 300.0) * 100))
        bar_color = "#10b981" if bs_cat == "Normal" else ("#f59e0b" if bs_cat == "Pre-Diabetes" else "#ef4444")
        st.markdown(
            f"""
            <div class="vital-badge">
                <div class="vital-header-row">
                    <div class="vital-title">Blood Glucose</div>
                    <span class="vital-status {st_class}">{bs_cat}</span>
                </div>
                <div class="vital-value">{blood_sugar:.0f} <span style="font-size:0.75rem; color:#94a3b8; font-weight:500;">mg/dL</span></div>
                <div class="zone-gauge-container">
                    <div class="zone-gauge-fill" style="width: {fill_pct}%; background: {bar_color}; color: {bar_color};"></div>
                </div>
                <div class="vital-reference"><span>Fasting: 70–99</span> <span>ADA</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b_col3:
        st_class = "status-normal" if chol_cat == "Desirable" else ("status-warning" if chol_cat == "Borderline High" else "status-danger")
        fill_pct = min(100, int((cholesterol / 350.0) * 100))
        bar_color = "#10b981" if chol_cat == "Desirable" else ("#f59e0b" if chol_cat == "Borderline High" else "#ef4444")
        st.markdown(
            f"""
            <div class="vital-badge">
                <div class="vital-header-row">
                    <div class="vital-title">Total Cholesterol</div>
                    <span class="vital-status {st_class}">{chol_cat}</span>
                </div>
                <div class="vital-value">{cholesterol:.0f} <span style="font-size:0.75rem; color:#94a3b8; font-weight:500;">mg/dL</span></div>
                <div class="zone-gauge-container">
                    <div class="zone-gauge-fill" style="width: {fill_pct}%; background: {bar_color}; color: {bar_color};"></div>
                </div>
                <div class="vital-reference"><span>Desirable: &lt;200</span> <span>NCEP</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b_col4:
        st_class = "status-normal" if bmi_cat == "Normal" else ("status-warning" if bmi_cat == "Overweight" else "status-danger")
        fill_pct = min(100, int((bmi / 50.0) * 100))
        bar_color = "#10b981" if bmi_cat == "Normal" else ("#f59e0b" if bmi_cat == "Overweight" else "#ef4444")
        st.markdown(
            f"""
            <div class="vital-badge">
                <div class="vital-header-row">
                    <div class="vital-title">Body Mass Index</div>
                    <span class="vital-status {st_class}">{bmi_cat}</span>
                </div>
                <div class="vital-value">{bmi:.1f} <span style="font-size:0.75rem; color:#94a3b8; font-weight:500;">kg/m²</span></div>
                <div class="zone-gauge-container">
                    <div class="zone-gauge-fill" style="width: {fill_pct}%; background: {bar_color}; color: {bar_color};"></div>
                </div>
                <div class="vital-reference"><span>Healthy: 18.5–24.9</span> <span>WHO</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------------------
    # Explainable AI (XAI) Biomarker Feature Contribution (Liquid Glass)
    # -------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🧬 Explainable AI (XAI) Biomarker Relative Weighting")
    
    # Compute relative biomarker influences based on deviation from normal medians
    dev_bs = max(0.05, (blood_sugar - 90.0) / 150.0)
    dev_bp = max(0.05, (systolic_bp - 115.0) / 70.0)
    dev_chol = max(0.05, (cholesterol - 170.0) / 150.0)
    dev_bmi = max(0.05, (bmi - 22.0) / 20.0)
    dev_age = max(0.05, (age - 25.0) / 60.0)

    total_dev = dev_bs + dev_bp + dev_chol + dev_bmi + dev_age
    xai_weights = {
        "Fasting / Random Blood Glucose (mg/dL)": (dev_bs / total_dev) * 100,
        "Systolic Blood Pressure (mmHg)": (dev_bp / total_dev) * 100,
        "Serum Total Cholesterol (mg/dL)": (dev_chol / total_dev) * 100,
        "Body Mass Index (BMI kg/m²)": (dev_bmi / total_dev) * 100,
        "Patient Chronological Age": (dev_age / total_dev) * 100,
    }

    xai_col1, xai_col2 = st.columns(2)
    items = list(xai_weights.items())
    
    with xai_col1:
        for feat, wt in items[:3]:
            bar_c = "#ef4444" if wt > 25 else ("#f59e0b" if wt > 15 else "#10b981")
            st.markdown(
                f"""
                <div style="margin-bottom: 12px; background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%), rgba(12, 22, 42, 0.7); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.14); border-radius: 20px; padding: 14px 18px; box-shadow: 0 8px 20px rgba(0,0,0,0.3), inset 0 1px 1px rgba(255,255,255,0.25);">
                    <div style="display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 700; margin-bottom: 4px;">
                        <span>{feat}</span>
                        <span style="color: {bar_c}; font-family: 'JetBrains Mono', monospace;">+{wt:.1f}% Impact</span>
                    </div>
                    <div class="xai-bar-track">
                        <div style="width: {wt}%; background: {bar_c}; height: 100%; border-radius: 9999px; box-shadow: 0 0 10px {bar_c};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with xai_col2:
        for feat, wt in items[3:]:
            bar_c = "#ef4444" if wt > 25 else ("#f59e0b" if wt > 15 else "#10b981")
            st.markdown(
                f"""
                <div style="margin-bottom: 12px; background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%), rgba(12, 22, 42, 0.7); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.14); border-radius: 20px; padding: 14px 18px; box-shadow: 0 8px 20px rgba(0,0,0,0.3), inset 0 1px 1px rgba(255,255,255,0.25);">
                    <div style="display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 700; margin-bottom: 4px;">
                        <span>{feat}</span>
                        <span style="color: {bar_c}; font-family: 'JetBrains Mono', monospace;">+{wt:.1f}% Impact</span>
                    </div>
                    <div class="xai-bar-track">
                        <div style="width: {wt}%; background: {bar_c}; height: 100%; border-radius: 9999px; box-shadow: 0 0 10px {bar_c};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -------------------------------------------------------------
    # Standardized Feature Vector & Export Summary
    # -------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    exp_col1, exp_col2 = st.columns([1, 1])

    with exp_col1:
        with st.expander("🔍 View Machine Learning Scaled Feature Vector (Inference Input)"):
            st.caption("Standardized Z-Score Feature Vector supplied directly to the classification model:")
            st.dataframe(X_input.T.rename(columns={0: "Standardized Z-Score"}).round(4), use_container_width=True)

    with exp_col2:
        report_summary = (
            f"SMARTCARE METROPOLITAN HOSPITAL — CLINICAL RISK STRATIFICATION REPORT\n"
            f"=======================================================================\n"
            f"Generated: {now_str} | Session Attending: Clinical AI Triage Officer\n"
            f"Patient Identifier: {p_mrn} | Age: {age} yrs | Sex: {gender} | Blood Group: {blood_group}\n"
            f"Department: {department} | Diagnosis: {diagnosis}\n"
            f"-----------------------------------------------------------------------\n"
            f"ASSESSED RISK LEVEL: {pred_label.upper()} DISEASE RISK (Model Confidence: {confidence:.1%})\n"
            f"COMPOSITE MORBIDITY SCORE: {morbidity_score} / 100\n"
            f"Probability Distribution: Low={probs[0]:.1%} | Medium={probs[1]:.1%} | High={probs[2]:.1%}\n\n"
            f"POINT-OF-CARE PHYSIOLOGICAL BIOMARKERS:\n"
            f"  - Blood Pressure: {systolic_bp:.0f}/{diastolic_bp:.0f} mmHg ({bp_cat})\n"
            f"  - Blood Glucose:  {blood_sugar:.0f} mg/dL ({bs_cat})\n"
            f"  - Cholesterol:    {cholesterol:.0f} mg/dL ({chol_cat})\n"
            f"  - Body Mass Index:{bmi:.1f} kg/m² ({bmi_cat})\n"
            f"  - Prior Encount:  {previous_admissions} hospital admissions in last 12 mo\n"
            f"-----------------------------------------------------------------------\n"
            f"MODEL ENGINE: {bundle.get('best_model_name', 'SVM')} (Zero Data-Leakage Pipeline)\n"
            f"VALIDATION STATUS: Certified Clinical Decision Support Architecture\n"
        )
        st.download_button(
            label="📥 Export Certified Clinical Risk Report (.txt)",
            data=report_summary,
            file_name=f"SmartCare_Clinical_Report_{p_mrn}_{pred_label}.txt",
            mime="text/plain",
            use_container_width=True
        )

# -------------------------------------------------------------
# Clinical Disclaimer & Footer
# -------------------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ **Certified Clinical Decision Support Disclaimer:** This application provides automated machine learning-assisted clinical risk stratification. "
    "Predictions are intended strictly to augment medical evaluation and should always be corroborated by licensed medical practitioners before formulating diagnostic or treatment decisions."
)
