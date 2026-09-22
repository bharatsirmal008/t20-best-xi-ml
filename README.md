# T20 Best XI ML 🏏🤖

An end-to-end **Machine Learning system for predicting the Best Playing XI in T20 cricket** using historical player performance, team context, venue information, match conditions, and explainable AI.

The system combines an ML-based **player suitability score** with a **constraint-based optimizer** to generate a valid Playing XI for an IPL franchise or a national team.

---

## 🚀 Live Demo

🌐 **Streamlit App:**  
https://t20-best-xi-ml.streamlit.app/

💻 **GitHub Repository:**  
https://github.com/bharatsirmal008/t20-best-xi-ml

---

## 📌 Project Overview

Selecting a Playing XI is a complex decision involving:

- Player performance
- Recent form
- Batting and bowling contribution
- Team history
- Venue conditions
- Match context
- Squad eligibility
- Role balance
- Overseas-player restrictions in IPL

This project automates this process using a two-stage pipeline:

```text
User selects Team
        ↓
User provides eligible squad
        ↓
Pre-match feature generation
        ↓
ML player suitability prediction
        ↓
Constraint-based XI optimizer
        ↓
Best Playing XI (11 Players)
