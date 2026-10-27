# AI-Driven Real-Time Contingency Mitigation for IEEE 9-Bus Systems Using Transformer-Coupled BESS

An advanced, multi-platform power systems engineering project that implements an artificial intelligence-supervised Battery Energy Storage System (BESS) for real-time thermal overload relief and contingency mitigation on the standard IEEE 9-bus benchmark system.

---

## 🚀 Project Overview

During severe power system contingencies (such as N-1 line outages), transmission lines can experience dangerous thermal overloads exceeding 100% capacity. Traditional remediation relies on slow, iterative manual adjustments or heavy AC power flow computations. 

This project bridges traditional power systems theory with applied machine learning by introducing an ultra-fast, data-driven BESS controller. By mapping real-time network telemetry directly to optimal active power injections, the system achieves sub-millisecond decision-making to secure the grid against cascading failures.

---

## ⚡ System Architecture

* **Transmission Network:** Standard IEEE 9-bus test system ($138\text{ kV}$ high-voltage base).
* **Storage Location:** **Bus 7**, chosen strategically due to its high Power Transfer Distribution Factor (PTDF) sensitivity and proximity to heavy load corridors.
* **3-Tier Configuration:** 
  * Utility-scale BESS ($40\text{--}50\text{ MW}$).
  * Dedicated medium-voltage collector bus ($6.6\text{ kV}$).
  * Step-up power transformer ($6.6\text{ kV} / 138\text{ kV}$ with $V_k = 10\%$, $V_{kr} = 0.5\%$).

---

## 📊 Key Performance Highlights

* **Inference Speed:** **$9.73\text{ ms}$** via the ML controller vs. **$118.5\text{ ms}$** for traditional iterative physical AC power flow solvers.
* **Thermal Relief:** Successfully drives post-contingency line loadings from dangerous overloads ($>148\%$) down to safe operating limits ($<95\%$).
* **Cross-Platform Validation:** Rigorously cross-checked across Python, MATLAB, and PowerWorld to guarantee academic and technical precision.

---

## 🛠️ Tech Stack & Tools

* **Python (Pandapower & Scikit-Learn):** Automated N-1 contingency generation, multi-scenario load scaling, dataset creation, and Random Forest Regressor training (`bess_ml_controller
