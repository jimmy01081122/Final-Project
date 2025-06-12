# 🧠 TSP Genetic Algorithm Solver

A Traveling Salesman Problem (TSP) solver using Genetic Algorithm (GA), with auto-tuning and animation support.
This is a modified version of [Renato Maynard's TSP-GA project](https://github.com/RenatoMaynard/TSP-Genetic-Algorithm/blob/main/ga_interactive.py), modified by Jimmy Chang.

## 📁 Repository Structure

```
.
├── ga.py                     # Main GA solver with self-tuning & animation (latest)
├── ga_c.py                  # Older GA version (find-closest-only)
├── abctest/                 # Verilog + AIG models for testing ABC tools (N=4 TSP example)
├── GAtest_old_version/      # Intermediate test versions
├── GA_basic_self_coding_old_version/ # Early self-coded GA version
├── tsp/                     # Output logs from .tsp test cases
├── tspcase_file/            # Input .tsp problem files
```

## 🚀 How to Run

```bash
python3 ga.py
```

### Modes

* **Mode 1 (Self-Tuning Solver)**
  Input your target cost, and the algorithm will attempt to find matching or closest results using auto-tuned GA parameters.

* **Mode 2 (Fixed GA Solver)**
  Choose between **simple** or **advanced** GA version to find the best route possible.

## 🧩 Dependencies

* Python 3.8+
* `deap`
* `numpy`
* `matplotlib`
* `tsplib95`
* `IPython`

You can install all requirements using:
```
pip install deap numpy matplotlib tsplib95 ipython
```

## 📝 Metadata

| Field             | Info                                                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- |
| **File Name**     | `ga.py`                                                                                                          |
| **Description**   | TSP Solver using Genetic Algorithm with auto parameter tuning                                                    |
| **Origin Source** | [https://github.com/RenatoMaynard/TSP-Genetic-Algorithm](https://github.com/RenatoMaynard/TSP-Genetic-Algorithm) |
| **Modified by**   | Jimmy Chang                                                                                                      |
| **Storage**       | [https://github.com/jimmy01081122/Final-Project](https://github.com/jimmy01081122/Final-Project)                 |
| **Version**       | 2.0.0                                                                                                            |
| **Updated**       | 2025-06-12                                                                                                       |
| **License**       | MIT                                                                                                              |

## 📌 Notes

* Animation for route evolution is shown directly in **Jupyter Notebook** via `HTML` embedding.
* Supports `.tsp` files compatible with [TSPLIB format](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).

--------------------------
# 🧠 TSP 遺傳演算法求解器（GA Solver）

本專案是一個針對旅行推銷員問題（Traveling Salesman Problem, TSP）所設計的 **遺傳演算法（Genetic Algorithm, GA）求解器**。
在 Renato Maynard 的原始專案基礎上進行強化，加入了 **參數自動調整功能、自動化動畫繪製** 與更多測試案例。

> 🔗 原始來源：[RenatoMaynard/TSP-Genetic-Algorithm](https://github.com/RenatoMaynard/TSP-Genetic-Algorithm/blob/main/ga_interactive.py)
> ✏️ 修改作者：Jimmy Chang
> 📦 專案倉庫：[github.com/jimmy01081122/Final-Project](https://github.com/jimmy01081122/Final-Project)

---

## 📂 專案結構

```
.
├── ga.py                         # 最新版主程式（支援自調參數與動畫）
├── ga_c.py                       # 舊版僅支援最接近目標解
├── abctest/                      # Verilog 及 AIG 格式模型（含 N=4 的 TSP 測試）
├── GAtest_old_version/           # 測試用中期版本
├── GA_basic_self_coding_old_version/  # 最初自製版本
├── tsp/                          # 執行結果 log 紀錄
├── tspcase_file/                 # 放置 .tsp 測試案例
```

---

## ▶️ 執行方式

使用指令執行：

```bash
python3 ga.py
```

### 模式選擇

* **模式 1：目標成本自調參數（Self-Tuning Solver）**
  輸入想達到的目標成本，系統會自動調整 GA 參數，嘗試找到目標解或最接近的解。

* **模式 2：固定參數版本（Simple/Advanced GA）**
  可選擇簡單版或進階版 GA 執行，找出最佳路徑。

---

## 📦 相依套件

需先安裝以下套件（建議使用 Python 3.8+）：

```bash
pip install deap numpy matplotlib tsplib95 ipython
```

---

## 📌 檔案資訊

| 欄位名稱      | 內容                                                                                               |
| --------- | ------------------------------------------------------------------------------------------------ |
| **檔案名稱**  | `ga.py`                                                                                          |
| **說明**    | TSP 遺傳演算法求解器（支援動畫與參數自動調整）                                                                        |
| **原始作者**  | Renato Maynard                                                                                   |
| **修改者**   | Jimmy Chang                                                                                      |
| **儲存位置**  | [https://github.com/jimmy01081122/Final-Project](https://github.com/jimmy01081122/Final-Project) |
| **版本**    | 2.0.0                                                                                            |
| **最後更新日** | 2025-06-12                                                                                       |
| **授權**    | MIT License                                                                                      |

---

## 📈 額外功能

* ✅ 支援 `.tsp` 檔案（符合 TSPLIB 規格）自動解析距離矩陣
* ✅ 自動生成路徑動畫（支援 Jupyter Notebook 互動視覺化）
* ✅ 執行過程會產生 `.log` 檔，紀錄所有 print 資訊
* ✅ 成本演化圖與參數調整結果自動視覺化


## License
This repository is licensed under the MIT License. You are free to modify, share, and use this code for your own projects.

