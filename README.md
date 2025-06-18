# 💭 Dream Symbol Interpreter

> Unlock the hidden meaning of your dreams using Machine Learning and Large Language Models.

**Dream Symbol Interpreter** is a Streamlit-based web application that takes a user's dream description and provides a personalized 3-paragraph interpretation using AI. It combines symbol recognition via XGBoost and expressive natural language generation using the **Mistral AI API**.

---

## 🌟 Features

- 🧠 **Smart Symbol Detection**: Uses TF-IDF and an XGBoost model to detect symbolic keywords from the user’s input.
- 🧾 **Dynamic Interpretation**: Generates insightful interpretations in three paragraphs using **Mistral LLMs**.
- 🎨 **Custom Tone Selection**:
  - ✨ Mystical
  - 🎨 Poetic
  - 🧠 Psychological
- 🎨 **Stylized UI**: A beautifully designed blurred-gradient interface for an immersive experience.
- 🧠 **Fast, Real-Time Inference**: Integrated caching, optimized ML pipeline.

---

## 🛠 Tech Stack

| Tool        | Role                                |
|-------------|-------------------------------------|
| Python      | Core programming language           |
| Streamlit   | Web application framework           |
| TF-IDF      | Feature extraction from text        |
| XGBoost     | Dream symbol classification         |
| Mistral AI  | Language model for interpretation   |
| Pandas      | Data processing                     |
| Requests    | API integration                     |

---

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.8+
- A Mistral AI API Key (get one from https://mistral.ai)
- `combined_dataset.csv` file with `Word`, `Interpretation`, `Alphabet` columns
- Background image (optional): `bg.jpg`

### 📦 Installation

Clone the repository:


## git clone https://github.com/your-username/dream-symbol-interpreter.git
cd dream-symbol-interpreter
