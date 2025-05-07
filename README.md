
# 🌱 SWPA Innovation Ecosystem Rolodex Web App

This web application helps visualize, analyze, and match entrepreneurial needs with available support programs across Southwestern Pennsylvania. Built using **Streamlit**, **PostgreSQL**, and **OpenAI's LLMs**, it empowers ecosystem stakeholders to explore service providers, program offerings, and entrepreneurial challenges—all in one place.

---

## 📌 Project Overview

* 🗺 **Interactive Map**: Shows the geographic distribution of service providers and entrepreneurs.
* 📊 **Needs Dashboard**: Analyzes the frequency and type of support needs by county and service category.
* 🔍 **Program Explorer**: Filters programs by provider, county, vertical, and product type.
* 🤖 **LLM Assistant**: Recommends the best programs for entrepreneurs using OpenAI and LangChain.
* ☁️ **Real-time Sync**: Connects to a live **Supabase** database for data insertion and retrieval.

---

## 💻 Local Setup Instructions

### 1. Install Anaconda (or use a preferred Python environment manager)

* [Download Anaconda](https://www.anaconda.com/products/distribution) and install it.
* Alternatively, use `venv` or `virtualenv`.

### 2. Clone the repository

```bash
git clone https://github.com/sisazavi/rolodex
cd rolodex
```

### 3. Create and activate a virtual environment

```bash
conda create -n rolodex_env python=3.12
conda activate rolodex_env
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Set up environment variables

Create a `.streamlit/secrets.toml` file:

```toml
OPENAI_API_KEY = "your-openai-key"
SUPABASE_USER = "your-supabase-username"
SUPABASE_PASSWORD = "your-supabase-password"
SUPABASE_HOST = "your-supabase-host"
SUPABASE_PORT = "5432"
SUPABASE_DB = "your-database-name"
```

### 6. Run the app

```bash
streamlit run rolodex_webapp.py
```

---

## 📁 Project Structure

```bash
.
├── rolodex_webapp.py        # Main Streamlit app
├── functions.py             # Helper functions for querying, formatting, and matching
├── requirements.txt         # List of Python dependencies
├── .streamlit/
│   └── secrets.toml         # Environment variables (do not push to GitHub)
└── README.md                # You're here!
```

* `rolodex_webapp.py`: Organizes the app into tabs: About, Overview, Needs, Programs, and Matching Tool.
* `functions.py`: Contains utility functions for data retrieval, formatting, and LLM-based recommendations.
* `.streamlit/secrets.toml`: Safely stores sensitive environment variables for local and cloud deployment.

---

## 🚀 How to Use the App

Once the app is running:

1. **About Tab**: Learn what the app does and how to contribute.
2. **Rolodex Overview**: Visualize provider and entrepreneur locations.
3. **Needs Dashboard**: Explore the distribution of needs by county.
4. **Programs Tab**: Browse and filter programs by vertical, county, and product type.
5. **Matching Tool**: Select an entrepreneur and get tailored recommendations.

---

## 📦 Prerequisites

* Python 3.12+
* Streamlit
* OpenAI API Key
* Supabase PostgreSQL connection credentials

---

## 🛠 Troubleshooting Tips

* **App restarts unexpectedly**: Streamlit might be re-running due to file changes or memory usage. Try reducing large queries.
* **No data showing**: Check that your Supabase credentials are correct and the database is accessible.
* **API errors**: Ensure your `OPENAI_API_KEY` is valid and your usage limits are not exceeded.

---

## 🤝 Contribution Guidelines

We welcome contributions! To suggest improvements:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a detailed explanation

---

## 📄 License

MIT License. See `LICENSE` file for more information.

---

Let me know if you'd like me to generate a `LICENSE`, `.gitignore`, or Docker setup next.
