# Graph-Based Recommendation System Using User–Item Networks

A college demonstration project that shows how graph theory can power a recommendation engine. Users and items are nodes in a bipartite graph; edges represent interactions. The system finds similar users (via shared items) and recommends unseen items ranked by popularity among those similar users.

---

## Project Structure

```
lak/
├── app.py                  # Main Streamlit application
├── data/
│   └── interactions.csv    # Sample dataset (user_id, item_id, interaction)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Dataset

`data/interactions.csv` has three columns:

| Column      | Description                          |
|-------------|--------------------------------------|
| user_id     | User identifier (U1 – U10)           |
| item_id     | Item identifier (I1 – I10)           |
| interaction | Always 1 (indicates an interaction)  |

The dataset is fixed so results are fully reproducible.

---

## How the Recommendation Logic Works

1. **Build a bipartite graph** — user nodes on one side, item nodes on the other; an edge means the user interacted with that item.
2. **Find similar users** — other users who share at least one item with the selected user. The more shared items, the more similar.
3. **Score candidate items** — for every item a similar user touched that the selected user has NOT touched, increment that item's score by 1.
4. **Rank and return** — items are sorted by descending score; ties are broken alphabetically by item ID.

---

## Setup & Running Locally (Windows)

### Prerequisites
- Python 3.9 or higher installed
- `pip` available in your terminal

### Steps

Open **Command Prompt** or **PowerShell** and run:

```powershell
# 1. Navigate to the project folder
cd "C:\Users\ARVIN\OneDrive\Attachments\Desktop\lak"

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Streamlit Community Cloud Deployment

1. Push the project to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"** → select your repository and set the main file to `app.py`.
4. Click **"Deploy"** — Streamlit Cloud reads `requirements.txt` automatically.

> **Note:** Make sure `data/interactions.csv` is committed to the repository.

---

## Edge Cases Handled

| Situation                                | Behaviour                                       |
|------------------------------------------|-------------------------------------------------|
| User with no interactions                | Warning shown; no recommendations generated     |
| No similar users found                   | Info message; no recommendations generated      |
| All items already interacted with        | Info message; nothing new to recommend          |
| Fewer items available than requested top-N | Shows only available recommendations          |

---

## Technologies Used

- **Python** — core language
- **Streamlit** — interactive web interface
- **Pandas** — data loading and manipulation
- **NetworkX** — bipartite graph construction and traversal
- **Matplotlib** — graph visualisation
