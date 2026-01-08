# Quick Start Guide

### Step 1: Run the Setup Script (Easiest)

```bash
cd /Users/mohith/Desktop/Projects/Airbnb-Price-Analysis-and-Prediction
./setup_environment.sh
```

This will:
- Create a virtual environment
- Install all required packages
- Set up everything automatically

### Step 2: Activate Environment & Start Jupyter

```bash
# Activate the environment
source venv/bin/activate

# Start Jupyter Notebook
jupyter notebook
```

### Step 3: Open the Notebook

- Click on `Airbnb_Listing_Price_Analysis_and_Prediction.ipynb`
- The first cell will check your environment and show any issues

---

## Manual Setup (If Script Doesn't Work)

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Install packages
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install NLTK data (for textblob)
python3 -c "import nltk; nltk.download('punkt'); nltk.download('brown'); nltk.download('wordnet')"

# 5. Start Jupyter
jupyter notebook
```

---

## Verify Everything Works

Run the first cell in the notebook. You should see:
- Python environment information
- All libraries imported successfully
- Package versions displayed

If you see errors, check `ENVIRONMENT_SETUP.md` for troubleshooting.

---

## Important Notes

1. **Always activate your environment** before running Jupyter:
   ```bash
   source venv/bin/activate
   ```

2. **Data files**: Make sure you have `calendar.csv` and `listings.csv` in a `data/` folder, or update the `DATA_PATH` variable in the notebook.

3. **If Jupyter doesn't see your environment**: Install ipykernel:
   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name=airbnb-env
   ```
   Then select the kernel in Jupyter: Kernel → Change Kernel → airbnb-env

