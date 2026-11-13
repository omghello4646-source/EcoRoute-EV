# EcoRoute EV

**Driving-Style-Aware Machine Learning and Deep Learning for EV Energy Prediction, Routing, and Eco-Driving Feedback**

EcoRoute EV is a beginner-friendly Python + Streamlit educational prototype. It demonstrates route planning, public EV charging-station lookup, a transparent energy estimate, analytics, local trip history, CSV export, and a built-in rule-based assistant.

> **Important model note:** The included energy estimator is a transparent demonstration calculation, not a trained or validated ML/DL model. If you already have a working model in your original `app.py`, back up that file and merge its model-loading/prediction code before replacing it. Do not claim ML/DL accuracy unless you have evaluated a real model on an appropriate dataset.

## Features

- Local demo signup/login with salted PBKDF2 password hashes (not production authentication).
- Sidebar navigation: Dashboard, Route Planner, Charging Stations, Energy Prediction, Analytics, Trip History, EcoRoute Assistant, About.
- Route results stored in `st.session_state`, so they persist across Streamlit reruns and page changes in the same session.
- A failed route request does not clear the previous successful route.
- Route distance, estimated duration, and a Leaflet/OpenStreetMap map.
- Charging-station lookup from OpenStreetMap Overpass data.
- Transparent energy calculation and eco-driving suggestions.
- Per-user CSV trip history and CSV download.
- Friendly messages for common network and input failures.
- White-and-green styling.

## Technology

- Python
- Streamlit
- pandas
- Requests
- OpenStreetMap Nominatim (geocoding)
- OSRM public routing service
- OpenStreetMap Overpass API (charging stations)
- JSON and CSV for local storage

## Project structure

```text
EcoRouteEV/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── user_data/              # created automatically at runtime
    ├── users.json          # local password hashes
    └── history_<username>.csv
```

## Windows installation

### 1. Open Command Prompt

Press `Win + R`, type `cmd`, and press Enter.

### 2. Go to the project folder

```bat
cd /d "C:\Users\Gladish T\OneDrive\Desktop\EcoRouteEV"
```

If your folder is somewhere else, change the path to match the actual location.

### 3. Inspect the current files first

```bat
dir
```

Make a backup of the existing `app.py` before replacing it, especially if it contains your current prediction model. In File Explorer, turn on **View → Show → File name extensions** to spot names such as `requirements.txt.txt`.

### 4. Create and activate a virtual environment

If a working `.venv` already exists, you can activate it instead of creating another one.

```bat
python -m venv .venv
.venv\Scripts\activate
```

If `python` is not recognised, try `py -3 -m venv .venv`. If neither command works, install Python from the official Python website and ensure the PATH option is selected where available. Follow any device administrator policies.

### 5. Install packages

Save the supplied dependency list as `requirements.txt`, then run:

```bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If installation is blocked by a managed-device policy, contact the device administrator or use an approved Python environment. Do not bypass organisational security controls.

### 6. Start the application

```bat
python -m streamlit run app.py
```

A successful start normally prints a local URL such as `http://localhost:8501`. Open that URL in your browser. Keep the Command Prompt window open while using the app.

### 7. Stop the application

Click the Command Prompt window running Streamlit and press `Ctrl + C`.

## How to use it

1. Create an account from the signup tab, then log in.
2. Use **Route Planner** to enter a start and destination. Route output stays in session state after reruns. Save a successful route to history using the save button.
3. Use **Charging Stations** to search near a place and choose a radius.
4. Use **Energy Prediction** to enter distance, vehicle consumption in kWh/100 km, and driving style. Save the estimate to history if desired.
5. Use **Analytics** for charts based on saved records.
6. Use **Trip History** to review records and download CSV.
7. Ask the built-in **EcoRoute Assistant** about supported topics.

## Energy calculation and limitations

The included estimator calculates:

`energy_kWh = distance_km × efficiency_kWh_per_100_km ÷ 100 × driving_style_multiplier`

Illustrative multipliers:
- Eco: `0.90`
- Normal: `1.00`
- Aggressive: `1.20`

These multipliers are assumptions for demonstration only. Actual EV energy use depends on the vehicle, speed, traffic, weather, road elevation, payload, tyres, cabin heating/cooling, battery condition, and driving behaviour. This code does not establish real-world accuracy and does not implement deep learning.

## Public API limitations

Nominatim, OSRM, Overpass, and OpenStreetMap tile servers are public services. They can be unavailable, slow, rate-limited, or incomplete. This prototype uses request timeouts and error messages, but availability is not guaranteed. Use services respectfully and check their usage policies before deployment or high-volume use.

Charging station data may be missing or outdated. The app does not verify operational status, live availability, connector compatibility, or prices.

## Local data and privacy

- Account hashes: `user_data/users.json`
- Trip history: `user_data/history_<username>.csv`

These files are stored on the computer running the app. Back up the `user_data` folder privately. Do not commit it to a public GitHub repository. The JSON login system is for a local educational prototype only; it is not designed for production authentication or untrusted multi-user hosting.

## Troubleshooting

- **`app.py` not found:** run `dir`, check the folder path, and confirm the filename is exactly `app.py`.
- **`requirements.txt` not found:** enable file extensions in File Explorer and rename `requirements.txt.txt` to `requirements.txt`.
- **Wrong directory:** run the `cd /d "..."` command above again.
- **Python not recognised:** try `py -3 --version`; otherwise install Python through an approved method.
- **Missing package:** activate `.venv` and rerun `python -m pip install -r requirements.txt`.
- **Package install blocked:** follow the device's approved software-installation process; do not bypass policy restrictions.
- **Port already in use:** run `python -m streamlit run app.py --server.port 8502`.
- **Route timeout:** retry later; public routing services can be busy.
- **Location not found:** add a city/state/country to make the place more specific.
- **Overpass unavailable:** try again later or use a smaller search radius.
- **No charging stations:** mapped data may be incomplete; try another radius or nearby area.
- **Route disappears:** this app stores route output in `st.session_state`, which persists through reruns in the same session. It does not persist across a brand-new browser session or server restart.
- **History not created:** save a route/prediction and confirm the app can write to the project folder. Check `user_data/` exists.
- **Broken map tiles:** the map needs internet access to load Leaflet and OpenStreetMap tiles.

## Testing checklist

- [ ] App starts without syntax errors.
- [ ] Signup accepts a valid username and password.
- [ ] Duplicate username is rejected.
- [ ] Login and logout work.
- [ ] `users.json` contains password hashes, not raw passwords.
- [ ] Energy estimate displays kWh and kWh/km.
- [ ] A valid route displays distance, duration, and map.
- [ ] Route remains visible after changing pages and returning in the same session.
- [ ] Invalid location displays an understandable error.
- [ ] Failed route request does not erase a previously successful route.
- [ ] Second valid route replaces the displayed route.
- [ ] Charging-station lookup handles both results and empty results.
- [ ] Analytics shows an empty state before history exists.
- [ ] Chatbot responds without an API key.
- [ ] Records are saved to the correct username's history file.
- [ ] CSV export downloads successfully.
- [ ] UI follows the white-and-green theme.
- [ ] `user_data/` is excluded from Git.

## GitHub upload

1. Create a private or public GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore`.
3. Do **not** upload `.venv/`, `user_data/`, account records, or private trip history.
4. If using Git, run:

```bat
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Build EcoRoute EV prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the example GitHub URL with your own repository URL.

## Possible future improvements

- Integrate and document the user's existing trained model.
- Use real EV telemetry with appropriate evaluation and train/test separation.
- Add unit tests for authentication, route-state handling, and energy calculations.
- Add vehicle-specific efficiency presets and charging-stop planning.
- Consider a production identity provider and database before any public multi-user deployment.
