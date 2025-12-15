Here’s a concise README you can drop into your repo and tweak as needed.

***

# LinkedIn Job Market Intelligence Dashboard

## Overview
This project is an interactive visual analytics dashboard built for the CS7DS4 / CSU44065 “Addressing Complexity” assignment. It explores LinkedIn job postings to reveal how skills cluster together and how they relate to salary levels, geographic markets, hiring organizations, and industries.

The key idea is to use a **skill co‑occurrence network as the main control surface**: clicking a skill node filters the entire dashboard and recomputes all statistics in real time.

## Dataset
- Source: LinkedIn Job Postings (Kaggle)  
- Tables used:  
  - Job postings  
  - Salaries  
  - Companies  
  - Skills and industries  
  - Job–skill mappings  
- Integrated analytical subset: ~25,000 jobs with complete salary, location, company, industry, and skill attributes.

Basic preprocessing:
- Join postings, salaries, companies, and taxonomies on their keys.  
- Remove implausible salary outliers (<20k, >500k).  
- Impute missing salaries with a log‑normal model.  
- Derive `skill_count` per job.  
- Extract US state codes from location strings.  
- Build a skill–skill co‑occurrence graph and prune edges with fewer than 25 co‑occurrences.

## Features
The dashboard presents six coordinated views on a single page:

1. **Skill Co‑Occurrence Network**  
   - Nodes: skills (size = frequency in postings)  
   - Edges: co‑occurrence strength (pruned to avoid hairball)  
   - Radial layout for stability  
   - Click a node to filter the entire dashboard.

2. **Salary Distribution (Box Plot)**  
   - Five‑number summary of salaries.  
   - Updates to show distribution for the selected skill.

3. **Geographic Salary Distribution (US Choropleth)**  
   - State‑level median salary for the current filter.  

4. **Top Hiring Organizations (Bar Chart)**  
   - Horizontal bars sorted by job count for the selected skill.

5. **Industry Composition (Treemap)**  
   - Area encodes job count by industry under the current filter.

6. **Skill Density vs Salary (Scatter + Trendline)**  
   - X: number of required skills; Y: salary.  
   - Polynomial trendline to show diminishing returns.

## Tech Stack
- Python  
- Plotly Dash  
- Pandas / NumPy for data processing

The app uses Dash callbacks and a shared data store so that a single click on the network triggers filtering and regeneration of all six views with low latency.

## Running the App

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd linkedin-job-intelligence-dashboard
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the dataset files in the `data/` folder (or update the paths in the code).

5. Run the Dash app:
   ```bash
   python app.py
   ```

6. Open a browser and go to:
   ```
   http://127.0.0.1:8050
   ```

## Usage

- Hover over nodes in the skill network to see skill labels.  
- Click a skill node to filter all views to jobs containing that skill.  
- Compare different skills by clicking nodes in turn and observing how salary, geography, companies, industries, and skill density patterns change.

## License / Academic Note

This project was developed as part of a university coursework assignment. If reusing or extending the code, please respect the dataset’s terms of use on Kaggle and acknowledge this project where appropriate.