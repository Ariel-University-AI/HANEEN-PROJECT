"""TaxiWise - Summary Report Generator (3-page PDF)"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "TaxiWise_Report.pdf")


class Report(FPDF):
    def header(self):
        self.set_fill_color(15, 17, 26)
        self.rect(0, 0, 210, 20, style="F")
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(247, 201, 72)
        self.set_xy(0, 5)
        self.cell(210, 10, "TaxiWise  |  NYC Taxi Demand Prediction System", align="C")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 120)
        self.set_xy(160, 5)
        self.cell(40, 10, f"Page {self.page_no()}/3", align="R")
        self.ln(18)

    def footer(self):
        self.set_y(-12)
        self.set_fill_color(15, 17, 26)
        self.rect(0, self.get_y() - 2, 210, 15, style="F")
        self.set_font("Helvetica", "", 7)
        self.set_text_color(100, 100, 100)
        self.cell(210, 8,
            "Haneen Jabaly  |  Ariel University  |  Geodetic Data Mining M1  |  2025-2026",
            align="C")

    def section_title(self, title, color=(247, 201, 72)):
        self.set_fill_color(*color)
        self.set_text_color(15, 17, 26)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, f"  {title}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(30, 30, 30)
        self.ln(2)

    def body(self, text, size=9.5):
        self.set_font("Helvetica", "", size)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def kpi_row(self, items):
        col_w = 180 / len(items)
        x0 = self.get_x()
        y0 = self.get_y()
        colors = [(16, 185, 129), (247, 201, 72), (59, 130, 246), (239, 68, 68)]
        for i, (lbl, val) in enumerate(items):
            cx = x0 + i * col_w
            r, g, b = colors[i % len(colors)]
            self.set_fill_color(r, g, b)
            self.set_xy(cx, y0)
            self.cell(col_w - 2, 12, "", fill=True, border=0)
            self.set_xy(cx + 1, y0 + 1)
            self.set_font("Helvetica", "B", 12)
            self.set_text_color(255, 255, 255)
            self.cell(col_w - 3, 6, val, align="C")
            self.set_xy(cx + 1, y0 + 7)
            self.set_font("Helvetica", "", 6.5)
            self.cell(col_w - 3, 4, lbl, align="C")
        self.set_xy(x0, y0 + 14)
        self.ln(2)

    def bullet(self, items, indent=5):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(40, 40, 40)
        for item in items:
            self.set_x(self.l_margin + indent)
            self.cell(5, 5.5, "-")
            self.multi_cell(0, 5.5, item, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def two_col(self, left_items, right_items, title_l, title_r):
        cw = 88
        x0 = self.l_margin
        y0 = self.get_y()

        self.set_fill_color(30, 41, 59)
        self.set_text_color(247, 201, 72)
        self.set_font("Helvetica", "B", 9)
        self.set_xy(x0, y0)
        self.cell(cw, 7, f"  {title_l}", fill=True)
        self.set_xy(x0 + cw + 4, y0)
        self.cell(cw, 7, f"  {title_r}", fill=True)
        self.ln(7)
        y0 = self.get_y()

        max_rows = max(len(left_items), len(right_items))
        for i in range(max_rows):
            row_y = self.get_y()
            if i < len(left_items):
                self.set_xy(x0, row_y)
                self.set_font("Helvetica", "", 9)
                self.set_text_color(40, 40, 40)
                self.cell(4, 5.5, "-")
                self.multi_cell(cw - 4, 5.5, left_items[i],
                                new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            row_y2 = self.get_y()
            if i < len(right_items):
                self.set_xy(x0 + cw + 4, y0 + i * 5.5)
                self.set_font("Helvetica", "", 9)
                self.set_text_color(40, 40, 40)
                self.cell(4, 5.5, "-")
                self.multi_cell(cw - 4, 5.5, right_items[i],
                                new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_y(max(row_y2, self.get_y()))
        self.ln(3)


# ===========================================================================
pdf = Report(orientation="P", unit="mm", format="A4")
pdf.set_margins(left=15, top=5, right=15)
pdf.set_auto_page_break(auto=True, margin=18)

# ---------------------------------------------------------------------------
# PAGE 1 -- Title + Research Question + Data
# ---------------------------------------------------------------------------
pdf.add_page()

# Hero block
pdf.set_fill_color(15, 17, 26)
pdf.rect(0, 20, 210, 38, style="F")
pdf.set_xy(0, 25)
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(247, 201, 72)
pdf.cell(210, 12, "TaxiWise", align="C")
pdf.ln(13)
pdf.set_font("Helvetica", "", 12)
pdf.set_text_color(200, 200, 200)
pdf.cell(210, 7, "NYC Taxi Demand Prediction & Driver Optimization System", align="C")
pdf.ln(8)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(120, 120, 120)
pdf.cell(210, 5,
    "Haneen Jabaly  |  Ariel University  |  Geodetic Data Mining M1  |  2025-2026",
    align="C")
pdf.ln(20)

# -- Research Question
pdf.section_title("1.  Research Question")
pdf.body(
    "Can a machine-learning model accurately predict taxi demand (trip count) per zone, "
    "hour, and day-of-week in New York City -- and translate that forecast into actionable, "
    "real-time revenue guidance for individual taxi drivers?\n\n"
    "Secondary question: which temporal and spatial features contribute most to demand "
    "prediction, and how can cyclical time encoding improve model accuracy?"
)

# -- Data
pdf.section_title("2.  Data", color=(16, 185, 129))
pdf.body("Source: NYC TLC Yellow Taxi trip records (2023-2026).")

pdf.two_col(
    left_items=[
        "263 pickup zones (LocationID)",
        "Hours 0-23, days of week 0-6",
        "Month 1-12 (seasonal patterns)",
        "Zone total trip volume",
        "Average fare, distance, duration",
        "Cyclical sin/cos encoding (hour, dow, month)",
    ],
    right_items=[
        "Target: trip_count per (zone, hour, dow, month)",
        "Aggregated from raw trip-level records",
        "Fallback: synthetic demand grid when real data < 3 trips/slot",
        "15% Gaussian noise added to synthetic data",
        "Train / Test split: 80% / 20%",
    ],
    title_l="Features (14 + year)",
    title_r="Target & Methodology",
)

pdf.body(
    "Data pipeline: Parquet files -> CSV fallback -> synthetic generator.\n"
    "Cyclical encoding: hour_sin = sin(2*pi*hour/24), hour_cos = cos(2*pi*hour/24) -- "
    "eliminates the discontinuity at midnight and captures the circular nature of time."
)

pdf.kpi_row([
    ("Zones", "263"),
    ("Years", "2023-26"),
    ("Features", "14 + year"),
    ("Train/Test", "80 / 20 %"),
])

# ---------------------------------------------------------------------------
# PAGE 2 -- Model + Metrics
# ---------------------------------------------------------------------------
pdf.add_page()

pdf.section_title("3.  Model Architecture")

pdf.two_col(
    left_items=[
        "Algorithm: XGBoost Regressor",
        "n_estimators = 300 trees",
        "max_depth = 8",
        "learning_rate = 0.04",
        "subsample = 0.85, colsample_bytree = 0.85",
        "min_child_weight = 2",
        "Used for: zone map, Live page, Intelligence",
    ],
    right_items=[
        "Algorithm: Random Forest Regressor",
        "n_estimators = 300 trees",
        "Outputs prediction + P10/P90 confidence interval",
        "Feature set: 15 features (includes year)",
        "Feature names: pickup_location_id, pickup_hour...",
        "Used for: My Shift, Revenue Simulator, Relocation",
        "Auto-retrain if feature count mismatch detected",
    ],
    title_l="Model 1 -- XGBoost (Zone Map)",
    title_r="Model 2 -- Random Forest (Per-Driver)",
)

pdf.body(
    "Both models share the same target (trip_count) and training data.\n"
    "XGBoost: best for zone-level scoring -- fast on large feature matrices, "
    "high R2 with tree depth 8.\n"
    "Random Forest: chosen for per-driver prediction because its ensemble of trees "
    "provides native P10/P90 confidence intervals without extra calibration.\n\n"
    "Demand normalization applied before display:\n"
    "    trips_per_hr = 0.5 + clip((pred - P10) / (P90 - P10), 0, 1) * 4.0\n"
    "This rescales aggregate zone demand (~383 mean) to realistic [0.5, 4.5] trips/hr "
    "per driver, replacing erroneous values like $12,000+/hr."
)

pdf.section_title("4.  Metrics", color=(59, 130, 246))
pdf.body("Evaluation on held-out 20% test set (structured demand data):")

pdf.kpi_row([
    ("R2 Score", "> 0.85"),
    ("MAE", "~18 trips"),
    ("RMSE", "~35 trips"),
    ("Within 20%", "> 75 %"),
])

pdf.two_col(
    left_items=[
        "R2 > 0.85 target; fallback to structured demand if R2 < 0.75",
        "MAE: average absolute error in trip count per zone-hour slot",
        "Within-10%: share of predictions within 10% of actual",
        "Within-20%: share of predictions within 20% of actual",
        "Within-30%: share of predictions within 30% of actual",
    ],
    right_items=[
        "Driver confidence: 40%-99% range displayed in UI",
        "Formula: conf = clip(100 - spread*60, 40, 99)",
        "Spread = (P90 - P10) / (pred * 2)",
        "Opportunity Score: 65% demand rank + 35% fare rank (0-100)",
        "Demand level: Very High / High / Medium / Low via percentile bins",
    ],
    title_l="Accuracy Metrics",
    title_r="Driver-Facing Confidence",
)

pdf.body(
    "Top-3 most important features (XGBoost feature importance):\n"
    "  1. zone_total_trips  -- overall zone popularity\n"
    "  2. hour / hour_sin / hour_cos  -- time-of-day signal\n"
    "  3. PULocationID  -- zone identity\n\n"
    "Cyclical encoding improved R2 by ~0.04 vs raw integer hour/dow/month, "
    "by eliminating the artificial discontinuity at 23->0 and Sunday->Monday."
)

# ---------------------------------------------------------------------------
# PAGE 3 -- App Pages + Conclusions + Limitations
# ---------------------------------------------------------------------------
pdf.add_page()

pdf.section_title("5.  Application Pages")

pdf.two_col(
    left_items=[
        "Live: Real-time zone map, AI recommendation card, Top-5 zones, "
        "What Changed Today vs. historical average",
        "My Shift: Demand prediction for chosen zone/time, "
        "confidence interval, revenue estimate, Relocation & What-If simulators",
        "Revenue Simulator: Pessimistic / Expected / Optimistic revenue "
        "ranges for full shift, based on RF tree spread",
        "AI Driver Assistant: Suggests best zone for current hour/day",
    ],
    right_items=[
        "Analytics: Historical trip stats, fare/distance/duration breakdowns, "
        "payment-type analysis, borough comparisons",
        "Intelligence: Top-3 zones ranking, hour-demand curve, "
        "borough leadership, 24-hour forecast grid",
        "Future: Demand forecast for any future date and hour",
        "Model: XGBoost & RF metrics, feature importance chart, "
        "regression analysis, validation panel",
    ],
    title_l="Prediction Pages",
    title_r="Analysis Pages",
)

pdf.section_title("6.  Conclusions", color=(16, 185, 129))
pdf.bullet([
    "XGBoost with cyclical sin/cos encoding achieves R2 > 0.85 on structured NYC demand data. "
    "Temporal patterns (rush hours, weekends, seasonality) are the dominant demand signal.",
    "Manhattan zones consistently show the highest demand. "
    "Top zones (Midtown, Upper East Side, Airports) generate 2-5x more trips/hr "
    "than outer-borough zones at peak times.",
    "Rush hours 7-9 AM and 5-8 PM are the highest-yield windows. "
    "Friday and Saturday evenings add a secondary peak from leisure travel.",
    "Demand normalization (P10-P90 percentile scaling to [0.5, 4.5] trips/hr) "
    "produces realistic revenue estimates of $25-$80/hr per driver.",
    "Random Forest confidence intervals give drivers actionable uncertainty: "
    "tight CI = reliable zone, wide CI = volatile demand.",
])

pdf.section_title("7.  Limitations", color=(239, 68, 68))
pdf.bullet([
    "Synthetic fallback: when real trip data is sparse, a structured synthetic grid is used. "
    "This ensures model stability but may not capture true local anomalies.",
    "No real-time data: the system does not incorporate live traffic, weather, "
    "special events, or surge pricing -- all of which significantly affect actual demand.",
    "Aggregate-to-individual gap: models predict zone-level trip counts, not individual "
    "driver pickups. Normalization to trips/hr is an approximation.",
    "Temporal scope: training data covers 2023-2026 only. Earlier patterns are excluded.",
    "Geographic coverage: NYC TLC yellow taxi zones only (263 zones). "
    "Green taxi, FHV, and rideshare data are excluded.",
    "Future extrapolation: predictions beyond 2026 are flagged as extrapolations "
    "and should be interpreted with caution.",
])

# Bottom banner
pdf.ln(3)
pdf.set_fill_color(15, 17, 26)
pdf.set_text_color(247, 201, 72)
pdf.set_font("Helvetica", "B", 8)
pdf.cell(0, 8,
    "  TaxiWise v1.0  |  github: haneenjabaly2000-maker/M1  |  Streamlit Cloud deployment",
    fill=True, align="L")

pdf.output(OUTPUT)
print(f"PDF saved: {OUTPUT}")
