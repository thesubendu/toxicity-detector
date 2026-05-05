import tkinter as tk
from tkinter import ttk, messagebox
import threading
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─── Detector Core ────────────────────────────────────────────────────────────

class SimpleToxicityDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.accuracy = 0.0

    def calculate_simple_descriptors(self, smiles):
        try:
            if not smiles or not isinstance(smiles, str):
                return None
            smiles_upper = smiles.upper()
            descriptors = {
                'length': len(smiles),
                'carbon_count': smiles_upper.count('C'),
                'oxygen_count': smiles_upper.count('O'),
                'nitrogen_count': smiles_upper.count('N'),
                'sulfur_count': smiles_upper.count('S'),
                'phosphorus_count': smiles_upper.count('P'),
                'fluorine_count': smiles_upper.count('F'),
                'chlorine_count': smiles_upper.count('CL'),
                'bromine_count': smiles_upper.count('BR'),
                'iodine_count': smiles_upper.count('I'),
                'ring_count': sum(1 for c in smiles if c.isdigit()),
                'aromatic_count': sum(1 for c in smiles if c in 'cnops'),
                'double_bond_count': smiles.count('='),
                'triple_bond_count': smiles.count('#'),
                'branch_count': smiles.count('('),
                'charge_count': smiles.count('+') + smiles.count('-'),
                'hydroxyl_count': smiles_upper.count('OH'),
                'carbonyl_count': smiles.count('C=O') + smiles.count('O=C'),
            }
            total_atoms = (descriptors['carbon_count'] + descriptors['oxygen_count'] +
                           descriptors['nitrogen_count'] + descriptors['sulfur_count'] +
                           descriptors['phosphorus_count'])
            if total_atoms > 0:
                descriptors['oxygen_ratio'] = descriptors['oxygen_count'] / total_atoms
                descriptors['nitrogen_ratio'] = descriptors['nitrogen_count'] / total_atoms
                descriptors['halogen_ratio'] = (descriptors['fluorine_count'] +
                                                descriptors['chlorine_count'] +
                                                descriptors['bromine_count']) / total_atoms
                descriptors['complexity'] = descriptors['length'] / max(total_atoms, 1)
                descriptors['heteroatom_ratio'] = (total_atoms - descriptors['carbon_count']) / total_atoms
            else:
                descriptors['oxygen_ratio'] = 0
                descriptors['nitrogen_ratio'] = 0
                descriptors['halogen_ratio'] = 0
                descriptors['complexity'] = descriptors['length']
                descriptors['heteroatom_ratio'] = 0
            return descriptors
        except Exception:
            return None

    def create_sample_data(self, n_samples=800):
        known_compounds = [
            ('CCO', 1), ('CC(=O)Oc1ccccc1C(=O)O', 1),
            ('CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 1), ('c1ccccc1', 1),
            ('CCl4', 1), ('CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 1),
            ('COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1', 1),
            ('OC1C(O)C(O)C(CO)OC1O', 0), ('CC(C)(C)C(=O)O', 0),
            ('CCCCCCCCCCCCCCCC(=O)O', 0), ('NC(CC(=O)O)C(=O)O', 0),
            ('NC(CCC(=O)O)C(=O)O', 0), ('CC(C)CCCC(C)CCCC(C)CCCC(C)C', 0),
            ('OC(CO)(CO)CO', 0),
        ]
        np.random.seed(42)
        data = []
        for smiles, toxicity in known_compounds:
            d = self.calculate_simple_descriptors(smiles)
            if d:
                d['toxicity'] = toxicity
                data.append(d)
        base_toxic = [c for c, t in known_compounds if t == 1]
        base_safe  = [c for c, t in known_compounds if t == 0]
        for label, pool in [(1, base_toxic), (0, base_safe)]:
            for _ in range((n_samples - len(known_compounds)) // 2):
                d = self.calculate_simple_descriptors(np.random.choice(pool))
                if d:
                    for k in d:
                        d[k] = max(0, d[k] + np.random.normal(0, 0.2))
                    d['toxicity'] = label
                    data.append(d)
        return pd.DataFrame(data)

    def train(self, n_samples=800):
        data = self.create_sample_data(n_samples)
        X = data.drop('toxicity', axis=1)
        y = data['toxicity']
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)
        self.model.fit(X_train_s, y_train)
        y_pred = self.model.predict(X_test_s)
        self.accuracy = accuracy_score(y_test, y_pred)

    def predict(self, smiles):
        d = self.calculate_simple_descriptors(smiles)
        if d is None:
            return None, "Invalid SMILES string"
        df = pd.DataFrame([d])[self.feature_names]
        fs = self.scaler.transform(df)
        pred = self.model.predict(fs)[0]
        prob = self.model.predict_proba(fs)[0]
        return {
            'prediction': 'Toxic' if pred == 1 else 'Safe',
            'confidence': float(max(prob)),
            'safe_prob':  float(prob[0]),
            'toxic_prob': float(prob[1]),
            'descriptors': d,
        }, None


# ─── Colours & fonts ──────────────────────────────────────────────────────────

BG       = "#0f1117"
CARD     = "#1a1d27"
BORDER   = "#2a2d3e"
ACCENT   = "#6c63ff"
SAFE_COL = "#22c55e"
TOXIC_COL= "#ef4444"
TEXT     = "#e2e8f0"
SUBTEXT  = "#94a3b8"
FONT_FAM = "Times New Roman"

EXAMPLES = [
    ("Aspirin",  "CC(=O)Oc1ccccc1C(=O)O"),
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("Glucose",  "OC1C(O)C(O)C(CO)OC1O"),
    ("Ethanol",  "CCO"),
    ("Benzene",  "c1ccccc1"),
    ("Methane",  "C"),
]


# ─── Main App ─────────────────────────────────────────────────────────────────

class ToxicityApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Drug Toxicity Detector")
        self.configure(bg=BG)
        self.geometry("960x720")
        self.minsize(800, 620)
        self.detector = SimpleToxicityDetector()
        self.trained   = False
        self.history   = []   # list of dicts

        self._build_ui()
        self._start_training()

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Header ──
        hdr = tk.Frame(self, bg=CARD, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🧪  Drug Toxicity Detector",
                 font=(FONT_FAM, 18, "bold"), bg=CARD, fg=TEXT).pack(side="left", padx=20)
        self.status_lbl = tk.Label(hdr, text="⏳  Training model…",
                                   font=(FONT_FAM, 10), bg=CARD, fg=SUBTEXT)
        self.status_lbl.pack(side="right", padx=20)

        # ── Progress bar (shown during training) ──
        self.prog_frame = tk.Frame(self, bg=BG)
        self.prog_frame.pack(fill="x", padx=20, pady=(8, 0))
        self.progress = ttk.Progressbar(self.prog_frame, mode="indeterminate",
                                        style="Accent.Horizontal.TProgressbar")
        self.progress.pack(fill="x")

        # ── Body: left panel + right panel ──
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=12)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_right(body)

        # ── History table at bottom ──
        self._build_history()

        # ── Style ──
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Accent.Horizontal.TProgressbar",
                        troughcolor=CARD, background=ACCENT, bordercolor=CARD)
        style.configure("History.Treeview",
                        background=CARD, foreground=TEXT, fieldbackground=CARD,
                        rowheight=26, font=(FONT_FAM, 9))
        style.configure("History.Treeview.Heading",
                        background=BORDER, foreground=SUBTEXT,
                        font=(FONT_FAM, 9, "bold"))
        style.map("History.Treeview", background=[("selected", ACCENT)])

    def _card(self, parent, title, **grid_kw):
        outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        outer.grid(**grid_kw, sticky="nsew", padx=6, pady=6)
        inner = tk.Frame(outer, bg=CARD, padx=14, pady=12)
        inner.pack(fill="both", expand=True)
        tk.Label(inner, text=title, font=(FONT_FAM, 10, "bold"),
                 bg=CARD, fg=SUBTEXT).pack(anchor="w", pady=(0, 8))
        return inner

    def _build_left(self, body):
        frame = self._card(body, "ANALYSE COMPOUND", row=0, column=0)
        frame.rowconfigure(6, weight=1)

        # Compound name
        tk.Label(frame, text="Compound Name (optional)",
                 font=(FONT_FAM, 9), bg=CARD, fg=SUBTEXT).pack(anchor="w")
        self.name_var = tk.StringVar()
        tk.Entry(frame, textvariable=self.name_var,
                 font=(FONT_FAM, 11), bg=BORDER, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT).pack(fill="x", ipady=6, pady=(2, 10))

        # SMILES input
        tk.Label(frame, text="SMILES String",
                 font=(FONT_FAM, 9), bg=CARD, fg=SUBTEXT).pack(anchor="w")
        self.smiles_var = tk.StringVar()
        smiles_entry = tk.Entry(frame, textvariable=self.smiles_var,
                                font=(FONT_FAM, 11), bg=BORDER, fg=TEXT,
                                insertbackground=TEXT, relief="flat",
                                highlightthickness=1, highlightbackground=BORDER,
                                highlightcolor=ACCENT)
        smiles_entry.pack(fill="x", ipady=6, pady=(2, 4))
        smiles_entry.bind("<Return>", lambda e: self._analyse())

        # Analyse button
        self.analyse_btn = tk.Button(
            frame, text="Analyse", font=(FONT_FAM, 11, "bold"),
            bg=ACCENT, fg="white", relief="flat", cursor="hand2",
            activebackground="#5a52e0", activeforeground="white",
            state="disabled", command=self._analyse)
        self.analyse_btn.pack(fill="x", ipady=8, pady=(6, 14))

        # Quick examples
        tk.Label(frame, text="Quick Examples",
                 font=(FONT_FAM, 9), bg=CARD, fg=SUBTEXT).pack(anchor="w", pady=(0, 4))
        eg_grid = tk.Frame(frame, bg=CARD)
        eg_grid.pack(fill="x")
        for i, (name, smi) in enumerate(EXAMPLES):
            btn = tk.Button(
                eg_grid, text=name, font=(FONT_FAM, 9),
                bg=BORDER, fg=TEXT, relief="flat", cursor="hand2",
                activebackground=ACCENT, activeforeground="white",
                command=lambda n=name, s=smi: self._load_example(n, s))
            btn.grid(row=i//2, column=i%2, padx=3, pady=3, sticky="ew")
        eg_grid.columnconfigure(0, weight=1)
        eg_grid.columnconfigure(1, weight=1)

    def _build_right(self, body):
        frame = self._card(body, "RESULT", row=0, column=1)

        # Big verdict label
        self.verdict_lbl = tk.Label(
            frame, text="—", font=(FONT_FAM, 36, "bold"),
            bg=CARD, fg=SUBTEXT)
        self.verdict_lbl.pack(pady=(4, 2))

        self.conf_lbl = tk.Label(frame, text="Enter a SMILES string to begin",
                                 font=(FONT_FAM, 10), bg=CARD, fg=SUBTEXT)
        self.conf_lbl.pack()

        # Probability bars
        bars_frame = tk.Frame(frame, bg=CARD)
        bars_frame.pack(fill="x", pady=14)
        bars_frame.columnconfigure(1, weight=1)

        tk.Label(bars_frame, text="Safe",  font=(FONT_FAM, 9), bg=CARD, fg=SAFE_COL,  width=6, anchor="e").grid(row=0, column=0, padx=(0,8))
        tk.Label(bars_frame, text="Toxic", font=(FONT_FAM, 9), bg=CARD, fg=TOXIC_COL, width=6, anchor="e").grid(row=1, column=0, padx=(0,8))

        self.safe_bar  = self._bar(bars_frame, SAFE_COL,  row=0)
        self.toxic_bar = self._bar(bars_frame, TOXIC_COL, row=1)

        self.safe_pct  = tk.Label(bars_frame, text="—", font=(FONT_FAM, 9), bg=CARD, fg=TEXT, width=5)
        self.safe_pct.grid(row=0, column=2, padx=(8,0))
        self.toxic_pct = tk.Label(bars_frame, text="—", font=(FONT_FAM, 9), bg=CARD, fg=TEXT, width=5)
        self.toxic_pct.grid(row=1, column=2, padx=(8,0))

        # Molecular descriptors
        tk.Label(frame, text="Molecular Descriptors",
                 font=(FONT_FAM, 9, "bold"), bg=CARD, fg=SUBTEXT).pack(anchor="w", pady=(6, 4))
        desc_outer = tk.Frame(frame, bg=BORDER, padx=1, pady=1)
        desc_outer.pack(fill="both", expand=True)
        desc_inner = tk.Frame(desc_outer, bg=CARD)
        desc_inner.pack(fill="both", expand=True)

        self.desc_canvas = tk.Canvas(desc_inner, bg=CARD, highlightthickness=0)
        sb = tk.Scrollbar(desc_inner, orient="vertical", command=self.desc_canvas.yview)
        self.desc_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.desc_canvas.pack(side="left", fill="both", expand=True)
        self.desc_frame = tk.Frame(self.desc_canvas, bg=CARD)
        self.desc_canvas.create_window((0, 0), window=self.desc_frame, anchor="nw")
        self.desc_frame.bind("<Configure>",
            lambda e: self.desc_canvas.configure(scrollregion=self.desc_canvas.bbox("all")))

    def _bar(self, parent, color, row):
        bg_bar = tk.Frame(parent, bg=BORDER, height=14)
        bg_bar.grid(row=row, column=1, sticky="ew", pady=4)
        fill = tk.Frame(bg_bar, bg=color, height=14, width=0)
        fill.place(x=0, y=0, relheight=1)
        # Store reference on bg_bar
        bg_bar.fill = fill
        return bg_bar

    def _build_history(self):
        outer = tk.Frame(self, bg=BORDER, padx=1, pady=1)
        outer.pack(fill="x", padx=16, pady=(0, 12))
        inner = tk.Frame(outer, bg=CARD, padx=14, pady=10)
        inner.pack(fill="both")

        hdr = tk.Frame(inner, bg=CARD)
        hdr.pack(fill="x")
        tk.Label(hdr, text="ANALYSIS HISTORY",
                 font=(FONT_FAM, 9, "bold"), bg=CARD, fg=SUBTEXT).pack(side="left")
        tk.Button(hdr, text="Clear", font=(FONT_FAM, 9),
                  bg=BORDER, fg=SUBTEXT, relief="flat", cursor="hand2",
                  activebackground=TOXIC_COL, activeforeground="white",
                  command=self._clear_history).pack(side="right")

        cols = ("name", "smiles", "prediction", "confidence", "safe%", "toxic%")
        self.tree = ttk.Treeview(inner, columns=cols, show="headings",
                                  height=5, style="History.Treeview")
        heads = {"name": ("Name", 120), "smiles": ("SMILES", 220),
                 "prediction": ("Prediction", 90), "confidence": ("Confidence", 90),
                 "safe%": ("Safe %", 80), "toxic%": ("Toxic %", 80)}
        for col, (label, w) in heads.items():
            self.tree.heading(col, text=label)
            self.tree.column(col, width=w, anchor="center")
        self.tree.tag_configure("safe",  foreground=SAFE_COL)
        self.tree.tag_configure("toxic", foreground=TOXIC_COL)

        vsb = ttk.Scrollbar(inner, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.tree.pack(fill="x")

    # ── Training ─────────────────────────────────────────────────────────────

    def _start_training(self):
        self.progress.start(12)
        t = threading.Thread(target=self._train_thread, daemon=True)
        t.start()

    def _train_thread(self):
        self.detector.train()
        self.after(0, self._on_trained)

    def _on_trained(self):
        self.progress.stop()
        self.prog_frame.pack_forget()
        self.trained = True
        self.analyse_btn.config(state="normal")
        acc = self.detector.accuracy
        self.status_lbl.config(
            text=f" Model ready  •  Accuracy: {acc:.1%}",
            fg=SAFE_COL)

    # ── Analysis ─────────────────────────────────────────────────────────────

    def _load_example(self, name, smiles):
        self.name_var.set(name)
        self.smiles_var.set(smiles)
        if self.trained:
            self._analyse()

    def _analyse(self):
        if not self.trained:
            return
        smiles = self.smiles_var.get().strip()
        if not smiles:
            messagebox.showwarning("Input needed", "Please enter a SMILES string.")
            return
        name = self.name_var.get().strip() or smiles

        result, error = self.detector.predict(smiles)
        if error:
            messagebox.showerror("Error", error)
            return

        self._update_result(result)
        self._add_history(name, smiles, result)

    def _update_result(self, r):
        pred   = r['prediction']
        color  = SAFE_COL if pred == "Safe" else TOXIC_COL
        icon   = "" if pred == "Safe" else ""
        self.verdict_lbl.config(text=f"{icon}  {pred}", fg=color)
        self.conf_lbl.config(
            text=f"Confidence: {r['confidence']:.1%}", fg=TEXT)

        # Update bars
        self._set_bar(self.safe_bar,  r['safe_prob'],  SAFE_COL)
        self._set_bar(self.toxic_bar, r['toxic_prob'], TOXIC_COL)
        self.safe_pct.config( text=f"{r['safe_prob']:.1%}")
        self.toxic_pct.config(text=f"{r['toxic_prob']:.1%}")

        # Descriptors grid
        for w in self.desc_frame.winfo_children():
            w.destroy()
        show = ['length','carbon_count','oxygen_count','nitrogen_count',
                'ring_count','aromatic_count','double_bond_count',
                'branch_count','complexity','heteroatom_ratio',
                'halogen_ratio','hydroxyl_count']
        labels = {
            'length':'SMILES Length','carbon_count':'Carbons',
            'oxygen_count':'Oxygens','nitrogen_count':'Nitrogens',
            'ring_count':'Ring Count','aromatic_count':'Aromatic Atoms',
            'double_bond_count':'Double Bonds','branch_count':'Branches',
            'complexity':'Complexity','heteroatom_ratio':'Heteroatom Ratio',
            'halogen_ratio':'Halogen Ratio','hydroxyl_count':'Hydroxyl Groups',
        }
        for i, key in enumerate(show):
            val = r['descriptors'].get(key, 0)
            row, col = divmod(i, 2)
            cell = tk.Frame(self.desc_frame, bg=CARD)
            cell.grid(row=row, column=col, sticky="ew", padx=4, pady=2)
            tk.Label(cell, text=labels.get(key, key),
                     font=(FONT_FAM, 8), bg=CARD, fg=SUBTEXT,
                     anchor="w", width=18).pack(side="left")
            tk.Label(cell, text=f"{val:.2f}" if isinstance(val, float) else str(val),
                     font=(FONT_FAM, 8, "bold"), bg=CARD, fg=TEXT).pack(side="right")
        self.desc_frame.columnconfigure(0, weight=1)
        self.desc_frame.columnconfigure(1, weight=1)

    def _set_bar(self, bar_widget, prob, color):
        bar_widget.update_idletasks()
        total_w = bar_widget.winfo_width()
        fill_w  = int(total_w * prob)
        bar_widget.fill.config(width=fill_w, bg=color)

    def _add_history(self, name, smiles, r):
        tag = "safe" if r['prediction'] == "Safe" else "toxic"
        values = (
            name,
            smiles[:40] + ("…" if len(smiles) > 40 else ""),
            r['prediction'],
            f"{r['confidence']:.1%}",
            f"{r['safe_prob']:.1%}",
            f"{r['toxic_prob']:.1%}",
        )
        self.tree.insert("", 0, values=values, tags=(tag,))

    def _clear_history(self):
        for item in self.tree.get_children():
            self.tree.delete(item)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ToxicityApp()
    app.mainloop()
