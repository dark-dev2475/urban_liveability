import pandas as pd
import matplotlib.pyplot as plt
import io
import sys

# ==========================================
# ⚙️ CONFIGURATION (USE YOUR ORIGINAL FILES)
# ==========================================
# Use 'r' to handle backslashes correctly
FILE_1_PATH = r'C:\Users\gagan\OneDrive\Desktop\urban_bot\dataset_comparison\file1_synced_final.csv'
FILE_2_PATH = r'C:\Users\gagan\OneDrive\Desktop\urban_bot\dataset_comparison\file2_synced_final.csv'
FILE_3_PATH = r'C:\Users\gagan\OneDrive\Desktop\urban_bot\dataset_comparison\MASTER_REFERENCE.csv' # Note: .xlsx

# ==========================================
# 1. THE MASTER REFERENCE (Your Provided List)
# ==========================================
master_csv_data = """Location_Name,Livability Score
Mumbai,0
Dharavi,0
Koliwada,0
Worli,1
Bandra,1
Sion,0
Kurla,0
Ghatkopar,1
Mulund,1
Kandivali,1
Borivali,1
Andheri East,0
Vihar,0
Nalasopara,0
Virar,0
Panvel,1
Bhandup,0
Thane City,1
Thane-Ghodbunder,1
Kalyan-Dombivli,1
Navi Mumbai-Vashi,1
Navi Mumbai-Nerul,1
Navi Mumbai-Ghansoli,0
Navi Mumbai-Taloja,0
Ambernath,0
Badlapur,1
Ulhasnagar,0
Vasai-Virar,0
Mira-Bhayandar,1
Bhiwandi,0
Palghar,1
Bhusawal,0
Raver,0
Jalgaon,1
Amalner,0
Chalisgaon,0
Nashik,1
Nashik-Informal Pockets,0
Manmad,0
Yeola,1
Sinnar,0
Sangamner,0
Igatpuri,1
Malegaon,0
Dondaicha,1
Shirpur,1
Jamner,1
Chopda,1
Aurangabad,0
Paithan,0
Vaijapur,1
Sillod,1
Phulambri,1
Partur,1
Jalna,0
Ambejogai,1
Majalgaon,1
Latur,0
Udgir,0
Beed,0
Parli Vaijnath,0
Ashti,1
Ambajogai,1
Parbhani,0
Gangakhed,1
Jintur,1
Sailu,1
Selu,1
Manvat,1
Nanded,0
Biloli,1
Kinwat,1
Loha,1
Hadgaon,1
Mudkhed,1
Deglur,1
Basmatnagar,1
Hingoli,0
Washim,0
Karanja,1
Mangrulpir,1
Akola,1
Akot,1
Balapur,1
Murtizapur,1
Amravati,0
Achalpur,1
Chandurbazar,1
Anjangaon,1
Morshi,1
Warud,1
Yavatmal,0
Wani,1
Umarkhed,1
Ghatanji,1
Pusad,1
Digras,1
Buldhana,1
Malkapur,1
Khamgaon,1
Nandura,1
Shegaon,1
Chikhli,1
Nagpur,1
Nagpur-Squatter Areas,0
Ambazari,0
Wardha,1
Hinganghat,1
Arvi,1
Chandrapur,0
Warora,0
Bhadravati,1
Ballarpur,1
Gondiya,1
Bhandara,1
Tumsar,1
Gadchiroli,0
Pune,1
Pune-Shivajinagar,1
Pune-Yerwada,0
Pune-Hadapsar,0
Pune-Hinjewadi,0
Pune-Koregaon Park,1
Pune-Magarpatta,1
Pimpri-Chinchwad,0
Baramati,1
Daund,1
Chakan,1
Talegaon Dhabade,0
Lonavala,1
Kamshet,1
Satara,1
Phaltan,0
Karad,0
Vita,0
Islampur,1
Solapur,0
Barshi,1
Akkalkot,1
Pandharpur,0
Jaysingpur,1
Kolhapur,1
Ichalkaranji,0
Sangli,0
Miraj,1
Tasgaon,0
Ratnagiri,0
Chiplun,0
Dapoli,0
Guhagar,0
Sindhudurg-Vengurla,1
Kankavli,1
Kudal,1
Raigad,1
Roha,0
Pen,0
Mahad,0
Uran,0
Palghar,0
Boisar,0
Jawhar,1
Mokhada,1
Nashik-City,1
Dhule,0
Nandurbar,1
Taloda,1
Shahada,1
Shirdi,1
Shirpur-Varwade,1
Ahmednagar,1
Shrirampur,0
Kopargaon,0
Pathardi,1
Rahuri,1
Shrigonda,1
Rahata,1
Newasa,1
Akola (City),1
Telhara,1
Khandesh Region,1
Panchgani,0
Mahabaleshwar,0
Matheran,0
Malshej Ghat,0
Bhandardara,0
Aurangabad-City,1
Jalgaon-City,1
Amravati-City,0
Nagpur-City,1
Thane-City,1
Navi-Mumbai-City,1
Pimpri-Chinchwad-City,0
Kalyan,1
Dombivli,1
Murbad,1
Yavat,1
Katol,1
Kamptee,1
Umred,1
Bhandara-City,1
Gondia-City,1
Gadchiroli-City,0
Wardha-City,1
Chandrapur-City,0
Usmanabad,1
Khandesh-Division,1
Marathwada-Region,0
Vidarbha-Region,1
Konkan-Region,1
Desh-Region,1
North-Maharashtra,1
Talegaon MIDC belt,0
Ratnagiri-Sindhudurg,0
Satara-Kolhapur,1
Solapur-Sangli,1"""

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def load_file(path):
    """Loads CSV or Excel safely."""
    try:
        if path.endswith('.csv'): return pd.read_csv(path)
        elif path.endswith(('.xls', '.xlsx')): return pd.read_excel(path)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {path}")
        sys.exit()
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        sys.exit()

def get_label_col(df):
    """Finds the label column (slums, score, etc)."""
    for col in df.columns:
        if str(col).lower().strip() in ['slums', 'livability score', 'score', 'label']:
            return col
    return df.columns[1] # Fallback to 2nd column

def clean_val(val):
    """Converts 0.0, '0', etc. to integer."""
    try:
        return int(float(str(val).strip()))
    except:
        return -1

# ==========================================
# 3. PROCESSING & SYNCING
# ==========================================
print("🚀 Loading Files...")
# Load Master Reference
master_df = pd.read_csv(io.StringIO(master_csv_data))
master_df['Location_Name'] = master_df['Location_Name'].astype(str).str.strip()
master_df['_key'] = master_df['Location_Name'].str.lower()

# Load User Files
df1_raw = load_file(FILE_1_PATH)
df2_raw = load_file(FILE_2_PATH)
df3_raw = load_file(FILE_3_PATH)

print("🔄 Synchronizing Data...")
# Function to sync a dataframe to the master list
def sync_data(master, target, label_col):
    target['_key'] = target[target.columns[0]].astype(str).str.strip().str.lower()
    merged = pd.merge(master[['Location_Name', '_key']], target[['_key', label_col]], on='_key', how='left')
    return merged[label_col].fillna(-1).apply(clean_val) # Fill missing with -1

# Extract columns and sync
h1_col = get_label_col(df1_raw)
h2_col = get_label_col(df2_raw)
h3_col = get_label_col(df3_raw)

master_df['H1'] = sync_data(master_df, df1_raw, h1_col)
master_df['H2'] = sync_data(master_df, df2_raw, h2_col)
master_df['H3'] = sync_data(master_df, df3_raw, h3_col)

# ==========================================
# 4. ANALYSIS & GRAPH
# ==========================================
def check_status(row):
    votes = [row['H1'], row['H2'], row['H3']]
    if -1 in votes: return "Missing Data"
    
    # Check for unanimity
    if len(set(votes)) == 1:
        return "Unanimous"
    
    # Check for Majority (2 vs 1)
    return "Majority"

master_df['Consensus'] = master_df.apply(check_status, axis=1)

# Filter out missing data for stats
valid_df = master_df[master_df['Consensus'] != "Missing Data"]
counts = valid_df['Consensus'].value_counts()

print("\n📊 Generating Graph...")
plt.figure(figsize=(8, 5))
colors = ['#27ae60', '#f39c12', '#c0392b']
bars = plt.bar(counts.index, counts.values, color=colors[:len(counts)])
plt.title("Human Verification Agreement Levels")
plt.ylabel("Number of Locations")
plt.grid(axis='y', alpha=0.3)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), int(bar.get_height()), ha='center', va='bottom', fontweight='bold')
plt.show()

# ==========================================
# 5. LATEX GENERATION (FIXED SYNTAX)
# ==========================================
print("\n📝 Generating LaTeX...")

total = len(valid_df)
unanimous = len(valid_df[valid_df['Consensus'] == "Unanimous"])
majority = len(valid_df[valid_df['Consensus'] == "Majority"])
u_pct = (unanimous / total * 100) if total > 0 else 0
m_pct = (majority / total * 100) if total > 0 else 0

latex_code = r"""
% PACKAGES REQUIRED: \usepackage{booktabs}, \usepackage{xcolor}, \usepackage{pifont}

\begin{table}[htbp]
\centering
\caption{Inter-Annotator Agreement Statistics}
\label{tab:agreement}
\begin{tabular}{l c c}
\toprule
\textbf{Category} & \textbf{Count} & \textbf{Percentage} \\
\midrule
Unanimous Agreement (3/3) & """ + str(unanimous) + r""" & """ + f"{u_pct:.1f}" + r"\% \\" + r"""
Majority Consensus (2/3) & """ + str(majority) + r""" & """ + f"{m_pct:.1f}" + r"\% \\" + r"""
\midrule
\textbf{Total Valid Samples} & \textbf{""" + str(total) + r"""} & \textbf{100\%} \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Detailed Human Comparison (First 15 Rows)}
\label{tab:detailed_comparison}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{l c c c | l}
\toprule
\textbf{Location} & \textbf{H1} & \textbf{H2} & \textbf{H3} & \textbf{Consensus} \\
\midrule
"""

for i, row in valid_df.head(15).iterrows():
    loc = str(row['Location_Name']).replace('&', r'\&')
    h1, h2, h3 = row['H1'], row['H2'], row['H3']
    
    if row['Consensus'] == "Unanimous":
        status = r"\textcolor{green!60!black}{\textbf{Unanimous}}"
    else:
        status = r"\textcolor{orange}{\textbf{Majority}}"
        
    latex_code += f"{loc} & {h1} & {h2} & {h3} & {status} \\\\\n"

latex_code += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""

print(latex_code)
print("\n✅ DONE! Copy the LaTeX above.")