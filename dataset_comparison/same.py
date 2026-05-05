import pandas as pd
import io
import os
import sys

# ==========================================
# 1. THE NEW MASTER DATA (Pasted from your request)
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
Solapur-Sangli,1
Bandra Kurla Complex,0
Dadar,1
Mahim,1
Matunga,1
Lower Parel,1
Prabhadevi,1
Parel,1
Shivaji Park area,1
Andheri West,1
Juhu,1
Versova,1
Powai,1
Vikhroli,0
Dahisar,0
Colaba,1
Cuffe Parade,1
Fort,1
Churchgate,1
Marine Drive area,1
Malabar Hill,1
Tardeo,1
Byculla,0
Chembur,0
Goregaon,1
Malad,1
Ville Parle,1
Mira Road,1
Bhayandar,1
Vasai,0
Kalyan,0
Dombivli,0
Badlapur,0
Thane city (Ghodbunder),1
Majiwada,1
Vasant Vihar belt,1
Navi Mumbai (Vashi),1
Nerul,1
Kharghar,1
Seawoods,1
Belapur,1
Airoli,1
Ghansoli,0
Koparkhairane,0
Taloja,0
Shivajinagar,1
Deccan Gymkhana,1
Kothrud,1
Karve Nagar,1
Baner,1
Balewadi,1
Aundh,1
Pashan,1
Wakad,1
Hinjewadi,0
Kharadi,1
Viman Nagar,1
Koregaon Park,1
Kalyani Nagar,1
Magarpatta,1
Hadapsar,0
Yerwada,0
Camp,1
Fatima Nagar,0
Mundhwa,0
Wagholi,0
Bavdhan,1
Pimpri,1
Chinchwad,1
Nigdi,1
Talegaon Dabhade,1
Lonavala,0
Chakan,0
Rajgurunagar,0
Mumbai,1
Thane,1
Navi Mumbai,1
Ahmednagar,0
Jalgaon,0
Nandurbar,0
Akola,0
Wardha,0
Bhandara,0
Gondia,0
Osmanabad,0
Sindhudurg,1
Alibaug,0
Malshej Ghat area,0
Igatpuri,0
Kamshet,0
Tuljapur,0
Kankavli,0
Vengurla,1"""

# ==========================================
# 2. CONFIGURATION (UPDATE PATHS HERE)
# ==========================================
# Use 'r' before the string to avoid errors
FILE_1_PATH = r'C:\Users\gagan\OneDrive\Desktop\urban_bot\dataset_comparison\file1.csv'
FILE_2_PATH = r'C:\Users\gagan\OneDrive\Desktop\urban_bot\dataset_comparison\file2.csv'
FILE_3_PATH = r'C:\Users\gagan\OneDrive\Desktop\urban_bot\dataset_comparison\file3.xlsx'

# What is the header name of the Name column in your EXISTING files?
# (e.g., 'Location_Name' or 'place_name')
OLD_NAME_COL = 'Location_Name' 

# What is the header name of the Label/Score column in your EXISTING files?
# (e.g. 'slums', 'Livability Score', 'Score')
OLD_LABEL_COL = 'slums' 

# ==========================================
# 3. PROCESSING LOGIC
# ==========================================
def process_synchronization():
    print("🚀 Starting Process...")

    # 1. Load the NEW Master Reference
    master_df = pd.read_csv(io.StringIO(master_csv_data))
    # Standardize Master Names (Strip whitespace)
    master_df['Location_Name'] = master_df['Location_Name'].astype(str).str.strip()
    
    # Create a helper key for matching (lowercase)
    master_df['_key'] = master_df['Location_Name'].str.lower()

    # 2. Function to Sync a single file
    def sync_file(filepath, file_label):
        print(f"\n📂 Processing {file_label}...")
        
        # Load File
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")
            return

        # Check Columns
        if OLD_NAME_COL not in df.columns:
            # Fallback: Try identifying column 0
            print(f"⚠️ Column '{OLD_NAME_COL}' not found. Using first column as Name.")
            df.rename(columns={df.columns[0]: OLD_NAME_COL}, inplace=True)

        if OLD_LABEL_COL not in df.columns:
            # Fallback: Try identifying column 1
            print(f"⚠️ Column '{OLD_LABEL_COL}' not found. Using second column as Label.")
            df.rename(columns={df.columns[1]: OLD_LABEL_COL}, inplace=True)

        # Standardize Target Names
        df['_key'] = df[OLD_NAME_COL].astype(str).str.strip().str.lower()

        # 3. MERGE (The Magic Step)
        # Left Merge on Master: Keeps Master rows, pulls data from Target
        merged = pd.merge(master_df[['Location_Name', '_key']], 
                          df[['_key', OLD_LABEL_COL]], 
                          on='_key', 
                          how='left')

        # 4. Clean up
        final_df = merged.drop(columns=['_key'])
        
        # Fill missing values (If the old file didn't have "Mumbai", what score?)
        # We will mark it as "MISSING" so you can spot it, or fill with 0
        final_df[OLD_LABEL_COL] = final_df[OLD_LABEL_COL].fillna('MISSING')

        # Save
        folder = os.path.dirname(filepath)
        new_filename = f"{file_label}_synced_final.csv"
        save_path = os.path.join(folder, new_filename)
        
        final_df.to_csv(save_path, index=False)
        print(f"✅ Saved: {new_filename}")

    # Run for all 3 files
    sync_file(FILE_1_PATH, "file1")
    sync_file(FILE_2_PATH, "file2")
    sync_file(FILE_3_PATH, "file3")

    # Also save the Master Reference itself as a CSV for your records
    master_save_path = os.path.join(os.path.dirname(FILE_1_PATH), "MASTER_REFERENCE.csv")
    master_df[['Location_Name', 'Livability Score']].to_csv(master_save_path, index=False)
    print(f"\n✅ Saved Master Reference to: {master_save_path}")

if __name__ == "__main__":
    process_synchronization()