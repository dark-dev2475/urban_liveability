"""
run_pipeline.py -- Execute the entire Urban Liveability Pipeline
================================================================
Runs all steps in sequence: NLP scoring -> dataset build -> augmentation -> training -> evaluation.
Usage: python run_pipeline.py
"""
import os, sys, time, io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    start = time.time()
    print("\n" + "=" * 60)
    print("  URBAN LIVEABILITY MULTI-MODAL PIPELINE")
    print("=" * 60 + "\n")
    
    # Step 1: NLP Amenity Scores
    print("\n" + "-" * 60)
    from pipeline.step_01_nlp_amenity_scores import process_all_places as step1
    step1()
    
    # Step 2: NLP Context Scores
    print("\n" + "-" * 60)
    from pipeline.step_02_nlp_context_scores import process_all_places as step2
    step2()

    # Step 2b: Satellite Scores
    print("\n" + "-" * 60)
    from pipeline.step_02b_satellite_scores import generate_satellite_scores as step2b
    step2b()
    
    # Step 3: Build Master Dataset
    print("\n" + "-" * 60)
    from pipeline.step_03_build_master_dataset import build_master_dataset as step3
    step3()
    
    # Step 4: Data Augmentation
    print("\n" + "-" * 60)
    from pipeline.step_04_data_augmentation import augment_dataset as step4
    step4()
    
    # Step 5: Train Models
    print("\n" + "-" * 60)
    from pipeline.step_05_train_models import train_all_models as step5
    step5()
    
    # Step 6: Evaluate & Compare
    print("\n" + "-" * 60)
    from pipeline.step_06_evaluate import run_evaluation as step6
    step6()
    
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE -- {elapsed:.1f}s total")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Scores:  pipeline_output/")
    print(f"  Models:  models/")
    print(f"  Figures: figures/")
    print(f"\nTo predict a place:")
    print(f"  python pipeline/step_07_predict.py [place_name]")

if __name__ == "__main__":
    main()
