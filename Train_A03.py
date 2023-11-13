from General_A03 import *
import A03

out_dir = base_dir + "/" + "output_wbc"

###############################################################################
# MAIN
###############################################################################

def main():
    # Load datasets
    train_data, test_data = load_and_prepare_BCCD_data()
    
    # Do directory exist?
    if os.path.exists(out_dir):
        check_overwrite = input("Output folder exists; do you wish to overwrite it? (y/n) ")
        if check_overwrite == "y":
            shutil.rmtree(out_dir)
        else:
            print("Exiting...")
            exit(1)
            
    # Create output directory
    os.makedirs(out_dir)
        
    # Predict for training
    train_metrics = predict_dataset(train_data, "TRAIN", out_dir, 
                                    BCCD_TYPES.WBC.value, A03.find_WBC)

    # Predict for testing
    test_metrics = predict_dataset(test_data, "TEST", out_dir, 
                                   BCCD_TYPES.WBC.value, A03.find_WBC)

    # Save metrics
    print_metrics(train_metrics, test_metrics)
    with open(out_dir + "/RESULTS_WBC.txt", "w") as f:
        print_metrics(train_metrics, test_metrics, f)
    
if __name__ == "__main__": 
    main()
    