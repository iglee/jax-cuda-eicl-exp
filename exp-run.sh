while true; do
    # Run the Python script
    python -m emergent_in_context_learning.experiment.experiment --config emergent_in_context_learning/experiment/configs/images_all_exemplars.py --jaxline_mode train --logtostderr
    
    # Check the exit status
    if [ $? -eq 0 ]; then
        echo "Script succeeded."
        break
    else
        echo "Script failed. Retrying..."
    fi
done