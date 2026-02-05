conda create -n rDIQKD python=3.8 -y
echo "Conda environment 'rDIQKD' created."
conda activate rDIQKD
echo "Updating pip and installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "All packages installed successfully in the 'rDIQKD' environment."
else
    echo "There was an error installing the packages."
fi