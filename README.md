Matthew Chimitt
mmc200005
Project Readme

## LINK TO DATASET: https://finance.yahoo.com/quote/DELL/history?p=DELL 
## Link to Where I host it: https://mchimitt.github.io/DELL_STOCKS.csv

How to build and run the code:
1) Create a virtual environment
- in the command line, run "pip install virtualenv"
- run "python -m venv venv" to create an environment called env
- run ".\venv\Scripts\activate" to enter into the virtual environment

2) install the necessary packages
- pandas - pip install pandas
- Numpy - pip install numpy
- DateTime - pip install DateTime
- Matplotlib - pip install matplotlib
- SkLearn - pip install -U scikit-learn
- tabulate (to create the table) - pip install tabulate

3) Running lstm.py
- Inside the virtual environment, run "python lstm.py"
- When running, the terminal will display the current experiment, which upon opening the file, should just be the model with the best hyper parameters.
    - TO VIEW THE OTHER EXPERIMENTS, UNCOMMENT THEM INSIDE OF THE MAIN METHOD. I commented them because many of the experiments take at least 5 minutes to finish, and there are 21 total experiments.
    The logs of these experiments are found in the Log file in my submission.
- After the model trains, the code will show the MSE results, the MSE Curve, the Error Curve, and the actual vs predicted data curve for both the training and testing data. 
    - FOR EACH MATPLOTLIB GRAPH: when it displays, you must close the graph for the code to continue.
- If you do choose to uncomment each of the other 20 experiments, upon completion of all of the experiments a table of results will be printed. This table matches the tables found in the report and in the Log.