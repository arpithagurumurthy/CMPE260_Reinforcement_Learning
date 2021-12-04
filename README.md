# CMPE260_Reinforcement_Learning
## Deep Reinforcement Learning for Automated Stock Trading

Stock trading plays an important role in making investment decisions. Automated stock trading is extremely beneficial for investment firms, hedge funds or any individual with an avid interest in trading. The goal of this project is to maximize the return on investments while minimizing the risks. Given the unpredictabilities and complications associated with the stock market, it is a daunting task to design such a system. The goal of Deep Reinforcement Learning (DRL) is to maximize the total rewards by taking into account all future actions over a period of time. Stock trading can be modeled as a partially observable Markov Decision Process as it involves a lot of unknowns, hence DRL can be successfully applied to solve this use case.

## Steps to run

After cloning the repo, run the following to install dependencies:

```pip3 install -r requirements.txt```

To train the DQN agent:

```python3 <train_file>.py --model_name=custom_value --stock_name=custom_value --window_size=custom_value --num_episode=custom_value --initial_balance=custom_value```

If we don't specify the custom_values in the arguments, default values that are set in the train file are automatically passed.

We then use our streamlit app to evaluate the models against the below 4 datasets:
* Apple (AAPL_2020-2021)
* Facebook (FB_2020-2021.csv)  
* Nasdaq Composite (NasdaqComposite_2020-2021)
* ^GSPC_2018

To run the streamlit application, execute the following command to select the dataset, model and window size:
```streamlit run app.py```

Below are the sample execution results for 'FB_2020-2021' on model 'DQN_v1':

* Streamlit results:

<img src="https://github.com/arpithagurumurthy/CMPE260_Reinforcement_Learning/blob/main/Screenshots/Streamlit_app.png">

* Execution summary:

<img src="https://github.com/arpithagurumurthy/CMPE260_Reinforcement_Learning/blob/main/Screenshots/Execution_results.png" width=500>








