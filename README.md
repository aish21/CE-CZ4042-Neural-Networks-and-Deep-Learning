# CE-CZ4042-Neural-Networks-and-Deep-Learning
Repository for NTU Course - CE/CZ 4042 - Neural Networks and Deep Learning
In recent times, social networking has become a huge part of our day-to-day activities. From WhatsApp to Instagram, we always stay connected and updated about the people whom we care about. One such social networking service is Twitter that has connected people from different parts of the world and supported numerous businesses. On this platform, users post and interact with messages known as “tweets”. Users are exposed to tweets that are of various categories such as excited, angry, sad, happy, etc. Thus, our team felt the need to identify the sentiment of a given tweet to help the user identify the type of content they are being exposed to on a daily basis. Hence, we decided to use the Long Short Term Memory networks to fulfill our mission. 

The original data can be found in '/data/Original Data'. To follow along in the training process, please follow the following order:
- `/scripts/Data-Cleaning.ipynb`
- `/scripts/Exploratory-Data-Analysis.ipynb`
- `/scripts/Feature-Engineering.ipynb`
- `/scripts/LSTM-Vanilla-First-Trial.ipynb`
- Following the above Jupyter Notebook, please execute the following files for model construction, training and tuning.
```bash 
python LSTM-Trial-Architectures.py       # Train different model architectures to see model progression
python LSTM-Final.py                     # Train the final model architectures ( with 9 and 3 labels )
python Hyperparameter-Tuning.py          # Identify the best Hyperparameters for the final model
```
- For post - processing, execute `/scripts/Postprocessing.ipynb`
