# NFL Play Type Prediction

## Project Overview
This project applies machine learning techniques to classify NFL plays as either "Run" or "Pass" based on game situations. Two models were tested—XGBoost and Random Forest—using over 33,000 rows of real NFL play-by-play data.

## Data Overview
- **Training Set:** 33,957 rows, 18 columns  
- **Test Set:** 1,643 rows, 27 columns  
- Missing values handled via **median/mode** imputation and column removal based on null ratio. Outliers removed from **'TO GO'** variable.

## Feature Engineering
- **Remain_Time_in_Seconds:** Converts remaining time in the quarter to seconds for numerical analysis.
- **red_zone:** Indicates if the team is within the 20-yard line (1 = in red zone, 0 = outside).
- **late_quarter:** Flags plays occurring in the last 2 minutes of a quarter (1 = late, 0 = early).
- **TO_GO_DOWN_INTERACTION:** Multiplies yards to go by current down to capture play urgency and context.
- **previous_pass_attempts:** Counts the number of previous pass plays by the team in the current game.
- **previous_run_attempts:** Counts the number of previous run plays by the team in the current game.

## Exploratory Data Analysis
![EDA](https://github.com/MethapatV/NFL-Play-Type-Prediction/blob/main/EDA_NFL.png)
The analysis shows that teams run and pass at nearly equal rates in the red zone, disproving the initial hypothesis that running is favored. Outside the red zone, passing is more common. These findings suggest that proximity to the end zone does not significantly influence play-calling tendencies, and that teams maintain a balanced strategy regardless of field position.

## Models
- XGB Classifier
- Random Forest

**Building Best Model**
XGBoost and Random Forest are both effective tree-based models, each with its own strengths. While Random Forest is faster and more interpretable, XGBoost typically provides higher accuracy by capturing complex patterns through sequential learning and regularization. After testing both models, XGBoost outperformed Random Forest in predictive accuracy. Therefore, XGBoost was selected as the final model for this project.

## XGB Classifier
![XGB](https://github.com/MethapatV/NFL-Play-Type-Prediction/blob/main/CONF_NFL.png)

The confusion matrix shows that the model predicts "Run" plays more accurately than "Pass" plays, with an overall accuracy of 66.5%. Precision, recall, and F1-score for "Run" are 72%, while those for "Pass" are lower at 58%, indicating a bias toward predicting "Run." This imbalance may result from class distribution, limited "Pass"-specific features, or model configuration. To improve performance, techniques like class balancing, enhanced feature engineering, and hyperparameter tuning are recommended.

A Variance Inflation Factor (VIF) analysis was also performed to assess multicollinearity among input variables. The highest VIF value observed was 10 (Quarter), followed by To Go (9.7), TO_GO_DOWN_INTERACTION (8.9), and Down (7.9). While these values are on the higher end, they remain within the commonly accepted threshold (VIF < 10), suggesting that multicollinearity is present but not severe. Monitoring and potentially combining related features could still enhance model robustness and interpretability.

## Insight
![Feature Importance](images/pm25_prediction_plot.png)
The XGBoost model identified Remain_Time_in_Seconds as the most influential feature in predicting play calls, emphasizing the impact of time pressure in late-game situations. YARD LINE 0-100 followed, showing the importance of field position, especially near the end zone. Historical trends, like previous_pass_attempts and previous_run_attempts, also ranked highly, reflecting teams’ behavioral patterns. The TO_GO_DOWN_INTERACTION feature provided more predictive value than TO GO or DOWN alone, while QUARTER was found to be the least impactful. These findings suggest that situational context and time dynamics are key drivers in play-calling, and refining top-ranked features may further improve model performance.

## Author
Methapat Vorakamolpisit  
Master's in Analytics – Northeastern University
