# Basketball Analytics Project
Using ML to generate expected PPS, opportunity grade classification, and prescription analysis for users.


# Exploratory Data Analysis
We explored the multiple pickle files we had to work with, to better understand the variables available. We explored the variables' ranges, mean and median values, variances, distributions, data types, and missing values, as well as the relationships between the variables.
Moreover, this part was very helpful for the project in total, as it inspired many ideas for the feature engineering stage.

## Data Discovery Highlights
### Example 1: Misspecified Zones

In some cases, outliers were incorrect data points and they had to be either corrected or removed, leading to the next stage of the project, the data cleaning part.
As an example, in the x & y coordinates strip plots shown here, some zones have many data points that seem to belong in different zones: e.g. x_coordinate of 5-2 zone and y_coordinate of 4-2 zone

![image](https://github.com/sakshamarora97/basketball-analytics-project/assets/62840042/3b8a06da-7fb0-4321-a441-ea6b9660a3db)

### Example 2: Misspecified 2 Pointers

In some cases, visualizing the data was really helpful in understanding the problem with the data and how to fix it, if possible.
Plotting all 2-point shots –
The shots marked in the red rectangle can’t be 2 pointer shots
![image](https://github.com/sakshamarora97/basketball-analytics-project/assets/62840042/358bbfd2-d53d-4dea-896d-4bce1fb145f6)

### Example 3: Incorrectly tagged Fastbreaks
Another major issue we discovered was that when 3-point shots were labeled as "Fastbreak", their accuracy was approximately 98%, which can by no means be true. However, since the number of those data points was around 8,000, we decided to keep them in the data to avoid losing the rest of the information contained in those rows, and just exclude the "Fastbreak" variable from the features used in the model for the 3-point shots.

<img width="488" alt="image" src="https://github.com/sakshamarora97/basketball-analytics-project/assets/62840042/f5ab2082-1b47-4410-8884-dd2cd3a9ca50">

### Example 4: Time remaining matters!
Another interesting insight we came across is that there’s a high peak in the number of three pointers missed in the last few seconds of a period. That’s because when time on the clock is running out, many teams will attempt a desperation three from long range, just to not end up with the ball in their hands, or even if the ball it’s closer to the 3point line, the shot will be heavily contested because it’s easy for the defense to predict that shot.

<img width="471" alt="image" src="https://github.com/sakshamarora97/basketball-analytics-project/assets/62840042/115c068f-88ac-4adc-bd91-c583a16c4dbe">

# Feature Engineering

We experimented with a wide variety of features to use in our model, falling within three main categories: player attributes, shot attributes, and team attributes.
|                          |     Feature                                                                                                                      |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------|
|     Player attributes    |     Height,   Position, College Year                                                                                             |
|                          |     Past   FG% from zone x (Last 1/3/5/10 games, current season, previous season)                                                |
|                          |     Past   FG% for type of shot x (Last 1/3/5/10 games, current season, previous season)                                         |
|                          |     Number   of makes and attempts per game from zone x (Last 1/3/5/10 games, current   season, previous season)                 |
|                          |     Number   of makes and attempts per game for type of shot x (Last 1/3/5/10 games,   current season, previous season)          |
|     Shot attributes      |     Shot   distance                                                                                                              |
|                          |     Shot   angle                                                                                                                 |
|                          |     Shot type                                                                                                                    |
|                          |     Zone                                                                                                                         |
|                          |     Other   tags (fastbreak, off turnover, second chance, paint, etc.)                                                           |
|                          |     X-coordinate                                                                                                                 |
|                          |     Y-coordinate                                                                                                                 |
|                          |     Period                                                                                                                       |
|                          |     Number of Big, Small, Med players in opposition   team on court                                                              |
|                          |     Time remaining on clock                                                                                                      |
|                          |     Score   differential                                                                                                         |
|                          |     Clutch time: True/False                                                                                                      |
|                          |     Past Avg. usage rates of shooter (Last 1/3/5/10 games,   current season, previous season)                                    |
|     Team attributes      |     Opponent team past Win/Loss ratio (Last   1/3/5/10 games, current season, previous season)                                   |
|                          |     Team past Win/Loss ratio (Last   1/3/5/10 games, current season, previous season)                                            |
|                          |     Number of high blocking rate opponents on court     Division I team: True/False                                              |
|                          |     Location:   Home/Away/Neutral                                                                                                |


# Modeling
Using the created features, we modeled the problem as binary classification problem and used a train validation split of 80%-20% 
Logistic regression and decision tree can’t include features with missing data and we didn’t want to introduce bias with imputation (hence reduced feature set).


### Recursive Feature Elimination
To reduce the number of fitting (to improve model generalizability), we performed recursive feature elimination. We determined the number of features post which adding additional features didn’t improve model performance. We took these features for our process ahead. As you can see from the illustration below, in the 3 pointer model, the train performance keeps on gradually improving. We chose 25 features for the next step. In the 2 pointer model, we see that the performance clearly saturates in the train and test, we chose 26 features to take ahead. 

3 pointer recursive feature elimination (NCAAM1)

<img width="331" alt="image" src="https://github.com/sakshamarora97/basketball-analytics-project/assets/62840042/ddb02a57-dd6b-43cb-9214-21ed519c97e5">

2 pointer recursive feature elimination (NCAAM1)

<img width="332" alt="image" src="https://github.com/sakshamarora97/basketball-analytics-project/assets/62840042/39d66016-e04d-4973-8206-f0b50d6ef732">


### Correlation treatment 
Post this step we performed multi-collinearity treatment, by iteratively removing features with correlation higher than the threshold one at a time. Between 2-5 features were removed for each model

### Results
2 pointer model
|     Model                  |     Feature Set                                        |     Train F1    |     Train ROC-AUC    |     Test F1    |     Test ROC-AUC    |
|----------------------------|--------------------------------------------------------|-----------------|----------------------|----------------|---------------------|
|     Logistic Regression    |     Only including features with no   missing data     |     63.8        |     62.8             |     63.7       |     62.8            |
|     Decision Tree          |     Only including features with no   missing data     |     64.2        |     66.4             |     62.2       |     64.6            |
|     LightGBM               |     Only including features with no   missing data     |     68.9        |     72.1             |     68.6       |     71.6            |
|     LightGBM               |     Full Feature Set                                   |     69.2        |     73               |     68.9       |     72.2            |
|     LightGBM               |     Most important features     (final model)          |     69.1        |     72.8             |     68.8       |     72.1            |

3 pointer model
|     Model       |     Feature Set                                        |     Train F1    |     Train ROC-AUC    |     Test F1    |     Test ROC-AUC    |
|-----------------|--------------------------------------------------------|-----------------|----------------------|----------------|---------------------|
|     LightGBM    |     Full Feature Set                                   |     51.1        |     60.2             |     50.9       |     55              |
|     LightGBM    |     Most important features     (final model)          |     51.2        |     59.2             |     50.8       |     54.9            |
|     LightGBM    |     Only including features with no   missing data     |     68.9        |     72.1             |     68.6       |     71.6            |
