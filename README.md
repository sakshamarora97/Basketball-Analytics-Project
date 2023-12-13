# Basketball Analytics Project
Using ML to generate expected PPS, opportunity grade classification, and prescription analysis for users.


# Exploratory Data Analysis
We explored the multiple pickle files we had to work with, to better understand the variables available. We explored the variables' ranges, mean and median values, variances, distributions, data types, and missing values, as well as the relationships between the variables.
Moreover, this part was very helpful for the project in total, as it inspired many ideas for the feature engineering stage.

## Data Discovery Highlights
### Example 1 Misspecified Zones

In some cases, outliers were incorrect data points and they had to be either corrected or removed, leading to the next stage of the project, the data cleaning part.
As an example, in the x & y coordinates strip plots shown here, some zones have many data points that seem to belong in different zones: e.g. x_coordinate of 5-2 zone and y_coordinate of 4-2 zone

![image](https://github.com/sakshamarora97/basketball-analytics-project/assets/62840042/3b8a06da-7fb0-4321-a441-ea6b9660a3db)

### Example 2 Misspecified 2 Pointers

In some cases, visualizing the data was really helpful in understanding the problem with the data and how to fix it, if possible.
Plotting all 2-point shots –
The shots marked in the red rectangle can’t be 2 pointer shots
![image](https://github.com/sakshamarora97/basketball-analytics-project/assets/62840042/358bbfd2-d53d-4dea-896d-4bce1fb145f6)

### Example 3 Incorrectly tagged Fastbreaks
Another major issue we discovered was that when 3-point shots were labeled as "Fastbreak", their accuracy was approximately 98%, which can by no means be true. However, since the number of those data points was around 8,000, we decided to keep them in the data to avoid losing the rest of the information contained in those rows, and just exclude the "Fastbreak" variable from the features used in the model for the 3-point shots.

<img width="488" alt="image" src="https://github.com/sakshamarora97/basketball-analytics-project/assets/62840042/f5ab2082-1b47-4410-8884-dd2cd3a9ca50">

### Time remaining matters!
Another interesting insight we came across is that there’s a high peak in the number of three pointers missed in the last few seconds of a period. That’s because when time on the clock is running out, many teams will attempt a desperation three from long range, just to not end up with the ball in their hands, or even if the ball it’s closer to the 3point line, the shot will be heavily contested because it’s easy for the defense to predict that shot.

<img width="488" alt="image" src="https://github.com/sakshamarora97/basketball-analytics-project/assets/62840042/bc71a4e1-93a0-461a-a58a-d3ded90a2816">

