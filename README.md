# APML Mini Project
Small project from the [Advanced Probibalistic Machine Learning course](https://www.uu.se/en/admissions/freestanding-courses/course-syllabus/?kKod=1RT705&lasar=) at Uppsala University. Mathematical calculations and evaluation of results can be found in the [report](https://github.com/johanssonkarin/apml-miniproject/blob/master/APML_project_report.pdf).

The purpose was work through the whole process of solving a real-world problem using probabilistic machine learning. The task was to estimate the skill of teams involved in a competition based on the results of matches between pairs of teams. My group and I first defined a probabilistic model, where all the quantities were represented as random variables, based on the Trueskill Bayesian ranking system developed by Microsoft research for raking online matches. The model assigns a skill to each player, in the form of a Gaussian random variable. When two teams meet in a match, the outcome of that match is a Gaussian random variable with mean equal to the difference between the two teams’ skills. The result of the match is 1 (indicating the victory of Team 1) if the outcome is greater than zero, -1 if it is less than zero (indicating the victory of Team 2).

We used Bayesian inference to find the posterior distribution of the teams’ skills given observations of the results of matches. Because the posterior distribution was intractable, we used two different approximation methods based on graphical models.

The SerieA.csv dataset containing the results of the Italian 2018/2019 Serie A elite football (soccer) division was primarily used. However, the Trueskill model can be used to solve a variety of skill and ranking problems. Later in the project we also tried the model on a NBA dataset. 

